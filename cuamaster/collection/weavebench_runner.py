"""Compatibility wrapper that leaves the imported WeaveBench tree untouched."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import struct
import sys
import textwrap
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from cuamaster.benchmarks import (
    claude_code_tool_policy,
)
from cuamaster.collection.collection_policy import append_collection_policy
from cuamaster.endpoints import DynamicEndpointPool

logger = logging.getLogger("cuamaster.collection.weavebench_runner")

_SHELL_SCREENSHOT_HOOK = (
    Path(__file__).resolve().parent
    / "assets"
    / "claudecode_shell_screenshot_hook.py"
)
_REMOTE_SHELL_SCREENSHOT_HOOK = (
    "/tmp_workspace/_trajectory/claudecode_shell_screenshot_hook.py"
)
_REMOTE_SHELL_EVENT_LOG = "/tmp_workspace/_trajectory/shell_events.jsonl"
_REMOTE_SHELL_SCREENSHOT_DIR = "/tmp_workspace/_shell_screenshots"
_ENDPOINT_FAILURE_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "timeout",
    "timed out",
    "connection error",
    "connection refused",
    "service unavailable",
    "internal server error",
    "internal_server_error",
    "bad gateway",
    "gateway timeout",
    "retry_exhausted",
    "empty_run",
    "api error",
    "model request",
    "openclaw runtime exited non-zero",
    "agent runner exited non-zero",
)
_NON_ENDPOINT_FAILURE_MARKERS = (
    "/setup/execute",
    "/setup/upload",
    "failed to download",
    "failed to upload file",
    "no space left on device",
    "osworld_evaluator_failed",
    "osworld setup command",
    "setup execute failed",
    "setup step ",
)
_SLOW_SETUP_TIMEOUT_SECONDS = 600
_OPENCLAW_BOOTSTRAP_HARDENING_MARKER = "# gui-trajectory openclaw bootstrap hardening"
_endpoint_pool_instance: DynamicEndpointPool | None = None
_proxy_secret_filter = None
_worker_monitor_failure: str | None = None


class VMTransportUnavailable(RuntimeError):
    """Raised when an active VM lease no longer serves controller requests."""


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _is_vm_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, VMTransportUnavailable):
        return True
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        status = getattr(exc.response, "status_code", None)
        return status in {500, 502, 503, 504}
    return False


def _vm_transport_context(env) -> dict[str, object]:
    provider = getattr(env, "provider", None)
    return {
        "task_id": getattr(env, "_gui_trajectory_task_id", None),
        "profile_id": os.environ.get("WEAVEBENCH_VM_PROFILE_ID"),
        "vm_base_url": getattr(env, "vm_base_url", None),
        "resource_id": getattr(provider, "resource_id", None),
        "endpoint_host": getattr(provider, "endpoint_host", None),
        "ports": dict(getattr(provider, "ports", {}) or {}),
    }


def _vm_command_label(command: object) -> str:
    if not isinstance(command, (list, tuple)):
        return type(command).__name__
    values = [str(value) for value in command[:2]]
    return " ".join(values) or "<empty>"


def _vm_transport_failure(
    env,
    operation: str,
    exc: BaseException,
    *,
    command: object = None,
) -> VMTransportUnavailable:
    context = _vm_transport_context(env)
    details = [
        f"operation={operation}",
        f"task_id={context['task_id'] or '<unknown>'}",
        f"profile_id={context['profile_id'] or '<unknown>'}",
        f"vm_base_url={context['vm_base_url'] or '<unknown>'}",
        f"resource_id={context['resource_id'] or '<unknown>'}",
    ]
    if command is not None:
        details.append(f"command={_vm_command_label(command)}")
    return VMTransportUnavailable(
        "VM transport unavailable (" + ", ".join(details) + f"): {exc}"
    )


def _install_vm_transport_guard(upstream) -> None:
    if getattr(upstream, "_gui_trajectory_transport_guard", False):
        return
    original_vm_exec = getattr(upstream, "_vm_exec", None)
    original_vm_fetch = getattr(upstream, "_vm_fetch", None)
    original_vm_upload = getattr(upstream, "_vm_upload", None)
    original_vm_upload_bytes = getattr(upstream, "_vm_upload_bytes", None)
    if not callable(original_vm_exec) or not callable(original_vm_fetch):
        return

    def guarded_vm_exec(env, command, *args, **kwargs):
        try:
            return original_vm_exec(env, command, *args, **kwargs)
        except Exception as exc:
            if not _is_vm_transport_error(exc):
                raise
            failure = _vm_transport_failure(
                env,
                "execute",
                exc,
                command=command,
            )
            logger.error("%s", failure)
            raise failure from exc

    def guarded_vm_fetch(env, remote, local):
        retries = _positive_int_env("GUI_TRAJECTORY_VM_FETCH_RETRIES", 3)
        backoff = _positive_float_env(
            "GUI_TRAJECTORY_VM_FETCH_RETRY_BACKOFF_SECONDS", 1.0
        )
        last_error: BaseException | None = None
        for attempt in range(1, retries + 1):
            try:
                return original_vm_fetch(env, remote, local)
            except Exception as exc:
                if not _is_vm_transport_error(exc):
                    raise
                last_error = exc
                if attempt < retries:
                    logger.warning(
                        "VM artifact fetch transport failure attempt=%d/%d "
                        "remote=%s context=%s: %s",
                        attempt,
                        retries,
                        remote,
                        _vm_transport_context(env),
                        exc,
                    )
                    time.sleep(backoff * (2 ** (attempt - 1)))
        assert last_error is not None
        failure = _vm_transport_failure(env, f"fetch:{remote}", last_error)
        logger.error("%s", failure)
        raise failure from last_error

    def guarded_vm_upload(operation, upload, env, payload, remote, *args, **kwargs):
        retries = _positive_int_env("GUI_TRAJECTORY_VM_UPLOAD_RETRIES", 3)
        backoff = _positive_float_env(
            "GUI_TRAJECTORY_VM_UPLOAD_RETRY_BACKOFF_SECONDS", 2.0
        )
        last_error: BaseException | None = None
        for attempt in range(1, retries + 1):
            try:
                return upload(env, payload, remote, *args, **kwargs)
            except Exception as exc:
                if not _is_vm_transport_error(exc):
                    raise
                last_error = exc
                if attempt < retries:
                    logger.warning(
                        "VM artifact upload transport failure attempt=%d/%d "
                        "remote=%s context=%s: %s",
                        attempt,
                        retries,
                        remote,
                        _vm_transport_context(env),
                        exc,
                    )
                    time.sleep(backoff * (2 ** (attempt - 1)))
        assert last_error is not None
        failure = _vm_transport_failure(env, f"{operation}:{remote}", last_error)
        logger.error("%s", failure)
        raise failure from last_error

    def guarded_vm_upload_text(env, content, remote, *args, **kwargs):
        return guarded_vm_upload(
            "upload", original_vm_upload, env, content, remote, *args, **kwargs
        )

    def guarded_vm_upload_bytes(env, path, remote, *args, **kwargs):
        return guarded_vm_upload(
            "upload-bytes",
            original_vm_upload_bytes,
            env,
            path,
            remote,
            *args,
            **kwargs,
        )

    def guarded_wait_file(env, remote: str, timeout: int) -> bool:
        failure_limit = _positive_int_env(
            "GUI_TRAJECTORY_VM_TRANSPORT_FAILURE_LIMIT", 3
        )
        poll_timeout = _positive_float_env(
            "GUI_TRAJECTORY_VM_TRANSPORT_POLL_TIMEOUT_SECONDS", 10.0
        )
        poll_interval = _positive_float_env(
            "GUI_TRAJECTORY_VM_TRANSPORT_POLL_INTERVAL_SECONDS", 5.0
        )
        deadline = time.monotonic() + timeout
        consecutive_failures = 0
        while time.monotonic() < deadline:
            try:
                result = guarded_vm_exec(
                    env,
                    ["bash", "-c", f"test -f {remote} && echo YES || echo NO"],
                    timeout=min(poll_timeout, max(1.0, deadline - time.monotonic())),
                )
            except VMTransportUnavailable as exc:
                consecutive_failures += 1
                logger.warning(
                    "VM wait probe transport failure consecutive=%d/%d "
                    "remote=%s context=%s",
                    consecutive_failures,
                    failure_limit,
                    remote,
                    _vm_transport_context(env),
                )
                if consecutive_failures >= failure_limit:
                    failure = _vm_transport_failure(
                        env,
                        f"wait:{remote}",
                        exc,
                    )
                    logger.error("%s", failure)
                    raise failure from exc
            else:
                consecutive_failures = 0
                if "YES" in (result.get("output") or ""):
                    return True
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_interval, remaining))
        return False

    guarded_vm_exec._gui_trajectory_transport_guard = True
    guarded_vm_fetch._gui_trajectory_transport_guard = True
    guarded_wait_file._gui_trajectory_transport_guard = True
    upstream._vm_exec = guarded_vm_exec
    upstream._vm_fetch = guarded_vm_fetch
    if callable(original_vm_upload):
        guarded_vm_upload_text._gui_trajectory_transport_guard = True
        upstream._vm_upload = guarded_vm_upload_text
    if callable(original_vm_upload_bytes):
        guarded_vm_upload_bytes._gui_trajectory_transport_guard = True
        upstream._vm_upload_bytes = guarded_vm_upload_bytes
    upstream._wait_file = guarded_wait_file
    upstream._gui_trajectory_transport_guard = True


def _write_vm_transport_failure(
    output_dir: Path,
    env,
    agent,
    exc: BaseException,
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _vm_transport_context(env)
    status = {
        "agent_done": False,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "timed_out": False,
        "exit_code": None,
        "model": getattr(agent, "model", None),
        "gui": bool(getattr(agent, "gui", False)),
        "transport_failure": True,
        "transport_error": f"{type(exc).__name__}: {exc}",
        "vm_transport": context,
    }
    (output_dir / "execution_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    failure = {
        "schema_version": "gui_trajectory_collection.vm_transport_failure.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "error": status["transport_error"],
        **context,
    }
    (output_dir / "vm_transport_failure.json").write_text(
        json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fingerprint_path = output_dir / "environment_fingerprint.json"
    if not fingerprint_path.is_file():
        fingerprint_path.write_text(
            json.dumps(
                {
                    "schema_version": "gui_trajectory_collection.environment.v1",
                    "captured_at": failure["captured_at"],
                    "profile_id": context["profile_id"],
                    "platform_type": None,
                    "probes": {},
                    "transport_failure": failure,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def _worker_aware_monitor(
    progress,
    total: int,
    stop_after: float,
    *,
    poll_interval: float = 2.0,
    active_children_fn=None,
) -> str | None:
    active_children_fn = active_children_fn or multiprocessing.active_children
    seen_index = 0
    finished = 0
    started_at = time.monotonic()
    active_tasks: dict[str, tuple[str, str]] = {}
    known_workers: dict[str, object] = {}
    failures: list[str] = []

    while finished < total and time.monotonic() - started_at < stop_after:
        time.sleep(poll_interval)
        current_length = len(progress)
        for index in range(seen_index, current_length):
            event = progress[index]
            kind = event["event"]
            worker = str(event["name"])
            mode = str(event["mode"])
            task = str(event["task"])
            tag = f"[{worker}][{mode}] {task}"
            if kind == "start":
                active_tasks[worker] = (mode, task)
                logger.info("▶ START  %s", tag)
            elif kind == "skip":
                active_tasks.pop(worker, None)
                logger.info("⊘ SKIP   %s (already done)", tag)
                finished += 1
            elif kind == "done":
                active_tasks.pop(worker, None)
                error = event.get("error")
                if error:
                    logger.warning(
                        "✗ DONE   %s score=%.3f err=%s",
                        tag,
                        event.get("score", 0.0),
                        str(error)[:120],
                    )
                else:
                    logger.info(
                        "✓ DONE   %s score=%.3f", tag, event.get("score", 0.0)
                    )
                finished += 1
            elif kind == "crash":
                active_tasks.pop(worker, None)
                logger.error("☠ CRASH  %s err=%s", tag, event.get("error"))
                finished += 1
        seen_index = current_length

        if finished >= total:
            break

        live_workers = {}
        for process in active_children_fn():
            name = str(getattr(process, "name", ""))
            if re.fullmatch(r"env-\d+", name) and process.is_alive():
                live_workers[name] = process
                known_workers[name] = process

        exited_active_workers = sorted(set(active_tasks) - set(live_workers))
        for worker in exited_active_workers:
            mode, task = active_tasks.pop(worker)
            process = known_workers.get(worker)
            exit_code = getattr(process, "exitcode", None)
            message = (
                "worker exited without a terminal progress event"
                f" (exitcode={exit_code})"
            )
            logger.error("☠ CRASH  [%s][%s] %s err=%s", worker, mode, task, message)
            failures.append(f"{worker}:{task}:{message}")
            finished += 1

        if known_workers and not live_workers and finished < total:
            remaining = total - finished
            message = (
                f"all workers exited with {remaining} run(s) still unaccounted for"
            )
            logger.error(message)
            failures.append(message)
            break

    elapsed = time.monotonic() - started_at
    if finished < total and elapsed >= stop_after:
        message = f"monitor timed out with {total - finished} unfinished run(s)"
        logger.error(message)
        failures.append(message)
    logger.info("monitor exit: finished=%d/%d, elapsed=%.0fs", finished, total, elapsed)
    return "; ".join(failures) or None


def _install_worker_exit_monitor(runner_module) -> None:
    if not callable(getattr(runner_module, "_monitor", None)):
        return

    def monitor(progress, total, stop_after):
        global _worker_monitor_failure
        _worker_monitor_failure = _worker_aware_monitor(progress, total, stop_after)

    runner_module._monitor = monitor


class _ConfiguredProxy:
    def __init__(self, proxy_url: str) -> None:
        parsed = urlparse(proxy_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("legacy OSWorld proxy URL must use http or https")
        if not parsed.hostname or parsed.port is None:
            raise ValueError("legacy OSWorld proxy URL must include host and port")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("legacy OSWorld proxy URL cannot include a path or query")
        self.host = parsed.hostname
        self.port = parsed.port
        self.username = unquote(parsed.username or "")
        self.password = unquote(parsed.password or "")
        self.protocol = parsed.scheme


class _ConfiguredProxyPool:
    def __init__(self, proxy_url: str) -> None:
        self.proxy = _ConfiguredProxy(proxy_url)

    def get_next_proxy(self):
        return self.proxy

    def mark_proxy_failed(self, proxy) -> None:
        del proxy

    def mark_proxy_success(self, proxy) -> None:
        del proxy


class _ProxySecretFilter(logging.Filter):
    def __init__(self, secrets: set[str]) -> None:
        super().__init__()
        self.secrets = tuple(
            sorted((value for value in secrets if value), key=len, reverse=True)
        )

    def add_secrets(self, secrets: set[str]) -> None:
        self.secrets = tuple(
            sorted({*self.secrets, *(value for value in secrets if value)}, key=len, reverse=True)
        )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = message
        for secret in self.secrets:
            redacted = redacted.replace(secret, "<redacted-proxy>")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def _install_proxy_log_redaction(proxy_url: str) -> None:
    global _proxy_secret_filter
    parsed = urlparse(proxy_url)
    secrets = {proxy_url, parsed.netloc}
    if _proxy_secret_filter is None:
        _proxy_secret_filter = _ProxySecretFilter(secrets)
    else:
        _proxy_secret_filter.add_secrets(secrets)
    root_logger = logging.getLogger()
    if _proxy_secret_filter not in root_logger.filters:
        root_logger.addFilter(_proxy_secret_filter)
    for handler in root_logger.handlers:
        if _proxy_secret_filter not in handler.filters:
            handler.addFilter(_proxy_secret_filter)
    setup_logger = logging.getLogger("desktopenv.setup")
    if _proxy_secret_filter not in setup_logger.filters:
        setup_logger.addFilter(_proxy_secret_filter)


def _enable_legacy_osworld_proxy(env) -> None:
    if not _truthy(os.environ.get("GUI_TRAJECTORY_LEGACY_OSWORLD_PROXY_ENABLED")):
        raise RuntimeError(
            "legacy OSWorld task requires proxy support; enable "
            "backend_options.legacy_osworld_enable_proxy and provide its configured "
            "proxy URL environment variable"
        )
    proxy_url = os.environ.get("GUI_TRAJECTORY_LEGACY_OSWORLD_PROXY_URL", "").strip()
    if not proxy_url:
        raise RuntimeError("legacy OSWorld proxy is enabled but no proxy URL was provided")

    proxy_pool = _ConfiguredProxyPool(proxy_url)
    _install_proxy_log_redaction(proxy_url)
    setup_module = importlib.import_module(
        "weavebench.desktop_env.controllers.setup"
    )
    setup_module.get_global_proxy_pool = lambda: proxy_pool
    env.enable_proxy = True
    if bool(os.environ.get("CUAMASTER_SANDBOX_PROVIDER", "").strip()) or _truthy(
        os.environ.get("GUI_TRAJECTORY_REMOTE_SSH_PROVIDER")
    ):
        env._setup_docker_proxy_forwarding = lambda: None
    logger.info(
        "enabled legacy OSWorld proxy host=%s port=%s",
        proxy_pool.proxy.host,
        proxy_pool.proxy.port,
    )


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_endpoint_pool() -> DynamicEndpointPool | None:
    global _endpoint_pool_instance
    path = os.environ.get("GUI_TRAJECTORY_ENDPOINT_POOL")
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if (
        _endpoint_pool_instance is None
        or _endpoint_pool_instance.config_path != resolved
    ):
        _endpoint_pool_instance = DynamicEndpointPool(config_path=resolved)
    return _endpoint_pool_instance


def _record_has_endpoint_failure(record: dict) -> bool:
    error = str(record.get("error") or "").lower()
    if any(marker in error for marker in _NON_ENDPOINT_FAILURE_MARKERS):
        return False
    text = " ".join(
        str(value or "")
        for value in (
            error,
            record.get("agent_status"),
            record.get("agent_exit"),
        )
    ).lower()
    return bool(text.strip()) and any(marker in text for marker in _ENDPOINT_FAILURE_MARKERS)


def _write_endpoint_attempts(output_dir: Path, attempts: list[dict]) -> None:
    if not attempts:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "endpoint_failover.json").write_text(
        json.dumps({"attempts": attempts}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _is_osworld_task(task: dict, tasks_root: Path) -> bool:
    row = _task_manifest_rows(tasks_root).get(str(task.get("task_id")))
    evaluator = row.get("evaluator") if isinstance(row, dict) else None
    return bool(
        isinstance(row, dict)
        and (
            row.get("execution_adapter") == "legacy_osworld"
            or (
                isinstance(evaluator, dict)
                and evaluator.get("type") == "osworld_v2_task_class"
            )
        )
    )


def _dedent_claudecode_retry_parser(script: str) -> str:
    marker = 'RETRY_REASON=$(echo "$LAST_RESULT" | python3 -c \'\n'
    terminator = "' 2>/dev/null || echo PARSE_ERR)"
    start = script.find(marker)
    if start < 0:
        return script
    body_start = start + len(marker)
    body_end = script.find(terminator, body_start)
    if body_end < 0:
        return script
    body = textwrap.dedent(script[body_start:body_end])
    return script[:body_start] + body + script[body_end:]


def _harden_claudecode_tool_policy(script: str) -> str:
    marker = 'DISALLOWED_TOOLS="WebFetch,WebSearch"'
    if marker not in script:
        raise RuntimeError("Claude Code run script has no recognized tool policy marker")
    requested_mode = os.environ.get(
        "CUA_INTERACTION_HARNESS",
        os.environ.get("GUI_TRAJECTORY_REQUESTED_MODE", "gui"),
    ).strip().lower()
    try:
        policy = claude_code_tool_policy(requested_mode, runtime_location="guest")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    replacement = (
        f'ALLOWED_TOOLS="{policy.allowed_argument}"\n'
        f'DISALLOWED_TOOLS="{policy.disallowed_argument}"'
    )
    hardened = script.replace(marker, replacement, 1)
    unsafe_flag = "--dangerously-skip-permissions"
    if unsafe_flag not in hardened:
        raise RuntimeError("Claude Code run script has no recognized permission mode")
    hardened = hardened.replace(unsafe_flag, "")
    disallowed_arg = '--disallowedTools "$DISALLOWED_TOOLS" \\\n'
    if disallowed_arg not in hardened:
        raise RuntimeError("Claude Code run script has no recognized tool arguments")
    allowed_args = (
        '--allowedTools "$ALLOWED_TOOLS" \\\n'
        '      --disallowedTools "$DISALLOWED_TOOLS" \\\n'
    )
    return hardened.replace(disallowed_arg, allowed_args)


@contextmanager
def _shared_remote_ssh_control():
    if not _truthy(os.environ.get("GUI_TRAJECTORY_REMOTE_SSH_PROVIDER")):
        yield
        return

    from cuamaster.collection.providers.ssh import (
        RemoteSSHConfig,
        SSHClient,
    )

    previous = os.environ.get("GUI_TRAJECTORY_SSH_CONTROL_PATH")
    config = RemoteSSHConfig.from_env()
    client = SSHClient(config)
    try:
        try:
            client.run("true", timeout=config.connect_timeout + 5)
        except Exception as exc:
            raise RuntimeError(
                "remote SSH preflight failed for "
                f"{config.user}@{config.host}:{config.port}; verify the configured "
                "password or identity before starting collection"
            ) from exc
        os.environ["GUI_TRAJECTORY_SSH_CONTROL_PATH"] = str(client.control_path)
        yield
    finally:
        if previous is None:
            os.environ.pop("GUI_TRAJECTORY_SSH_CONTROL_PATH", None)
        else:
            os.environ["GUI_TRAJECTORY_SSH_CONTROL_PATH"] = previous
        client.close()


def _install_task_endpoint_failover(runner_module) -> None:
    pool = _configured_endpoint_pool()
    if pool is None:
        return
    inner_run_one = runner_module.ds.run_one

    def run_one_with_failover(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        audit_output_dir = _compatible_output_dir(task, mode, Path(output_dir))
        attempted: set[str] = set()
        audit: list[dict] = []
        last_record: dict = {}
        while len(attempted) < len(pool.base_urls):
            retry_after = pool.retry_after(exclude=attempted)
            if retry_after > 0:
                logger.warning(
                    "all candidate model endpoints are cooling down; waiting %.1fs",
                    retry_after,
                )
                time.sleep(retry_after)
            base_url = pool.acquire(exclude=attempted)
            attempted.add(base_url)
            agent.litellm_base_url = base_url
            started = datetime.now(timezone.utc)
            try:
                record = inner_run_one(
                    env, agent, task, mode, output_dir, tasks_root, **kwargs
                )
            except Exception as exc:
                record = {"error": f"{type(exc).__name__}: {exc}"}
            last_record = record
            endpoint_failure = _record_has_endpoint_failure(record)
            audit.append(
                {
                    "base_url": base_url,
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "endpoint_failure": endpoint_failure,
                    "error": record.get("error"),
                }
            )
            if not endpoint_failure:
                pool.mark_success(base_url)
                record["model_base_url"] = base_url
                _write_endpoint_attempts(audit_output_dir, audit)
                return record
            pool.mark_failure(base_url, str(record.get("error") or "endpoint failure"))
            if len(attempted) >= len(pool.base_urls):
                break
            logger.warning(
                "task %s endpoint failed at %s; retrying with another endpoint",
                task.get("task_id"),
                base_url,
            )
            reset = getattr(env, "reset", None)
            if callable(reset) and not _is_osworld_task(task, Path(tasks_root)):
                reset(task_config=task)
        last_record["model_base_url"] = audit[-1]["base_url"] if audit else None
        _write_endpoint_attempts(audit_output_dir, audit)
        return last_record

    runner_module.ds.run_one = run_one_with_failover


def _prepare_guest_artifact_dirs(upstream, env, client_password: str) -> None:
    paths = (
        "/tmp_workspace/_trajectory",
        _REMOTE_SHELL_SCREENSHOT_DIR,
        "/home/user/.claude",
    )
    shell_command = (
        f"mkdir -p {' '.join(paths)} && "
        f"chown -R user:user {' '.join(paths)}"
    )
    command = (
        f"printf '%s\\n' {shlex.quote(client_password)} | "
        f"sudo -S -p '' sh -c {shlex.quote(shell_command)}"
    )
    result = upstream._vm_exec(env, ["bash", "-c", command])
    if result.get("returncode") not in (0, None):
        raise RuntimeError(
            "failed to prepare Claude Code artifact directories: "
            f"{result.get('error') or result.get('output') or result}"
        )


def _normalized(value: object) -> str:
    return str(value or "").strip().casefold().replace(" ", "-")


def _os_family(value: object) -> str:
    normalized = _normalized(value)
    if normalized in {"linux", "ubuntu", "debian"}:
        return "linux"
    if normalized in {"darwin", "mac", "macos", "osx"}:
        return "macos"
    if normalized in {"windows", "win"}:
        return "windows"
    return normalized


def _profile_capabilities(profile: dict) -> set[str]:
    os_spec = profile.get("os") if isinstance(profile.get("os"), dict) else {}
    software = profile.get("software") if isinstance(profile.get("software"), dict) else {}
    capabilities = {
        "computer",
        "filesystem",
        "terminal",
        f"os:{_normalized(os_spec.get('type'))}",
        f"arch:{_normalized(os_spec.get('architecture'))}",
    }
    for name in software:
        normalized = _normalized(name)
        capabilities.update({normalized, f"app:{normalized}"})
    return capabilities


def _profile_image_ids(profile: dict) -> set[str]:
    image = profile.get("image") if isinstance(profile.get("image"), dict) else {}
    identifiers = {str(image.get("id") or "").strip()}
    metadata = image.get("metadata")
    if isinstance(metadata, dict):
        identifiers.update(
            str(metadata.get(key) or "").strip()
            for key in ("id", "image_id", "derived_from")
        )
        build = metadata.get("build")
        if isinstance(build, dict):
            identifiers.add(str(build.get("base_image_id") or "").strip())
    return identifiers - {""}


def _manifest_task_matches(row: dict, profile: dict) -> bool:
    os_spec = profile.get("os") if isinstance(profile.get("os"), dict) else {}
    target_environments = row.get("target_environments") or []
    if target_environments:
        os_type = _os_family(os_spec.get("type"))
        os_version = _normalized(os_spec.get("version"))
        architecture = _normalized(os_spec.get("architecture"))
        if not any(
            isinstance(target, dict)
            and _os_family(target.get("os_type")) == os_type
            and (
                not target.get("os_versions")
                or os_version
                in {_normalized(value) for value in target.get("os_versions") or []}
            )
            and (
                not target.get("architectures")
                or architecture
                in {_normalized(value) for value in target.get("architectures") or []}
            )
            for target in target_environments
        ):
            return False
    environment_id = str(row.get("environment_id") or "").strip()
    if environment_id:
        return environment_id in _profile_image_ids(profile)
    required = {str(value) for value in row.get("required_capabilities") or []}
    return required <= _profile_capabilities(profile)


def _install_task_platform_filter(messages_gui) -> None:
    if _truthy(os.environ.get("GUI_TRAJECTORY_FORCE_ALL_TASKS_ON_PLATFORM")):
        logger.warning(
            "platform compatibility filtering is disabled for cross-platform replay"
        )
        return
    declared = os.environ.get("WEAVEBENCH_VM_PROFILE_JSON")
    if not declared:
        return
    profile = json.loads(declared)
    require_explicit = _truthy(
        os.environ.get("GUI_TRAJECTORY_REQUIRE_EXPLICIT_PLATFORM")
    )
    discover = messages_gui.ds.discover_tasks

    def discover_compatible(tasks_root, categories):
        tasks = discover(tasks_root, categories)
        manifest_path = Path(tasks_root) / "taskset_manifest.json"
        if not manifest_path.is_file():
            if require_explicit:
                raise ValueError(
                    f"non-Ubuntu collection requires {manifest_path} with target_environments"
                )
            return tasks
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("tasks") if isinstance(manifest, dict) else None
        by_id = {
            str(row.get("id")): row
            for row in rows or []
            if isinstance(row, dict) and row.get("id")
        }
        compatible = []
        for task in tasks:
            row = by_id.get(str(task.get("task_id")))
            if row is None:
                if not require_explicit:
                    compatible.append(task)
                continue
            if require_explicit and not row.get("target_environments"):
                continue
            if _manifest_task_matches(row, profile):
                compatible.append(task)
        logger.info(
            "platform filter retained %d/%d tasks for profile %s",
            len(compatible),
            len(tasks),
            profile.get("id"),
        )
        if require_explicit and not compatible:
            raise ValueError(
                f"no tasks explicitly support profile {profile.get('id') or '<unknown>'}"
            )
        return compatible

    messages_gui.ds.discover_tasks = discover_compatible


def _install_vm_provider() -> None:
    sandbox_provider_name = os.environ.get("CUAMASTER_SANDBOX_PROVIDER", "").strip()
    use_remote_ssh = _truthy(os.environ.get("GUI_TRAJECTORY_REMOTE_SSH_PROVIDER"))
    if not (sandbox_provider_name or use_remote_ssh):
        return
    import weavebench.desktop_env.desktop_env as desktop_module
    import weavebench.desktop_env.providers as providers_module

    if sandbox_provider_name:
        from cuamaster.collection.providers.sandbox import create_sandbox_provider

        def provider_factory(region):
            return create_sandbox_provider(sandbox_provider_name, region)

    else:
        from cuamaster.collection.providers.ssh import create_remote_ssh_provider

        provider_factory = create_remote_ssh_provider

    original_factory = providers_module.create_vm_manager_and_provider

    def create_vm_manager_and_provider(provider_name, region, use_proxy=False):
        if str(provider_name).strip().lower() == "docker":
            return provider_factory(region)
        return original_factory(provider_name, region, use_proxy=use_proxy)

    providers_module.create_vm_manager_and_provider = create_vm_manager_and_provider
    desktop_module.create_vm_manager_and_provider = create_vm_manager_and_provider


def _screenshot_size(payload: bytes | None) -> dict[str, int] | None:
    if not payload:
        return None
    if payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24:
        width, height = struct.unpack(">II", payload[16:24])
        return {"width": width, "height": height}
    if payload.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 9 <= len(payload):
            if payload[offset] != 0xFF:
                offset += 1
                continue
            marker = payload[offset + 1]
            offset += 2
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if offset + 2 > len(payload):
                break
            segment_length = int.from_bytes(payload[offset : offset + 2], "big")
            if segment_length < 2 or offset + segment_length > len(payload):
                break
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            } and segment_length >= 7:
                height = int.from_bytes(payload[offset + 3 : offset + 5], "big")
                width = int.from_bytes(payload[offset + 5 : offset + 7], "big")
                return {"width": width, "height": height}
            offset += segment_length
    return None


def _install_screen_size_fallback() -> None:
    if not (
        bool(os.environ.get("CUAMASTER_SANDBOX_PROVIDER", "").strip())
        or _truthy(os.environ.get("GUI_TRAJECTORY_REMOTE_SSH_PROVIDER"))
    ):
        return
    controller_module = importlib.import_module(
        "weavebench.desktop_env.controllers.python"
    )
    controller_class = controller_module.PythonController
    original = controller_class.get_vm_screen_size
    if getattr(original, "_gui_trajectory_screenshot_fallback", False):
        return

    def get_vm_screen_size(self):
        size = _screenshot_size(self.get_screenshot())
        if size is not None:
            return size
        return original(self)

    get_vm_screen_size._gui_trajectory_screenshot_fallback = True
    controller_class.get_vm_screen_size = get_vm_screen_size


def _install_agent_adapter(harness: str):
    if harness == "openclaw":
        transport = os.environ.get("WEAVEBENCH_TRANSPORT", "auto").strip().lower()
        if transport == "auto":
            model = os.environ.get("WEAVEBENCH_MODEL", "").casefold()
            transport = "messages" if "claude" in model else "chat"
        module_name = {
            "chat": "weavebench.agents._openclaw._chat",
            "messages": "weavebench.agents._openclaw._messages",
            "responses": "weavebench.agents._openclaw._responses",
        }.get(transport)
        if module_name is None:
            raise ValueError(f"unsupported OpenClaw transport: {transport!r}")
        upstream = importlib.import_module(module_name)
        _install_vm_transport_guard(upstream)
        upstream.BOOTSTRAP_SH = _harden_openclaw_bootstrap(upstream.BOOTSTRAP_SH)

    if harness != "claudecode":
        runtime_location = os.environ.get(
            "GUI_TRAJECTORY_RUNTIME_LOCATION", "guest"
        ).strip().lower()
        if runtime_location != "guest":
            raise ValueError(
                f"the {harness} harness requires GUI_TRAJECTORY_RUNTIME_LOCATION=guest"
            )
        return None

    import weavebench.agents.claudecode_agent as upstream

    _install_vm_transport_guard(upstream)

    runtime_location = os.environ.get(
        "GUI_TRAJECTORY_RUNTIME_LOCATION", "guest"
    ).strip().lower()
    if runtime_location == "host":
        from cuamaster.collection.host_claude_agent import build_host_claude_agent

        host_agent = build_host_claude_agent(upstream)
        upstream.ClaudeCodeAgent = host_agent
        return host_agent
    if runtime_location != "guest":
        raise ValueError(
            f"unsupported GUI_TRAJECTORY_RUNTIME_LOCATION: {runtime_location!r}"
        )

    BaseAgent = upstream.ClaudeCodeAgent

    class IsolatedClaudeCodeAgent(BaseAgent):
        """Adds artifact recovery even when the upstream adapter lacks it."""

        def __init__(self, *args, **kwargs):
            configured_timeout = os.environ.get(
                "GUI_TRAJECTORY_TASK_TIMEOUT_SECONDS", ""
            ).strip()
            if configured_timeout:
                timeout = int(configured_timeout)
                if timeout < 1:
                    raise ValueError(
                        "GUI_TRAJECTORY_TASK_TIMEOUT_SECONDS must be >= 1"
                    )
                kwargs["timeout"] = timeout
            super().__init__(*args, **kwargs)

        def _render_run_script(self, system_override):
            return _harden_claudecode_tool_policy(
                _dedent_claudecode_retry_parser(
                    super()._render_run_script(system_override)
                )
            )

        def _build_prompt(self, task_prompt, system_override):
            return append_collection_policy(
                super()._build_prompt(task_prompt, system_override)
            )

        def configure(self, env) -> None:
            super().configure(env)
            if not self.gui:
                return
            _prepare_guest_artifact_dirs(upstream, env, self.client_password)
            upstream._vm_upload_text(
                env,
                _SHELL_SCREENSHOT_HOOK.read_text(encoding="utf-8"),
                _REMOTE_SHELL_SCREENSHOT_HOOK,
            )
            upstream._vm_exec(env, [
                "bash", "-c",
                f"chmod 0700 {_REMOTE_SHELL_SCREENSHOT_HOOK} && "
                f"/usr/bin/python3 {_REMOTE_SHELL_SCREENSHOT_HOOK} "
                "install /home/user/.claude/settings.json",
            ])
            logger.info("configure: installed Bash post-action screenshot hook")

        def _fetch_gui_artifacts(self, env, output_dir: Path) -> None:
            self._isolated_fetch_called = True
            if not self.gui:
                return
            output_dir = Path(output_dir)
            try:
                upstream._vm_fetch(
                    env,
                    "/tmp_workspace/_trajectory/computer_events.jsonl",
                    output_dir / "computer_events.jsonl",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("event log fetch failed: %s", exc)
            try:
                upstream._vm_fetch(
                    env,
                    _REMOTE_SHELL_EVENT_LOG,
                    output_dir / "shell_events.jsonl",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("shell event log fetch failed: %s", exc)
            try:
                listing = upstream._vm_exec(env, [
                    "bash", "-c",
                    "find /tmp_workspace/_screenshots "
                    f"{_REMOTE_SHELL_SCREENSHOT_DIR} -maxdepth 1 -type f "
                    "-name '*.png' -print 2>/dev/null | sort || true",
                ])
                remotes = [
                    line.strip()
                    for line in (listing.get("output") or "").splitlines()
                    if line.strip().startswith((
                        "/tmp_workspace/_screenshots/",
                        f"{_REMOTE_SHELL_SCREENSHOT_DIR}/",
                    ))
                    and line.strip().endswith(".png")
                ]
                screenshots = output_dir / "screenshots"
                if remotes:
                    screenshots.mkdir(parents=True, exist_ok=True)
                for remote in remotes:
                    upstream._vm_fetch(env, remote, screenshots / Path(remote).name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("screenshot fetch failed: %s", exc)
            try:
                screenshot = env.controller.get_screenshot()
                if screenshot:
                    (output_dir / "final_screenshot.png").write_bytes(screenshot)
            except Exception as exc:  # noqa: BLE001
                logger.warning("final screenshot failed: %s", exc)

        def _capture_environment_fingerprint(self, env, output_dir: Path) -> None:
            self._isolated_fingerprint_called = True

            # Determine platform from profile
            declared = os.environ.get("WEAVEBENCH_VM_PROFILE_JSON")
            platform_type = "linux"
            if declared:
                try:
                    profile = json.loads(declared)
                    os_spec = profile.get("os") if isinstance(profile, dict) else {}
                    os_type = str(os_spec.get("type", "")).strip().lower()
                    if os_type in {"windows", "win32"}:
                        platform_type = "windows"
                    elif os_type in {"darwin", "macos", "mac"}:
                        platform_type = "darwin"
                except json.JSONDecodeError:
                    pass

            # Platform-specific probes
            if platform_type == "windows":
                probes = {
                    "os_version": ["cmd", "/c", "ver"],
                    "python": ["python", "--version"],
                    "node": ["node", "--version"],
                    "powershell": ["powershell", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
                    "chrome": ["powershell", "-NoProfile", "-Command", "(Get-Command chrome -ErrorAction SilentlyContinue).Source"],
                    "edge": ["powershell", "-NoProfile", "-Command", "(Get-Command msedge -ErrorAction SilentlyContinue).Source"],
                    "office": ["powershell", "-NoProfile", "-Command", "(Get-Command winword -ErrorAction SilentlyContinue).Source"],
                }
            elif platform_type == "darwin":
                probes = {
                    "os_version": ["/bin/sh", "-c", "sw_vers"],
                    "kernel": ["/bin/sh", "-c", "uname -a"],
                    "python": ["/bin/sh", "-c", "python3 --version 2>&1"],
                    "node": ["/bin/sh", "-c", "node --version 2>&1"],
                    "chrome": ["/bin/sh", "-c", "test -d '/Applications/Google Chrome.app' && echo '/Applications/Google Chrome.app'"],
                    "safari": ["/bin/sh", "-c", "test -d '/Applications/Safari.app' && echo '/Applications/Safari.app'"],
                    "office": ["/bin/sh", "-c", "test -d '/Applications/Microsoft Word.app' && echo '/Applications/Microsoft Word.app'"],
                    "libreoffice": ["/bin/sh", "-c", "test -d '/Applications/LibreOffice.app' && echo '/Applications/LibreOffice.app'"],
                }
            else:
                probes = {
                    "os_release": ["bash", "-c", "cat /etc/os-release 2>/dev/null"],
                    "kernel": ["bash", "-c", "uname -a 2>/dev/null"],
                    "claude": ["bash", "-c", "/usr/local/bin/claude --version 2>&1"],
                    "python": ["bash", "-c", "python3 --version 2>&1"],
                    "node": ["bash", "-c", "node --version 2>&1"],
                    "chrome": ["bash", "-c", "(google-chrome --version || chromium --version) 2>&1"],
                    "libreoffice": ["bash", "-c", "libreoffice --version 2>&1"],
                    "gimp": ["bash", "-c", "gimp --version 2>&1 | head -1"],
                }

            fingerprint = {
                "schema_version": "gui_trajectory_collection.environment.v1",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "profile_id": os.environ.get("WEAVEBENCH_VM_PROFILE_ID"),
                "declared_profile": None,
                "platform_type": platform_type,
                "probes": {},
            }
            if declared:
                try:
                    fingerprint["declared_profile"] = json.loads(declared)
                except json.JSONDecodeError:
                    fingerprint["declared_profile"] = declared

            for name, command in probes.items():
                try:
                    result = upstream._vm_exec(env, command, timeout=30)
                    fingerprint["probes"][name] = {
                        "returncode": result.get("returncode"),
                        "output": (result.get("output") or "").strip(),
                        "error": (result.get("error") or "").strip(),
                    }
                except Exception as exc:  # noqa: BLE001
                    fingerprint["probes"][name] = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }
            (Path(output_dir) / "environment_fingerprint.json").write_text(
                json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        def run(self, env, instruction: str, output_dir: Path, **kwargs):
            self._isolated_fetch_called = False
            self._isolated_fingerprint_called = False
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                screenshot = env.controller.get_screenshot()
                if screenshot:
                    (output_dir / "init_screenshot.png").write_bytes(screenshot)
            except Exception as exc:  # noqa: BLE001
                logger.warning("initial screenshot failed: %s", exc)
            started_at = time.monotonic()
            transport_failure: BaseException | None = None
            try:
                env._gui_trajectory_task_id = output_dir.name
            except Exception:  # noqa: BLE001
                pass
            try:
                result = super().run(env, instruction, output_dir, **kwargs)
            except Exception as exc:
                if not _is_vm_transport_error(exc):
                    raise
                transport_failure = exc
                _write_vm_transport_failure(
                    output_dir,
                    env,
                    self,
                    exc,
                    time.monotonic() - started_at,
                )
                if isinstance(exc, VMTransportUnavailable):
                    raise
                failure = _vm_transport_failure(env, "agent-run", exc)
                raise failure from exc
            finally:
                if transport_failure is None and not self._isolated_fetch_called:
                    self._fetch_gui_artifacts(env, output_dir)
                if transport_failure is None and not self._isolated_fingerprint_called:
                    self._capture_environment_fingerprint(env, output_dir)
                auth = output_dir / "auth_setup.sh"
                if auth.is_file() and self.litellm_api_key:
                    text = auth.read_text(encoding="utf-8", errors="replace")
                    auth.write_text(
                        text.replace(self.litellm_api_key, "<redacted>"), encoding="utf-8"
                    )
            return result

    upstream.ClaudeCodeAgent = IsolatedClaudeCodeAgent
    return IsolatedClaudeCodeAgent


def _harden_openclaw_bootstrap(script: str) -> str:
    if _OPENCLAW_BOOTSTRAP_HARDENING_MARKER in script:
        return script
    done_check = "if [ -f /home/user/.openclaw_bootstrap.done ]; then"
    node_check = "NODE_MAJ=$(/opt/node22/bin/node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\\1/' || echo 0)"
    node_extract = 'SUDO tar -xJf "$NODE_TAR" -C /opt/node22 --strip-components=1 || { echo "node extract failed"; exit 12; }'
    openclaw_extract = 'SUDO tar xzf /tmp/openclaw.tar.gz -C /usr/lib/node_modules || { echo "extract failed"; exit 17; }'
    for marker in (done_check, node_check, node_extract, openclaw_extract):
        if marker not in script:
            raise ValueError(f"unsupported OpenClaw bootstrap script; missing {marker!r}")
    preflight = f"""{_OPENCLAW_BOOTSTRAP_HARDENING_MARKER}
SUDO chmod 1777 /tmp
OPENCLAW_RUNTIME_TMP="/tmp/openclaw-$(id -u)"
SUDO rm -rf "$OPENCLAW_RUNTIME_TMP"
mkdir -p "$OPENCLAW_RUNTIME_TMP"
chmod 0700 "$OPENCLAW_RUNTIME_TMP"
SUDO rm -rf /home/user/.npm/_cacache /home/user/.cache/pip

"""
    reuse_system_node = """SYSTEM_NODE=$(command -v node 2>/dev/null || true)
SYSTEM_NODE_MAJ=$($SYSTEM_NODE --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\\1/' || echo 0)
if [ -n "$SYSTEM_NODE" ] && [ "${SYSTEM_NODE_MAJ:-0}" -ge 22 ] && [ ! -x /opt/node22/bin/node ]; then
  SUDO mkdir -p /opt/node22/bin
  SUDO ln -sf "$SYSTEM_NODE" /opt/node22/bin/node
  SYSTEM_NPM=$(command -v npm 2>/dev/null || true)
  SYSTEM_NPX=$(command -v npx 2>/dev/null || true)
  [ -z "$SYSTEM_NPM" ] || SUDO ln -sf "$SYSTEM_NPM" /opt/node22/bin/npm
  [ -z "$SYSTEM_NPX" ] || SUDO ln -sf "$SYSTEM_NPX" /opt/node22/bin/npx
fi
"""
    script = script.replace(done_check, preflight + done_check, 1)
    script = script.replace(node_check, reuse_system_node + node_check, 1)
    script = script.replace(node_extract, node_extract + '\n  rm -f "$NODE_TAR"', 1)
    script = script.replace(
        openclaw_extract,
        openclaw_extract + "\nrm -f /tmp/openclaw.tar.gz",
        1,
    )
    return script


_TASK_MANIFEST_CACHE: dict[Path, dict[str, dict]] = {}
_TASK_CLASS_CACHE: dict[tuple[str, str], type] = {}
_OSWORLD_EVALUATOR_CACHE: dict[str, tuple[object, object]] = {}


class _OSWorldSetupControllerAdapter:
    def __init__(self, controller) -> None:
        self._controller = controller

    def __getattr__(self, name: str):
        return getattr(self._controller, name)

    def setup(self, config, use_proxy: bool = False) -> bool:
        self._controller.use_proxy = use_proxy
        for item in config or []:
            config_type = str(item["type"])
            parameters = item.get("parameters") or {}
            method = getattr(self, config_type, None)
            if callable(method):
                method(**parameters)
                continue
            setup_method = getattr(self._controller, f"_{config_type}_setup")
            setup_method(**parameters)
        return True

    def _proxy_setup(self, client_password: str = "", is_docker: bool = False):
        if bool(os.environ.get("CUAMASTER_SANDBOX_PROVIDER", "").strip()) or _truthy(os.environ.get("GUI_TRAJECTORY_REMOTE_SSH_PROVIDER")):
            is_docker = False
        return self._controller._proxy_setup(
            client_password, is_docker=is_docker
        )

    def download(self, files) -> None:
        remote_files = []
        upload = getattr(self._controller, "_upload_file_setup", None)
        for item in files:
            source = _resolve_local_setup_source(str(item.get("url") or ""))
            if source is None or not callable(upload):
                remote_files.append(item)
                continue
            upload([{"local_path": source, "path": item["path"]}])
        if not remote_files:
            return
        method = getattr(self._controller, "download", None)
        if callable(method):
            method(remote_files)
            return
        self._controller._download_setup(remote_files)

    def execute(
        self,
        command,
        stdout: str = "",
        stderr: str = "",
        shell: bool = False,
        until=None,
        quiet: bool = False,
        timeout: int = 120,
        check: bool = False,
    ):
        command, timeout, launch, background_wait = _prepare_osworld_setup_command(
            command, timeout
        )
        if launch:
            self.launch(command, shell=shell)
            sleep = getattr(self._controller, "_sleep_setup", None)
            if callable(sleep):
                sleep(5)
            return
        if background_wait:
            self._execute_long_running(
                command,
                stdout=stdout,
                stderr=stderr,
                shell=shell,
                quiet=quiet,
                timeout=timeout,
            )
            return
        method = getattr(self._controller, "execute", None)
        if callable(method):
            kwargs = {
                "stdout": stdout,
                "stderr": stderr,
                "shell": shell,
                "until": until,
                "quiet": quiet,
                "timeout": timeout,
            }
            if check:
                kwargs["check"] = True
            try:
                return method(command, **kwargs)
            except TypeError as exc:
                if not check or "check" not in str(exc):
                    raise
                kwargs.pop("check")
                result = method(command, **kwargs)
                self._raise_for_failed_result(command, result)
                return result
        if getattr(self._controller, "http_server", None):
            return self._execute_via_rest(
                command,
                stdout=stdout,
                stderr=stderr,
                shell=shell,
                until=until,
                quiet=quiet,
                timeout=timeout,
                check=check,
            )
        result = self._controller._execute_setup(
            command,
            stdout=stdout,
            stderr=stderr,
            shell=shell,
            until=until,
        )
        if check:
            self._raise_for_failed_result(command, result)
        return result

    @staticmethod
    def _raise_for_failed_result(command, result) -> None:
        if not isinstance(result, dict):
            return
        returncode = result.get("returncode")
        if returncode not in {None, 0}:
            detail = str(result.get("error") or result.get("output") or "")
            raise RuntimeError(
                f"OSWorld setup command failed with exit code {returncode}: "
                f"{detail[-1000:]}"
            )

    def _execute_via_rest(
        self,
        command,
        *,
        stdout: str,
        stderr: str,
        shell: bool,
        until,
        quiet: bool,
        timeout: int,
        check: bool = False,
    ) -> dict:
        command = _replace_setup_placeholders(command, self._controller)
        conditions = until or {}
        failures = 0
        while True:
            try:
                response = requests.post(
                    f"{self._controller.http_server.rstrip('/')}/setup/execute",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({
                        "command": command,
                        "shell": shell,
                        "timeout": timeout,
                    }),
                    timeout=(10, max(30, timeout + 30)),
                )
                if response.status_code != 200:
                    raise requests.RequestException(
                        f"Setup execute failed with status {response.status_code}: "
                        f"{response.text}"
                    )
                result = response.json()
                if stdout:
                    Path(self._controller.cache_dir, stdout).write_text(
                        result.get("output") or "", encoding="utf-8"
                    )
                if stderr:
                    Path(self._controller.cache_dir, stderr).write_text(
                        result.get("error") or "", encoding="utf-8"
                    )
                if not quiet:
                    logger.info("OSWorld setup command completed: %s", command)
                if check:
                    self._raise_for_failed_result(command, result)
                if not conditions or (
                    "returncode" in conditions
                    and result.get("returncode") == conditions["returncode"]
                ) or (
                    "stdout" in conditions
                    and conditions["stdout"] in (result.get("output") or "")
                ) or (
                    "stderr" in conditions
                    and conditions["stderr"] in (result.get("error") or "")
                ):
                    return result
            except requests.RequestException:
                failures += 1
                if failures >= 5:
                    raise
            time.sleep(min(3.0, 0.3 * (2 ** (failures - 1))))

    def _execute_long_running(
        self,
        command,
        *,
        stdout: str,
        stderr: str,
        shell: bool,
        quiet: bool,
        timeout: int,
    ) -> None:
        command = _replace_setup_placeholders(command, self._controller)
        command_text = command if isinstance(command, str) else shlex.join(command)
        token = uuid.uuid4().hex
        status_path = f"/tmp/weavebench_setup_{token}.status"
        pid_path = f"/tmp/weavebench_setup_{token}.pid"
        log_path = f"/tmp/weavebench_setup_{token}.log"
        script = (
            f"{command_text}; rc=$?; "
            f"printf '%s\\n' \"$rc\" > {shlex.quote(status_path)}"
        )
        start_command = [
            "bash",
            "-c",
            f"rm -f {shlex.quote(status_path)} {shlex.quote(pid_path)} "
            f"{shlex.quote(log_path)}; "
            f"nohup bash -c {shlex.quote(script)} > {shlex.quote(log_path)} "
            f"2>&1 </dev/null & echo $! > {shlex.quote(pid_path)}",
        ]
        self._execute_via_rest(
            start_command,
            stdout="",
            stderr="",
            shell=shell,
            until=None,
            quiet=True,
            timeout=30,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self._execute_via_rest(
                [
                    "bash",
                    "-c",
                    f"test -f {shlex.quote(status_path)} && "
                    f"cat {shlex.quote(status_path)} || echo RUNNING",
                ],
                stdout="",
                stderr="",
                shell=False,
                until=None,
                quiet=True,
                timeout=30,
            )
            value = str(status.get("output") or "").strip()
            if value != "RUNNING":
                details = self._execute_via_rest(
                    ["bash", "-c", f"cat {shlex.quote(log_path)} 2>/dev/null || true"],
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    until=None,
                    quiet=True,
                    timeout=30,
                )
                self._execute_via_rest(
                    ["rm", "-f", status_path, pid_path, log_path],
                    stdout="",
                    stderr="",
                    shell=False,
                    until=None,
                    quiet=True,
                    timeout=30,
                )
                try:
                    returncode = int(value.splitlines()[-1])
                except (IndexError, ValueError):
                    returncode = 1
                if returncode != 0:
                    detail = str(details.get("output") or details.get("error") or "")
                    raise RuntimeError(
                        f"OSWorld setup command failed with exit code {returncode}: "
                        f"{detail[-1000:]}"
                    )
                if not quiet:
                    logger.info("OSWorld setup command completed: %s", command)
                return
            time.sleep(5)

        self._execute_via_rest(
            [
                "bash",
                "-c",
                f"test -f {shlex.quote(pid_path)} && "
                f"kill -TERM $(cat {shlex.quote(pid_path)}) 2>/dev/null || true",
            ],
            stdout="",
            stderr="",
            shell=False,
            until=None,
            quiet=True,
            timeout=30,
        )
        raise TimeoutError(
            f"OSWorld setup command timed out after {timeout}s: {command_text}"
        )

    def launch(self, command, shell: bool = False) -> None:
        if _launches_gimp(command):
            self.execute(
                [
                    "bash",
                    "-c",
                    "set -eu; "
                    'config_dir="$HOME/.config/GIMP/2.10"; '
                    'config_file="$config_dir/gimprc"; '
                    'mkdir -p "$config_dir"; touch "$config_file"; '
                    "sed -i '/^[[:space:]]*(color-profile-policy[[:space:]]/d' "
                    '"$config_file"; '
                    "printf '\\n(color-profile-policy keep)\\n' >> \"$config_file\"",
                ],
                quiet=True,
                timeout=30,
            )
        method = getattr(self._controller, "launch", None)
        if callable(method):
            method(command, shell=shell)
            return
        self._controller._launch_setup(command, shell=shell)


def _osworld_setup_controller(controller):
    if isinstance(controller, _OSWorldSetupControllerAdapter):
        return controller
    return _OSWorldSetupControllerAdapter(controller)


def _launches_gimp(command) -> bool:
    if isinstance(command, str):
        values = shlex.split(command)
    else:
        values = [str(item) for item in command or []]
    return bool(values) and Path(values[0]).name.casefold() in {"gimp", "gimp-2.10"}


def _resolve_local_setup_source(value: str) -> str | None:
    if not value:
        return None
    if value.startswith("file://"):
        candidate = unquote(urlparse(value).path)
    elif "://" not in value:
        candidate = value if Path(value).exists() else unquote(value)
    else:
        return None
    return candidate if Path(candidate).exists() else None


def _replace_setup_placeholders(command, controller):
    replacements = {
        "{CLIENT_PASSWORD}": str(getattr(controller, "client_password", "")),
        "{SCREEN_WIDTH}": str(getattr(controller, "screen_width", 1920)),
        "{SCREEN_HEIGHT}": str(getattr(controller, "screen_height", 1080)),
        "{SCREEN_WIDTH_HALF}": str(getattr(controller, "screen_width", 1920) // 2),
        "{SCREEN_HEIGHT_HALF}": str(getattr(controller, "screen_height", 1080) // 2),
    }

    def replace(value: str) -> str:
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value

    if isinstance(command, str):
        return replace(command)
    return [replace(str(item)) for item in command]


def _prepare_osworld_setup_command(command, timeout: int):
    values = [command] if isinstance(command, str) else [str(item) for item in command]
    command_text = " ".join(values).casefold()
    launch = bool(re.search(r"(?:^|[;&|]\s*)npm\s+start(?:\s|$)", command_text))
    slow = any(
        marker in command_text
        for marker in (
            "pip install",
            "pip3 install",
            "npm install",
            "git clone",
            "miniconda3-latest",
        )
    )
    if not isinstance(command, str) and len(command) >= 2:
        executable = Path(str(command[0])).name.casefold()
        if executable in {"pip", "pip3"} and command[1] == "install":
            if "--no-cache-dir" not in command:
                command = [*command[:2], "--no-cache-dir", *command[2:]]
    return (
        command,
        max(timeout, _SLOW_SETUP_TIMEOUT_SECONDS) if slow else timeout,
        launch,
        slow,
    )


def _cleanup_guest_runtime(setup_controller) -> None:
    command = [
        "bash",
        "-c",
        "rm -rf /tmp/weavebench_claudecode_install /tmp/claudecode.tar.gz "
        "/tmp/weavebench-npm-cache /tmp/pip-* /tmp/openclaw.tar.gz "
        "/tmp/node-v22.13.0-linux-x64.tar.xz /tmp/openclaw-$(id -u) "
        "/home/user/.cache/pip /home/user/.npm/_cacache; "
        "mkdir -p /tmp/weavebench_claudecode_install",
    ]
    try:
        setup_controller.execute(command, quiet=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        logger.warning("guest runtime cleanup failed; continuing: %s", exc)


def _task_manifest_rows(tasks_root: Path) -> dict[str, dict]:
    manifest_path = tasks_root / "taskset_manifest.json"
    cached = _TASK_MANIFEST_CACHE.get(manifest_path)
    if cached is not None:
        return cached
    if not manifest_path.is_file():
        rows = {}
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = {
            str(row.get("id")): row
            for row in payload.get("tasks") or []
            if isinstance(row, dict) and row.get("id")
        }
    _TASK_MANIFEST_CACHE[manifest_path] = rows
    return rows


def _load_legacy_osworld_evaluators() -> tuple[object, object]:
    root_value = os.environ.get("GUI_TRAJECTORY_OSWORLD_EVALUATOR_ROOT", "").strip()
    root = Path(root_value).expanduser().resolve() if root_value else None
    cache_key = str(root or "installed")
    cached = _OSWORLD_EVALUATOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if root is not None:
        package = root / "desktop_env" / "evaluators"
        if not package.is_dir():
            raise FileNotFoundError(
                f"OSWorld evaluator package not found under {root}: {package}"
            )
        loaded = sys.modules.get("desktop_env")
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve() if loaded else None
        if loaded_path and root not in loaded_path.parents:
            raise RuntimeError(
                "desktop_env was already imported from a different repository: "
                f"{loaded_path}; expected evaluator root {root}"
            )
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    metrics = importlib.import_module("desktop_env.evaluators.metrics")
    getters = importlib.import_module("desktop_env.evaluators.getters")
    _OSWORLD_EVALUATOR_CACHE[cache_key] = (metrics, getters)
    return metrics, getters


def _install_legacy_osworld_evaluators(env) -> None:
    metrics, getters = _load_legacy_osworld_evaluators()
    module = sys.modules.get(type(env).__module__)
    if module is None:
        module = importlib.import_module(type(env).__module__)
    module.metrics = metrics
    module.getters = getters


def _legacy_osworld_task_config(task: dict, row: dict) -> dict:
    return {
        "id": str(task.get("task_id") or row.get("id") or ""),
        "instruction": str(task.get("prompt") or task.get("instruction") or ""),
        "config": row.get("runtime_config") or [],
        "evaluator": row.get("evaluator") or {},
        "proxy": bool(row.get("proxy", False)),
    }


def _legacy_in_place_image_paths(
    task_config: dict,
) -> tuple[str, str] | None:
    evaluator = task_config.get("evaluator") or {}
    result = evaluator.get("result")
    expected = evaluator.get("expected")
    if not isinstance(result, dict) or result.get("type") != "vm_file":
        return None
    if not isinstance(expected, dict) or expected.get("type") != "cloud_file":
        return None
    result_path = result.get("path")
    expected_dest = expected.get("dest")
    if not isinstance(result_path, str) or not isinstance(expected_dest, str):
        return None
    image_suffixes = {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
    if (
        Path(result_path).suffix.casefold() not in image_suffixes
        or Path(expected_dest).suffix.casefold() not in image_suffixes
    ):
        return None

    candidates: list[str] = []
    for item in task_config.get("config") or []:
        if not isinstance(item, dict) or item.get("type") != "download":
            continue
        parameters = item.get("parameters") or {}
        for file_config in parameters.get("files") or []:
            if not isinstance(file_config, dict):
                continue
            input_path = file_config.get("path")
            if (
                isinstance(input_path, str)
                and Path(input_path).name == Path(expected_dest).name
            ):
                candidates.append(input_path)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1 or candidates[0] == result_path:
        return None
    return candidates[0], result_path


def _capture_legacy_in_place_image(
    task_config: dict, env
) -> tuple[str, str, str] | None:
    paths = _legacy_in_place_image_paths(task_config)
    if paths is None:
        return None
    input_path, result_path = paths
    try:
        payload = env.controller.get_file(input_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed to fingerprint legacy image input %s: %s", input_path, exc
        )
        return None
    if payload is None:
        return None
    return input_path, result_path, hashlib.sha256(payload).hexdigest()


def _materialize_changed_legacy_image(
    env,
    snapshot: tuple[str, str, str] | None,
) -> bool:
    if snapshot is None:
        return False
    input_path, result_path, original_digest = snapshot
    try:
        payload = env.controller.get_file(input_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to re-read legacy image input %s: %s", input_path, exc)
        return False
    if payload is None or hashlib.sha256(payload).hexdigest() == original_digest:
        return False

    destination = str(Path(result_path).parent)
    env.setup_controller.execute(
        [
            "bash",
            "-c",
            f"mkdir -p {shlex.quote(destination)} && "
            f"cp -- {shlex.quote(input_path)} {shlex.quote(result_path)}",
        ],
        quiet=True,
        timeout=30,
    )
    evaluator = getattr(env, "evaluator", None)
    if isinstance(evaluator, dict):
        evaluator["postconfig"] = []
    logger.info(
        "materialized in-place image edit for evaluator: %s -> %s",
        input_path,
        result_path,
    )
    return True


def _stage_legacy_osworld_expected_files(task_config: dict, env) -> None:
    evaluator = task_config.get("evaluator") or {}
    expected = evaluator.get("expected")
    entries = expected if isinstance(expected, list) else [expected]
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "cloud_file":
            continue
        cache_dir = Path(str(env.cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = entry.get("path")
        destinations = entry.get("dest")
        paths = paths if isinstance(paths, list) else [paths]
        destinations = destinations if isinstance(destinations, list) else [destinations]
        for source_value, destination_value in zip(paths, destinations):
            source = _resolve_local_setup_source(str(source_value or ""))
            if source is None or not destination_value:
                continue
            destination = cache_dir / str(destination_value)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def _load_osworld_task_class(evaluator: dict) -> type:
    path = Path(str(evaluator.get("path") or "")).expanduser().resolve()
    repository_root = Path(
        str(evaluator.get("repository_root") or path.parents[2])
    ).expanduser().resolve()
    class_name = str(evaluator.get("class_name") or "TASK_CLASS")
    cache_key = (str(path), class_name)
    cached = _TASK_CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(f"OSWorld task class not found: {path}")
    repository_value = str(repository_root)
    if repository_value not in sys.path:
        sys.path.insert(0, repository_value)
    module_name = f"_gui_trajectory_osworld_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load OSWorld task module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task_class = getattr(module, class_name, None)
    if task_class is None and class_name != "TASK_CLASS":
        task_class = getattr(module, "TASK_CLASS", None)
    if task_class is None:
        raise AttributeError(f"{path} does not define {class_name} or TASK_CLASS")
    _TASK_CLASS_CACHE[cache_key] = task_class
    return task_class


def _evaluator_score(result) -> tuple[float, object]:
    if isinstance(result, dict):
        raw_score = result.get("overall_score", result.get("score", 0.0))
        return float(raw_score or 0.0), result
    return float(result or 0.0), result


def _agent_execution_failed(record: dict) -> bool:
    error_parts = [
        part.strip()
        for part in str(record.get("error") or "").split("|")
        if part.strip()
    ]
    if any(not part.casefold().startswith("judge_failed:") for part in error_parts):
        return True
    if record.get("agent_done") is False:
        return True
    agent_status = str(record.get("agent_status") or "").strip().lower()
    if agent_status and agent_status != "ok":
        return True
    agent_exit = record.get("agent_exit")
    return agent_exit not in (None, 0, "0")


def _write_task_metadata(output_dir: Path, task: dict, row: dict | None) -> None:
    path = output_dir / "task.json"
    if path.is_file():
        return
    manifest_row = row if isinstance(row, dict) else {}
    instruction = str(task.get("prompt") or task.get("instruction") or "").strip()
    payload = {
        "schema_version": "gui_trajectory.task.v2",
        "id": str(task.get("task_id") or manifest_row.get("id") or output_dir.name),
        "instruction": instruction,
        "snapshot": str(
            task.get("snapshot")
            or manifest_row.get("environment_id")
            or "clean"
        ),
        "config": manifest_row.get("runtime_config") or task.get("config") or [],
        "evaluator": manifest_row.get("evaluator") or task.get("evaluator") or {},
        "required_capabilities": manifest_row.get("required_capabilities") or [],
        "target_environments": manifest_row.get("target_environments") or [],
        "metadata": {
            "category": task.get("category"),
            "name": task.get("name"),
            "source_markdown": manifest_row.get("markdown"),
            "environment_id": manifest_row.get("environment_id"),
            "environment_digest": manifest_row.get("environment_digest"),
        },
        "version": manifest_row.get("version"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _compatible_output_dir(task: dict, mode: str, output_dir: Path) -> Path:
    model = os.environ.get("WEAVEBENCH_MODEL", "")
    result_root = os.environ.get("GUI_TRAJECTORY_RESULT_DIR", "")
    if not result_root or not Path(model).is_absolute():
        return Path(output_dir)
    result_mode = (
        os.environ.get("GUI_TRAJECTORY_REQUESTED_MODE", "").strip().lower()
        or mode
    )
    model_name = Path(model).name
    if not model_name:
        raise ValueError(f"model path has no usable name: {model!r}")
    return (
        Path(result_root)
        / result_mode
        / model_name
        / str(task["category"])
        / str(task["task_id"])
    )


def _score_allows_resume_skip(
    task: dict, mode: str, model: str, result_root: Path
) -> bool:
    candidates = [
        Path(result_root)
        / mode
        / model
        / str(task["category"])
        / str(task["task_id"])
    ]
    for subdir in Path(result_root).iterdir() if Path(result_root).exists() else []:
        if subdir.is_dir():
            candidates.append(
                subdir
                / mode
                / model
                / str(task["category"])
                / str(task["task_id"])
            )
    return any(
        path.is_file() and path.stat().st_size > 0
        for task_dir in candidates
        for path in (
            task_dir / "trajectory.jsonl",
            task_dir / "canonical_trajectory.jsonl",
            task_dir / "computer_events.jsonl",
        )
    )


def _install_osworld_task_adapter(runner_module) -> None:
    inner_run_one = runner_module.ds.run_one
    osworld_run_one = getattr(runner_module.ds, "_original_run_one", inner_run_one)
    inner_already_done = getattr(runner_module.ds, "already_done", None)

    def already_done_with_model_path(task, mode, model, result_root):
        mode = (
            os.environ.get("GUI_TRAJECTORY_REQUESTED_MODE", "").strip().lower()
            or mode
        )
        if Path(model).is_absolute():
            model = Path(model).name
        inner_already_done(task, mode, model, result_root)
        return _score_allows_resume_skip(task, mode, model, Path(result_root))

    def run_one_with_osworld_task(
        env, agent, task, mode, output_dir, tasks_root, **kwargs
    ):
        output_dir = _compatible_output_dir(task, mode, Path(output_dir))
        row = _task_manifest_rows(Path(tasks_root)).get(str(task.get("task_id")))
        _write_task_metadata(output_dir, task, row)
        evaluator = row.get("evaluator") if isinstance(row, dict) else None
        adapter = row.get("execution_adapter") if isinstance(row, dict) else None
        if adapter != "legacy_osworld" and (
            not isinstance(evaluator, dict)
            or evaluator.get("type") != "osworld_v2_task_class"
        ):
            return inner_run_one(
                env, agent, task, mode, output_dir, tasks_root, **kwargs
            )

        original_setup_controller = env.setup_controller
        setup_controller = _osworld_setup_controller(original_setup_controller)
        env.setup_controller = setup_controller
        legacy_image_snapshot = None
        try:
            _cleanup_guest_runtime(setup_controller)
            if adapter == "legacy_osworld":
                _install_legacy_osworld_evaluators(env)
                task_config = _legacy_osworld_task_config(task, row)
                if task_config["proxy"]:
                    _enable_legacy_osworld_proxy(env)
                env.reset(task_config)
                _stage_legacy_osworld_expected_files(task_config, env)
                legacy_image_snapshot = _capture_legacy_in_place_image(
                    task_config, env
                )
                evaluator_name = "legacy_osworld"
                task_instance = None
            else:
                task_class = _load_osworld_task_class(evaluator)
                task_instance = task_class()
                if hasattr(env, "reset"):
                    env.reset(None)
                task_instance.setup(
                    setup_controller,
                    use_proxy=bool(getattr(task_instance, "proxy", False)),
                )
                evaluator_name = "osworld_v2_task_class"
            if not hasattr(env, "cache_dir") and hasattr(setup_controller, "cache_dir"):
                env.cache_dir = setup_controller.cache_dir
            if hasattr(env, "is_environment_used"):
                env.is_environment_used = True

            record = osworld_run_one(
                env, agent, task, mode, output_dir, tasks_root, **kwargs
            )
            judge_score = record.get("score")
            scores = dict(record.get("scores") or {})
            scores["agent_judge_score"] = judge_score
            if _agent_execution_failed(record):
                record.update({
                    "score": 0.0,
                    "scores": scores,
                    "success": False,
                    "evaluator": evaluator_name,
                    "evaluator_version": row.get("version"),
                    "evaluator_skipped": "agent_execution_failed",
                })
                logger.warning(
                    "skipping OSWorld evaluator for %s after agent execution failure",
                    task.get("task_id"),
                )
            else:
                try:
                    if task_instance is None:
                        _materialize_changed_legacy_image(
                            env,
                            legacy_image_snapshot,
                        )
                    result = (
                        env.evaluate()
                        if task_instance is None
                        else task_instance.evaluate(env)
                    )
                    score, detail = _evaluator_score(result)
                    scores[evaluator_name] = detail
                    record.update({
                        "score": score,
                        "scores": scores,
                        "success": score >= 1.0,
                        "evaluator": evaluator_name,
                        "evaluator_version": row.get("version"),
                    })
                except Exception as exc:  # noqa: BLE001
                    message = f"osworld_evaluator_failed: {type(exc).__name__}: {exc}"
                    record["error"] = " | ".join(
                        value for value in (record.get("error"), message) if value
                    )
                    record["success"] = False
                    logger.exception(
                        "OSWorld evaluator failed for %s", task.get("task_id")
                    )
            output_path = Path(output_dir) / "score.json"
            output_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            return record
        finally:
            env.setup_controller = original_setup_controller

    if inner_already_done is not None:
        runner_module.ds.already_done = already_done_with_model_path
    runner_module.ds.run_one = run_one_with_osworld_task


def _runner_module(harness: str, transport: str):
    if harness == "openclaw":
        resolved = transport
        if resolved == "auto":
            model = os.environ.get("WEAVEBENCH_MODEL", "").casefold()
            resolved = "messages" if "claude" in model else "chat"
        module_name = {
            "chat": "weavebench.eval._runners.chat_gui",
            "messages": "weavebench.eval._runners.messages_gui",
            "responses": "weavebench.eval._runners.responses_gui",
        }.get(resolved)
        if module_name is None:
            raise ValueError(f"unsupported OpenClaw transport: {transport!r}")
    elif harness == "claudecode":
        module_name = "weavebench.eval._runners.messages_gui"
    elif harness in {"codex", "hermes"}:
        module_name = "weavebench.eval._runners.responses_gui"
    else:
        raise ValueError(f"unsupported WEAVEBENCH_HARNESS: {harness!r}")
    return importlib.import_module(module_name)


def _install_hybrid_mode(messages_gui, argv: list[str] | None = None) -> None:
    """Translate --mode hybrid for the upstream messages_gui runner.

    messages_gui only accepts cli/gui/both. In the guest runtime the
    claudecode agent exposes both the GUI MCP tool and Bash, so hybrid must run
    with upstream mode="gui" to select the GUI prompt and initialize
    screenshots. The requested hybrid mode is retained for result paths and
    score records.
    """
    argv = sys.argv if argv is None else argv
    try:
        mode_index = argv.index("--mode")
    except ValueError:
        return
    if mode_index + 1 >= len(argv):
        return
    if argv[mode_index + 1].strip().lower() != "hybrid":
        return
    runtime_location = os.environ.get(
        "GUI_TRAJECTORY_RUNTIME_LOCATION", "guest"
    ).strip().lower()
    if runtime_location == "host" and not _truthy(
        os.environ.get("GUI_TRAJECTORY_ENABLE_GUEST_SHELL")
    ):
        raise ValueError(
            "mode=hybrid with the host runtime requires the guest shell MCP bridge"
        )
    if runtime_location not in {"guest", "host"}:
        raise ValueError(f"unsupported GUI_TRAJECTORY_RUNTIME_LOCATION: {runtime_location!r}")
    os.environ["GUI_TRAJECTORY_REQUESTED_MODE"] = "hybrid"
    argv[mode_index + 1] = "gui"
    inner_run_one = messages_gui.ds.run_one

    def run_one_hybrid(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        record = inner_run_one(
            env, agent, task, "gui", output_dir, tasks_root, **kwargs
        )
        record["mode"] = "hybrid"
        score_path = (
            _compatible_output_dir(task, "gui", Path(output_dir)) / "score.json"
        )
        if score_path.exists():
            score_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
        return record

    messages_gui.ds.run_one = run_one_hybrid


def main() -> int:
    global _worker_monitor_failure
    _worker_monitor_failure = None
    harness = os.environ.get("WEAVEBENCH_HARNESS", "claudecode").strip().lower()
    if harness not in {"openclaw", "codex", "claudecode", "hermes"}:
        raise ValueError(f"unsupported WEAVEBENCH_HARNESS: {harness!r}")
    os.environ["WEAVEBENCH_HARNESS"] = harness
    _install_vm_provider()
    _install_screen_size_fallback()
    _install_agent_adapter(harness)
    transport = os.environ.get("WEAVEBENCH_TRANSPORT", "auto").strip().lower()
    runner_module = _runner_module(harness, transport)

    _install_task_platform_filter(runner_module)
    _install_osworld_task_adapter(runner_module)
    _install_hybrid_mode(runner_module)
    _install_task_endpoint_failover(runner_module)
    _install_worker_exit_monitor(runner_module)

    with _shared_remote_ssh_control():
        result = runner_module.main()
    result_code = int(result or 0)
    if _worker_monitor_failure and result_code == 0:
        logger.error("worker monitor failed: %s", _worker_monitor_failure)
        return 1
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
