"""Backend-agnostic CLI+GUI CUA collection over a versioned VM matrix."""
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import create_backend
from .export import export_result_tree
from .image_store import ImageStore
from .ledger import EpisodeLedger
from .profiles import (
    MatrixConfig,
    VMProfile,
    expand_platform_profiles,
    load_matrix_config,
)
from .providers.sandbox import SANDBOX_PROVIDER_NAMES
from .schemas import EpisodeStatus, content_digest
from .trace_store import TraceStore, atomic_write_json

DEFAULT_IMAGE_STORE = Path(__file__).resolve().parents[2] / "vm_images"
_OUTPUT_LOCK = threading.Lock()
_TOTAL_RUNS_PATTERN = re.compile(r"\btotal_runs=(\d+)")
_TASK_EVENT_PATTERN = re.compile(
    r"(?P<event>▶ START|⊘ SKIP|[✓✗] DONE|☠ CRASH)\s+"
    r"\[(?P<worker>[^]]+)]\[(?P<mode>[^]]+)]\s+(?P<task>\S+)"
)
_RUNTIME_ENDPOINT_KEYS = ("litellm_base_url", "model_endpoint")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _stable_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stable = copy.deepcopy(payload)
    defaults = stable.get("defaults") or {}
    default_options = defaults.get("backend_options") or {}
    default_pool = str(default_options.get("model_endpoint_pool") or "").strip()
    if default_pool:
        for key in _RUNTIME_ENDPOINT_KEYS:
            default_options.pop(key, None)

    for profile in stable.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        options = profile.get("backend_options") or {}
        endpoint_pool = str(
            options.get("model_endpoint_pool") or default_pool
        ).strip()
        if endpoint_pool:
            for key in _RUNTIME_ENDPOINT_KEYS:
                options.pop(key, None)
    return stable


def _legacy_runtime_endpoints(root: Path, run_id: str) -> set[str]:
    ledger_path = root / ".collection_state" / "ledger.sqlite3"
    if not ledger_path.is_file():
        return set()
    try:
        with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT metadata_json FROM episodes WHERE run_id=?",
                (run_id,),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return set()

    endpoints: set[str] = set()
    for (metadata_json,) in rows:
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            continue
        backend = metadata.get("backend") if isinstance(metadata, dict) else None
        if not isinstance(backend, dict):
            continue
        for key in _RUNTIME_ENDPOINT_KEYS:
            endpoint = str(backend.get(key) or "").strip()
            if endpoint:
                endpoints.add(endpoint)
    return endpoints


def _matches_legacy_config_digest(
    payload: dict[str, Any],
    expected_digest: str,
    root: Path,
    run_id: str,
) -> bool:
    stable = _stable_config_payload(payload)
    defaults = stable.get("defaults") or {}
    default_options = defaults.get("backend_options") or {}
    if not str(default_options.get("model_endpoint_pool") or "").strip():
        return False
    for endpoint in _legacy_runtime_endpoints(root, run_id):
        candidate = copy.deepcopy(stable)
        candidate_options = candidate.setdefault("defaults", {}).setdefault(
            "backend_options", {}
        )
        candidate_options["litellm_base_url"] = endpoint
        if content_digest(candidate) == expected_digest:
            return True
    return False


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _progress(message: str) -> None:
    with _OUTPUT_LOCK:
        print(f"[trajectory-collection] {message}", file=sys.stderr, flush=True)


class _BackendProgress:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.last_output_at = self.started_at
        self.total: int | None = None
        self.completed = 0
        self.active: dict[str, tuple[str, float]] = {}

    def observe(self, line: str) -> bool:
        now = time.monotonic()
        self.last_output_at = now
        total_match = _TOTAL_RUNS_PATTERN.search(line)
        if total_match:
            self.total = int(total_match.group(1))
        event_match = _TASK_EVENT_PATTERN.search(line)
        if not event_match:
            return False
        worker = event_match.group("worker")
        task = event_match.group("task")
        if event_match.group("event") == "▶ START":
            self.active[worker] = (task, now)
            return False
        self.active.pop(worker, None)
        self.completed += 1
        return True

    def message(self, *, heartbeat: bool = False) -> str:
        now = time.monotonic()
        elapsed = now - self.started_at
        throughput = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, (self.total or 0) - self.completed)
        eta = remaining / throughput if self.total is not None and throughput > 0 else None
        total = str(self.total) if self.total is not None else "?"
        active = ",".join(
            f"{worker}:{task}({_format_duration(now - started_at)})"
            for worker, (task, started_at) in sorted(self.active.items())
        ) or "none"
        parts = [
            f"progress={self.completed}/{total}",
            f"active={active}",
            f"throughput={throughput * 60:.2f}/min",
            f"elapsed={_format_duration(elapsed)}",
        ]
        if eta is not None:
            parts.append(f"eta={_format_duration(eta)}")
        if heartbeat:
            parts.append(
                f"silent_for={_format_duration(now - self.last_output_at)}"
            )
        return " ".join(parts)


def _finish_aborted_run(
    ledger: EpisodeLedger,
    run_id: str,
    error: BaseException,
) -> None:
    episode_status = (
        EpisodeStatus.CANCELLED
        if isinstance(error, KeyboardInterrupt)
        else EpisodeStatus.INFRA_FAILED
    )
    message = f"{type(error).__name__}: {error}".rstrip(": ")
    for row in ledger.rows(run_id=run_id):
        if row.get("status") == EpisodeStatus.RUNNING.value:
            ledger.finish_episode(
                str(row["episode_id"]),
                episode_status,
                error=message,
            )
    ledger.finish_run(
        run_id,
        "cancelled" if episode_status is EpisodeStatus.CANCELLED else "failed",
    )


def _run_streaming_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    prefix: str,
    progress_interval_seconds: float = 30.0,
) -> int:
    process_env = dict(env)
    process_env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        tracker = _BackendProgress()
        heartbeat_interval = max(0.0, float(progress_interval_seconds))
        next_heartbeat = time.monotonic() + heartbeat_interval
        while True:
            timeout = None
            if heartbeat_interval > 0:
                timeout = max(0.0, next_heartbeat - time.monotonic())
            try:
                line = output_queue.get(timeout=timeout)
            except queue.Empty:
                _progress(f"profile={prefix} {tracker.message(heartbeat=True)}")
                next_heartbeat = time.monotonic() + heartbeat_interval
                continue
            if line is None:
                break
            log.write(line)
            log.flush()
            with _OUTPUT_LOCK:
                print(f"[{prefix}] {line}", end="", file=sys.stderr, flush=True)
            if tracker.observe(line):
                _progress(f"profile={prefix} {tracker.message()}")
            if heartbeat_interval > 0:
                next_heartbeat = time.monotonic() + heartbeat_interval
        reader.join()
        return process.wait()


def _resolve_images(
    config: MatrixConfig,
    profiles: list[VMProfile],
    image_store_arg: str | None,
) -> list[VMProfile]:
    store_value = image_store_arg or config.defaults.get("image_store") or DEFAULT_IMAGE_STORE
    store_path = Path(str(store_value)).expanduser()
    if not store_path.is_absolute():
        store_path = config.path.parent / store_path
    store = ImageStore(store_path)
    resolved = []
    for profile in profiles:
        if not profile.image_id:
            resolved.append(profile)
            continue
        remote_catalog = (
            profile.provider.strip().lower() in SANDBOX_PROVIDER_NAMES
            and _truthy(
                profile.backend_options.get("sandbox_remote_catalog", True)
            )
        )
        try:
            image = store.resolve(profile.image_id, verify=not remote_catalog)
        except (FileNotFoundError, KeyError):
            if remote_catalog:
                resolved.append(profile)
                continue
            raise
        if profile.os_type.lower() not in {"unknown", image.os_type.lower()}:
            raise ValueError(
                f"profile {profile.id}: os.type={profile.os_type} conflicts with "
                f"image catalog os.type={image.os_type}"
            )
        resolved.append(replace(
            profile,
            image_path=None if remote_catalog else str(image.path),
            os_type=image.os_type if profile.os_type.lower() == "unknown" else profile.os_type,
            os_version=(
                image.os_version if profile.os_version == "unknown" else profile.os_version
            ),
            architecture=(
                image.architecture
                if profile.architecture == "unknown"
                else profile.architecture
            ),
            snapshot=image.snapshot if profile.snapshot == "clean" else profile.snapshot,
            software={**image.software, **profile.software},
            image_metadata=image.as_metadata(),
        ))
    return resolved


def _run_profile(
    config: MatrixConfig,
    profile: VMProfile,
    root: Path,
    args: argparse.Namespace,
    run_id: str,
    ledger: EpisodeLedger,
    trace_store: TraceStore,
) -> dict[str, Any]:
    output_dir = root / profile.id
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = create_backend(profile.backend)
    spec = backend.prepare(
        config=config,
        profile=profile,
        output_dir=output_dir,
        cli=args,
        base_env=os.environ,
    )
    if args.dry_run:
        return {
            "profile": profile.id,
            "status": "dry_run",
            "command": spec.command,
            "cwd": str(spec.cwd),
            "backend": spec.metadata,
            "metadata": profile.as_metadata(),
        }
    work_id = content_digest({"run_id": run_id, "profile_id": profile.id})[:32]
    existing = ledger.episode(work_id)
    if existing and existing.get("status") == EpisodeStatus.SUCCEEDED.value and not args.force_rerun:
        exported = export_result_tree(
            output_dir,
            profile,
            run_id=run_id,
            ledger=ledger,
            trace_store=trace_store,
        )
        return {
            "profile": profile.id,
            "status": "skipped_completed",
            "exported_trajectories": exported,
            "result_dir": str(output_dir),
            "backend": spec.metadata,
            "metadata": profile.as_metadata(),
        }
    ledger.register_episode(
        work_id,
        run_id=run_id,
        profile_id=profile.id,
        task_id="__profile__",
        agent_id=str(spec.metadata.get("harness") or spec.metadata.get("backend") or "backend"),
        output_dir=str(output_dir),
        metadata={"kind": "profile_backend_run", "backend": spec.metadata},
    )
    if existing and existing.get("status") not in {
        EpisodeStatus.PENDING.value, EpisodeStatus.RUNNING.value
    }:
        ledger.reset_episode(work_id)
    ledger.mark_running(work_id, worker_id="collector")
    try:
        if spec.env.pop("GUI_TRAJECTORY_MISSING_API_KEY", None):
            raise ValueError("set OPENROUTER_API_KEY or WEAVEBENCH_LITELLM_KEY")
        started = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        log_path = output_dir / "collector.log"
        _progress(f"profile={profile.id} start log={log_path}")
        returncode = _run_streaming_process(
            spec.command,
            cwd=spec.cwd,
            env=spec.env,
            log_path=log_path,
            prefix=profile.id,
            progress_interval_seconds=getattr(
                args, "progress_interval_seconds", 30.0
            ),
        )
        _progress(
            f"profile={profile.id} backend_done returncode={returncode} "
            f"elapsed={_format_duration(time.monotonic() - started_monotonic)}"
        )
        exported = export_result_tree(
            output_dir,
            profile,
            run_id=run_id,
            ledger=ledger,
            trace_store=trace_store,
        )
        ledger.finish_episode(
            work_id,
            EpisodeStatus.SUCCEEDED if returncode == 0 else EpisodeStatus.INFRA_FAILED,
            error=None if returncode == 0 else f"backend exited {returncode}",
        )
        return {
            "profile": profile.id,
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "exported_trajectories": exported,
            "result_dir": str(output_dir),
            "log": str(log_path),
            "backend": spec.metadata,
            "metadata": profile.as_metadata(),
        }
    except BaseException as exc:
        status = (
            EpisodeStatus.CANCELLED
            if isinstance(exc, KeyboardInterrupt)
            else EpisodeStatus.INFRA_FAILED
        )
        ledger.finish_episode(
            work_id,
            status,
            error=f"{type(exc).__name__}: {exc}".rstrip(": "),
        )
        raise


def _resolve_vm_concurrency(
    config: MatrixConfig,
    profiles: list[VMProfile],
    requested: int | None,
) -> int:
    configured = requested
    if configured is None:
        configured = config.defaults.get("concurrency")
    if configured is None:
        configured = config.defaults.get("max_concurrent_vms")
    if configured is None:
        configured = max(profile.num_envs for profile in profiles)
    concurrency = int(configured)
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    oversized = [profile for profile in profiles if profile.num_envs > concurrency]
    if oversized:
        details = ", ".join(
            f"{profile.id}={profile.num_envs}" for profile in oversized
        )
        raise ValueError(
            f"profile VM demand exceeds concurrency={concurrency}: {details}"
        )
    return concurrency


def _cap_profile_vm_concurrency(
    profiles: list[VMProfile],
    concurrency: int | None,
    *,
    enabled: bool,
) -> list[VMProfile]:
    if not enabled or concurrency is None:
        return profiles
    return [
        replace(profile, num_envs=min(profile.num_envs, concurrency))
        for profile in profiles
    ]


def _run_profiles(
    config: MatrixConfig,
    profiles: list[VMProfile],
    root: Path,
    args: argparse.Namespace,
    run_id: str,
    ledger: EpisodeLedger,
    trace_store: TraceStore,
) -> list[dict[str, Any]]:
    vm_concurrency = int(args.concurrency)
    max_parallel_profiles = args.max_parallel_profiles or len(profiles)
    worker_count = min(len(profiles), max_parallel_profiles)
    pending = list(profiles)
    futures = {}
    results: list[dict[str, Any]] = []
    active_vms = 0
    peak_vms = 0
    completed_profiles = 0
    collection_started = time.monotonic()
    _progress(
        f"start profiles={len(profiles)} concurrency={vm_concurrency} "
        f"max_parallel_profiles={max_parallel_profiles}"
    )

    def submit_ready(pool: ThreadPoolExecutor) -> None:
        nonlocal active_vms, peak_vms
        while pending and len(futures) < max_parallel_profiles:
            available = vm_concurrency - active_vms
            profile_index = next(
                (
                    index
                    for index, profile in enumerate(pending)
                    if profile.num_envs <= available
                ),
                None,
            )
            if profile_index is None:
                return
            profile = pending.pop(profile_index)
            future = pool.submit(
                _run_profile,
                config,
                profile,
                root,
                args,
                run_id,
                ledger,
                trace_store,
            )
            futures[future] = profile
            active_vms += profile.num_envs
            peak_vms = max(peak_vms, active_vms)
            _progress(
                f"profile={profile.id} scheduled vm_slots={profile.num_envs} "
                f"active_vms={active_vms}/{vm_concurrency}"
            )

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        submit_ready(pool)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                profile = futures.pop(future)
                active_vms -= profile.num_envs
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "profile": profile.id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "metadata": profile.as_metadata(),
                    }
                results.append(result)
                completed_profiles += 1
                elapsed = time.monotonic() - collection_started
                throughput = completed_profiles / elapsed if elapsed > 0 else 0.0
                eta = (
                    (len(profiles) - completed_profiles) / throughput
                    if throughput > 0
                    else 0.0
                )
                _progress(
                    f"[{completed_profiles}/{len(profiles)}] profile={profile.id} "
                    f"status={result.get('status')} active_vms={active_vms}/"
                    f"{vm_concurrency} peak_vms={peak_vms} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)
            submit_ready(pool)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cuamaster-collect",
        description="Collect CLI+GUI CUA trajectories over versioned VM profiles.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--image-store", default=None)
    parser.add_argument("--profiles", default=None, help="Optional CSV profile ids.")
    parser.add_argument(
        "--platform-mode",
        choices=("configured", "all"),
        default=None,
        help=(
            "Override defaults.platform_mode. 'all' runs every selected logical "
            "profile once on Linux, Windows, and macOS."
        ),
    )
    parser.add_argument(
        "--concurrency",
        "--max-concurrent-vms",
        dest="concurrency",
        type=int,
        default=None,
        help="Maximum number of VM environments active across all profiles.",
    )
    parser.add_argument(
        "--max-parallel-profiles",
        type=int,
        default=None,
        help="Optional legacy cap on concurrently running profiles.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=30.0,
        help="Emit a heartbeat while a backend task is still running; 0 disables it.",
    )
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Generic backend option override; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Run profiles again even when the durable ledger marks them completed.",
    )
    parser.add_argument(
        "--allow-config-change",
        action="store_true",
        help=(
            "Allow an existing output root to resume with an intentionally changed "
            "matrix configuration. The previous digest is recorded in run.json."
        ),
    )
    parser.add_argument(
        "--cap-profile-concurrency",
        action="store_true",
        help=(
            "Cap each selected profile's VM count at --concurrency instead of "
            "rejecting profiles configured with a larger num_envs value."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {}
    for value in args.backend_option:
        if "=" not in value:
            raise SystemExit(f"--backend-option must be KEY=VALUE: {value}")
        key, item = value.split("=", 1)
        overrides[key.strip()] = item
    args.backend_options = overrides
    if args.max_parallel_profiles is not None and args.max_parallel_profiles < 1:
        raise SystemExit("--max-parallel-profiles must be >= 1")
    if args.progress_interval_seconds < 0:
        raise SystemExit("--progress-interval-seconds must be >= 0")
    config = load_matrix_config(args.config)
    selected = (
        {value.strip() for value in args.profiles.split(",") if value.strip()}
        if args.profiles else None
    )
    try:
        profiles = expand_platform_profiles(
            config,
            config.enabled_profiles(selected),
            args.platform_mode,
        )
        profiles = _resolve_images(config, profiles, args.image_store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not profiles:
        raise SystemExit("no enabled VM profiles selected")
    profiles = _cap_profile_vm_concurrency(
        profiles,
        args.concurrency,
        enabled=args.cap_profile_concurrency,
    )
    try:
        args.concurrency = _resolve_vm_concurrency(config, profiles, args.concurrency)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_marker = root / ".collection_state" / "run.json"
    config_payload = json.loads(config.path.read_text(encoding="utf-8"))
    stable_config_payload = _stable_config_payload(config_payload)
    config_digest = content_digest(
        stable_config_payload
        if args.platform_mode is None
        else {"config": stable_config_payload, "platform_mode": args.platform_mode}
    )
    if run_marker.is_file():
        marker = json.loads(run_marker.read_text(encoding="utf-8"))
        marker_digest = marker.get("config_digest")
        legacy_matches = (
            args.platform_mode is None
            and isinstance(marker_digest, str)
            and _matches_legacy_config_digest(
                config_payload,
                marker_digest,
                root,
                str(marker.get("run_id") or ""),
            )
        )
        config_changed = marker_digest not in {None, config_digest} and not legacy_matches
        if config_changed and not args.allow_config_change:
            raise SystemExit(
                "output root belongs to a different config; choose a new output root"
            )
        if config_changed:
            history = marker.setdefault("config_digest_history", [])
            history.append({
                "config_digest": marker_digest,
                "replaced_at": datetime.now(timezone.utc).isoformat(),
            })
            _progress(
                "resuming existing output root with an explicitly allowed "
                "configuration change"
            )
        run_id = str(marker["run_id"])
        if marker_digest != config_digest:
            marker["config_digest"] = config_digest
            marker["config_digest_scope"] = "stable_runtime_routing_v1"
            atomic_write_json(run_marker, marker)
    else:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
            + content_digest({
                "config": stable_config_payload,
                "output_root": str(root),
            })[:12]
        )
        atomic_write_json(run_marker, {
            "run_id": run_id,
            "config": str(config.path),
            "config_digest": config_digest,
            "config_digest_scope": "stable_runtime_routing_v1",
        })
    ledger = EpisodeLedger(root / ".collection_state" / "ledger.sqlite3")
    ledger.create_run(run_id, config_path=str(config.path), metadata={"output_root": str(root)})
    ledger.recover_stale(24 * 60 * 60)
    trace_store = TraceStore(root / ".trace-index")
    try:
        if args.export_only:
            results = [
                {
                    "profile": profile.id,
                    "status": "exported",
                    "exported_trajectories": export_result_tree(
                        root / profile.id,
                        profile,
                        run_id=run_id,
                        ledger=ledger,
                        trace_store=trace_store,
                    ),
                }
                for profile in profiles
            ]
        else:
            results = _run_profiles(
                config,
                profiles,
                root,
                args,
                run_id,
                ledger,
                trace_store,
            )
    except BaseException as exc:
        _finish_aborted_run(ledger, run_id, exc)
        raise
    summary = {
        "schema_version": "gui_trajectory_collection.run.v2",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config.path),
        "concurrency": args.concurrency,
        "results": sorted(results, key=lambda item: item["profile"]),
    }
    atomic_write_json(root / "collection_summary.json", summary)
    successful = all(item.get("status") != "failed" for item in results)
    ledger.finish_run(run_id, "completed" if successful else "failed")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
