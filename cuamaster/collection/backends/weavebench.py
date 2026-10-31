"""Optional compatibility backend for the existing WeaveBench runtime."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from cuamaster.benchmarks import get_interaction_harness
from cuamaster.runtime import DEFAULT_ACTION_PROTOCOL, SUPPORTED_ACTION_PROTOCOLS

from .base import CollectionBackend, RunSpec
from ..profiles import MatrixConfig, VMProfile

from ..providers.sandbox import SANDBOX_PROVIDER_NAMES, SANDBOX_PROVIDER_ENV

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sandbox_commands(
    options: Mapping[str, Any], base_env: Mapping[str, str]
) -> tuple[str, str, dict[str, Any]]:
    """Resolve the orchestrator start/stop command templates for the sandbox
    provider. The templates are expanded by the provider with the fields
    documented on ``CommandSandboxConfig``; the start command must print the
    leased VM's endpoints as one JSON object on its final stdout line."""
    start_command = str(
        options.get("sandbox_start_command")
        or base_env.get("CUAMASTER_SANDBOX_START_COMMAND")
        or ""
    ).strip()
    stop_command = str(
        options.get("sandbox_stop_command")
        or base_env.get("CUAMASTER_SANDBOX_STOP_COMMAND")
        or ""
    ).strip()
    if not start_command or not stop_command:
        raise ValueError(
            "the sandbox provider requires backend_options.sandbox_start_command "
            "and backend_options.sandbox_stop_command (or the matching "
            "CUAMASTER_SANDBOX_START_COMMAND/CUAMASTER_SANDBOX_STOP_COMMAND "
            "environment variables) pointing at your VM orchestrator CLI"
        )
    metadata = {
        "start_command": start_command,
        "stop_command": stop_command,
    }
    return start_command, stop_command, metadata


class WeaveBenchBackend(CollectionBackend):
    def prepare(
        self,
        *,
        config: MatrixConfig,
        profile: VMProfile,
        output_dir: Path,
        cli: Any,
        base_env: Mapping[str, str],
    ) -> RunSpec:
        defaults = config.defaults
        options = {
            **(defaults.get("backend_options") or {}),
            **profile.backend_options,
            **getattr(cli, "backend_options", {}),
        }
        action_protocol = str(
            options.get("action_protocol") or DEFAULT_ACTION_PROTOCOL
        ).strip()
        if action_protocol not in SUPPORTED_ACTION_PROTOCOLS - {"auto"}:
            raise ValueError(
                "WeaveBench action_protocol must be canonical or qwen_xml"
            )
        weavebench_root_value = (
            options.get("root")
            or defaults.get("weavebench_root")
            or base_env.get("WEAVEBENCH_ROOT")
        )
        if not weavebench_root_value:
            raise ValueError(
                "WeaveBench backend requires backend_options.root, "
                "defaults.weavebench_root, or the WEAVEBENCH_ROOT environment "
                "variable pointing at a WeaveBench checkout"
            )
        weavebench_root = _path(weavebench_root_value, config.path.parent)
        if not (weavebench_root / "weavebench/eval/_runners/messages_gui.py").is_file():
            raise FileNotFoundError(f"invalid WeaveBench root: {weavebench_root}")
        configured_provider = profile.provider.strip().lower()
        if configured_provider == "manager":
            raise ValueError(
                "WeaveBench provider=manager is disabled because it uses the legacy "
                "pre-provisioned manager service; use provider=remote_ssh"
            )
        sandbox = configured_provider in SANDBOX_PROVIDER_NAMES
        remote_ssh = configured_provider in {"remote_ssh", "ssh"}
        external_vm_provider = remote_ssh or sandbox
        manager_remote_catalog = (
            sandbox
            and bool(profile.image_id)
            and _truthy(options.get("sandbox_remote_catalog", True))
        )
        if (
            profile.image_path
            and not Path(profile.image_path).is_file()
            and not manager_remote_catalog
        ):
            raise FileNotFoundError(
                f"profile {profile.id}: image not found: {profile.image_path}"
            )
        pool_metadata: dict[str, Any] = {}
        if sandbox:
            start_command, stop_command, pool_metadata = _sandbox_commands(
                options, base_env
            )
            options.setdefault("start_timeout_seconds", 2400)
            options.setdefault("remote_start_command", start_command)
            options.setdefault("remote_stop_command", stop_command)
        configured_interaction_harness = options.get("interaction_harness")
        legacy_mode = options.get("mode")
        if configured_interaction_harness and legacy_mode:
            configured_name = str(configured_interaction_harness).strip().lower()
            legacy_name = str(legacy_mode).strip().lower()
            if configured_name != legacy_name:
                raise ValueError(
                    f"profile {profile.id}: interaction_harness={configured_name!r} "
                    f"conflicts with legacy mode={legacy_name!r}"
                )
        try:
            interaction_harness = get_interaction_harness(
                configured_interaction_harness or legacy_mode or "gui"
            )
        except ValueError as exc:
            raise ValueError(
                f"profile {profile.id}: {exc}"
            ) from exc
        mode = interaction_harness.name
        remote_image_path = options.get("remote_image_path") or profile.image_path
        manager_image_path = (
            str(options.get("sandbox_image_path") or "").strip()
            if manager_remote_catalog
            else str(remote_image_path or "").strip()
        )
        endpoint_pool = str(options.get("model_endpoint_pool") or "").strip()
        litellm_base_url = str(
            options.get("litellm_base_url")
            or options.get("model_endpoint")
            or ""
        ).strip()
        model = options.get("model") or defaults.get("model")
        endpoint_pool_api_key_env = None
        if endpoint_pool:
            from cuamaster.endpoints import (
                load_endpoint_pool_config,
                resolve_endpoint_pool_path,
            )

            endpoint_pool_path = resolve_endpoint_pool_path(
                endpoint_pool, config.path.parent
            )
            endpoint_pool_config = load_endpoint_pool_config(endpoint_pool_path)
            endpoint_pool = str(endpoint_pool_path)
            model = model or endpoint_pool_config.model
            endpoint_pool_api_key_env = endpoint_pool_config.api_key_env
            if not litellm_base_url:
                litellm_base_url = endpoint_pool_config.base_urls[0]
        harness = str(options.get("harness") or "claudecode").strip().lower()
        if harness not in {"openclaw", "codex", "claudecode", "hermes"}:
            raise ValueError(
                "WeaveBench backend harness must be openclaw, codex, claudecode, or hermes"
            )
        transport = str(options.get("transport") or "auto").strip().lower()
        if transport not in {"auto", "chat", "messages", "responses"}:
            raise ValueError(
                "WeaveBench backend transport must be auto, chat, messages, or responses"
            )
        if harness != "openclaw" and transport != "auto":
            raise ValueError(
                f"profile {profile.id}: transport is only configurable for openclaw"
            )
        runtime_location = str(
            options.get("runtime_location")
            or (
                "host"
                if external_vm_provider or profile.os_type.casefold() != "ubuntu"
                else "guest"
            )
        ).strip().lower()
        if runtime_location not in {"guest", "host"}:
            raise ValueError("WeaveBench runtime_location must be guest or host")
        if runtime_location == "guest" and profile.os_type.casefold() not in {
            "ubuntu", "windows", "darwin"
        }:
            raise ValueError(
                f"profile {profile.id}: guest runtime supports Ubuntu, Windows, and Darwin only"
            )
        if runtime_location == "host" and harness != "claudecode":
            raise ValueError(
                "WeaveBench host runtime currently supports the claudecode harness only"
            )
        host_guest_shell = (
            runtime_location == "host"
            and harness == "claudecode"
            and interaction_harness.supports_cli
        )
        if (
            interaction_harness.supports_cli
            and runtime_location != "guest"
            and not host_guest_shell
        ):
            raise ValueError(
                f"profile {profile.id}: mode={mode!r} requires runtime_location=guest; "
                "only the Claude Code host runtime can proxy shell commands into the guest"
            )
        image_strategy = str(
            options.get("image_strategy") or "runtime-overlay"
        ).strip().lower()
        if image_strategy not in {"runtime-overlay", "baked"}:
            raise ValueError(
                "WeaveBench backend image_strategy must be runtime-overlay or baked"
            )
        if image_strategy == "runtime-overlay" and not (
            configured_provider == "docker" or external_vm_provider
        ):
            raise ValueError(
                "WeaveBench runtime-overlay image strategy requires docker, remote_ssh, "
                "or the sandbox provider"
            )
        if remote_ssh:
            if options.get("ssh_password"):
                raise ValueError(
                    "do not store ssh_password in config; use the environment variable "
                    "named by ssh_password_env"
                )
            if not options.get("ssh_host"):
                raise ValueError(
                    f"{configured_provider} provider requires backend_options.ssh_host"
                )
            if not remote_image_path:
                raise ValueError(
                    f"{configured_provider} provider requires an image resolved from "
                    "image.path/image.id "
                    "or backend_options.remote_image_path"
                )
        if sandbox and not (profile.image_id or manager_image_path):
            raise ValueError(
                "the sandbox provider requires image.id, image.path, "
                "backend_options.remote_image_path, or "
                "backend_options.sandbox_image_path"
            )
        tasks_value = (
            options.get("tasks_root")
            or defaults.get("tasks_root")
        )
        if not model or not tasks_value:
            raise ValueError("WeaveBench backend requires model and tasks_root")
        tasks_root = _path(tasks_value, config.path.parent)
        provider_name = "docker" if external_vm_provider else profile.provider
        command = [
            sys.executable,
            "-m", "cuamaster.collection.weavebench_runner",
            "--provider_name", provider_name,
            "--os_type", profile.os_type,
            "--num_envs", str(profile.num_envs),
            "--model", str(model),
            "--tasks_root", str(tasks_root),
            "--bench_subdirs", str(options.get("bench_subdirs") or defaults.get("bench_subdirs") or "."),
            "--categories", str(
                options.get("domains")
                or defaults.get("domains")
                or "DAV,DES,DOC,DSK,GAM,OPS,SPA,WEB"
            ),
            "--mode", mode,
            "--result_dir", str(output_dir),
            "--max_steps", str(int(options.get("max_steps") or defaults.get("max_steps") or 300)),
        ]
        monitor_max_seconds = options.get(
            "monitor_max_seconds", defaults.get("monitor_max_seconds")
        )
        if monitor_max_seconds is not None:
            monitor_max_seconds = int(monitor_max_seconds)
            if monitor_max_seconds < 1:
                raise ValueError("monitor_max_seconds must be >= 1")
            command += ["--monitor_max_seconds", str(monitor_max_seconds)]
        if profile.image_path and not manager_remote_catalog:
            command += ["--path_to_vm", profile.image_path]
        if litellm_base_url:
            command += ["--litellm_base_url", litellm_base_url]
        for key in ("task_filter", "limit", "region", "client_password"):
            value = options.get(key) or defaults.get(key)
            if value not in (None, "", 0):
                command += [f"--{key}", str(value)]
        env = dict(base_env)
        env.update(profile.env)
        python_paths = [str(REPO_ROOT), str(weavebench_root)]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env.update({
            "PYTHONPATH": os.pathsep.join(python_paths),
            "WEAVEBENCH_HARNESS": harness,
            "WEAVEBENCH_TRANSPORT": transport,
            "WEAVEBENCH_MODEL": str(model),
            "GUI_TRAJECTORY_RESULT_DIR": str(output_dir),
            "GUI_TRAJECTORY_REQUESTED_MODE": mode,
            "CUA_INTERACTION_HARNESS": mode,
            "GUI_TRAJECTORY_RUNTIME_LOCATION": runtime_location,
            "GUI_TRAJECTORY_ACTION_PROTOCOL": action_protocol,
            "CUA_ACTION_PROTOCOL": action_protocol,
            "GUI_TRAJECTORY_ENABLE_GUEST_SHELL": str(host_guest_shell).lower(),
            "GUI_TRAJECTORY_REQUIRE_EXPLICIT_PLATFORM": str(
                options.get(
                    "require_explicit_platform",
                    profile.os_type.casefold() != "ubuntu",
                )
            ).lower(),
            "GUI_TRAJECTORY_FORCE_ALL_TASKS_ON_PLATFORM": str(
                _truthy(options.get("force_all_tasks_on_platform"))
            ).lower(),
            "WEAVEBENCH_ASSETS_DIR": str(PACKAGE_ROOT / "assets"),
            "WEAVEBENCH_IMAGE_STRATEGY": image_strategy,
            "WEAVEBENCH_VM_PROFILE_ID": profile.id,
            "WEAVEBENCH_VM_PROFILE_JSON": json.dumps(
                profile.as_metadata(), ensure_ascii=False, separators=(",", ":")
            ),
        })
        legacy_proxy_enabled = _truthy(
            options.get("legacy_osworld_enable_proxy")
        )
        legacy_proxy_url_env = str(
            options.get("legacy_osworld_proxy_url_env") or "OSWORLD_PROXY_URL"
        ).strip()
        if legacy_proxy_enabled:
            if not legacy_proxy_url_env:
                raise ValueError(
                    "legacy_osworld_proxy_url_env must name an environment variable"
                )
            legacy_proxy_url = str(env.get(legacy_proxy_url_env) or "").strip()
            parsed_proxy = urlparse(legacy_proxy_url)
            if (
                parsed_proxy.scheme not in {"http", "https"}
                or not parsed_proxy.hostname
                or parsed_proxy.port is None
            ):
                raise ValueError(
                    "legacy OSWorld proxy is enabled, but environment variable "
                    f"{legacy_proxy_url_env} does not contain a valid HTTP proxy URL"
                )
            env["GUI_TRAJECTORY_LEGACY_OSWORLD_PROXY_ENABLED"] = "true"
            env["GUI_TRAJECTORY_LEGACY_OSWORLD_PROXY_URL"] = legacy_proxy_url
        task_timeout_seconds = options.get(
            "task_timeout_seconds", defaults.get("task_timeout_seconds")
        )
        if task_timeout_seconds is not None:
            task_timeout_seconds = int(task_timeout_seconds)
            if task_timeout_seconds < 1:
                raise ValueError("task_timeout_seconds must be >= 1")
            env["GUI_TRAJECTORY_TASK_TIMEOUT_SECONDS"] = str(task_timeout_seconds)
        openclaw_tarball = weavebench_root / "cache/runtime_assets/openclaw.tar.gz"
        if harness == "openclaw" and openclaw_tarball.is_file():
            env.setdefault("OSWORLD_LOCAL_OPENCLAW_TAR_GZ", str(openclaw_tarball))
        if litellm_base_url:
            env["WEAVEBENCH_LITELLM_BASE_URL"] = litellm_base_url
            env["OPENROUTER_BASE_URL"] = litellm_base_url
        if endpoint_pool:
            env["GUI_TRAJECTORY_ENDPOINT_POOL"] = endpoint_pool
        osworld_evaluator_root = options.get(
            "osworld_evaluator_root", defaults.get("osworld_evaluator_root")
        )
        if osworld_evaluator_root:
            env["GUI_TRAJECTORY_OSWORLD_EVALUATOR_ROOT"] = str(
                _path(osworld_evaluator_root, config.path.parent)
            )
        api_key_env = str(
            options.get("litellm_api_key_env")
            or endpoint_pool_api_key_env
            or "WEAVEBENCH_LITELLM_KEY"
        ).strip()
        if api_key_env and env.get(api_key_env):
            env["WEAVEBENCH_LITELLM_KEY"] = env[api_key_env]
        elif options.get("allow_unauthenticated_model_endpoint"):
            env["WEAVEBENCH_LITELLM_KEY"] = "EMPTY"
        if litellm_base_url and env.get("WEAVEBENCH_LITELLM_KEY"):
            env["OPENROUTER_API_KEY"] = env["WEAVEBENCH_LITELLM_KEY"]
            env.setdefault("OPENAI_API_KEY", env["WEAVEBENCH_LITELLM_KEY"])
            env.setdefault("OPENAI_BASE_URL", litellm_base_url)
        judge_model = options.get("judge_model")
        if judge_model:
            env["WEAVEBENCH_JUDGE_MODEL"] = str(judge_model)
        if sandbox:
            vm_backend = str(
                options.get("sandbox_vm_backend")
                or ("macos_kvm" if profile.os_type.casefold() in {"darwin", "macos", "mac"} else "desktop_env")
            )
            env.update({
                SANDBOX_PROVIDER_ENV: configured_provider,
                "CUAMASTER_SANDBOX_IMAGE_ID": str(profile.image_id or ""),
                "CUAMASTER_SANDBOX_IMAGE_PATH": manager_image_path,
                "CUAMASTER_SANDBOX_VM_BACKEND": vm_backend,
                "CUAMASTER_SANDBOX_ARCHITECTURE": profile.architecture,
                "CUAMASTER_SANDBOX_START_COMMAND": str(
                    options["remote_start_command"]
                ),
                "CUAMASTER_SANDBOX_STOP_COMMAND": str(
                    options["remote_stop_command"]
                ),
                "CUAMASTER_SANDBOX_START_TIMEOUT_S": str(
                    options.get("start_timeout_seconds", 2400)
                ),
            })
        elif remote_ssh:
            password_env = str(
                options.get("ssh_password_env") or "GUI_TRAJECTORY_SSH_PASSWORD"
            )
            if env.get(password_env):
                env["GUI_TRAJECTORY_SSH_PASSWORD"] = env[password_env]
            remote_env = {
                "GUI_TRAJECTORY_REMOTE_SSH_PROVIDER": "1",
                "GUI_TRAJECTORY_SSH_HOST": options["ssh_host"],
                "GUI_TRAJECTORY_SSH_PORT": options.get("ssh_port", 22),
                "GUI_TRAJECTORY_SSH_USER": options.get("ssh_user", "root"),
                "GUI_TRAJECTORY_SSH_REMOTE_IMAGE_PATH": remote_image_path,
                "GUI_TRAJECTORY_SSH_DOCKER_IMAGE": options.get(
                    "remote_docker_image", "happysixd/osworld-docker:latest"
                ),
                "GUI_TRAJECTORY_SSH_STAGE_IMAGE": options.get("stage_image", False),
                "GUI_TRAJECTORY_SSH_ALLOW_PULL": options.get("allow_pull", False),
                "GUI_TRAJECTORY_SSH_START_TIMEOUT_S": options.get(
                    "start_timeout_seconds", 1800
                ),
                "GUI_TRAJECTORY_SSH_CONNECT_TIMEOUT_S": options.get(
                    "ssh_connect_timeout_seconds", 15
                ),
                "GUI_TRAJECTORY_SSH_STRICT_HOST_KEY_CHECKING": options.get(
                    "strict_host_key_checking", "accept-new"
                ),
                "GUI_TRAJECTORY_SSH_RAM_SIZE": options.get("ram_size", "8G"),
                "GUI_TRAJECTORY_SSH_CPU_CORES": options.get("cpu_cores", 4),
                "GUI_TRAJECTORY_SSH_DISK_SIZE": options.get("disk_size", "64G"),
            }
            optional_remote_env = {
                "GUI_TRAJECTORY_SSH_IDENTITY_FILE": options.get("ssh_identity_file"),
                "GUI_TRAJECTORY_SSH_KNOWN_HOSTS_FILE": options.get("known_hosts_file"),
                "GUI_TRAJECTORY_SSH_DOCKER_IMAGE_TAR": options.get(
                    "remote_docker_image_tar"
                ),
                "GUI_TRAJECTORY_SSH_REMOTE_START_COMMAND": options.get(
                    "remote_start_command"
                ),
                "GUI_TRAJECTORY_SSH_REMOTE_STOP_COMMAND": options.get(
                    "remote_stop_command"
                ),
            }
            env.update({str(key): str(value) for key, value in remote_env.items()})
            env.update({
                key: str(value)
                for key, value in optional_remote_env.items()
                if value not in (None, "")
            })
        if profile.image_id:
            env["WEAVEBENCH_VM_IMAGE_ID"] = profile.image_id
        if not (env.get("WEAVEBENCH_LITELLM_KEY") or env.get("OPENROUTER_API_KEY")):
            # Dry runs are handled by the collector before this becomes fatal.
            env["GUI_TRAJECTORY_MISSING_API_KEY"] = "1"
        return RunSpec(
            command=command,
            env=env,
            cwd=REPO_ROOT,
            metadata={
                "backend": "weavebench",
                "mode": mode,
                "interaction_harness": interaction_harness.as_dict(),
                "root": str(weavebench_root),
                "tasks_root": str(tasks_root),
                "harness": harness,
                "transport": transport,
                "action_protocol": action_protocol,
                "litellm_base_url": litellm_base_url or None,
                "model_endpoint_pool": endpoint_pool or None,
                "image_strategy": image_strategy,
                "runtime_location": runtime_location,
                "shell_runtime_location": "guest" if host_guest_shell else None,
                "controller_transport": "osworld-http" if runtime_location == "host" else "stdio",
                "vm_transport": (
                    "sandbox"
                    if sandbox
                    else "ssh" if remote_ssh else profile.provider
                ),
                "ssh_host": str(options["ssh_host"]) if remote_ssh else None,
                "ssh_port": int(options.get("ssh_port", 22)) if remote_ssh else None,
                "sandbox": pool_metadata or None,
                "legacy_osworld_proxy_enabled": legacy_proxy_enabled,
                "legacy_osworld_proxy_url_env": (
                    legacy_proxy_url_env if legacy_proxy_enabled else None
                ),
            },
        )
