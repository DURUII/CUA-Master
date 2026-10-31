#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cuamaster.runtime import DEFAULT_ACTION_PROTOCOL, SUPPORTED_ACTION_PROTOCOLS  # noqa: E402
from cuamaster.endpoints import (  # noqa: E402
    DynamicEndpointPool,
    load_endpoint_pool_config,
    resolve_endpoint_pool_path,
)
from cuamaster.collection.judge import score_collection  # noqa: E402


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
    print(f"[task-trajectory] {message}", file=sys.stderr, flush=True)


def _load_config_data(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML pipeline configs require PyYAML") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("pipeline config must be an object")
    return value


def _config_path(value: Any, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"pipeline config {name} must be an object")
    return value


def _embedded_matrix(raw: dict[str, Any], base_dir: Path) -> dict[str, Any] | None:
    value = raw.get("vm_matrix")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("pipeline config vm_matrix must be an object")

    matrix = copy.deepcopy(value)
    matrix.setdefault("schema_version", 1)
    defaults = matrix.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("pipeline config vm_matrix.defaults must be an object")
    if defaults.get("image_store"):
        defaults["image_store"] = str(
            _config_path(defaults["image_store"], base_dir)
        )
    matrix["defaults"] = defaults

    profiles = matrix.get("profiles") or []
    if not isinstance(profiles, list):
        raise ValueError("pipeline config vm_matrix.profiles must be an array")
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        image = profile.get("image") or {}
        if isinstance(image, dict) and image.get("path"):
            image["path"] = str(_config_path(image["path"], base_dir))
            profile["image"] = image
    return matrix


def _pipeline_defaults(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    raw = _load_config_data(config_path)
    if raw.get("schema_version") != 1:
        raise ValueError("pipeline config schema_version must be 1")

    generation = _section(raw, "generation")
    collection = _section(raw, "collection")
    api = _section(raw, "api")
    scoring = _section(raw, "scoring")
    export_config = _section(raw, "export")
    configured_protocols = {
        str(value).strip()
        for value in (
            raw.get("action_protocol"),
            collection.get("action_protocol"),
            export_config.get("action_protocol"),
        )
        if value not in (None, "")
    }
    if len(configured_protocols) > 1:
        raise ValueError(
            "pipeline action_protocol must be identical at the top level, "
            "collection, and export stages"
        )
    action_protocol = next(iter(configured_protocols), DEFAULT_ACTION_PROTOCOL)
    if action_protocol not in SUPPORTED_ACTION_PROTOCOLS - {"auto"}:
        raise ValueError(
            "pipeline action_protocol must be canonical or qwen_xml"
        )
    base_dir = config_path.parent
    endpoint_pool_value = api.get("endpoint_pool")
    if endpoint_pool_value and (api.get("endpoint") or api.get("base_urls")):
        raise ValueError(
            "pipeline api.endpoint_pool cannot be combined with api.endpoint or api.base_urls"
        )
    endpoint_pool = None
    endpoint_pool_config = None
    if endpoint_pool_value:
        endpoint_pool = resolve_endpoint_pool_path(endpoint_pool_value, base_dir)
        endpoint_pool_config = load_endpoint_pool_config(endpoint_pool)
    scoring_endpoint_pool_value = scoring.get("endpoint_pool") or endpoint_pool_value
    scoring_endpoint_pool = (
        resolve_endpoint_pool_path(scoring_endpoint_pool_value, base_dir)
        if scoring_endpoint_pool_value
        else None
    )
    scoring_endpoint_pool_config = (
        load_endpoint_pool_config(scoring_endpoint_pool)
        if scoring_endpoint_pool is not None
        else None
    )
    api_key_env = api.get("api_key_env") or (
        endpoint_pool_config.api_key_env if endpoint_pool_config else None
    )
    values = {
        "run_dir": _config_path(raw.get("run_dir"), base_dir),
        "output_root": _config_path(raw.get("output_root"), base_dir),
        "generation_config": _config_path(generation.get("config"), base_dir),
        "pool": generation.get("pool"),
        "compile_format": generation.get("format"),
        "matrix_config": _config_path(collection.get("matrix_config"), base_dir),
        "embedded_matrix": _embedded_matrix(raw, base_dir),
        "concurrency": collection.get("concurrency"),
        "max_parallel_profiles": collection.get("max_parallel_profiles"),
        "progress_interval_seconds": collection.get("progress_interval_seconds"),
        "retry_failed_attempts": collection.get("retry_failed_attempts"),
        "action_protocol": action_protocol,
        "endpoint": api.get("endpoint"),
        "base_urls": (
            api.get("base_urls")
            if endpoint_pool_config is None
            else list(endpoint_pool_config.base_urls)
        ),
        "endpoint_pool": endpoint_pool,
        "model": api.get("model") or (
            endpoint_pool_config.model if endpoint_pool_config else None
        ),
        "api_key_env": api_key_env,
        "allow_unauthenticated": api.get(
            "allow_unauthenticated", not bool(api_key_env)
        ),
        "api_timeout": api.get("timeout_seconds"),
        "skip_api_check": api.get("skip_check"),
        "score_with_model": scoring.get("enabled"),
        "scoring_endpoint_pool": scoring_endpoint_pool,
        "scoring_model": scoring.get("model") or (
            scoring_endpoint_pool_config.model
            if scoring_endpoint_pool_config
            else None
        ),
        "scoring_api_key_env": scoring.get("api_key_env") or (
            scoring_endpoint_pool_config.api_key_env
            if scoring_endpoint_pool_config
            else None
        ),
        "scoring_allow_unauthenticated": scoring.get(
            "allow_unauthenticated",
            not bool(
                scoring.get("api_key_env")
                or (
                    scoring_endpoint_pool_config.api_key_env
                    if scoring_endpoint_pool_config
                    else None
                )
            ),
        ),
        "scoring_concurrency": scoring.get("concurrency"),
        "scoring_threshold": scoring.get("threshold"),
        "scoring_attempts": scoring.get("attempts"),
        "scoring_timeout": scoring.get("timeout_seconds"),
        "scoring_max_output_tokens": scoring.get("max_output_tokens"),
        "scoring_max_evidence_chars": scoring.get("max_evidence_chars"),
        "scoring_include_final_screenshot": scoring.get(
            "include_final_screenshot"
        ),
        "bundle_format": export_config.get("bundle_format"),
        "copy_images": export_config.get("copy_images"),
        "require_pre_screenshot": export_config.get("require_pre_screenshot"),
        "visual_window": export_config.get("visual_window"),
        "max_history_actions": export_config.get("max_history_actions"),
        "quality_policy": export_config.get("quality_policy"),
        "max_execution_output_chars": export_config.get(
            "max_execution_output_chars"
        ),
    }
    return {key: value for key, value in values.items() if value is not None}


def _configured_base_urls(args: argparse.Namespace) -> list[str]:
    if args.endpoint:
        return [str(args.endpoint)]
    value = args.base_urls or []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("pipeline config api.base_urls must be an array")
    return [str(item).strip() for item in value if str(item).strip()]


def _inject_api_backend_options(
    matrix: dict[str, Any], args: argparse.Namespace
) -> None:
    defaults = matrix.setdefault("defaults", {})
    default_options = defaults.setdefault("backend_options", {})
    if args.model:
        default_options["model"] = args.model
    if args.api_key_env:
        default_options["litellm_api_key_env"] = args.api_key_env
    if args.allow_unauthenticated:
        default_options["allow_unauthenticated_model_endpoint"] = True
    if args.endpoint_pool and not args.endpoint:
        default_options["model_endpoint_pool"] = str(args.endpoint_pool)
    default_options["action_protocol"] = args.action_protocol

    base_urls = _configured_base_urls(args)
    if not base_urls:
        return
    if args.endpoint_pool:
        default_options["litellm_base_url"] = base_urls[0]
        return
    if len(base_urls) == 1:
        default_options["litellm_base_url"] = base_urls[0]
        return

    default_options.pop("litellm_base_url", None)
    profiles = matrix.get("profiles") or []
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            continue
        options = profile.setdefault("backend_options", {})
        options["litellm_base_url"] = base_urls[index % len(base_urls)]


def _ensure_matrix_config(args: argparse.Namespace) -> Path:
    if args.matrix_config is not None:
        return args.matrix_config
    if args.embedded_matrix is None:
        raise RuntimeError("no VM matrix configured")

    matrix = copy.deepcopy(args.embedded_matrix)
    _inject_api_backend_options(matrix, args)
    matrix_path = args.output_root / ".pipeline" / "vm_matrix.json"
    content = json.dumps(matrix, ensure_ascii=False, indent=2) + "\n"
    if not matrix_path.is_file() or matrix_path.read_text(encoding="utf-8") != content:
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = matrix_path.with_suffix(".json.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(matrix_path)
    args.matrix_config = matrix_path
    return matrix_path


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    python_paths = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=_python_env(), check=True)


def _environment_id(task: dict[str, Any], recipe: dict[str, Any]) -> str:
    metadata = task.get("metadata") or {}
    return str(recipe.get("environment_id") or metadata.get("environment_id") or "")


def _validate_promoted(run_dir: Path, pool: str) -> dict[str, Any]:
    pool_dir = run_dir / pool
    files = sorted(pool_dir.rglob("generated_task.json"))
    if not files:
        raise RuntimeError(f"no generated tasks found under {pool_dir}")

    environment_counts: Counter[str] = Counter()
    errors: list[str] = []
    task_ids: set[str] = set()
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid generated task: {exc}")
            continue
        task = payload.get("task")
        recipe = payload.get("recipe") or {}
        if not isinstance(task, dict):
            errors.append(f"{path}: missing task object")
            continue
        task_id = str(task.get("id") or "")
        if not task_id:
            errors.append(f"{path}: task.id is required")
        elif task_id in task_ids:
            errors.append(f"{path}: duplicate task.id {task_id}")
        task_ids.add(task_id)
        environment_counts[_environment_id(task, recipe)] += 1

    if errors:
        raise RuntimeError("generated-task validation failed:\n- " + "\n- ".join(errors[:20]))
    return {
        "source_tasks": len(files),
        "environment_counts": dict(sorted(environment_counts.items())),
    }


def _validate_taskset(taskset: Path, expected_tasks: int) -> dict[str, Any]:
    manifest_path = taskset / "taskset_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"compiled taskset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("tasks") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("compiled taskset contains no tasks")
    if len(rows) != expected_tasks:
        raise RuntimeError(
            f"compiled task count differs from source: expected {expected_tasks}, found {len(rows)}"
        )

    task_ids = [str(row.get("id") or "") for row in rows if isinstance(row, dict)]
    if len(task_ids) != len(rows) or any(not task_id for task_id in task_ids):
        raise RuntimeError("compiled taskset contains a task without an id")
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("compiled taskset contains duplicate task ids")

    counts = Counter(
        str(row.get("environment_id") or "")
        for row in rows
        if isinstance(row, dict)
    )
    return {
        "compiled_tasks": len(rows),
        "environment_counts": dict(sorted(counts.items())),
    }


def _collection_api_protocol(options: dict[str, Any], model: str) -> str:
    harness = str(options.get("harness") or "claudecode").strip().lower()
    if harness == "claudecode":
        return "messages"
    if harness in {"codex", "hermes"}:
        return "responses"
    transport = str(options.get("transport") or "auto").strip().lower()
    if transport == "auto":
        return "messages" if "claude" in model.casefold() else "chat"
    return transport


def _matrix_api_targets(args: argparse.Namespace) -> list[dict[str, str]]:
    matrix_config = _ensure_matrix_config(args)
    raw = json.loads(matrix_config.read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    default_options = defaults.get("backend_options") or {}
    profiles = raw.get("profiles") or []
    targets: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for profile in profiles:
        if not isinstance(profile, dict) or not profile.get("enabled", True):
            continue
        options = {**default_options, **(profile.get("backend_options") or {})}
        endpoint = str(args.endpoint or options.get("litellm_base_url") or "")
        model = str(args.model or options.get("model") or "")
        api_key_env = str(
            args.api_key_env or options.get("litellm_api_key_env") or ""
        )
        if not endpoint or not model:
            continue
        protocol = _collection_api_protocol(options, model)
        key = (endpoint, model, api_key_env, protocol)
        if key not in seen:
            seen.add(key)
            targets.append({
                "endpoint": endpoint,
                "model": model,
                "api_key_env": api_key_env,
                "protocol": protocol,
            })
    return targets


def prepare(args: argparse.Namespace) -> None:
    stats = _validate_promoted(args.run_dir, args.pool)
    if args.taskset.exists():
        shutil.rmtree(args.taskset)
    args.taskset.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        "-m",
        "cuamaster.propagation.cli",
        "compile",
        "--config",
        str(args.generation_config),
        "--run-dir",
        str(args.run_dir),
        "--pool",
        args.pool,
        "--format",
        args.compile_format,
        "--output",
        str(args.taskset),
    ])
    stats.update(_validate_taskset(args.taskset, stats["source_tasks"]))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def check_api(args: argparse.Namespace) -> None:
    if args.endpoint_pool and not args.endpoint:
        pool = DynamicEndpointPool(config_path=args.endpoint_pool)
        configured_targets = _matrix_api_targets(args)
        protocols = {"chat"}
        protocols.update(
            target.get("protocol", "chat") for target in configured_targets
        )
        attempted: set[str] = set()
        errors: dict[str, str] = {}
        while len(attempted) < len(pool.base_urls):
            endpoint = pool.acquire(exclude=attempted)
            attempted.add(endpoint)
            try:
                for protocol in sorted(protocols):
                    target = {
                        "endpoint": endpoint,
                        "model": str(args.model or pool.model or ""),
                        "api_key_env": str(args.api_key_env or ""),
                        "protocol": protocol,
                    }
                    _check_api_target(target, args.api_timeout)
            except RuntimeError as exc:
                pool.mark_failure(endpoint, exc)
                errors[endpoint] = str(exc)
                continue
            pool.mark_success(endpoint)
            print(json.dumps({
                "endpoint": endpoint,
                "model": str(args.model or pool.model or ""),
                "protocols": sorted(protocols),
                "ok": True,
            }, ensure_ascii=False, indent=2))
            return
        raise RuntimeError(
            "all model endpoints failed API check: "
            + "; ".join(f"{url}: {error}" for url, error in errors.items())
        )

    targets = _matrix_api_targets(args)
    if not targets:
        print("No OpenAI-compatible model endpoint found; skipping API check.")
        return

    for target in targets:
        _check_api_target(target, args.api_timeout)
        print(json.dumps({**target, "ok": True}, ensure_ascii=False, indent=2))


def _check_api_target(target: dict[str, str], timeout: float) -> None:
    protocol = target.get("protocol", "chat")
    endpoint = target["endpoint"].rstrip("/")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(target["api_key_env"], "EMPTY")
    if protocol == "messages":
        url = endpoint + ("/messages" if endpoint.endswith("/v1") else "/v1/messages")
        payload_data = {
            "model": target["model"],
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
        }
        headers.update({
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        })
    elif protocol == "responses":
        url = endpoint + "/responses"
        payload_data = {
            "model": target["model"],
            "input": "Reply with OK.",
            "max_output_tokens": 8,
        }
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        url = endpoint + "/chat/completions"
        payload_data = {
            "model": target["model"],
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
            "temperature": 0,
        }
        headers["Authorization"] = f"Bearer {api_key}"
    payload = json.dumps(payload_data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"model {protocol} endpoint check failed for {url}: "
            f"HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"model {protocol} endpoint check failed for {url}: {exc}"
        ) from exc
    if protocol == "chat":
        valid = bool(result.get("choices")) if isinstance(result, dict) else False
    elif protocol == "messages":
        valid = bool(result.get("content")) if isinstance(result, dict) else False
    else:
        valid = bool(result.get("output")) if isinstance(result, dict) else False
    if not valid:
        raise RuntimeError(
            f"model {protocol} endpoint returned an invalid response for {url}: {result}"
        )


def _taskset_uses_legacy_osworld(taskset: Path) -> bool:
    manifest_path = taskset / "taskset_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    return isinstance(tasks, list) and any(
        isinstance(task, dict)
        and task.get("execution_adapter") == "legacy_osworld"
        for task in tasks
    )


def _legacy_evaluator_roots(matrix_config: Path) -> list[tuple[str, Path | None]]:
    raw = _load_config_data(matrix_config)
    defaults = raw.get("defaults") or {}
    default_options = defaults.get("backend_options") or {}
    targets: list[tuple[str, Path | None]] = []
    seen: set[Path | None] = set()
    for profile in raw.get("profiles") or []:
        if not isinstance(profile, dict) or not profile.get("enabled", True):
            continue
        backend = str(
            profile.get("backend") or defaults.get("backend") or "weavebench"
        ).strip().lower()
        if backend != "weavebench":
            continue
        options = {
            **default_options,
            **(profile.get("backend_options") or {}),
        }
        root = _config_path(
            options.get("osworld_evaluator_root"), matrix_config.parent
        )
        if root in seen:
            continue
        seen.add(root)
        targets.append((str(profile.get("id") or "weavebench"), root))
    return targets


def check_collection_dependencies(args: argparse.Namespace) -> None:
    if not _taskset_uses_legacy_osworld(args.taskset):
        return
    matrix_config = _ensure_matrix_config(args)
    targets = _legacy_evaluator_roots(matrix_config)
    if not targets:
        return

    probe = (
        "import importlib, os, sys\n"
        "root = os.environ.get('GUI_TRAJECTORY_OSWORLD_EVALUATOR_ROOT', '').strip()\n"
        "if root:\n"
        "    sys.path.insert(0, root)\n"
        "cv2 = importlib.import_module('cv2')\n"
        "gui = next((line.split(':', 1)[1].strip() for line in "
        "cv2.getBuildInformation().splitlines() if line.strip().startswith('GUI:')), '')\n"
        "if gui and gui.upper() != 'NONE':\n"
        "    raise RuntimeError(f'GUI-enabled OpenCV build detected: GUI={gui}')\n"
        "importlib.import_module('desktop_env.evaluators.metrics')\n"
        "importlib.import_module('desktop_env.evaluators.getters')\n"
    )
    for profile_id, evaluator_root in targets:
        env = _python_env()
        if evaluator_root is not None:
            env["GUI_TRAJECTORY_OSWORLD_EVALUATOR_ROOT"] = str(evaluator_root)
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            continue
        detail = (completed.stderr or completed.stdout).strip()
        advice = "install the missing evaluator dependency in the collector runtime"
        if "libGL.so.1" in detail or "GUI-enabled OpenCV build" in detail:
            advice = (
                "remove opencv-python and install opencv-python-headless>=4.8.1.78,<5; "
                "the collector evaluator runtime must use a headless OpenCV build"
            )
        root_label = str(evaluator_root) if evaluator_root is not None else "installed"
        raise RuntimeError(
            "collection dependency preflight failed for "
            f"profile {profile_id!r} using OSWorld evaluators from {root_label}:\n"
            f"{detail[-4000:]}\n"
            f"Remediation: {advice}."
        )
    _progress(f"collection dependency check passed profiles={len(targets)}")


def collect(
    args: argparse.Namespace,
    *,
    dry_run: bool,
    task_filter: list[str] | None = None,
    profile_filter: list[str] | None = None,
    force_rerun: bool = False,
    allow_config_change: bool = False,
    cap_profile_concurrency: bool = False,
) -> None:
    if not (args.taskset / "taskset_manifest.json").is_file():
        raise RuntimeError("taskset is missing; run the prepare stage first")
    matrix_config = _ensure_matrix_config(args)
    command = [
        sys.executable,
        str(REPO_ROOT / "run.py"),
        "--config",
        str(matrix_config),
        "--output-root",
        str(args.collection),
        "--backend-option",
        f"tasks_root={args.taskset}",
        "--backend-option",
        f"action_protocol={args.action_protocol}",
    ]
    if args.concurrency is not None:
        command.extend(("--concurrency", str(args.concurrency)))
    if args.max_parallel_profiles is not None:
        command.extend(("--max-parallel-profiles", str(args.max_parallel_profiles)))
    progress_interval_seconds = getattr(args, "progress_interval_seconds", None)
    if progress_interval_seconds is not None:
        command.extend(
            ("--progress-interval-seconds", str(progress_interval_seconds))
        )
    if args.model:
        command.extend(("--backend-option", f"model={args.model}"))
    if args.endpoint:
        command.extend(("--backend-option", f"litellm_base_url={args.endpoint}"))
    if task_filter:
        command.extend(("--backend-option", f"task_filter={','.join(task_filter)}"))
    if profile_filter:
        command.extend(("--profiles", ",".join(profile_filter)))
    if dry_run:
        command.append("--dry-run")
    if force_rerun or args.force_rerun:
        command.append("--force-rerun")
    if allow_config_change:
        command.append("--allow-config-change")
    if cap_profile_concurrency:
        command.append("--cap-profile-concurrency")
    _run(command)


def _has_trajectory_evidence(task_dir: Path) -> bool:
    return any(
        path.is_file() and path.stat().st_size > 0
        for path in (
            task_dir / "trajectory.jsonl",
            task_dir / "canonical_trajectory.jsonl",
            task_dir / "computer_events.jsonl",
        )
    )


def _retry_reason(task_dir: Path) -> str | None:
    has_trajectory = _has_trajectory_evidence(task_dir)
    reasons: list[str] = []
    if not has_trajectory:
        reasons.append("missing_or_empty_trajectory")
    execution_path = task_dir / "execution_status.json"
    if execution_path.is_file():
        try:
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"invalid_execution_status:{type(exc).__name__}")
        else:
            if execution.get("transport_failure") is True:
                reasons.append("vm_transport_unavailable")
            if execution.get("timed_out") is True:
                reasons.append("agent_timed_out")
            if execution.get("agent_done") is False:
                reasons.append("agent_not_done")
            exit_code = execution.get("exit_code")
            if exit_code not in (None, 0):
                reasons.append(f"agent_exit_code={exit_code}")

    score_path = task_dir / "score.json"
    if score_path.is_file():
        try:
            score_record = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if "missing_or_empty_trajectory" in reasons:
                reasons.append(f"invalid_score:{type(exc).__name__}")
        else:
            if score_record.get("agent_done") is False:
                reasons.append("score_agent_not_done")
            agent_status = str(score_record.get("agent_status") or "").casefold()
            if agent_status in {
                "empty_run",
                "error",
                "failed",
                "retry_exhausted",
                "timed_out",
                "timeout",
            }:
                reasons.append(f"agent_status={agent_status}")
            agent_exit = score_record.get("agent_exit")
            if agent_exit not in (None, 0, "0"):
                reasons.append(f"agent_exit={agent_exit}")
            error = str(score_record.get("error") or "")
            retryable_error_markers = {
                "agent_empty_run": "agent_empty_run",
                "agent_retry_exhausted": "agent_retry_exhausted",
                "agent_timed_out": "agent_timed_out",
                "VM transport unavailable": "vm_transport_unavailable",
                "transport_failure": "vm_transport_unavailable",
                "/setup/upload": "vm_transport_unavailable",
                "/setup/launch": "vm_transport_unavailable",
                "/setup/execute": "vm_transport_unavailable",
                "/screenshot": "vm_transport_unavailable",
            }
            for marker, reason in retryable_error_markers.items():
                if marker.casefold() in error.casefold():
                    reasons.append(reason)

    if has_trajectory:
        reasons = [reason for reason in reasons if reason == "vm_transport_unavailable"]
    return " | ".join(dict.fromkeys(reasons)) or None


def _taskset_task_ids(taskset: Path | None) -> set[str] | None:
    if taskset is None:
        return None
    manifest_path = taskset / "taskset_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(tasks, list):
        raise ValueError("taskset manifest tasks must be an array")
    return {
        str(task["id"])
        for task in tasks
        if isinstance(task, dict) and task.get("id")
    }


def _failed_collection_tasks(
    collection_root: Path,
    taskset: Path | None = None,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    if not collection_root.is_dir():
        return failures
    allowed_task_ids = _taskset_task_ids(taskset)
    for profile_dir in sorted(collection_root.iterdir()):
        if not profile_dir.is_dir() or profile_dir.name.startswith("."):
            continue
        for task_path in sorted(profile_dir.rglob("task.json")):
            task_dir = task_path.parent
            if task_dir.name == "canonical":
                continue
            reason = _retry_reason(task_dir)
            if reason is None:
                continue
            task_id = task_dir.name
            try:
                task_record = json.loads(task_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            else:
                task_id = str(
                    task_record.get("task_id")
                    or task_record.get("id")
                    or task_id
                )
            if allowed_task_ids is not None and task_id not in allowed_task_ids:
                continue
            failures.append({
                "profile_id": profile_dir.name,
                "task_id": task_id,
                "output_dir": str(task_dir),
                "reason": reason,
            })
    known_outputs = {failure["output_dir"] for failure in failures}
    retry_root = collection_root / ".collection_state" / "retry_attempts"
    if retry_root.is_dir():
        for manifest_path in sorted(retry_root.glob("round-*/retry_manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("task_id") or "")
                output_dir = str(task.get("output_dir") or "")
                profile_id = str(task.get("profile_id") or "")
                if not task_id or not output_dir or not profile_id:
                    continue
                if allowed_task_ids is not None and task_id not in allowed_task_ids:
                    continue
                output_path = Path(output_dir)
                if (output_path / "task.json").is_file():
                    continue
                if output_dir in known_outputs:
                    continue
                failures.append({
                    "profile_id": profile_id,
                    "task_id": task_id,
                    "output_dir": output_dir,
                    "reason": "missing_retry_output",
                })
                known_outputs.add(output_dir)
    return failures


def _unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _archive_failed_tasks(
    collection_root: Path,
    failures: list[dict[str, str]],
    retry_round: int,
) -> Path:
    archive_root = (
        collection_root
        / ".collection_state"
        / "retry_attempts"
        / f"round-{retry_round:02d}"
    )
    archived: list[dict[str, Any]] = []
    for failure in failures:
        task_dir = Path(failure["output_dir"])
        relative = task_dir.relative_to(collection_root)
        archived_failure: dict[str, Any] = dict(failure)
        if task_dir.exists():
            destination = _unique_archive_path(archive_root / relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(task_dir), str(destination))
            archived_failure["archive_dir"] = str(destination)
        else:
            archived_failure["archive_dir"] = None
        archived.append(archived_failure)
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "retry_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "task_trajectory_pipeline.retry.v1",
                "retry_round": retry_round,
                "tasks": archived,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive_root


def _remaining_retry_failures(
    collection_root: Path,
    taskset: Path | None,
    attempted_failures: list[dict[str, str]],
) -> list[dict[str, str]]:
    failures = _failed_collection_tasks(collection_root, taskset)
    failed_outputs = {failure["output_dir"] for failure in failures}
    for attempted in attempted_failures:
        output_dir = Path(attempted["output_dir"])
        if output_dir.is_dir() and (output_dir / "task.json").is_file():
            continue
        output_dir_text = str(output_dir)
        if output_dir_text in failed_outputs:
            continue
        failures.append({
            "profile_id": attempted["profile_id"],
            "task_id": attempted["task_id"],
            "output_dir": output_dir_text,
            "reason": "missing_retry_output",
        })
        failed_outputs.add(output_dir_text)
    return failures


def _retry_failed_collection(args: argparse.Namespace) -> list[dict[str, str]]:
    retry_attempts = int(getattr(args, "retry_failed_attempts", 0) or 0)
    failures = _failed_collection_tasks(
        args.collection,
        getattr(args, "taskset", None),
    )
    retry_root = args.collection / ".collection_state" / "retry_attempts"
    existing_rounds = [
        int(match.group(1))
        for path in retry_root.glob("round-*")
        if (match := re.fullmatch(r"round-(\d+)", path.name))
    ] if retry_root.is_dir() else []
    first_retry_round = max(existing_rounds, default=0) + 1
    for retry_attempt in range(1, retry_attempts + 1):
        retry_round = first_retry_round + retry_attempt - 1
        if not failures:
            _progress(
                f"retry attempt={retry_attempt}/{retry_attempts} "
                "skipped failed_tasks=0"
            )
            return []
        task_ids = sorted({failure["task_id"] for failure in failures})
        archive_root = _archive_failed_tasks(
            args.collection,
            failures,
            retry_round,
        )
        _progress(
            f"retry attempt={retry_attempt}/{retry_attempts} "
            f"archive_round={retry_round} "
            f"failed_outputs={len(failures)} unique_tasks={len(task_ids)} "
            f"archive={archive_root}"
        )
        tasks_by_profile: dict[str, set[str]] = {}
        for failure in failures:
            tasks_by_profile.setdefault(failure["profile_id"], set()).add(
                failure["task_id"]
            )
        for profile_id, profile_tasks in sorted(tasks_by_profile.items()):
            try:
                collect(
                    args,
                    dry_run=False,
                    task_filter=sorted(profile_tasks),
                    profile_filter=[profile_id],
                    force_rerun=True,
                    allow_config_change=True,
                    cap_profile_concurrency=True,
                )
            except subprocess.CalledProcessError as exc:
                _progress(
                    f"retry attempt={retry_attempt}/{retry_attempts} "
                    f"profile={profile_id} collector exited "
                    f"returncode={exc.returncode}; inspecting persisted task outputs"
                )
        failures = _remaining_retry_failures(
            args.collection,
            getattr(args, "taskset", None),
            failures,
        )
    return failures


def _raise_remaining_collection_failures(
    failures: list[dict[str, str]],
    *,
    cause: BaseException | None = None,
) -> None:
    if not failures:
        return
    task_ids = sorted({failure["task_id"] for failure in failures})
    preview = ",".join(task_ids[:20])
    if len(task_ids) > 20:
        preview += f",...(+{len(task_ids) - 20})"
    reasons = Counter(failure["reason"] for failure in failures)
    reason_summary = ",".join(
        f"{reason}={count}" for reason, count in sorted(reasons.items())
    )
    error = RuntimeError(
        "collection still has retryable failures after configured retries: "
        f"failed_outputs={len(failures)} unique_tasks={len(task_ids)} "
        f"reasons={reason_summary} tasks={preview}"
    )
    if cause is None:
        raise error
    raise error from cause


def _collect_with_retries(args: argparse.Namespace) -> None:
    check_collection_dependencies(args)
    collection_error: subprocess.CalledProcessError | None = None
    try:
        collect(args, dry_run=False)
    except subprocess.CalledProcessError as exc:
        collection_error = exc
        _progress(
            f"initial collector exited returncode={exc.returncode}; "
            "inspecting persisted task outputs before retrying"
        )
    remaining = _retry_failed_collection(args)
    if remaining and (
        collection_error is not None
        or int(getattr(args, "retry_failed_attempts", 0) or 0) > 0
    ):
        _raise_remaining_collection_failures(remaining, cause=collection_error)
    if collection_error is not None:
        _progress(
            "initial collector returned non-zero, but no retryable task outputs "
            "remain; continuing"
        )


def retry_collection(args: argparse.Namespace) -> None:
    check_collection_dependencies(args)
    failures = _failed_collection_tasks(
        args.collection,
        getattr(args, "taskset", None),
    )
    if not failures:
        _progress("retry-only collection skipped failed_tasks=0")
        return
    _progress(
        "retry-only collection start "
        f"failed_outputs={len(failures)} "
        f"unique_tasks={len({failure['task_id'] for failure in failures})}"
    )
    remaining = _retry_failed_collection(args)
    _raise_remaining_collection_failures(remaining)


def _collection_run_id(collection_root: Path) -> str:
    summary_path = collection_root / "collection_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, dict) and summary.get("run_id"):
            return str(summary["run_id"])
    return "model-judge-refresh"


def _refresh_scored_episodes(args: argparse.Namespace) -> int:
    from cuamaster.collection.export import export_result_tree
    from cuamaster.collection.profiles import load_matrix_config

    matrix = load_matrix_config(_ensure_matrix_config(args))
    run_id = _collection_run_id(args.collection)
    refreshed = 0
    for profile in matrix.enabled_profiles():
        profile_root = args.collection / profile.id
        if not profile_root.is_dir():
            continue
        refreshed += export_result_tree(profile_root, profile, run_id=run_id)
    return refreshed


def score(args: argparse.Namespace) -> None:
    if not getattr(args, "score_with_model", False):
        _progress("model scoring disabled")
        return
    endpoint_pool = getattr(args, "scoring_endpoint_pool", None)
    if endpoint_pool is None:
        raise RuntimeError(
            "model scoring requires scoring.endpoint_pool or api.endpoint_pool"
        )
    summary = score_collection(
        args.collection,
        endpoint_pool_path=endpoint_pool,
        model=getattr(args, "scoring_model", None),
        api_key_env=getattr(args, "scoring_api_key_env", None),
        allow_unauthenticated=getattr(
            args, "scoring_allow_unauthenticated", False
        ),
        concurrency=getattr(args, "scoring_concurrency", 4),
        threshold=getattr(args, "scoring_threshold", 0.6),
        attempts=getattr(args, "scoring_attempts", 3),
        timeout=getattr(args, "scoring_timeout", 300.0),
        max_output_tokens=getattr(args, "scoring_max_output_tokens", 4096),
        max_evidence_chars=getattr(args, "scoring_max_evidence_chars", 24000),
        include_final_screenshot=getattr(
            args, "scoring_include_final_screenshot", True
        ),
        force=getattr(args, "force_rescore", False),
        progress=_progress,
    )
    refreshed = _refresh_scored_episodes(args)
    _progress(
        f"model scoring candidates={summary['candidates']} "
        f"accepted={summary['accepted']} refreshed={refreshed}"
    )


def export(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "cuamaster.collection.dataset",
        "--collection-root",
        str(args.collection),
        "--output-dir",
        str(args.dataset),
        "--bundle-format",
        args.bundle_format,
        "--visual-window",
        str(args.visual_window),
        "--max-history-actions",
        str(args.max_history_actions),
        "--action-protocol",
        getattr(args, "action_protocol", DEFAULT_ACTION_PROTOCOL),
        "--quality-policy",
        getattr(args, "quality_policy", "high_quality"),
        "--max-execution-output-chars",
        str(getattr(args, "max_execution_output_chars", 4000)),
    ]
    if getattr(args, "taskset", None):
        command.extend(("--taskset-root", str(args.taskset)))
    if args.copy_images:
        command.append("--copy-images")
    if args.require_pre_screenshot:
        command.append("--require-pre-screenshot")
    _run(command)


def run_pipeline(args: argparse.Namespace) -> None:
    stages: list[tuple[str, Any]] = [("prepare", prepare)]
    if not args.skip_api_check:
        stages.append(("check-api", check_api))
    stages.extend((
        ("collect", _collect_with_retries),
        ("score", score),
        ("export", export),
    ))
    pipeline_started = time.monotonic()
    total = len(stages)
    for index, (name, function) in enumerate(stages, 1):
        stage_started = time.monotonic()
        _progress(f"[{index}/{total}] start stage={name}")
        try:
            function(args)
        except BaseException:
            _progress(
                f"[{index}/{total}] failed stage={name} "
                f"elapsed={_format_duration(time.monotonic() - stage_started)}"
            )
            raise
        _progress(
            f"[{index}/{total}] done stage={name} "
            f"elapsed={_format_duration(time.monotonic() - stage_started)}"
        )
    _progress(
        f"complete stages={total} "
        f"elapsed={_format_duration(time.monotonic() - pipeline_started)}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    config_path = (
        config_args.config.expanduser().resolve() if config_args.config else None
    )
    defaults = _pipeline_defaults(config_path)

    parser = argparse.ArgumentParser(
        description=(
            "Compile generated tasks, collect trajectories on real VMs, and export "
            "training data."
        )
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="run",
        choices=(
            "run",
            "prepare",
            "check-api",
            "dry-run",
            "collect",
            "retry",
            "score",
            "export",
        ),
        help=(
            "run executes prepare, API check, collection, model scoring, and "
            "export end to end"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="pipeline YAML/JSON config; command-line values override it",
    )
    parser.add_argument("--run-dir", type=Path, default=defaults.get("run_dir"))
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=defaults.get("generation_config"),
    )
    parser.add_argument(
        "--matrix-config", type=Path, default=defaults.get("matrix_config")
    )
    parser.add_argument("--output-root", type=Path, default=defaults.get("output_root"))
    parser.add_argument("--pool", default=defaults.get("pool", "promoted"))
    parser.add_argument(
        "--compile-format", default=defaults.get("compile_format", "weavebench")
    )
    parser.add_argument("--endpoint", default=defaults.get("endpoint"))
    parser.add_argument(
        "--endpoint-pool",
        type=Path,
        default=defaults.get("endpoint_pool"),
        help="YAML endpoint pool reloaded by collection workers",
    )
    parser.add_argument(
        "--base-url",
        dest="base_urls",
        action="append",
        default=None,
        help="OpenAI-compatible endpoint; may be repeated",
    )
    parser.add_argument("--model", default=defaults.get("model"))
    parser.add_argument("--api-key-env", default=defaults.get("api_key_env"))
    parser.add_argument(
        "--allow-unauthenticated",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("allow_unauthenticated", False),
    )
    parser.add_argument(
        "--api-timeout", type=float, default=defaults.get("api_timeout", 60.0)
    )
    parser.add_argument(
        "--concurrency",
        "--max-concurrent-vms",
        dest="concurrency",
        type=int,
        default=defaults.get("concurrency"),
        help="maximum number of active VM environments across all profiles",
    )
    parser.add_argument(
        "--max-parallel-profiles",
        type=int,
        default=defaults.get("max_parallel_profiles"),
        help="optional legacy cap on concurrently running profiles",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=defaults.get("progress_interval_seconds", 30.0),
        help="seconds between collection heartbeats; 0 disables them",
    )
    parser.add_argument(
        "--retry-failed-attempts",
        type=int,
        default=defaults.get("retry_failed_attempts", 1),
        help=(
            "additional collection rounds for empty agent runs, timeouts, VM "
            "transport failures, or missing trajectories; 0 disables retries"
        ),
    )
    parser.add_argument(
        "--score-with-model",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("score_with_model", False),
        help="score collected trajectories with an OpenAI-compatible model",
    )
    parser.add_argument(
        "--scoring-endpoint-pool",
        type=Path,
        default=defaults.get("scoring_endpoint_pool"),
    )
    parser.add_argument("--scoring-model", default=defaults.get("scoring_model"))
    parser.add_argument(
        "--scoring-api-key-env",
        default=defaults.get("scoring_api_key_env"),
    )
    parser.add_argument(
        "--scoring-allow-unauthenticated",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("scoring_allow_unauthenticated", False),
    )
    parser.add_argument(
        "--scoring-concurrency",
        type=int,
        default=defaults.get("scoring_concurrency", 4),
    )
    parser.add_argument(
        "--scoring-threshold",
        type=float,
        default=defaults.get("scoring_threshold", 0.6),
    )
    parser.add_argument(
        "--scoring-attempts",
        type=int,
        default=defaults.get("scoring_attempts", 3),
    )
    parser.add_argument(
        "--scoring-timeout",
        type=float,
        default=defaults.get("scoring_timeout", 300.0),
    )
    parser.add_argument(
        "--scoring-max-output-tokens",
        type=int,
        default=defaults.get("scoring_max_output_tokens", 4096),
    )
    parser.add_argument(
        "--scoring-max-evidence-chars",
        type=int,
        default=defaults.get("scoring_max_evidence_chars", 24000),
    )
    parser.add_argument(
        "--scoring-include-final-screenshot",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("scoring_include_final_screenshot", True),
    )
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument(
        "--action-protocol",
        choices=tuple(sorted(SUPPORTED_ACTION_PROTOCOLS - {"auto"})),
        default=defaults.get("action_protocol", DEFAULT_ACTION_PROTOCOL),
        help="Shared model/output protocol for collection and dataset export.",
    )
    parser.add_argument(
        "--bundle-format", default=defaults.get("bundle_format", "all")
    )
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("copy_images", True),
    )
    parser.add_argument(
        "--require-pre-screenshot",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("require_pre_screenshot", True),
    )
    parser.add_argument(
        "--visual-window", type=int, default=defaults.get("visual_window", 10)
    )
    parser.add_argument(
        "--max-history-actions",
        type=int,
        default=defaults.get("max_history_actions", 3),
    )
    parser.add_argument(
        "--quality-policy",
        choices=("structural", "strict", "high_quality"),
        default=defaults.get("quality_policy", "high_quality"),
    )
    parser.add_argument(
        "--max-execution-output-chars",
        type=int,
        default=defaults.get("max_execution_output_chars", 4000),
    )
    parser.add_argument(
        "--skip-api-check",
        action="store_true",
        default=defaults.get("skip_api_check", False),
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.base_urls is None:
        args.base_urls = defaults.get("base_urls")
    tokens = list(argv) if argv is not None else sys.argv[1:]
    if args.endpoint or any(
        token == "--base-url" or token.startswith("--base-url=") for token in tokens
    ):
        args.endpoint_pool = None
    args.embedded_matrix = defaults.get("embedded_matrix")
    missing = [
        option
        for option, value in (
            ("--run-dir", args.run_dir),
            ("--generation-config", args.generation_config),
            ("--output-root", args.output_root),
        )
        if value is None
    ]
    if missing:
        parser.error(
            "missing required configuration: "
            + ", ".join(missing)
            + " (set them in --config or on the command line)"
        )
    if args.matrix_config is None and args.embedded_matrix is None:
        parser.error(
            "missing VM matrix: set vm_matrix in --config or pass --matrix-config"
        )
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.max_parallel_profiles is not None and args.max_parallel_profiles < 1:
        parser.error("--max-parallel-profiles must be >= 1")
    if args.retry_failed_attempts < 0:
        parser.error("--retry-failed-attempts must be >= 0")
    if args.scoring_concurrency < 1:
        parser.error("--scoring-concurrency must be >= 1")
    if not 0 <= args.scoring_threshold <= 1:
        parser.error("--scoring-threshold must be between 0 and 1")
    if args.scoring_attempts < 1:
        parser.error("--scoring-attempts must be >= 1")
    if args.scoring_timeout <= 0:
        parser.error("--scoring-timeout must be > 0")
    if args.scoring_max_output_tokens < 1:
        parser.error("--scoring-max-output-tokens must be >= 1")
    if args.scoring_max_evidence_chars < 4000:
        parser.error("--scoring-max-evidence-chars must be >= 4000")
    args.run_dir = args.run_dir.expanduser().resolve()
    args.generation_config = args.generation_config.expanduser().resolve()
    if args.matrix_config is not None:
        args.matrix_config = args.matrix_config.expanduser().resolve()
    if args.scoring_endpoint_pool is not None:
        args.scoring_endpoint_pool = args.scoring_endpoint_pool.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.taskset = args.output_root / "taskset"
    args.collection = args.output_root / "collection"
    args.dataset = args.output_root / "dataset"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "run":
        run_pipeline(args)
    elif args.stage == "prepare":
        prepare(args)
    elif args.stage == "check-api":
        check_api(args)
    elif args.stage == "dry-run":
        collect(args, dry_run=True)
    elif args.stage == "collect":
        _collect_with_retries(args)
    elif args.stage == "retry":
        retry_collection(args)
    elif args.stage == "score":
        score(args)
    else:
        export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
