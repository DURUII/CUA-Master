"""Build portable canonical bundles and framework-specific SFT data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from cuamaster.runtime import (
    ACTION_PROTOCOL_CANONICAL,
    DEFAULT_ACTION_PROTOCOL,
    QwenXmlActionParser,
    SUPPORTED_ACTION_PROTOCOLS,
    format_qwen_xml_action,
)
from cuamaster.benchmarks import (
    HYBRID_CANONICAL_SYSTEM_PROMPT,
    HYBRID_QWEN_XML_SYSTEM_PROMPT,
)

from cuamaster.action_space import CUAAction
from .collection_policy import HARD_CODED_SUDO_PASSWORD_RE
from .export import _shell_timeout_seconds

from .schemas import canonical_json, content_digest
from .trace_store import atomic_write_json, atomic_write_text


CANONICAL_SYSTEM_PROMPT = HYBRID_CANONICAL_SYSTEM_PROMPT
DEFAULT_SYSTEM_PROMPT = HYBRID_QWEN_XML_SYSTEM_PROMPT
CANONICAL_STEP_SCHEMA = "cua_sft.step.v1"
CANONICAL_EPISODE_SCHEMA = "cua_sft.episode.v1"
CANONICAL_DATASET_SCHEMA = "cua_sft.dataset.v1"
HYBRID_STEP_SCHEMA = "hybrid_sft.step.v1"
HYBRID_EPISODE_SCHEMA = "hybrid_sft.episode.v1"
HYBRID_DATASET_SCHEMA = "hybrid_sft.dataset.v1"
SHELL_STEP_SCHEMA = "shell_sft.step.v1"
SHELL_EPISODE_SCHEMA = "shell_sft.episode.v1"
SHELL_DATASET_SCHEMA = "shell_sft.dataset.v1"
MIXED_SFT_SCHEMA = "hybrid_cua_sft.example.v2"
MIXED_SFT_DATASET_SCHEMA = "hybrid_cua_sft.dataset.v2"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_URL_CREDENTIAL_RE = re.compile(
    r"(?P<scheme>https?://)(?P<credentials>[^\s/@:]+:[^\s/@]+)@",
    re.I,
)
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}", re.I)
_API_SECRET_RE = re.compile(
    r"\b(?P<name>api[_-]?key|access[_-]?token|secret)"
    r"(?P<separator>\s*[=:]\s*['\"]?)[^\s'\"]{8,}",
    re.I,
)
_STRICT_BLOCKERS = {
    "collector_hook_tampering": re.compile(
        r"claudecode_shell_screenshot_hook|/tmp_workspace/_trajectory",
        re.I,
    ),
    "embedded_url_credentials": _URL_CREDENTIAL_RE,
    "hardcoded_sudo_password": HARD_CODED_SUDO_PASSWORD_RE,
    "evaluation_artifact_access": re.compile(
        r"(?:^|[\s/'\"])(?:_grade|score|quality|episode_manifest)\.json(?:$|[\s'\"])",
        re.I,
    ),
}
_QUALITY_WARNINGS = {
    "package_install": re.compile(
        r"(?:apt(?:-get)?|dnf|yum|pacman)\s+(?:[^\n;&]+\s+)?install\b|"
        r"(?:pip|pip3|npm)\s+install\b",
        re.I,
    ),
    "direct_browser_profile_mutation": re.compile(
        r"(?:>|tee\s+|rm\s+|mv\s+|cp\s+|sqlite3\s+)"
        r"[^\n;&]*(?:\.config/(?:google-chrome|chromium)|/Cookies|/Bookmarks)",
        re.I,
    ),
    "direct_office_package_mutation": re.compile(
        r"(?:openpyxl|python-docx|python-pptx|odfpy|zipfile).*"
        r"\.(?:xlsx|xlsm|docx|pptx)",
        re.I | re.S,
    ),
}
_HIGH_QUALITY_EXCLUSIONS = {
    "direct_browser_profile_mutation",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


@lru_cache(maxsize=512)
def _source_events(episode_dir: str) -> tuple[dict[str, Any], ...]:
    path = Path(episode_dir) / "trajectory.jsonl"
    return tuple(_jsonl(path)) if path.is_file() else ()


def _restore_source_shell_timeout(
    episode_dir: Path,
    step: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    if action.get("type") != "shell.exec":
        return action
    provenance = step.get("provenance") or {}
    raw_turn = provenance.get("raw_turn")
    if not isinstance(raw_turn, int) or raw_turn < 1:
        return action
    events = _source_events(str(episode_dir))
    if raw_turn > len(events):
        return action
    event = events[raw_turn - 1]
    request = event.get("request")
    if not isinstance(request, dict):
        return action
    tool_name = str(provenance.get("tool_name") or event.get("tool_name") or "")
    timeout = _shell_timeout_seconds(request, tool_name)
    if timeout is None:
        return action
    restored = dict(action)
    restored_args = dict(restored.get("args") or {})
    restored_args["timeout_sec"] = timeout
    restored["args"] = restored_args
    return restored


def _write_dataset_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)
    path.chmod(0o644)


def _write_dataset_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)
    path.chmod(0o644)


def discover_episodes(root: str | Path) -> Iterable[Path]:
    for manifest in sorted(Path(root).rglob("episode_manifest.json")):
        if ".trace-index" not in manifest.parts:
            yield manifest.parent


def _absolute_image(episode_dir: Path, relative: Any) -> str | None:
    if not relative:
        return None
    path = Path(str(relative))
    path = path if path.is_absolute() else episode_dir / path
    return str(path.resolve()) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as stream:
        header = stream.read(32)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        return struct.unpack(">II", header[16:24])
    if header[:6] in {b"GIF87a", b"GIF89a"} and len(header) >= 10:
        return struct.unpack("<HH", header[6:10])
    return None, None


class ImagePackager:
    def __init__(self, bundle_root: Path, *, copy_images: bool) -> None:
        self.bundle_root = bundle_root
        self.images_dir = bundle_root / "images"
        self.copy_images = copy_images
        if copy_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self.references: dict[str, dict[str, Any]] = {}
        self.images_copied = 0
        self.images_reused = 0

    def package(self, source: Path) -> dict[str, Any]:
        source = source.resolve()
        digest = _sha256(source)
        existing = self.references.get(digest)
        if existing is not None:
            self.images_reused += 1
            return dict(existing)
        suffix = source.suffix.lower()
        if suffix not in _IMAGE_SUFFIXES:
            suffix = ".bin"
        relative = Path("images") / f"{digest}{suffix}"
        destination = self.bundle_root / relative
        if self.copy_images:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                shutil.copy2(source, destination)
                self.images_copied += 1
            else:
                self.images_reused += 1
        else:
            try:
                relative = source.relative_to(self.bundle_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    "--no-copy-images requires source screenshots to already be inside "
                    f"the output bundle: {source}"
                ) from exc
        width, height = _image_size(source)
        reference = {
            "path": relative.as_posix(),
            "sha256": digest,
            "width": width,
            "height": height,
            "bytes": source.stat().st_size,
        }
        self.references[digest] = reference
        return dict(reference)


def _portable_task(manifest: dict[str, Any]) -> dict[str, Any]:
    task = manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return {
        "id": task.get("id"),
        "instruction": task.get("instruction"),
        "version": task.get("version"),
        "source_task_ids": task.get("source_task_ids")
        or metadata.get("source_task_ids")
        or [],
        "required_capabilities": task.get("required_capabilities") or [],
        "metadata": {
            key: metadata.get(key)
            for key in ("application", "category", "domain", "mode")
            if metadata.get(key) is not None
        },
    }


def _portable_agent(manifest: dict[str, Any]) -> dict[str, Any]:
    agent = manifest.get("agent") if isinstance(manifest.get("agent"), dict) else {}
    return {
        key: agent.get(key)
        for key in (
            "id",
            "harness",
            "model",
            "provider",
            "transport",
            "runtime_strategy",
        )
    }


def _portable_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    vm = manifest.get("vm") if isinstance(manifest.get("vm"), dict) else {}
    image = vm.get("image") if isinstance(vm.get("image"), dict) else {}
    image_metadata = (
        image.get("metadata") if isinstance(image.get("metadata"), dict) else {}
    )
    return {
        "profile_id": vm.get("id"),
        "backend": vm.get("backend"),
        "provider": vm.get("provider"),
        "os": vm.get("os") or {},
        "image": {
            "id": image.get("id") or image_metadata.get("id"),
            "snapshot": image.get("snapshot") or image_metadata.get("snapshot"),
            "format": image_metadata.get("format"),
        },
        "software": vm.get("software") or image_metadata.get("software") or {},
    }


def _portable_evaluation(manifest: dict[str, Any]) -> dict[str, Any]:
    evaluation = (
        manifest.get("evaluation")
        if isinstance(manifest.get("evaluation"), dict)
        else {}
    )
    return {
        key: evaluation.get(key)
        for key in (
            "score",
            "scores",
            "success",
            "evaluator_version",
            "agent_done",
            "agent_status",
            "error",
        )
    }


def _valid_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    try:
        return CUAAction.from_dict(action).to_dict()
    except (TypeError, ValueError):
        return None


def _execution_succeeded(step: dict[str, Any]) -> bool:
    execution = step.get("execution")
    return isinstance(execution, dict) and execution.get("ok") is True


def _action_modality(action: dict[str, Any]) -> str:
    action_type = str(action.get("type") or "")
    if action_type.startswith("computer."):
        return "visual"
    if action_type == "shell.exec":
        return "shell"
    if action_type == "finish":
        return "control"
    raise ValueError(f"unsupported GUI+CLI action type: {action_type}")


def _screenshot_digest(episode_dir: Path, value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, dict):
        digest = value.get("sha256")
        if isinstance(digest, str) and digest:
            return digest
        value = value.get("path")
    path = _absolute_image(episode_dir, value)
    return _cached_sha256(path) if path else None


@lru_cache(maxsize=None)
def _cached_sha256(path: str) -> str:
    return _sha256(Path(path))


def _unchanged_wait(episode_dir: Path, step: dict[str, Any]) -> bool:
    action = _valid_action(step.get("action"))
    if action is None or action.get("type") != "computer.wait":
        return False
    before = _screenshot_digest(
        episode_dir, (step.get("pre_observation") or {}).get("screenshot")
    )
    after = _screenshot_digest(
        episode_dir, (step.get("post_observation") or {}).get("screenshot")
    )
    return bool(before and after and before == after)


def _episode_bundle(steps: list[dict[str, Any]]) -> str:
    modalities = {
        _action_modality(action)
        for step in steps
        if (action := _valid_action(step.get("action"))) is not None
        and action.get("type") != "finish"
    }
    if modalities == {"visual"}:
        return "canonical"
    if modalities == {"shell"}:
        return "shell"
    return "hybrid"


def _step_decision_context(step: dict[str, Any]) -> dict[str, Any]:
    context = step.get("decision_context")
    if isinstance(context, dict) and context:
        return dict(context)
    intent = (step.get("provenance") or {}).get("assistant_intent")
    if not intent:
        return {}
    return {
        "source": "legacy_provenance",
        "visibility": "observable_assistant_output",
        "intent": str(intent),
        "text": str(intent),
        "chat_indices": [],
        "blocks": [],
    }


def _public_decision_context(step: dict[str, Any]) -> dict[str, Any]:
    context = _step_decision_context(step)
    intent = context.get("intent")
    if not intent or re.search(r"\[thinking\]|<\|close\|>think|<analysis>", str(intent), re.I):
        return {}
    return {
        "source": context.get("source"),
        "visibility": context.get("visibility"),
        "intent": _sanitize_text(str(intent)),
    }


def _public_provenance(step: dict[str, Any]) -> dict[str, Any]:
    provenance = step.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    return {
        key: _sanitize_value(provenance.get(key))
        for key in (
            "source",
            "raw_turn",
            "raw_action_index",
            "tool_call_id",
            "tool_name",
            "started_at",
            "finished_at",
        )
        if provenance.get(key) is not None
    }


def _sanitize_text(value: str) -> str:
    value = _URL_CREDENTIAL_RE.sub(r"\g<scheme><redacted>@", value)
    value = _BEARER_TOKEN_RE.sub("Bearer <redacted>", value)
    return _API_SECRET_RE.sub(
        lambda match: f"{match.group('name')}{match.group('separator')}<redacted>",
        value,
    )


def _sanitize_value(value: Any, *, max_chars: int = 0) -> Any:
    if isinstance(value, str):
        text = _sanitize_text(value)
        return _truncate_text(text, max_chars) if max_chars else text
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item, max_chars=max_chars)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item, max_chars=max_chars) for item in value]
    return value


def _truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}\n...[truncated {omitted} chars]"


def _compact_execution(execution: Any, *, max_output_chars: int) -> dict[str, Any]:
    if not isinstance(execution, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        "ok",
        "status_source",
        "returncode",
        "exit_code",
        "stdout",
        "stderr",
        "output",
        "result",
        "error",
        "synthetic",
    ):
        if key not in execution or execution[key] is None:
            continue
        value = _sanitize_value(execution[key], max_chars=max_output_chars)
        compact[key] = value
    if not compact and "action" in execution:
        compact["ok"] = execution.get("ok")
    return compact


def _action_text(action: dict[str, Any]) -> str:
    return canonical_json(_sanitize_value(action))


def _history_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return "None"
    blocks = []
    for item in history:
        blocks.append(
            "\n".join((
                f"Step {item['step_index']} ({item['modality']}):",
                f"Action: {_action_text(item['action'])}",
                f"Execution: {canonical_json(item['execution'])}",
            ))
        )
    return "\n\n".join(blocks)


def _episode_content_flags(steps: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    command_text = "\n".join(
        str(((step.get("action") or {}).get("args") or {}).get("command") or "")
        for step in steps
        if (step.get("action") or {}).get("type") == "shell.exec"
    )
    serialized_steps = json.dumps(steps, ensure_ascii=False)
    blockers = [
        name
        for name, pattern in _STRICT_BLOCKERS.items()
        if pattern.search(serialized_steps if name == "embedded_url_credentials" else command_text)
    ]
    warnings = [name for name, pattern in _QUALITY_WARNINGS.items() if pattern.search(command_text)]
    return blockers, warnings


def _policy_exclusion_reasons(
    blockers: list[str],
    warnings: list[str],
    quality_policy: str,
) -> list[str]:
    reasons: set[str] = set()
    if quality_policy in {"strict", "high_quality"}:
        reasons.update(blockers)
    if quality_policy == "high_quality":
        reasons.update(flag for flag in warnings if flag in _HIGH_QUALITY_EXCLUSIONS)
    return sorted(reasons)


def _load_source_task_groups(taskset_root: str | Path | None) -> dict[str, list[str]]:
    if taskset_root is None:
        return {}
    root = Path(taskset_root)
    manifest_path = root / "taskset_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = _json(manifest_path)
    groups: dict[str, list[str]] = {}
    for task in manifest.get("tasks") or []:
        if not isinstance(task, dict) or not task.get("id") or not task.get("markdown"):
            continue
        markdown_path = root / str(task["markdown"])
        if not markdown_path.is_file():
            continue
        text = markdown_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^Generated from:\s*(.+?)\s*$", text, re.MULTILINE)
        if match:
            groups[str(task["id"])] = [
                item.strip() for item in match.group(1).split(",") if item.strip()
            ]
    return groups


def _with_source_groups(
    manifest: dict[str, Any],
    source_task_groups: dict[str, list[str]],
) -> dict[str, Any]:
    task = manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
    task_id = str(task.get("id") or "")
    source_task_ids = source_task_groups.get(task_id)
    if not source_task_ids:
        return manifest
    updated_manifest = dict(manifest)
    updated_task = dict(task)
    updated_metadata = dict(updated_task.get("metadata") or {})
    updated_metadata["source_task_ids"] = source_task_ids
    updated_task["metadata"] = updated_metadata
    updated_manifest["task"] = updated_task
    return updated_manifest


def _training_trajectory(
    episode_dir: Path,
    manifest: dict[str, Any],
    quality: dict[str, Any],
    steps: list[dict[str, Any]],
    *,
    packager: ImagePackager,
    max_execution_output_chars: int,
    quality_flags: list[str],
) -> dict[str, Any]:
    task = manifest.get("task") or {}
    converted_steps = []
    for step in steps:
        decision_context = _public_decision_context(step)
        observation = dict(step.get("pre_observation") or {})
        post_observation = dict(step.get("post_observation") or {})
        if observation.get("screenshot"):
            source = _absolute_image(episode_dir, observation["screenshot"])
            observation["screenshot"] = packager.package(Path(source)) if source else None
        if post_observation.get("screenshot"):
            source = _absolute_image(
                episode_dir, post_observation["screenshot"]
            )
            post_observation["screenshot"] = packager.package(Path(source)) if source else None
        converted_steps.append({
            "index": step.get("step"),
            "intent": decision_context.get("intent"),
            "decision_context": decision_context,
            "action": _sanitize_value(step.get("action")),
            "observation": _sanitize_value(observation),
            "post_observation": _sanitize_value(post_observation),
            "execution": _compact_execution(
                step.get("execution"),
                max_output_chars=max_execution_output_chars,
            ),
            "provenance": _public_provenance(step),
        })
    source_task_ids = (task.get("metadata") or {}).get("source_task_ids") or []
    return {
        "schema_version": "hybrid_cua_trajectory.v2",
        "episode_id": manifest.get("episode_id"),
        "video_id": manifest.get("episode_id"),
        "task_id": task.get("id"),
        "source_task_ids": source_task_ids,
        "split_group_id": source_task_ids[0] if source_task_ids else task.get("id"),
        "application": (task.get("metadata") or {}).get("application"),
        "category": (task.get("metadata") or {}).get("category"),
        "source_video": None,
        "trajectory_path": "episodes.jsonl",
        "trajectory_selector": {"episode_id": manifest.get("episode_id")},
        "task": _sanitize_text(str(task.get("instruction") or "")),
        "num_steps": len(converted_steps),
        "step_count": len(converted_steps),
        "training_tier": quality.get("training_tier"),
        "recommended_train_weight": 1.0 if quality.get("training_tier") == "sft_accepted" else 0.0,
        "trajectory_valid": quality.get("valid"),
        "exportable_for_training": quality.get("training_tier") == "sft_accepted",
        "quality_score": manifest.get("evaluation", {}).get("score"),
        "quality_flags": quality_flags,
        "steps": converted_steps,
    }


def _sft_examples(
    episode_dir: Path,
    manifest: dict[str, Any],
    quality: dict[str, Any],
    steps: list[dict[str, Any]],
    *,
    packager: ImagePackager,
    system_prompt: str,
    action_protocol: str,
    max_history_actions: int,
    visual_window: int,
    max_execution_output_chars: int,
    quality_flags: list[str],
    drop_unchanged_waits: bool,
    include_empty_think: bool,
) -> list[dict[str, Any]]:
    task = manifest.get("task") or {}
    instruction = _sanitize_text(str(task.get("instruction") or ""))
    source_task_ids = (task.get("metadata") or {}).get("source_task_ids") or []
    history: list[dict[str, Any]] = []
    screenshot_history: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for step in steps:
        action = _valid_action(step.get("action"))
        if action is None or not _execution_succeeded(step):
            continue
        visual_action = _action_modality(action) == "visual"
        screenshot_path = _absolute_image(
            episode_dir, (step.get("pre_observation") or {}).get("screenshot")
        ) if visual_action else None
        if visual_action and not screenshot_path:
            continue
        if screenshot_path:
            screenshot = packager.package(Path(screenshot_path))
            if (
                not screenshot_history
                or screenshot_history[-1]["sha256"] != screenshot["sha256"]
            ):
                screenshot_history.append(screenshot)
        image_metadata = screenshot_history[-visual_window:] if visual_window > 0 else []
        if visual_action and not image_metadata:
            continue
        images = [image["path"] for image in image_metadata]
        placeholders = "\n".join(
            f"Visual observation {index}:\n<image>"
            for index, _ in enumerate(images, 1)
        )
        previous = _history_text(history[-max_history_actions:])
        if action_protocol == ACTION_PROTOCOL_CANONICAL:
            user = "\n".join(filter(None, (
                f"Task: {instruction}",
                f"Recent interaction history:\n{previous}",
                (
                    "Recent visual observations (oldest to newest):\n"
                    f"{placeholders}"
                    if placeholders else
                    "Recent visual observations: None"
                ),
                "Return exactly one canonical action JSON object.",
            )))
        else:
            user = "\n".join(filter(None, (
                placeholders,
                "Please generate the next move according to the UI screenshot, "
                "instruction, and previous actions.",
                f"Instruction: {instruction}",
                f"Previous canonical actions and observations:\n{previous}",
                "Return the next single action using the Qwen XML tool-call format.",
            )))
        sanitized_action = _restore_source_shell_timeout(
            episode_dir,
            step,
            _sanitize_value(action),
        )
        decision_context = _public_decision_context(step)
        step_index = int(step.get("step") or len(history) + 1)
        history_entry = {
            "step_index": step_index,
            "modality": _action_modality(action),
            "action": sanitized_action,
            "execution": _compact_execution(
                step.get("execution"),
                max_output_chars=max_execution_output_chars,
            ),
        }
        if drop_unchanged_waits and _unchanged_wait(episode_dir, step):
            history.append(history_entry)
            continue
        if action_protocol == ACTION_PROTOCOL_CANONICAL:
            assistant = canonical_json(sanitized_action)
        else:
            assistant = format_qwen_xml_action(
                sanitized_action,
                include_empty_think=include_empty_think,
            )
            sanitized_action = QwenXmlActionParser().parse(assistant).to_dict()
            history_entry["action"] = sanitized_action
        rows.append({
            "schema_version": MIXED_SFT_SCHEMA,
            "sample_id": content_digest({
                "schema_version": MIXED_SFT_SCHEMA,
                "episode_id": manifest.get("episode_id"),
                "step_index": step_index,
                "action": sanitized_action,
                "image_sha256": [image["sha256"] for image in image_metadata],
            })[:32],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "images": images,
            "image_metadata": image_metadata,
            "task_id": task.get("id"),
            "source_task_ids": source_task_ids,
            "split_group_id": source_task_ids[0] if source_task_ids else task.get("id"),
            "episode_id": manifest.get("episode_id"),
            "step_index": step_index,
            "modality": _action_modality(action),
            "action_protocol": action_protocol,
            "target_action": sanitized_action,
            "decision_context": decision_context,
            "quality": {
                "episode_training_tier": quality.get("training_tier"),
                "episode_score": (manifest.get("evaluation") or {}).get("score"),
                "observation_quality": step.get("observation_quality"),
                "flags": quality_flags,
            },
        })
        history.append(history_entry)
    return rows


def _prefix_bundle_paths(value: Any, prefix: str) -> Any:
    if isinstance(value, list):
        return [_prefix_bundle_paths(item, prefix) for item in value]
    if isinstance(value, dict):
        updated = {
            key: _prefix_bundle_paths(item, prefix)
            for key, item in value.items()
        }
        path = updated.get("path")
        if isinstance(path, str) and path.startswith("images/"):
            updated["path"] = f"{prefix}/{path}"
        return updated
    if isinstance(value, str) and value.startswith("images/"):
        return f"{prefix}/{value}"
    return value


def _build_mixed_sft_bundle(
    destination: Path,
    episodes: list[tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
    *,
    copy_images: bool,
    require_pre_screenshot: bool,
    system_prompt: str,
    action_protocol: str,
    max_history_actions: int,
    visual_window: int,
    max_execution_output_chars: int,
    quality_policy: str,
    drop_unchanged_waits: bool,
    include_empty_think: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bundle_root = destination / "mixed"
    bundle_root.mkdir(parents=True, exist_ok=True)
    packager = ImagePackager(bundle_root, copy_images=copy_images)
    episode_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    action_types: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    unchanged_waits_dropped = 0

    for episode_dir, manifest, quality, steps in episodes:
        task = manifest.get("task") or {}
        task_id = task.get("id")
        episode_id = manifest.get("episode_id")
        blockers, warnings = _episode_content_flags(steps)
        warning_counts.update(warnings)
        structural_issues: list[str] = []
        for step in steps:
            action = _valid_action(step.get("action"))
            if action is None:
                structural_issues.append("invalid_action")
                continue
            if not _execution_succeeded(step):
                structural_issues.append("unsuccessful_execution")
            if _action_modality(action) == "visual":
                screenshot = _absolute_image(
                    episode_dir,
                    (step.get("pre_observation") or {}).get("screenshot"),
                )
                if require_pre_screenshot and not screenshot:
                    structural_issues.append("missing_visual_pre_screenshot")
        exclusion_reasons = sorted(set(structural_issues))
        exclusion_reasons.extend(
            _policy_exclusion_reasons(blockers, warnings, quality_policy)
        )
        exclusion_reasons = sorted(set(exclusion_reasons))
        if exclusion_reasons:
            exclusion_counts.update(exclusion_reasons)
            exclusion_rows.append({
                "episode_id": episode_id,
                "task_id": task_id,
                "reasons": exclusion_reasons,
                "warnings": warnings,
            })
            continue

        quality_flags = sorted(set(warnings))
        trajectory = _training_trajectory(
            episode_dir,
            manifest,
            quality,
            steps,
            packager=packager,
            max_execution_output_chars=max_execution_output_chars,
            quality_flags=quality_flags,
        )
        examples = _sft_examples(
            episode_dir,
            manifest,
            quality,
            steps,
            packager=packager,
            system_prompt=system_prompt,
            action_protocol=action_protocol,
            max_history_actions=max_history_actions,
            visual_window=visual_window,
            max_execution_output_chars=max_execution_output_chars,
            quality_flags=quality_flags,
            drop_unchanged_waits=drop_unchanged_waits,
            include_empty_think=include_empty_think,
        )
        episode_unchanged_waits = sum(
            _unchanged_wait(episode_dir, step) for step in steps
        ) if drop_unchanged_waits else 0
        expected_examples = len(steps) - episode_unchanged_waits
        if len(examples) != expected_examples:
            reason = "incomplete_sft_examples"
            exclusion_counts[reason] += 1
            exclusion_rows.append({
                "episode_id": episode_id,
                "task_id": task_id,
                "reasons": [reason],
                "warnings": warnings,
            })
            continue
        unchanged_waits_dropped += episode_unchanged_waits
        trajectory_rows.append(trajectory)
        sft_rows.extend(examples)
        for step in steps:
            action_types[str((step.get("action") or {}).get("type") or "unknown")] += 1
        episode_rows.append({
            "episode_id": episode_id,
            "path": "episodes.jsonl",
            "selector": {"episode_id": episode_id},
            "status": manifest.get("status"),
            "training_tier": quality.get("training_tier"),
            "task_id": task_id,
            "source_task_ids": (task.get("metadata") or {}).get("source_task_ids") or [],
            "agent_id": (manifest.get("agent") or {}).get("id"),
            "score": (manifest.get("evaluation") or {}).get("score"),
            "quality_flags": quality_flags,
        })

    _write_dataset_text(
        bundle_root / "episodes.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trajectory_rows),
    )
    _write_dataset_text(
        bundle_root / "trajectories.sft.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sft_rows),
    )
    _write_dataset_text(
        bundle_root / "excluded.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exclusion_rows),
    )
    stats = {
        "schema_version": "hybrid_cua_sft.dataset_stats.v2",
        "episodes_seen": len(episodes),
        "episodes_exported": len(episode_rows),
        "episodes_excluded": len(exclusion_rows),
        "sft_examples_exported": len(sft_rows),
        "unique_images": len(packager.references),
        "images_copied": packager.images_copied,
        "image_references_reused": packager.images_reused,
        "action_types": dict(sorted(action_types.items())),
        "quality_policy": quality_policy,
        "drop_unchanged_waits": drop_unchanged_waits,
        "unchanged_waits_dropped": unchanged_waits_dropped,
        "include_empty_think": include_empty_think,
        "quality_warning_counts": dict(sorted(warning_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "max_history_actions": max_history_actions,
        "visual_window": visual_window,
        "action_protocol": action_protocol,
        "max_execution_output_chars": max_execution_output_chars,
        "copy_images": copy_images,
        "require_pre_screenshot": require_pre_screenshot,
    }
    _write_dataset_json(bundle_root / "dataset.stats.json", stats)
    _write_dataset_json(bundle_root / "dataset_manifest.json", {
        "schema_version": MIXED_SFT_DATASET_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "episodes": "episodes.jsonl",
            "sft": "trajectories.sft.jsonl",
            "excluded": "excluded.jsonl",
            "images": "images/",
            "stats": "dataset.stats.json",
        },
        "schemas": {
            "episode": "hybrid_cua_trajectory.v2",
            "sft": MIXED_SFT_SCHEMA,
        },
        "policies": {
            "action_protocol": action_protocol,
            "modalities": ["visual", "shell", "control"],
            "portable_relative_paths": True,
            "include_execution_results_in_history": True,
            "drop_unchanged_waits": drop_unchanged_waits,
            "include_empty_think": include_empty_think,
            "include_private_reasoning": False,
            "redact_secrets": True,
            "quality_policy": quality_policy,
            "source_group_aware": True,
        },
        "counts": stats,
    })
    return stats, episode_rows, trajectory_rows, sft_rows


_BUNDLE_SCHEMAS = {
    "canonical": (
        CANONICAL_STEP_SCHEMA,
        CANONICAL_EPISODE_SCHEMA,
        CANONICAL_DATASET_SCHEMA,
        "cua_sft.dataset_stats.v1",
        ["visual", "control"],
    ),
    "hybrid": (
        HYBRID_STEP_SCHEMA,
        HYBRID_EPISODE_SCHEMA,
        HYBRID_DATASET_SCHEMA,
        "hybrid_sft.dataset_stats.v1",
        ["visual", "shell", "control"],
    ),
    "shell": (
        SHELL_STEP_SCHEMA,
        SHELL_EPISODE_SCHEMA,
        SHELL_DATASET_SCHEMA,
        "shell_sft.dataset_stats.v1",
        ["shell", "control"],
    ),
}


def _build_portable_bundle(
    destination: Path,
    episodes: list[tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
    *,
    bundle_name: str,
    copy_images: bool,
    require_pre_screenshot: bool,
) -> dict[str, Any]:
    step_schema, episode_schema, dataset_schema, stats_schema, modalities = (
        _BUNDLE_SCHEMAS[bundle_name]
    )
    bundle_root = destination / bundle_name
    bundle_root.mkdir(parents=True, exist_ok=True)
    packager = ImagePackager(bundle_root, copy_images=copy_images)
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    action_types: Counter[str] = Counter()
    stats: Counter[str] = Counter()

    for episode_dir, manifest, quality, steps in episodes:
        stats["episodes_seen"] += 1
        episode_id = str(manifest.get("episode_id") or episode_dir.name)
        task = _portable_task(manifest)
        agent = _portable_agent(manifest)
        environment = _portable_environment(manifest)
        history: list[dict[str, Any]] = []
        history_steps: list[dict[str, Any]] = []
        pending_step_rows: list[dict[str, Any]] = []
        exported_steps: list[dict[str, Any]] = []
        episode_skips = 0

        for step in steps:
            stats["steps_seen"] += 1
            action = _valid_action(step.get("action"))
            if action is None:
                stats["skipped_invalid_action"] += 1
                episode_skips += 1
                continue
            if not _execution_succeeded(step):
                stats["skipped_failed_action"] += 1
                episode_skips += 1
                continue

            modality = _action_modality(action)
            screenshot_path_value = None
            screenshot = None
            if modality == "visual":
                raw_screenshot = (step.get("pre_observation") or {}).get("screenshot")
                screenshot_path_value = _absolute_image(episode_dir, raw_screenshot)
                if screenshot_path_value:
                    screenshot = packager.package(Path(screenshot_path_value))
                elif require_pre_screenshot:
                    stats["skipped_missing_pre_screenshot"] += 1
                    episode_skips += 1
                    continue

            step_index = int(step.get("step") or len(history) + 1)
            sample_id = content_digest({
                "schema_version": step_schema,
                "episode_id": episode_id,
                "step_index": step_index,
                "action": action,
                "screenshot_sha256": (screenshot or {}).get("sha256"),
            })[:32]
            step_quality = {
                "episode_training_tier": quality.get("training_tier"),
                "episode_valid": quality.get("valid"),
                "observation_quality": step.get("observation_quality"),
            }
            decision_context = _step_decision_context(step)
            observation = {
                "modality": modality,
                "screenshot": screenshot if modality == "visual" else None,
            }
            execution = dict(step.get("execution") or {})
            episode_step = {
                "step_index": step_index,
                "modality": modality,
                "observation": observation,
                "action": action,
                "execution": execution,
                "decision_context": decision_context,
                "provenance": step.get("provenance") or {},
                "quality": step_quality,
            }
            history_record: dict[str, Any] = {"actions": list(history)}
            if bundle_name != "canonical":
                history_record["steps"] = list(history_steps)
            pending_step_rows.append({
                "schema_version": step_schema,
                "sample_id": sample_id,
                "episode_id": episode_id,
                "step_index": step_index,
                "task": task,
                "agent": agent,
                "environment": environment,
                "modality": modality,
                "observation": observation,
                "history": history_record,
                "action": action,
                "execution": execution,
                "decision_context": decision_context,
                "provenance": step.get("provenance") or {},
                "quality": step_quality,
            })
            exported_steps.append(episode_step)
            history.append(action)
            history_steps.append({"action": action, "execution": execution})

        if not exported_steps:
            stats["episodes_without_exportable_steps"] += 1
            continue
        complete = episode_skips == 0 and len(exported_steps) == len(steps)
        if not complete:
            stats["episodes_incomplete"] += 1
            stats["episodes_rejected_incomplete"] += 1
            stats["steps_rejected_with_episode"] += len(exported_steps)
            continue
        step_rows.extend(pending_step_rows)
        for episode_step in exported_steps:
            action_types[episode_step["action"]["type"]] += 1
        stats["steps_exported"] += len(exported_steps)
        episode_rows.append({
            "schema_version": episode_schema,
            "episode_id": episode_id,
            "status": manifest.get("status"),
            "bundle": bundle_name,
            "task": task,
            "agent": agent,
            "environment": environment,
            "evaluation": _portable_evaluation(manifest),
            "training_tier": quality.get("training_tier"),
            "complete": complete,
            "original_step_count": len(steps),
            "exported_step_count": len(exported_steps),
            "steps": exported_steps,
        })
        stats["episodes_exported"] += 1

    _write_dataset_text(
        bundle_root / "steps.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in step_rows),
    )
    _write_dataset_text(
        bundle_root / "episodes.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_rows),
    )
    for key in (
        "episodes_seen",
        "episodes_exported",
        "episodes_incomplete",
        "episodes_rejected_incomplete",
        "episodes_without_exportable_steps",
        "steps_seen",
        "steps_exported",
        "steps_rejected_with_episode",
        "skipped_invalid_action",
        "skipped_failed_action",
        "skipped_missing_pre_screenshot",
    ):
        stats.setdefault(key, 0)
    bundle_stats = {
        "schema_version": stats_schema,
        **dict(sorted(stats.items())),
        "unique_images": len(packager.references),
        "images_copied": packager.images_copied,
        "image_references_reused": packager.images_reused,
        "action_types": dict(sorted(action_types.items())),
        "copy_images": copy_images,
        "require_pre_screenshot": require_pre_screenshot,
    }
    _write_dataset_json(bundle_root / "dataset.stats.json", bundle_stats)
    _write_dataset_json(bundle_root / "dataset_manifest.json", {
        "schema_version": dataset_schema,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "steps": "steps.jsonl",
            "episodes": "episodes.jsonl",
            "images": "images/",
            "stats": "dataset.stats.json",
        },
        "schemas": {
            "step": step_schema,
            "episode": episode_schema,
        },
        "policies": {
            "sample_granularity": ["step", "episode"],
            "modalities": modalities,
            "screenshot_binding": "pre_action_visual_only",
            "successful_actions_only": True,
            "complete_episodes_only": True,
            "include_finish": True,
            "include_wait": True,
            "copy_images": copy_images,
            "image_naming": "sha256",
            "require_pre_screenshot_for_visual_actions": require_pre_screenshot,
        },
        "counts": bundle_stats,
    })
    return bundle_stats


def build_dataset(
    collection_root: str | Path,
    output_dir: str | Path,
    *,
    accepted_only: bool = True,
    system_prompt: str | None = None,
    action_protocol: str = DEFAULT_ACTION_PROTOCOL,
    max_history_actions: int = 3,
    visual_window: int = 10,
    bundle_format: str = "all",
    copy_images: bool = True,
    require_pre_screenshot: bool = True,
    quality_policy: str = "high_quality",
    max_execution_output_chars: int = 4000,
    taskset_root: str | Path | None = None,
    drop_unchanged_waits: bool = True,
    include_empty_think: bool = False,
) -> dict[str, Any]:
    if action_protocol not in SUPPORTED_ACTION_PROTOCOLS - {"auto"}:
        raise ValueError(f"unsupported dataset action protocol: {action_protocol}")
    if system_prompt is None:
        system_prompt = (
            CANONICAL_SYSTEM_PROMPT
            if action_protocol == ACTION_PROTOCOL_CANONICAL
            else HYBRID_QWEN_XML_SYSTEM_PROMPT
        )
    if bundle_format not in {"all", "canonical", "hybrid", "shell", "none"}:
        raise ValueError(f"unsupported bundle format: {bundle_format}")
    if quality_policy not in {"structural", "strict", "high_quality"}:
        raise ValueError(f"unsupported quality policy: {quality_policy}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_task_groups = _load_source_task_groups(taskset_root)
    tiers: Counter[str] = Counter()
    candidate_episodes: list[
        tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]
    ] = []
    for episode_dir in discover_episodes(collection_root):
        manifest = _json(episode_dir / "episode_manifest.json")
        manifest = _with_source_groups(manifest, source_task_groups)
        quality = _json(episode_dir / "quality.json")
        tier = str(quality.get("training_tier") or "unknown")
        tiers[tier] += 1
        if accepted_only and tier != "sft_accepted":
            continue
        steps = _jsonl(episode_dir / "canonical" / "steps.jsonl")
        candidate_episodes.append((episode_dir, manifest, quality, steps))

    mixed_stats, episode_rows, trajectory_rows, sft_rows = _build_mixed_sft_bundle(
        destination,
        candidate_episodes,
        copy_images=copy_images,
        require_pre_screenshot=require_pre_screenshot,
        system_prompt=system_prompt,
        action_protocol=action_protocol,
        max_history_actions=max_history_actions,
        visual_window=visual_window,
        max_execution_output_chars=max_execution_output_chars,
        quality_policy=quality_policy,
        drop_unchanged_waits=drop_unchanged_waits,
        include_empty_think=include_empty_think,
    )
    selected_episodes = [
        episode
        for episode in candidate_episodes
        if not _policy_exclusion_reasons(
            *_episode_content_flags(episode[3]),
            quality_policy,
        )
    ]
    top_level_episode_rows = []
    for row in episode_rows:
        updated = dict(row)
        updated["path"] = "mixed/episodes.jsonl"
        top_level_episode_rows.append(updated)
    top_level_trajectory_rows = [
        _prefix_bundle_paths(row, "mixed") for row in trajectory_rows
    ]
    for row in top_level_trajectory_rows:
        row["trajectory_path"] = "mixed/episodes.jsonl"
    top_level_sft_rows = [_prefix_bundle_paths(row, "mixed") for row in sft_rows]
    _write_dataset_text(
        destination / "episodes.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in top_level_episode_rows
        ),
    )
    _write_dataset_text(
        destination / "trajectories.train.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in top_level_trajectory_rows
        ),
    )
    _write_dataset_text(
        destination / "trajectories.sft.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in top_level_sft_rows
        ),
    )
    routed_episodes: dict[str, list[
        tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]
    ]] = {"canonical": [], "hybrid": [], "shell": []}
    for episode in selected_episodes:
        routed_episodes[_episode_bundle(episode[3])].append(episode)
    requested_bundles = (
        {"canonical", "hybrid", "shell"}
        if bundle_format == "all"
        else ({bundle_format} if bundle_format != "none" else set())
    )
    bundle_stats = {
        name: _build_portable_bundle(
            destination,
            routed_episodes[name],
            bundle_name=name,
            copy_images=copy_images,
            require_pre_screenshot=require_pre_screenshot,
        )
        for name in ("canonical", "hybrid", "shell")
        if name in requested_bundles
    }
    stats = {
        "schema_version": "gui_trajectory.dataset_stats.v1",
        "episodes_seen_by_tier": dict(sorted(tiers.items())),
        "episodes_exported": len(top_level_episode_rows),
        "trajectories_exported": len(top_level_trajectory_rows),
        "sft_examples_exported": len(top_level_sft_rows),
        "action_types": mixed_stats["action_types"],
        "accepted_only": accepted_only,
        "quality_policy": quality_policy,
        "max_execution_output_chars": max_execution_output_chars,
        "source_task_groups_loaded": len(source_task_groups),
        "visual_window": visual_window,
        "max_history_actions": max_history_actions,
        "action_protocol": action_protocol,
        "drop_unchanged_waits": drop_unchanged_waits,
        "include_empty_think": include_empty_think,
        "bundle_format": bundle_format,
        "episodes_routed_by_bundle": {
            name: len(rows) for name, rows in routed_episodes.items()
        },
        "canonical_bundle": bundle_stats.get("canonical"),
        "hybrid_bundle": bundle_stats.get("hybrid"),
        "shell_bundle": bundle_stats.get("shell"),
        "mixed_bundle": mixed_stats,
    }
    _write_dataset_json(destination / "dataset.stats.json", stats)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build portable canonical and ms-swift CUA datasets."
    )
    parser.add_argument("--collection-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--taskset-root",
        help="Optional taskset root used to attach source-task split groups.",
    )
    parser.add_argument("--include-non-accepted", action="store_true")
    parser.add_argument("--max-history-actions", type=int, default=3)
    parser.add_argument("--visual-window", type=int, default=10)
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--action-protocol",
        choices=tuple(sorted(SUPPORTED_ACTION_PROTOCOLS - {"auto"})),
        default=DEFAULT_ACTION_PROTOCOL,
        help="Assistant target protocol; defaults to the shared Qwen XML contract.",
    )
    parser.add_argument(
        "--bundle-format",
        choices=("all", "canonical", "hybrid", "shell", "none"),
        default="all",
        help="Portable dataset bundle(s) to generate.",
    )
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy screenshots into visual-capable bundles using content hashes.",
    )
    parser.add_argument(
        "--require-pre-screenshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject episodes whose visual actions lack an action-input screenshot.",
    )
    parser.add_argument(
        "--quality-policy",
        choices=("structural", "strict", "high_quality"),
        default="high_quality",
        help=(
            "Strict excludes infrastructure tampering and leaked credentials; "
            "high_quality also excludes package installation and direct browser-profile edits."
        ),
    )
    parser.add_argument(
        "--max-execution-output-chars",
        type=int,
        default=4000,
        help="Maximum characters retained for each execution output field.",
    )
    parser.add_argument(
        "--drop-unchanged-waits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop wait targets whose pre/post screenshots are byte-identical.",
    )
    parser.add_argument(
        "--include-empty-think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include an empty <think> block before Qwen XML targets.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = build_dataset(
        args.collection_root,
        args.output_dir,
        accepted_only=not args.include_non_accepted,
        system_prompt=args.system_prompt,
        action_protocol=args.action_protocol,
        max_history_actions=args.max_history_actions,
        visual_window=args.visual_window,
        bundle_format=args.bundle_format,
        copy_images=args.copy_images,
        require_pre_screenshot=args.require_pre_screenshot,
        quality_policy=args.quality_policy,
        max_execution_output_chars=args.max_execution_output_chars,
        taskset_root=args.taskset_root,
        drop_unchanged_waits=args.drop_unchanged_waits,
        include_empty_think=args.include_empty_think,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
