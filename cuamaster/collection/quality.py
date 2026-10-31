"""Episode validation, secret scanning, and training-tier classification."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from cuamaster.action_space import CUAAction
from .collection_policy import (
    COLLECTOR_TAMPERING_RE,
    EVALUATION_ARTIFACT_ACCESS_RE,
    HARD_CODED_SUDO_PASSWORD_RE,
    HIDDEN_APPLICATION_STATE_RE,
    MAX_TRAINING_ACTION_CHARS,
    NETWORK_WORKAROUND_RE,
)
from .trace_store import atomic_write_json


_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|secret)\s*[=:]\s*['\"]?[^\s'\"]{8,}", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}", re.I),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.I),
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _referenced_images(steps: Iterable[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for step in steps:
        for key in ("pre_observation", "post_observation"):
            observation = step.get(key)
            if isinstance(observation, dict) and observation.get("screenshot"):
                paths.add(str(observation["screenshot"]))
    return paths


def _screenshot_digest(root: Path, value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, dict):
        digest = value.get("sha256")
        if isinstance(digest, str) and digest:
            return digest
        value = value.get("path")
    path = Path(str(value))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        return None
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _action_policy_findings(action: dict[str, Any], step: int) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    action_type = str(action.get("type") or "")
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if action_type == "tool.call":
        name = str(args.get("name") or "").strip()
        findings.append({
            "step": step,
            "reason": "structured_tool_forbidden",
            "tool": name,
        })
    def text_values(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [text for item in value.values() for text in text_values(item)]
        if isinstance(value, list):
            return [text for item in value for text in text_values(item)]
        return []

    policy_text = "\n".join(text_values(action))
    serialized = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    for reason, pattern in (
        ("hardcoded_sudo_password", HARD_CODED_SUDO_PASSWORD_RE),
        ("hidden_application_state_access", HIDDEN_APPLICATION_STATE_RE),
        ("collector_tampering", COLLECTOR_TAMPERING_RE),
        ("evaluation_artifact_access", EVALUATION_ARTIFACT_ACCESS_RE),
        ("network_workaround", NETWORK_WORKAROUND_RE),
    ):
        if pattern.search(policy_text):
            findings.append({"step": step, "reason": reason})
    if len(serialized) > MAX_TRAINING_ACTION_CHARS:
        findings.append({
            "step": step,
            "reason": "oversized_action",
            "chars": len(serialized),
            "limit": MAX_TRAINING_ACTION_CHARS,
        })
    return findings


def scan_secrets(root: Path, *, max_file_bytes: int = 5_000_000) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > max_file_bytes:
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".log", ".txt", ".sh", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                findings.append({"file": str(path.relative_to(root)), "pattern": pattern.pattern})
                break
    return findings


def validate_episode(episode_dir: str | Path) -> dict[str, Any]:
    root = Path(episode_dir)
    manifest_path = root / "episode_manifest.json"
    steps_path = root / "canonical" / "steps.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    training_blockers: list[str] = []
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        errors.append(f"invalid episode manifest: {exc}")
    try:
        steps = _jsonl(steps_path)
    except json.JSONDecodeError as exc:
        steps = []
        errors.append(f"invalid canonical steps: {exc}")
    if not steps:
        errors.append("episode has no canonical steps")
    task = manifest.get("task") if isinstance(manifest, dict) else {}
    if (task or {}).get("metadata", {}).get("instruction_missing"):
        errors.append("task instruction is missing")
    if not (task or {}).get("evaluator"):
        errors.append("task evaluator is missing")
    for expected, step in enumerate(steps, 1):
        if step.get("step") != expected:
            errors.append(f"non-contiguous step index at {expected}")
        action = step.get("action")
        if not isinstance(action, dict) or not action.get("type"):
            errors.append(f"step {expected} has invalid action")
        else:
            try:
                CUAAction.from_dict(action)
            except (TypeError, ValueError) as exc:
                errors.append(f"step {expected} has invalid action: {exc}")
    gui_steps = []
    missing_execution_steps = 0
    failed_execution_steps = 0
    missing_visual_pre_screenshot_steps = 0
    partial_visual_steps = 0
    unchanged_wait_steps = 0
    action_policy_findings: list[dict[str, Any]] = []
    for step in steps:
        action_type = str((step.get("action") or {}).get("type") or "")
        action = step.get("action")
        if isinstance(action, dict):
            action_policy_findings.extend(
                _action_policy_findings(action, int(step.get("step") or 0))
            )
        if action_type.startswith("computer."):
            gui_steps.append(step)
            if not (step.get("pre_observation") or {}).get("screenshot"):
                missing_visual_pre_screenshot_steps += 1
            if step.get("observation_quality") != "complete":
                partial_visual_steps += 1
        if action_type == "computer.wait":
            before = _screenshot_digest(
                root, (step.get("pre_observation") or {}).get("screenshot")
            )
            after = _screenshot_digest(
                root, (step.get("post_observation") or {}).get("screenshot")
            )
            if before and after and before == after:
                unchanged_wait_steps += 1
        if action_type == "finish":
            continue
        execution = step.get("execution")
        ok = execution.get("ok") if isinstance(execution, dict) else None
        if not isinstance(ok, bool):
            missing_execution_steps += 1
        elif not ok:
            failed_execution_steps += 1
    if missing_visual_pre_screenshot_steps:
        errors.append(
            f"{missing_visual_pre_screenshot_steps} GUI action(s) have no "
            "pre-observation screenshot"
        )
    if missing_execution_steps:
        training_blockers.append(
            f"{missing_execution_steps} action(s) lack verified execution status"
        )
    if failed_execution_steps:
        training_blockers.append(
            f"{failed_execution_steps} action(s) have failed execution"
        )
    if partial_visual_steps:
        training_blockers.append(
            f"{partial_visual_steps} GUI action(s) have partial visual alignment"
        )
    if action_policy_findings:
        reasons = sorted({finding["reason"] for finding in action_policy_findings})
        training_blockers.append(
            "action policy violations: " + ", ".join(reasons)
        )
    if unchanged_wait_steps:
        warnings.append(
            f"{unchanged_wait_steps} wait action(s) produced no visual change"
        )
    for relative in sorted(_referenced_images(steps)):
        if not (root / relative).is_file():
            errors.append(f"missing screenshot: {relative}")
    partial = sum(step.get("observation_quality") != "complete" for step in steps)
    if partial:
        warnings.append(f"{partial} step(s) have partial observation alignment")
    secret_findings = scan_secrets(root)
    if secret_findings:
        errors.append(f"secret scan found {len(secret_findings)} file(s)")
    evaluation = manifest.get("evaluation") if isinstance(manifest, dict) else {}
    success = bool((evaluation or {}).get("success"))
    if errors:
        tier = "rejected_corrupt"
    elif success and not training_blockers:
        tier = "sft_accepted"
    elif success or manifest.get("status") in {"task_failed", "agent_failed"}:
        tier = "preference_candidate"
    else:
        tier = "diagnostic_only"
    report = {
        "schema_version": "gui_trajectory.quality.v1",
        "valid": not errors,
        "training_tier": tier,
        "step_count": len(steps),
        "partial_observation_steps": partial,
        "partial_visual_steps": partial_visual_steps,
        "missing_execution_steps": missing_execution_steps,
        "failed_execution_steps": failed_execution_steps,
        "unchanged_wait_steps": unchanged_wait_steps,
        "action_policy_findings": action_policy_findings,
        "missing_visual_pre_screenshot_steps": missing_visual_pre_screenshot_steps,
        "errors": errors,
        "warnings": warnings,
        "training_blockers": training_blockers,
        "secret_findings": secret_findings,
    }
    atomic_write_json(root / "quality.json", report)
    return report
