"""Build canonical atomic episodes from collected backend artifacts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .profiles import VMProfile
from .schemas import (
    EPISODE_SCHEMA,
    AgentSpec,
    EpisodeStatus,
    StepRecord,
    TaskSpec,
    content_digest,
    episode_id,
)
from .trace_store import atomic_write_json, atomic_write_text


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _relative_screenshot(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        if path.parent.name == "screenshots" or path.parent.name == "_screenshots":
            return f"screenshots/{path.name}"
        return None
    if path.parts and path.parts[0] == "screenshots":
        return str(path)
    return f"screenshots/{path.name}"


def _task_spec(task_dir: Path, score: dict[str, Any], status: dict[str, Any]) -> TaskSpec:
    raw = _read_json(task_dir / "task.json")
    task_id = str(
        raw.get("id") or score.get("task_id") or status.get("task_id") or task_dir.name
    )
    instruction = str(
        raw.get("instruction")
        or score.get("instruction")
        or status.get("instruction")
        or ""
    ).strip()
    missing_instruction = not instruction
    if missing_instruction:
        instruction = f"[missing instruction] {task_id}"
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    metadata = {
        **metadata,
        "category": score.get("category", task_dir.parent.name),
        "mode": score.get("mode", "gui"),
        "instruction_missing": missing_instruction,
    }
    evaluator_value = raw.get("evaluator") or score.get("evaluator") or {}
    if isinstance(evaluator_value, str):
        evaluator_value = {"name": evaluator_value}
    if not isinstance(evaluator_value, dict):
        evaluator_value = {}
    return TaskSpec.from_dict({
        "id": task_id,
        "instruction": instruction,
        "snapshot": raw.get("snapshot") or "clean",
        "config": raw.get("config") or [],
        "evaluator": evaluator_value,
        "required_capabilities": raw.get("required_capabilities") or [],
        "target_environments": raw.get("target_environments") or [],
        "metadata": metadata,
        "version": raw.get("version"),
    })


def _agent_spec(profile: VMProfile, score: dict[str, Any], status: dict[str, Any]) -> AgentSpec:
    harness = str(score.get("harness") or status.get("harness") or "unknown")
    model = str(score.get("model") or status.get("model") or "unknown")
    options = profile.backend_options
    return AgentSpec(
        id=f"{harness}-{model}".replace("/", "_").replace(" ", "_"),
        harness=harness,
        model=model,
        provider=str(options.get("model_provider") or options.get("provider") or "unknown"),
        transport=str(options.get("transport") or "unknown"),
        reasoning={
            key: value for key, value in options.items()
            if "reason" in str(key).lower() or "thinking" in str(key).lower()
        },
        max_steps=int(options.get("max_steps") or 300),
        timeout_seconds=int(options.get("timeout_seconds") or 7200),
        runtime_strategy=str(options.get("image_strategy") or "runtime-overlay"),
        metadata={
            "harness_version": status.get("harness_version"),
            "resolved_model": status.get("resolved_model") or score.get("resolved_model"),
        },
    )


def _evaluation(score: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    raw_score = score.get("score")
    explicit = score.get("success")
    success = bool(explicit) if explicit is not None else (
        isinstance(raw_score, (int, float)) and float(raw_score) >= 1.0
    )
    return {
        "score": raw_score,
        "scores": score.get("scores"),
        "success": success,
        "evaluator": score.get("evaluator"),
        "evaluator_version": score.get("evaluator_version"),
        "agent_done": status.get("agent_done", score.get("agent_done")),
        "agent_status": score.get("agent_status") or status.get("agent_status"),
        "error": score.get("error") or status.get("error"),
    }


def _episode_status(evaluation: dict[str, Any], status: dict[str, Any]) -> EpisodeStatus:
    if status.get("timed_out"):
        return EpisodeStatus.TIMED_OUT
    if status.get("cancelled"):
        return EpisodeStatus.CANCELLED
    if evaluation.get("success"):
        return EpisodeStatus.SUCCEEDED
    if evaluation.get("agent_done") is False or evaluation.get("agent_status") in {"error", "failed"}:
        return EpisodeStatus.AGENT_FAILED
    return EpisodeStatus.TASK_FAILED


def canonical_steps(events: list[dict[str, Any]], *, success: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_screenshot: str | None = None
    for raw_turn, event in enumerate(events, 1):
        actions = event.get("actions") if isinstance(event.get("actions"), list) else []
        executions = event.get("executions") if isinstance(event.get("executions"), list) else []
        pre_screenshot = _relative_screenshot(event.get("pre_screenshot")) or previous_screenshot
        if pre_screenshot:
            previous_screenshot = pre_screenshot
        event_post = _relative_screenshot(event.get("screenshot"))
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            execution = executions[action_index] if action_index < len(executions) else {}
            if not isinstance(execution, dict):
                execution = {"output": execution}
            execution = dict(execution)
            execution_post = _relative_screenshot(execution.pop("screenshot", None))
            is_last = action_index == len(actions) - 1
            post_screenshot = execution_post or (event_post if is_last else None)
            action_type = str(action.get("type") or "")
            if action_type.startswith("computer."):
                quality = "complete" if pre_screenshot and post_screenshot else "partial"
            elif action_type == "finish":
                quality = "complete"
            else:
                quality = (
                    "complete"
                    if isinstance(execution.get("ok"), bool)
                    else "partial"
                )
            decision_context = (
                dict(event.get("decision_context"))
                if isinstance(event.get("decision_context"), dict)
                else {}
            )
            record = StepRecord(
                step=len(records) + 1,
                pre_observation={"screenshot": pre_screenshot} if pre_screenshot else {},
                action=dict(action),
                execution=execution,
                post_observation={"screenshot": post_screenshot} if post_screenshot else {},
                provenance={
                    "raw_turn": raw_turn,
                    "raw_action_index": action_index,
                    "source": event.get("source"),
                    "tool_call_id": event.get("tool_call_id"),
                    "tool_name": event.get("tool_name"),
                    "assistant_intent": decision_context.get("intent"),
                    "started_at": event.get("started_at"),
                    "finished_at": event.get("finished_at"),
                },
                decision_context=decision_context,
                observation_quality=quality,
            ).to_dict()
            records.append(record)
            if post_screenshot:
                previous_screenshot = post_screenshot
                pre_screenshot = post_screenshot
    if not records or records[-1]["action"].get("type") != "finish":
        terminal_screenshot = previous_screenshot
        records.append(StepRecord(
            step=len(records) + 1,
            pre_observation={"screenshot": terminal_screenshot} if terminal_screenshot else {},
            action={"type": "finish", "args": {"status": "success" if success else "failure"}},
            execution={"ok": True, "synthetic": True},
            post_observation={"screenshot": terminal_screenshot} if terminal_screenshot else {},
            provenance={"source": "synthetic", "reason": "episode terminal status"},
            observation_quality="complete",
        ).to_dict())
    return records


def build_episode(
    task_dir: str | Path,
    profile: VMProfile,
    events: list[dict[str, Any]],
    *,
    run_id: str = "standalone",
    attempt: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    root = Path(task_dir)
    score = _read_json(root / "score.json")
    execution_status = _read_json(root / "execution_status.json")
    task = _task_spec(root, score, execution_status)
    agent = _agent_spec(profile, score, execution_status)
    evaluation = _evaluation(score, execution_status)
    status = _episode_status(evaluation, execution_status)
    image_metadata = profile.image_metadata or profile.as_metadata().get("image", {})
    image_digest = str(
        image_metadata.get("sha256")
        or image_metadata.get("digest")
        or content_digest(image_metadata)
    )
    identifier = episode_id(
        task_version=task.resolved_version,
        image_digest=image_digest,
        agent=agent.to_dict(),
        seed=seed,
        attempt=attempt,
    )
    steps = canonical_steps(events, success=bool(evaluation["success"]))
    canonical_dir = root / "canonical"
    atomic_write_text(
        canonical_dir / "steps.jsonl",
        "".join(json.dumps(step, ensure_ascii=False) + "\n" for step in steps),
    )
    atomic_write_json(canonical_dir / "task.json", task.to_dict())
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": EPISODE_SCHEMA,
        "episode_id": identifier,
        "run_id": run_id,
        "created_at": created_at,
        "status": status.value,
        "attempt": attempt,
        "seed": seed,
        "task": task.to_dict(),
        "agent": agent.to_dict(),
        "vm": profile.as_metadata(),
        "image_digest": image_digest,
        "evaluation": evaluation,
        "step_count": len(steps),
        "files": {
            "steps": "canonical/steps.jsonl",
            "task": "canonical/task.json",
            "source_task": "task.json" if (root / "task.json").is_file() else None,
            "chat": "chat.jsonl" if (root / "chat.jsonl").is_file() else None,
            "computer_events": (
                "computer_events.jsonl" if (root / "computer_events.jsonl").is_file() else None
            ),
            "environment_fingerprint": (
                "environment_fingerprint.json"
                if (root / "environment_fingerprint.json").is_file() else None
            ),
        },
        "source_sha256": {
            "chat": _digest(root / "chat.jsonl"),
            "computer_events": _digest(root / "computer_events.jsonl"),
            "canonical_runtime": _digest(root / "canonical_trajectory.jsonl"),
        },
    }
    atomic_write_json(root / "episode_manifest.json", manifest)
    return manifest
