"""TaskSpec loaders and ScaleCUA-style static verification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Protocol

from .schemas import TaskSpec


class ProfileLike(Protocol):
    os_type: str
    os_version: str
    architecture: str
    software: dict[str, str]


def _normalized(value: str) -> str:
    return value.strip().casefold().replace(" ", "-")


def profile_capabilities(profile: ProfileLike) -> set[str]:
    capabilities = {
        "computer",
        "filesystem",
        "terminal",
        f"os:{_normalized(profile.os_type)}",
        f"arch:{_normalized(profile.architecture)}",
    }
    for name in profile.software:
        normalized = _normalized(name)
        capabilities.add(normalized)
        capabilities.add(f"app:{normalized}")
    return capabilities


def task_supports_profile(task: TaskSpec, profile: ProfileLike) -> bool:
    if task.target_environments:
        os_type = _normalized(profile.os_type)
        os_version = _normalized(profile.os_version)
        architecture = _normalized(profile.architecture)
        matching_environment = any(
            _normalized(environment.os_type) == os_type
            and (
                not environment.os_versions
                or os_version in {_normalized(item) for item in environment.os_versions}
            )
            and (
                not environment.architectures
                or architecture
                in {_normalized(item) for item in environment.architectures}
            )
            for environment in task.target_environments
        )
        if not matching_environment:
            return False
    return set(task.required_capabilities) <= profile_capabilities(profile)


def load_task_spec(path: str | Path) -> TaskSpec:
    task_path = Path(path)
    if task_path.suffix.lower() == ".json":
        raw = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"task file must contain an object: {task_path}")
        return TaskSpec.from_dict(raw)
    if task_path.suffix.lower() in {".md", ".txt"}:
        instruction = task_path.read_text(encoding="utf-8").strip()
        return TaskSpec(
            id=task_path.stem,
            instruction=instruction,
            metadata={"source": str(task_path), "format": task_path.suffix.lstrip(".")},
        )
    raise ValueError(f"unsupported task format: {task_path}")


def discover_task_specs(root: str | Path) -> list[TaskSpec]:
    task_root = Path(root)
    candidates = sorted({
        *task_root.rglob("*.json"),
        *task_root.rglob("*.md"),
    })
    specs: list[TaskSpec] = []
    for path in candidates:
        try:
            specs.append(load_task_spec(path))
        except (ValueError, json.JSONDecodeError):
            continue
    return specs


def verify_task_spec(
    task: TaskSpec,
    *,
    available_capabilities: Iterable[str] = (),
    judge_score: float | None = None,
    negative_control_scores: Iterable[float] = (),
    success_threshold: float = 1.0,
) -> dict[str, Any]:
    capabilities = set(available_capabilities)
    missing = sorted(set(task.required_capabilities) - capabilities)
    errors: list[str] = []
    if missing:
        errors.append(f"missing capabilities: {', '.join(missing)}")
    if not task.evaluator:
        errors.append("missing executable evaluator")
    if judge_score is not None and not isinstance(judge_score, (int, float)):
        errors.append("judge score is not numeric")
    false_positives = [
        float(score) for score in negative_control_scores if float(score) >= success_threshold
    ]
    if false_positives:
        errors.append("negative control reached success threshold")
    return {
        "schema_version": "gui_trajectory.task_validation.v1",
        "task_id": task.id,
        "task_version": task.resolved_version,
        "valid": not errors,
        "checks": {
            "schema": True,
            "capabilities": not missing,
            "judge_only": judge_score is None or isinstance(judge_score, (int, float)),
            "negative_control": not false_positives,
        },
        "missing_capabilities": missing,
        "errors": errors,
    }
