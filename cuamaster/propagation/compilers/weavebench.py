"""Compile canonical generated tasks into WeaveBench Markdown tasksets."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from cuamaster.contracts import TaskSpec, content_digest

from ..schemas import GeneratedTask, SourceTask, TaskRecipe
from ..utils import atomic_write_json, atomic_write_text


def _load_generated(path: Path) -> GeneratedTask:
    raw = json.loads(path.read_text(encoding="utf-8"))
    recipe_raw = raw["recipe"]
    recipe = TaskRecipe(
        id=str(recipe_raw["id"]),
        anchor_task_id=str(recipe_raw["anchor_task_id"]),
        primitive_ids=tuple(recipe_raw.get("primitive_ids") or []),
        application=str(recipe_raw.get("application") or "desktop_os"),
        strategy=str(recipe_raw.get("strategy") or "sequential"),
        expected_effects=tuple(recipe_raw.get("expected_effects") or []),
        required_vm_capabilities=tuple(recipe_raw.get("required_vm_capabilities") or []),
        source_task_ids=tuple(recipe_raw.get("source_task_ids") or []),
        environment_id=str(recipe_raw.get("environment_id") or ""),
        environment_digest=str(recipe_raw.get("environment_digest") or ""),
    )
    return GeneratedTask(
        task=TaskSpec.from_dict(raw["task"]),
        recipe=recipe,
        evaluator_plugins=tuple(raw.get("evaluator_plugins") or []),
        assets=tuple(raw.get("assets") or []),
    )


def _grade_code(evaluator: dict[str, Any]) -> str:
    if evaluator.get("type") == "weavebench_grade" and evaluator.get("code"):
        return str(evaluator["code"])
    components = evaluator.get("components") if evaluator.get("type") == "composite" else []
    payload = []
    for component in components or []:
        for fragment in component.get("fragments") or []:
            if fragment.get("type") == "weavebench_grade" and fragment.get("code"):
                payload.append({
                    "primitive_id": component.get("primitive_id"),
                    "code": fragment["code"],
                })
    serialized = repr(payload)
    return f'''def grade(workspace_path=None, **kwargs):
    components = {serialized}
    scores = {{}}
    for index, component in enumerate(components):
        namespace = {{}}
        try:
            exec(component["code"], namespace)
            result = namespace["grade"](workspace_path=workspace_path, **kwargs)
            if isinstance(result, dict):
                score = result.get("overall_score", result.get("score", 0.0))
            else:
                score = result
            scores[component.get("primitive_id") or str(index)] = float(score or 0.0)
        except Exception as exc:
            scores[component.get("primitive_id") or str(index)] = 0.0
            scores[f"error_{{index}}"] = f"{{type(exc).__name__}}: {{exc}}"
    numeric = [value for value in scores.values() if isinstance(value, (int, float))]
    overall = min(numeric) if numeric else 0.0
    return {{"overall_score": overall, "component_scores": scores}}
'''


def _uses_native_osworld_evaluator(task: TaskSpec) -> bool:
    return bool(
        isinstance(task.metadata.get("legacy_osworld"), dict)
        or task.evaluator.get("type") == "osworld_v2_task_class"
    )


def _warmup(task: TaskSpec) -> str:
    commands = []
    for item in task.config:
        if item.get("type") == "execute":
            command = (item.get("parameters") or {}).get("command")
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            if command:
                commands.append(str(command))
    return "\n".join(commands)


def _find_workspace(source: SourceTask) -> Path | None:
    if not source.workspace_path:
        return None
    workspace = Path(source.workspace_path)
    if workspace.is_absolute() and workspace.is_dir():
        return workspace
    source_path = Path(source.source_path)
    for parent in (source_path.parent, *source_path.parents):
        candidate = parent / workspace
        if candidate.is_dir():
            return candidate
    return None


def compile_weavebench_taskset(
    generated_root: str | Path,
    output_root: str | Path,
    *,
    sources: list[SourceTask] = (),
) -> dict[str, Any]:
    source_by_id = {source.task.id: source for source in sources}
    destination = Path(output_root)
    manifest_tasks = []
    for path in sorted(Path(generated_root).rglob("generated_task.json")):
        generated = _load_generated(path)
        task = generated.task
        category = str(task.metadata.get("category") or "GEN")
        workspace_rel = Path("workspace") / category / task.id
        markdown_path = destination / category / f"{task.id}.md"
        source = source_by_id.get(generated.recipe.anchor_task_id)
        if source:
            source_workspace = _find_workspace(source)
            if source_workspace and not (destination / workspace_rel).exists():
                shutil.copytree(
                    source_workspace,
                    destination / workspace_rel,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                )
        (destination / workspace_rel).mkdir(parents=True, exist_ok=True)
        for subdir in ("exec", "gt", "tmp"):
            (destination / workspace_rel / subdir).mkdir(exist_ok=True)
        workspace_absolute = (destination / workspace_rel).resolve()
        max_steps = int(task.metadata.get("estimated_steps") or 300)
        timeout_seconds = int(task.metadata.get("timeout_seconds") or 7200)
        legacy_osworld = isinstance(task.metadata.get("legacy_osworld"), dict)
        native_osworld_evaluator = _uses_native_osworld_evaluator(task)
        automated_checks = ""
        if not native_osworld_evaluator:
            automated_checks = f"```python\n{_grade_code(task.evaluator).rstrip()}\n```"
        markdown = f"""---
id: {task.id}
name: {json.dumps(task.instruction[:120], ensure_ascii=False)}
category: {category}
timeout_seconds: {timeout_seconds}
gui: true
max_steps: {max_steps}
---

# {task.id}

## Prompt

{task.instruction}

## Expected Behavior

Complete all generated capability requirements and leave the environment in the verifiable final state.

## Source

Generated from: {', '.join(generated.recipe.source_task_ids)}

## Grading Criteria

All component postconditions must pass. Expected effects: {', '.join(generated.recipe.expected_effects)}

## Automated Checks

{automated_checks}

## Workspace Path

`{workspace_absolute.as_posix()}/`

## Skills

```
```

## Env

```
```

## Warmup

```bash
{"" if legacy_osworld else _warmup(task)}
```
"""
        atomic_write_text(markdown_path, markdown)
        manifest_tasks.append({
            "id": task.id,
            "version": task.resolved_version,
            "category": category,
            "markdown": str(markdown_path.relative_to(destination)),
            "workspace": str(workspace_rel),
            "execution_adapter": (
                "legacy_osworld" if legacy_osworld else None
            ),
            "proxy": bool(
                (task.metadata.get("legacy_osworld") or {}).get("proxy", False)
            ),
            "environment_id": (
                generated.recipe.environment_id
                or str(task.metadata.get("environment_id") or "")
            ),
            "environment_digest": (
                generated.recipe.environment_digest
                or str(task.metadata.get("environment_digest") or "")
            ),
            "required_capabilities": list(task.required_capabilities),
            "target_environments": [
                environment.to_dict() for environment in task.target_environments
            ],
            "runtime_config": list(task.config),
            "evaluator": dict(task.evaluator),
        })
    manifest = {
        "schema_version": "propagation.weavebench_taskset.v1",
        "taskset_version": content_digest(manifest_tasks),
        "tasks": manifest_tasks,
    }
    atomic_write_json(destination / "taskset_manifest.json", manifest)
    return manifest
