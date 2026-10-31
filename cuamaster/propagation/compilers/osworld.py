"""Compile canonical generated tasks into OSWorld JSON or V2 task classes."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..platforms import platform_family
from ..quality import is_direct_task_class_evaluator
from ..utils import atomic_write_json


def _task_class_reference(task: dict[str, Any]) -> dict[str, Any] | None:
    for item in task.get("config") or []:
        if isinstance(item, dict) and item.get("type") == "osworld_v2_task_class":
            return item
    return None


def _class_name(task_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", task_id).strip("_")
    return f"GeneratedTask_{suffix or 'Task'}"


def _task_class_source(task: dict[str, Any], reference: dict[str, Any]) -> str:
    class_name = _class_name(str(task["id"]))
    related_apps = list((task.get("metadata") or {}).get("related_apps") or [])
    platform = platform_family((task.get("metadata") or {}).get("platform"))
    source_path = Path(str(reference["path"])).expanduser()
    repository_root = Path(str(reference.get("repository_root") or "")).expanduser()
    relative_path = ""
    if str(repository_root):
        try:
            relative_path = str(source_path.resolve().relative_to(repository_root.resolve()))
        except (OSError, ValueError):
            relative_path = ""
    return f'''from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

SOURCE_TASK_RELATIVE_PATH = {relative_path!r}
SOURCE_TASK_FALLBACK_PATH = {str(source_path)!r}
SOURCE_MODULE_NAME = "generated_task_source_{task["id"].replace("-", "_")}"


def _resolve_source_task_path() -> str:
    roots = []
    configured_root = os.environ.get("OSWORLD_REPOSITORY_ROOT")
    if configured_root:
        roots.append(Path(configured_root))
    roots.append(Path(__file__).resolve().parents[2])
    if SOURCE_TASK_RELATIVE_PATH:
        for root in roots:
            candidate = root / SOURCE_TASK_RELATIVE_PATH
            if candidate.is_file():
                return str(candidate)
    return SOURCE_TASK_FALLBACK_PATH


SOURCE_TASK_PATH = _resolve_source_task_path()
_spec = importlib.util.spec_from_file_location(SOURCE_MODULE_NAME, SOURCE_TASK_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load source OSWorld task: {{SOURCE_TASK_PATH}}")
_source_module = importlib.util.module_from_spec(_spec)
sys.modules[SOURCE_MODULE_NAME] = _source_module
_spec.loader.exec_module(_source_module)
_SourceTask = _source_module.TASK_CLASS


class {class_name}(_SourceTask):
    id = {str(task["id"])!r}
    snapshot = {str(task.get("snapshot") or "clean")!r}
    instruction = {str(task["instruction"])!r}
    related_apps = {related_apps!r}
    platform = {platform!r}


TASK_CLASS = {class_name}
'''


def compile_osworld_taskset(generated_root: str | Path, output_root: str | Path) -> dict:
    destination = Path(output_root)
    tasks = []
    for path in sorted(Path(generated_root).rglob("generated_task.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        task = raw["task"]
        output = {
            "id": task["id"],
            "snapshot": task.get("snapshot", "clean"),
            "instruction": task["instruction"],
            "config": task.get("config") or [],
            "evaluator": task.get("evaluator") or {},
            "metadata": task.get("metadata") or {},
        }
        reference = _task_class_reference(task)
        if reference and is_direct_task_class_evaluator(output["evaluator"], reference):
            target = destination / "task_class" / f"task_{output['id']}.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_task_class_source(task, reference), encoding="utf-8")
            output_format = "osworld_v2_task_class"
        else:
            if str(output["metadata"].get("category") or "").upper() == "OSWORLD_V2":
                if not isinstance(output["evaluator"], dict) or not output["evaluator"].get("func"):
                    raise ValueError(
                        f"task {output['id']} has an unsupported OSWorld evaluator schema"
                    )
            category = str(output["metadata"].get("category") or "GEN")
            target = destination / category / f"{output['id']}.json"
            atomic_write_json(target, output)
            output_format = "osworld_json"
        canonical = destination / "canonical" / f"{output['id']}.json"
        atomic_write_json(canonical, output)
        tasks.append({
            "id": output["id"],
            "path": str(target.relative_to(destination)),
            "format": output_format,
        })
    manifest = {"schema_version": "propagation.osworld_taskset.v1", "tasks": tasks}
    atomic_write_json(destination / "taskset_manifest.json", manifest)
    atomic_write_json(destination / "test_v2.json", {"tasks": [item["id"] for item in tasks]})
    return manifest
