"""Load WeaveBench Markdown, canonical TaskSpec JSON, and episode evidence."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cuamaster.contracts import TargetEnvironment, TaskSpec

from ..schemas import SourceTask


_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_APP_ALIASES = {
    "calc": "libreoffice_calc",
    "spreadsheet": "libreoffice_calc",
    "writer": "libreoffice_writer",
    "document": "libreoffice_writer",
    "impress": "libreoffice_impress",
    "presentation": "libreoffice_impress",
    "gimp": "gimp",
    "image": "gimp",
    "chrome": "google_chrome",
    "browser": "google_chrome",
    "firefox": "firefox",
    "vscode": "visual_studio_code",
    "code": "visual_studio_code",
    "vlc": "vlc",
    "video": "vlc",
    "thunderbird": "thunderbird",
    "email": "thunderbird",
    "terminal": "terminal",
    "calendar": "calendar",
    "freecad": "freecad",
    "kicad": "kicad",
    "mailhub": "google_chrome",
    "pdf_viewer": "document_viewer",
    "image_viewer": "image_viewer",
    "file-manager": "files",
    "3d-slicer": "3d_slicer",
}


@dataclass(frozen=True)
class EpisodeEvidence:
    episode_id: str
    task_id: str
    success: bool
    training_tier: str
    steps: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "success": self.success,
            "training_tier": self.training_tier,
            "manifest": self.manifest,
            "steps": list(self.steps),
        }


def _sections(text: str) -> dict[str, str]:
    matches = list(_SECTION.finditer(text))
    values: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        values[match.group(1).strip()] = text[start:end].strip()
    return values


def _strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return value


def infer_application(task_id: str, instruction: str, snapshot: str = "") -> str:
    haystack = f"{task_id} {instruction} {snapshot}".lower()
    for token, application in _APP_ALIASES.items():
        if token in haystack:
            return application
    return "desktop_os"


def _ast_value(node: ast.AST, symbols: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise ValueError(f"unknown AST name: {node.id}")
        return symbols[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _ast_value(node.left, symbols) + _ast_value(node.right, symbols)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                try:
                    parts.append(str(_ast_value(value.value, symbols)))
                except (ValueError, TypeError, SyntaxError):
                    parts.append("{" + ast.unparse(value.value) + "}")
            else:
                raise ValueError("unsupported f-string element")
        return "".join(parts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "strip"
        and not node.args
        and not node.keywords
    ):
        return str(_ast_value(node.func.value, symbols)).strip()
    if isinstance(node, ast.List):
        return [_ast_value(item, symbols) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ast_value(item, symbols) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _ast_value(key, symbols): _ast_value(value, symbols)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    return ast.literal_eval(node)


def _assignment(node: ast.AST) -> tuple[str, ast.AST] | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
        return node.target.id, node.value
    return None


def _normalize_related_app(value: str) -> str:
    token = value.strip().lower().replace(" ", "_")
    return _APP_ALIASES.get(token, token.replace("-", "_"))


def load_osworld_v2_task_class(path: Path) -> SourceTask:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: dict[str, Any] = {}
    for node in tree.body:
        assignment = _assignment(node)
        if not assignment:
            continue
        name, value_node = assignment
        try:
            symbols[name] = _ast_value(value_node, symbols)
        except (ValueError, TypeError, SyntaxError):
            continue

    task_fields: dict[str, Any] | None = None
    task_class_name = ""
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields: dict[str, Any] = {}
        class_symbols = dict(symbols)
        for item in node.body:
            assignment = _assignment(item)
            if not assignment:
                continue
            name, value_node = assignment
            try:
                fields[name] = _ast_value(value_node, class_symbols)
                class_symbols[name] = fields[name]
            except (ValueError, TypeError, SyntaxError):
                continue
        if fields.get("instruction") and (fields.get("id") or path.stem.startswith("task_")):
            task_fields = fields
            task_class_name = node.name
            break
    if task_fields is None:
        raise ValueError(f"OSWorld V2 task class not found: {path}")

    task_id = str(task_fields.get("id") or symbols.get("TASK_ID") or path.stem.removeprefix("task_"))
    instruction = str(task_fields.get("instruction") or "").strip()
    related_apps = tuple(
        _normalize_related_app(str(item))
        for item in task_fields.get("related_apps") or []
        if str(item).strip()
    )
    application = related_apps[0] if related_apps else infer_application(
        task_id, instruction, str(task_fields.get("snapshot") or "")
    )
    repository_root = path.parents[2]
    task_reference = {
        "type": "osworld_v2_task_class",
        "task_id": task_id,
        "class_name": task_class_name,
        "path": str(path.resolve()),
        "repository_root": str(repository_root.resolve()),
    }
    capabilities = {"computer", "filesystem", *related_apps}
    task = TaskSpec(
        id=task_id,
        instruction=instruction,
        snapshot=str(task_fields.get("snapshot") or "clean"),
        config=(task_reference,),
        evaluator={**task_reference, "method": "evaluate"},
        required_capabilities=tuple(sorted(capabilities)),
        metadata={
            "category": "OSWORLD_V2",
            "application": application,
            "related_apps": list(related_apps),
            "source_format": "osworld_v2_task_class",
            "platform": str(task_fields.get("platform") or "linux"),
        },
    )
    return SourceTask(
        task=task,
        source_path=str(path.resolve()),
        format="osworld_v2_task_class",
        application=application,
        category="OSWORLD_V2",
        raw_sections={"task_class": task_class_name},
    )


def load_weavebench_task(path: Path) -> SourceTask:
    text = path.read_text(encoding="utf-8")
    sections = _sections(text)
    instruction = _strip_fence(sections.get("Prompt", ""))
    if not instruction:
        raise ValueError(f"WeaveBench task has no Prompt section: {path}")
    warmup = _strip_fence(sections.get("Warmup", ""))
    env = _strip_fence(sections.get("Env", ""))
    checks = _strip_fence(sections.get("Automated Checks", ""))
    workspace = sections.get("Workspace Path", "").strip().strip("`") or None
    category = path.parent.name
    application = infer_application(path.stem, instruction)
    config: list[dict[str, Any]] = []
    if warmup:
        config.append({"type": "execute", "parameters": {"command": warmup, "shell": True}})
    evaluator = {
        "type": "weavebench_grade",
        "code": checks,
        "source_trust": "source",
    } if checks else {}
    capabilities = {"computer", "filesystem"}
    if warmup or "terminal" in instruction.lower() or "command" in instruction.lower():
        capabilities.add("shell")
    capabilities.add(application)
    task = TaskSpec(
        id=path.stem,
        instruction=instruction,
        snapshot="clean",
        config=tuple(config),
        evaluator=evaluator,
        required_capabilities=tuple(sorted(capabilities)),
        metadata={
            "category": category,
            "application": application,
            "source_format": "weavebench_markdown",
            "workspace_path": workspace,
            "expected_behavior": sections.get("Expected Behavior"),
            "grading_criteria": sections.get("Grading Criteria"),
            "env": env,
        },
    )
    return SourceTask(
        task=task,
        source_path=str(path.resolve()),
        format="weavebench_markdown",
        application=application,
        category=category,
        workspace_path=workspace,
        raw_sections=sections,
    )


def load_ale_task_card(path: Path) -> SourceTask:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"ALE task card must contain an object: {path}")
    task_id = str(raw.get("taskId") or "").strip()
    instruction = str(raw.get("taskPrompt") or "").strip()
    if not task_id or not instruction:
        raise ValueError(f"ALE task card is missing taskId or taskPrompt: {path}")
    main_path = path.with_name("main.py")
    if not main_path.is_file():
        raise ValueError(f"ALE task module is missing: {main_path}")
    tasks_root = next((parent for parent in path.parents if parent.name == "tasks"), None)
    repository_root = tasks_root.parent if tasks_root is not None else path.parent
    vm = raw.get("vm") if isinstance(raw.get("vm"), dict) else {}
    source_reference = {
        "type": "ale_task_module",
        "task_id": task_id,
        "path": str(main_path.resolve()),
        "task_card_path": str(path.resolve()),
        "repository_root": str(repository_root.resolve()),
    }
    task = TaskSpec(
        id=task_id,
        instruction=instruction,
        snapshot=str(vm.get("snapshot") or "clean"),
        config=(source_reference,),
        evaluator={
            **source_reference,
            "type": "task_evaluator",
            "source_type": "ale_task_module",
            "method": "evaluate",
        },
        required_capabilities=("computer", "desktop_os", "filesystem", "shell"),
        target_environments=(TargetEnvironment(
            os_type="Ubuntu",
            os_versions=("22.04",),
            architectures=("x86_64",),
        ),),
        metadata={
            "category": str(raw.get("category") or task_id.partition("/")[0] or "ALE"),
            "application": "desktop_os",
            "source_format": "ale_task_card",
            "platform": "linux",
            "title": raw.get("title"),
            "summary": raw.get("summary"),
            "software": list(raw.get("software") or []),
            "evaluation": raw.get("evaluation"),
            "vm": vm,
            "input_files": list(raw.get("inputFiles") or []),
            "reference_files": list(raw.get("referenceFiles") or []),
        },
    )
    return SourceTask(
        task=task,
        source_path=str(path.resolve()),
        format="ale_task_card",
        application="desktop_os",
        category=str(task.metadata["category"]),
        workspace_path=str(path.parent.resolve()),
        raw_sections={
            "title": str(raw.get("title") or ""),
            "summary": str(raw.get("summary") or ""),
            "evaluation": str(raw.get("evaluation") or ""),
        },
    )


def load_json_task(path: Path) -> SourceTask:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"task JSON must contain an object: {path}")
    if isinstance(raw.get("task"), dict):
        raw = raw["task"]
    task = TaskSpec.from_dict(raw)
    application = str(task.metadata.get("application") or infer_application(
        task.id, task.instruction, task.snapshot
    ))
    category = str(task.metadata.get("category") or path.parent.name)
    return SourceTask(
        task=task,
        source_path=str(path.resolve()),
        format="task_spec_json",
        application=application,
        category=category,
        workspace_path=(
            str(task.metadata["workspace_path"]) if task.metadata.get("workspace_path") else None
        ),
    )


def load_source_tasks(roots: Iterable[str | Path]) -> list[SourceTask]:
    tasks: list[SourceTask] = []
    seen: set[tuple[str, str]] = set()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        candidates = [root] if root.is_file() else sorted({
            *root.rglob("*.md"),
            *root.rglob("*.json"),
            *(path for path in root.rglob("task_*.py") if path.parent.name == "task_class"),
        })
        for path in candidates:
            if any(part in {"canonical", "validation", ".trace-index"} for part in path.parts):
                continue
            try:
                if path.suffix.lower() == ".md":
                    source = load_weavebench_task(path)
                elif path.suffix.lower() == ".py":
                    source = load_osworld_v2_task_class(path)
                elif path.name == "task_card.json":
                    source = load_ale_task_card(path)
                else:
                    source = load_json_task(path)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            key = (source.task.id, source.task.resolved_version)
            if key not in seen:
                seen.add(key)
                tasks.append(source)
    return tasks


def load_episodes(roots: Iterable[str | Path]) -> dict[str, list[EpisodeEvidence]]:
    by_task: dict[str, list[EpisodeEvidence]] = {}
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        legacy_metadata: dict[tuple[str, str], dict[str, Any]] = {}
        for collection_manifest in sorted(root.rglob("collection_manifest.jsonl")):
            try:
                records = [
                    json.loads(line)
                    for line in collection_manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except json.JSONDecodeError:
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                model = str(record.get("model") or "")
                episode = str(record.get("episode") or "")
                task_path = str(record.get("task_path") or "")
                if model and episode and task_path:
                    legacy_metadata[(model, episode)] = record
        for manifest_path in sorted(root.rglob("episode_manifest.json")):
            if ".trace-index" in manifest_path.parts:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                quality = json.loads((manifest_path.parent / "quality.json").read_text(encoding="utf-8"))
                steps = tuple(
                    json.loads(line)
                    for line in (manifest_path.parent / "canonical" / "steps.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                )
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            task_id = str((manifest.get("task") or {}).get("id") or "")
            if not task_id:
                continue
            evidence = EpisodeEvidence(
                episode_id=str(manifest.get("episode_id") or manifest_path.parent.name),
                task_id=task_id,
                success=bool((manifest.get("evaluation") or {}).get("success")),
                training_tier=str(quality.get("training_tier") or "unknown"),
                steps=steps,
                manifest=manifest,
            )
            by_task.setdefault(task_id, []).append(evidence)
        for trajectory_path in sorted(root.rglob("traj.jsonl")):
            result_path = trajectory_path.with_name("result.txt")
            if not result_path.is_file():
                continue
            try:
                score = float(result_path.read_text(encoding="utf-8").strip())
                rows = [
                    json.loads(line)
                    for line in trajectory_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (ValueError, json.JSONDecodeError):
                continue
            model = trajectory_path.parents[2].name if len(trajectory_path.parents) > 2 else "unknown"
            source_metadata = legacy_metadata.get((model, trajectory_path.parent.name), {})
            task_id = str(source_metadata.get("task_path") or trajectory_path.parent.name)
            steps = []
            for index, row in enumerate(rows, 1):
                screenshot = row.get("screenshot_file")
                if screenshot and not Path(str(screenshot)).is_absolute():
                    screenshot = str((trajectory_path.parent / str(screenshot)).resolve())
                steps.append({
                    "index": int(row.get("step_num") or index),
                    "response": str(row.get("response") or ""),
                    "tool_call": dict(row.get("tool_call") or {}),
                    "screenshot": screenshot,
                })
            success = score >= 1.0 - 1e-9
            manifest = {
                "schema_version": "propagation.osworld_v2_episode.v1",
                "task": {"id": task_id},
                "episode_id": str(
                    source_metadata.get("episode_id")
                    or f"legacy-{model}-{trajectory_path.parent.name}"
                ),
                "evaluation": {"score": score, "success": success},
                "model": model,
                "step_count": len(steps),
                "source_path": str(trajectory_path.resolve()),
                "source_metadata": source_metadata,
            }
            evidence = EpisodeEvidence(
                episode_id=manifest["episode_id"],
                task_id=task_id,
                success=success,
                training_tier=(
                    "verified_success" if success else "partial_score" if score > 0 else "failed"
                ),
                steps=tuple(steps),
                manifest=manifest,
            )
            by_task.setdefault(task_id, []).append(evidence)
    return by_task
