"""Static quality contracts for generated tasks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .platforms import environment_platform, platform_family, platform_text_errors
from .schemas import EnvironmentContext, GeneratedTask


_NON_GUI_STATE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\b(?:edit|modify|write|patch|replace|append|delete|remove)\b.{0,100}"
        r"(?:calendar\.ics|\.sqlite\b|\.db\b|appdata\b|application support\b|"
        r"\.local/share\b|profile directory\b|internal database\b)",
        "directly mutates hidden application state",
    ),
    (
        r"\b(?:kill|pkill|killall|taskkill|stop|restart)\b.{0,80}"
        r"(?:process|daemon|service|calendar factory|application)",
        "controls background processes instead of using the application UI",
    ),
    (
        r"\b(?:developer tools|devtools|execute javascript|run javascript|"
        r"dom manipulation|browser console)\b",
        "uses browser internals instead of the visible application UI",
    ),
)


def gui_realism_errors(text: str) -> list[str]:
    """Reject shortcuts that bypass the user-visible application workflow."""
    errors: list[str] = []
    sentences = re.split(r"(?<=[.!?;])\s+|\n+", text)
    for pattern, message in _NON_GUI_STATE_PATTERNS:
        for sentence in sentences:
            if not re.search(pattern, sentence, flags=re.IGNORECASE | re.DOTALL):
                continue
            if re.search(
                r"\b(?:do not|don't|must not|never|forbid(?:den)?|off-limits|"
                r"not allowed)\b|\bno\s+(?:direct|browser|developer|devtools|cdp|"
                r"process|service|database|profile)",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue
            errors.append(f"trajectory is not GUI-realistic: {message}")
            break
    return sorted(set(errors))


def solve_plan_quality_errors(plan: list[str] | tuple[str, ...]) -> list[str]:
    return gui_realism_errors("\n".join(str(item) for item in plan))


def task_class_reference(task: Any) -> dict[str, Any] | None:
    for item in task.config:
        if isinstance(item, dict) and item.get("type") in {
            "ale_task_module",
            "osworld_v2_task_class",
        }:
            return item
    return None


def is_direct_task_class_evaluator(
    evaluator: Any,
    reference: dict[str, Any],
) -> bool:
    if not isinstance(evaluator, dict):
        return False
    path = str(evaluator.get("path") or evaluator.get("repository_path") or "")
    class_name = str(evaluator.get("class_name") or "")
    matches = (
        bool(path) and path == str(reference.get("path") or "")
    ) or (
        bool(class_name) and class_name == str(reference.get("class_name") or "")
    )
    return matches and (
        evaluator.get("method") == "evaluate"
        or evaluator.get("type") in {"task_evaluator", "osworld_v2_task_class"}
    )


def instruction_quality_errors(
    instruction: str,
    *,
    max_chars: int,
    max_words: int,
) -> list[str]:
    errors: list[str] = []
    words = re.findall(r"\b\w+\b", instruction)
    if len(instruction) > max_chars:
        errors.append(
            f"instruction is too long: {len(instruction)} characters exceeds {max_chars}"
        )
    if len(words) > max_words:
        errors.append(f"instruction is too long: {len(words)} words exceeds {max_words}")
    if re.search(r"\b(?:pip3?|apt-get|brew)\s+install\b", instruction, re.IGNORECASE):
        errors.append("instruction must not install dependencies as part of task execution")
    return errors


def generated_task_quality(
    generated: GeneratedTask,
    environment: EnvironmentContext,
    *,
    max_instruction_chars: int,
    max_instruction_words: int,
    require_platform_consistency: bool,
    require_supported_evaluator: bool,
    require_gui_realism: bool = True,
) -> tuple[dict[str, bool], list[str], list[str]]:
    task = generated.task
    errors: list[str] = []
    warnings: list[str] = []
    instruction_errors = instruction_quality_errors(
        task.instruction,
        max_chars=max_instruction_chars,
        max_words=max_instruction_words,
    )
    errors.extend(instruction_errors)
    realism_errors = gui_realism_errors(task.instruction) if require_gui_realism else []
    errors.extend(realism_errors)

    platform_errors: list[str] = []
    target_platform = environment_platform(environment)
    if require_platform_consistency:
        platform_errors.extend(platform_text_errors(task.instruction, target_platform))
        metadata_platform = platform_family(task.metadata.get("platform"))
        if metadata_platform not in {"unknown", target_platform}:
            platform_errors.append(
                f"task metadata platform {metadata_platform} does not match {target_platform}"
            )
        if not any(
            platform_family(target.os_type) == target_platform
            for target in task.target_environments
        ):
            platform_errors.append(
                f"task target_environments does not include {target_platform}"
            )
    errors.extend(platform_errors)

    evaluator_errors: list[str] = []
    if require_supported_evaluator:
        reference = task_class_reference(task)
        if reference is not None:
            source_path = Path(str(reference.get("path") or ""))
            if not source_path.is_file():
                evaluator_errors.append(
                    f"task-class source path does not exist: {source_path}"
                )
            if not is_direct_task_class_evaluator(task.evaluator, reference):
                evaluator_errors.append(
                    "task-class evaluator must directly reuse its trusted source evaluator"
                )
            if generated.recipe.strategy != "source_complete":
                evaluator_errors.append(
                    "task-class evaluator requires a source_complete capability recipe"
                )
        elif str(task.metadata.get("category") or "").upper() == "OSWORLD_V2":
            if not isinstance(task.evaluator, dict) or not task.evaluator.get("func"):
                evaluator_errors.append(
                    "OSWorld JSON evaluator must use the supported func/result/expected schema"
                )
    errors.extend(evaluator_errors)

    procedural_markers = len(re.findall(
        r"\b(?:First|Next|Then|After that|Finally|Step \d+)\b",
        task.instruction,
        flags=re.IGNORECASE,
    ))
    if procedural_markers > 12:
        warnings.append(
            f"instruction is highly procedural ({procedural_markers} sequencing markers)"
        )
    checks = {
        "instruction_quality": not instruction_errors,
        "gui_realism": not realism_errors,
        "platform_consistency": not platform_errors,
        "evaluator_contract": not evaluator_errors,
    }
    return checks, errors, warnings
