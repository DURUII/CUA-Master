"""Instruction/evaluator alignment judging for generated tasks.

The evaluator is the only reward signal a collected trajectory receives, so an
instruction that demands work the evaluator never scores (or omits work the
evaluator does score) injects reward noise into training data. This module asks
the generator backend to compare the instruction against the trusted evaluator
source code and reports both directions of drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .quality import task_class_reference
from .schemas import GeneratedTask

# Evaluator sources are small task classes; cap the excerpt defensively so a
# pathological file cannot blow up the judging prompt.
MAX_EVALUATOR_SOURCE_CHARS = 12_000

_SOURCE_CACHE: dict[str, str | None] = {}


def evaluator_source_excerpt(task: Any) -> str | None:
    """Return the trusted task-class evaluator source, or None if unavailable.

    Accepts a TaskSpec or a GeneratedTask. Excerpts are cached per evaluator
    path so generation-time callers do not re-read the file per variant.
    """
    task = getattr(task, "task", task)
    reference = task_class_reference(task)
    if reference is None:
        return None
    path_text = str(reference.get("path") or "")
    if path_text in _SOURCE_CACHE:
        return _SOURCE_CACHE[path_text]
    path = Path(path_text)
    source: str | None
    if not path.is_file():
        source = None
    else:
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            source = None
    if source is not None and len(source) > MAX_EVALUATOR_SOURCE_CHARS:
        source = source[:MAX_EVALUATOR_SOURCE_CHARS] + "\n# ... truncated ..."
    _SOURCE_CACHE[path_text] = source
    return source


def instruction_alignment_report(
    generated: GeneratedTask,
    backend: Any,
    *,
    judgements: int = 1,
) -> tuple[list[str], list[str]]:
    """Judge instruction/evaluator alignment via the generator backend.

    Runs up to ``judgements`` independent LLM judgements; a candidate is
    aligned only when every judgement agrees, so a single lenient sample
    cannot pass a misaligned task. Returns (errors, warnings). Alignment
    degrades to a warning instead of an error whenever the judgement cannot
    be made (no task-class evaluator source, no backend, or a backend
    failure) so that the check never rejects a task it could not actually
    inspect.
    """
    source = evaluator_source_excerpt(generated)
    if source is None:
        return [], ["instruction alignment was not judged: no task-class evaluator source"]
    if backend is None:
        return [], ["instruction alignment was not judged: no generator backend"]
    errors: list[str] = []
    warnings: list[str] = []
    judged = 0
    for judgement_index in range(max(1, judgements)):
        try:
            verdict = backend.align_check(
                generated,
                source,
                judgement_index=judgement_index,
            )
        except TypeError:
            verdict = backend.align_check(generated, source)
        except Exception as exc:  # noqa: BLE001 - backend/network specific
            warnings.append(
                f"instruction alignment judgement failed: {type(exc).__name__}: {exc}"
            )
            continue
        if not isinstance(verdict, dict):
            warnings.append(
                "instruction alignment was not judged: backend returned no verdict"
            )
            continue
        judged += 1
        for requirement in verdict.get("unverified_requirements") or []:
            message = (
                "instruction requires work the evaluator does not score "
                f"(unverified requirement): {requirement}"
            )
            if message not in errors:
                errors.append(message)
        for requirement in verdict.get("missing_requirements") or []:
            message = (
                "evaluator scores a condition the instruction never states "
                f"(missing requirement): {requirement}"
            )
            if message not in errors:
                errors.append(message)
        if not errors and not verdict.get("aligned"):
            warnings.append("instruction alignment verdict was inconclusive")
    if errors:
        return errors, []
    if judged:
        return [], warnings if judged < max(1, judgements) else [
            warning for warning in warnings if "inconclusive" in warning
        ]
    return [], warnings or ["instruction alignment was not judged"]
