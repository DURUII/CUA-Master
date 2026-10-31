"""Deterministic semantic similarity, scoring, and coverage helpers."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

from .schemas import (
    AbilityPrimitive,
    CandidateQualityScore,
    EnvironmentContext,
    GeneratedTask,
    ValidationReport,
)
from .platforms import environment_platform


_TOKEN_ALIASES = {
    "create": "make",
    "generate": "make",
    "produce": "make",
    "build": "make",
    "add": "make",
    "new": "make",
    "delete": "remove",
    "erase": "remove",
    "clear": "remove",
    "discard": "remove",
    "trash": "remove",
    "exporting": "export",
    "saving": "save",
    "saved": "save",
    "store": "save",
    "opening": "open",
    "opened": "open",
    "launch": "open",
    "start": "open",
    "rename": "rename",
    "renaming": "rename",
    "move": "move",
    "relocate": "move",
    "copy": "copy",
    "duplicate": "copy",
    "modify": "change",
    "edit": "change",
    "update": "change",
    "adjust": "change",
    "set": "change",
    "configure": "change",
    "email": "mail",
    "e-mail": "mail",
    "folder": "directory",
    "chrome": "browser",
    "firefox": "browser",
    "writer": "libreoffice_writer",
    "calc": "libreoffice_calc",
    "impress": "libreoffice_impress",
    "spreadsheet": "libreoffice_calc",
    "presentation": "libreoffice_impress",
    "document": "libreoffice_writer",
}

_COLOR_WORDS = (
    "red|green|blue|yellow|orange|purple|pink|black|white|gray|grey|brown|cyan|magenta"
)


def normalize_semantic_text(value: str) -> str:
    text = value.casefold().replace("\\", "/")
    text = re.sub(r"(?:[a-z]:)?/(?:[^\s/]+/)+[^\s]+", " <path> ", text)
    text = re.sub(
        r"\b[\w.-]+\.(?:txt|pdf|docx?|xlsx?|pptx?|csv|json|png|jpe?g|gif|svg|mp[34]|"
        r"avi|mkv|zip|tar|gz|odt|ods|odp|html?|xml|md|py|sh)\b",
        " <file> ",
        text,
    )
    text = re.sub(
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[:.]\d{2}(?:\s?[ap]m)?\b",
        " <datetime> ",
        text,
    )
    text = re.sub(rf"\b(?:{_COLOR_WORDS})\b", " <color> ", text)
    text = re.sub(
        r"\b\d+(?:\.\d+)?\s?(?:px|pt|cm|mm|in|%|kb|mb|gb)\b",
        " <measure> ",
        text,
    )
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " <number> ", text)
    tokens = re.findall(r"[\w<>:+.-]+", text)
    return " ".join(_TOKEN_ALIASES.get(token, token) for token in tokens)


def _token_counts(value: str) -> Counter[str]:
    return Counter(normalize_semantic_text(value).split())


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(count * right.get(token, 0) for token, count in left.items())
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _character_ngrams(value: str, size: int = 3) -> set[str]:
    normalized = normalize_semantic_text(value).replace(" ", "_")
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    normalized_left = normalize_semantic_text(left)
    normalized_right = normalize_semantic_text(right)
    if normalized_left == normalized_right:
        return 1.0
    token_score = _cosine(_token_counts(normalized_left), _token_counts(normalized_right))
    left_grams = _character_ngrams(normalized_left)
    right_grams = _character_ngrams(normalized_right)
    union = left_grams | right_grams
    ngram_score = len(left_grams & right_grams) / len(union) if union else 0.0
    return max(token_score, ngram_score)


def primitive_text(capability: AbilityPrimitive) -> str:
    return " ".join((
        capability.application,
        capability.intent,
        *capability.preconditions,
        *capability.effects,
    ))


def primitive_similarity(left: AbilityPrimitive, right: AbilityPrimitive) -> float:
    if left.semantic_signature == right.semantic_signature:
        return 1.0
    if normalize_semantic_text(left.application) != normalize_semantic_text(right.application):
        return 0.0
    return text_similarity(primitive_text(left), primitive_text(right))


def recipe_similarity(
    left: Iterable[AbilityPrimitive],
    right: Iterable[AbilityPrimitive],
) -> float:
    left_signatures = {item.semantic_signature for item in left}
    right_signatures = {item.semantic_signature for item in right}
    union = left_signatures | right_signatures
    return len(left_signatures & right_signatures) / len(union) if union else 0.0


def generated_task_similarity(left: GeneratedTask, right: GeneratedTask) -> float:
    instruction_score = text_similarity(left.task.instruction, right.task.instruction)
    left_primitives = set(left.recipe.primitive_ids)
    right_primitives = set(right.recipe.primitive_ids)
    union = left_primitives | right_primitives
    primitive_score = (
        len(left_primitives & right_primitives) / len(union) if union else 0.0
    )
    effect_score = text_similarity(
        " ".join(left.recipe.expected_effects),
        " ".join(right.recipe.expected_effects),
    )
    return 0.6 * instruction_score + 0.25 * primitive_score + 0.15 * effect_score


def difficulty_bucket(generated: GeneratedTask) -> tuple[str, float]:
    primitive_count = len(generated.recipe.primitive_ids)
    strategy_bonus = {
        "source_complete": 0.05,
        "sequential": 0.1,
        "parameterized": 0.15,
        "state_dependent": 0.3,
        "cross_app_handoff": 0.4,
    }.get(generated.recipe.strategy, 0.1)
    application_count = int(generated.recipe.metadata.get("application_count") or 1)
    dependency_depth = int(generated.recipe.metadata.get("dependency_depth") or 0)
    # Cap the capability term so that full-capability source_complete recipes
    # (typically 8-12 capabilities) spread across medium/hard instead of
    # saturating the hard bucket, while small recipes keep the original slope.
    score = min(
        1.0,
        min(0.5, 0.12 * primitive_count)
        + strategy_bonus
        + 0.1 * max(0, application_count - 1)
        + 0.08 * dependency_depth,
    )
    if score < 0.4:
        return "easy", score
    if score < 0.7:
        return "medium", score
    return "hard", score


def _sandbox_score(sandbox: dict[str, Any]) -> tuple[float, float]:
    if not sandbox.get("ran"):
        return 0.8, 0.75
    solvability = 1.0 if sandbox.get("agent_solve") else 0.0
    robustness_checks = (
        "initial_negative",
        "partial_negative",
        "reset_determinism",
        "repeated_score",
    )
    robustness = sum(bool(sandbox.get(name)) for name in robustness_checks) / len(
        robustness_checks
    )
    return solvability, robustness


def _solve_check_solvability(value: dict[str, Any]) -> float | None:
    verdict = value.get("solve_check")
    if not isinstance(verdict, dict) or "solvable" not in verdict:
        return None
    if not verdict.get("solvable") or verdict.get("blockers"):
        return 0.0
    return 1.0


def score_candidate(
    generated: GeneratedTask,
    environment: EnvironmentContext,
    report: ValidationReport | dict[str, Any],
    *,
    minimum_total_score: float,
    minimum_grounding: float,
    minimum_alignment: float,
    minimum_solvability: float,
    weights: dict[str, float] | None = None,
) -> CandidateQualityScore:
    value = report.to_dict() if isinstance(report, ValidationReport) else report
    checks = dict(value.get("checks") or {})
    sandbox = dict(value.get("sandbox") or {})
    hard_gate_passed = bool(value.get("valid"))
    required = set(generated.task.required_capabilities)
    grounding = 1.0
    if not generated.recipe.primitive_ids or not generated.recipe.source_task_ids:
        grounding -= 0.4
    if not environment.supports(sorted(required)):
        grounding -= 0.4
    if generated.recipe.environment_id not in {"", environment.image_id}:
        grounding -= 0.2
    if environment.policy.get("inventory_verified") is False:
        grounding -= 0.15
    grounding = max(0.0, grounding)

    alignment_checks = (
        "evaluator",
        "evaluator_contract",
        "instruction_quality",
        "platform_consistency",
    )
    alignment = sum(bool(checks.get(name, True)) for name in alignment_checks) / len(
        alignment_checks
    )
    words = re.findall(r"\b\w+\b", generated.task.instruction)
    clarity = 1.0
    if len(words) > 350:
        clarity -= min(0.5, (len(words) - 350) / 700)
    procedural_markers = len(re.findall(
        r"\b(?:first|next|then|after that|finally|step \d+)\b",
        generated.task.instruction,
        flags=re.IGNORECASE,
    ))
    if procedural_markers > 12:
        clarity -= min(0.3, (procedural_markers - 12) * 0.03)
    clarity = max(0.0, clarity)

    solvability, robustness = _sandbox_score(sandbox)
    if not sandbox.get("ran"):
        judged_solvability = _solve_check_solvability(value)
        if judged_solvability is not None:
            solvability = judged_solvability
    bucket, difficulty = difficulty_bucket(generated)
    default_novelty = 0.5 if generated.recipe.strategy == "source_complete" else 1.0
    novelty = float(
        generated.recipe.metadata.get("selection_novelty", default_novelty)
    )
    trajectory_value = min(
        1.0,
        0.55 * difficulty
        + 0.25 * min(1.0, len(generated.recipe.expected_effects) / 3)
        + 0.2 * min(1.0, len(generated.recipe.primitive_ids) / 4),
    )
    dimensions = {
        "grounding": grounding,
        "alignment": alignment,
        "clarity": clarity,
        "novelty": novelty,
        "difficulty": difficulty,
        "robustness": robustness,
        "solvability": solvability,
        "trajectory_value": trajectory_value,
    }
    effective_weights = weights or {
        "grounding": 0.2,
        "alignment": 0.2,
        "clarity": 0.1,
        "novelty": 0.15,
        "difficulty": 0.1,
        "robustness": 0.1,
        "solvability": 0.1,
        "trajectory_value": 0.05,
    }
    weight_total = sum(max(0.0, value) for value in effective_weights.values()) or 1.0
    total_score = sum(
        dimensions.get(name, 0.0) * max(0.0, weight)
        for name, weight in effective_weights.items()
    ) / weight_total
    rejection_reasons: list[str] = []
    if not hard_gate_passed:
        rejection_reasons.append("hard validation gate failed")
    for name, score, threshold in (
        ("grounding", grounding, minimum_grounding),
        ("alignment", alignment, minimum_alignment),
        ("solvability", solvability, minimum_solvability),
    ):
        if score < threshold:
            rejection_reasons.append(
                f"{name} score {score:.3f} is below {threshold:.3f}"
            )
    if total_score < minimum_total_score:
        rejection_reasons.append(
            f"total score {total_score:.3f} is below {minimum_total_score:.3f}"
        )
    application = generated.recipe.application or str(
        generated.task.metadata.get("application") or "unknown"
    )
    return CandidateQualityScore(
        task_id=generated.task.id,
        total_score=total_score,
        dimensions=dimensions,
        hard_gate_passed=hard_gate_passed,
        eligible=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        coverage_bucket={
            "application": application,
            "strategy": generated.recipe.strategy or "unknown",
            "difficulty": bucket,
            "platform": environment_platform(environment),
            "environment": environment.image_id,
        },
    )
