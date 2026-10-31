"""Compose evaluator postconditions without mutating shared evaluator code."""
from __future__ import annotations

from typing import Any, Iterable

from ..schemas import AbilityPrimitive


def evaluator_component(capability: AbilityPrimitive) -> dict[str, Any]:
    if capability.evaluator_fragments:
        return {
            "primitive_id": capability.id,
            "mode": "all",
            "fragments": list(capability.evaluator_fragments),
        }
    return {
        "primitive_id": capability.id,
        "mode": "effect_only",
        "expected_effects": list(capability.effects),
    }


def build_composite_evaluator(capabilities: Iterable[AbilityPrimitive]) -> dict[str, Any]:
    components = [evaluator_component(capability) for capability in capabilities]
    return {
        "type": "composite",
        "mode": "all",
        "success_threshold": 1.0,
        "components": components,
    }
