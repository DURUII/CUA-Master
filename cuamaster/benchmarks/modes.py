"""Reusable GUI, CLI, and hybrid benchmark harnesses."""
from __future__ import annotations

from .base import InteractionCapability, InteractionHarness


GUI_HARNESS = InteractionHarness(
    name="gui",
    capabilities=frozenset({InteractionCapability.GUI}),
    description="Visible computer interaction only.",
)
CLI_HARNESS = InteractionHarness(
    name="cli",
    capabilities=frozenset({InteractionCapability.CLI}),
    description="Command-line interaction inside the benchmark environment only.",
)
HYBRID_HARNESS = InteractionHarness(
    name="hybrid",
    capabilities=frozenset({InteractionCapability.GUI, InteractionCapability.CLI}),
    description="Visible computer interaction plus environment-local command execution.",
)

INTERACTION_HARNESSES = {
    harness.name: harness
    for harness in (GUI_HARNESS, CLI_HARNESS, HYBRID_HARNESS)
}


def get_interaction_harness(value: str | InteractionHarness) -> InteractionHarness:
    if isinstance(value, InteractionHarness):
        return value
    name = str(value or "").strip().lower()
    try:
        return INTERACTION_HARNESSES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(INTERACTION_HARNESSES))
        raise ValueError(
            f"unknown interaction harness {value!r}; expected one of {supported}"
        ) from exc
