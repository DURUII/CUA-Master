"""Benchmark-independent interaction harness definitions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class InteractionCapability(str, Enum):
    GUI = "gui"
    CLI = "cli"


@dataclass(frozen=True)
class InteractionHarness:
    """Defines the action families exposed while evaluating a benchmark."""

    name: str
    capabilities: frozenset[InteractionCapability]
    description: str

    @property
    def supports_gui(self) -> bool:
        return InteractionCapability.GUI in self.capabilities

    @property
    def supports_cli(self) -> bool:
        return InteractionCapability.CLI in self.capabilities

    def allows_action_type(self, action_type: str) -> bool:
        normalized = str(action_type or "").strip().lower()
        if normalized == "finish":
            return True
        if normalized.startswith("computer."):
            return self.supports_gui
        if normalized == "shell.exec":
            return self.supports_cli
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capabilities": sorted(capability.value for capability in self.capabilities),
            "description": self.description,
            "allows_structured_tools": False,
        }
