"""Versioned task, agent, episode, and canonical-step schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from cuamaster.contracts import TargetEnvironment, TaskSpec, canonical_json, content_digest

AGENT_SCHEMA = "gui_trajectory.agent.v1"
EPISODE_SCHEMA = "gui_trajectory.episode.v1"
STEP_SCHEMA = "gui_trajectory.step.v1"

__all__ = [
    "AGENT_SCHEMA",
    "EPISODE_SCHEMA",
    "STEP_SCHEMA",
    "AgentSpec",
    "EpisodeStatus",
    "StepRecord",
    "TargetEnvironment",
    "TaskSpec",
    "canonical_json",
    "content_digest",
    "episode_id",
]


class EpisodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    TASK_FAILED = "task_failed"
    AGENT_FAILED = "agent_failed"
    INFRA_FAILED = "infra_failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class AgentSpec:
    id: str
    harness: str
    model: str
    provider: str = "unknown"
    transport: str = "unknown"
    reasoning: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 300
    timeout_seconds: int = 7200
    runtime_strategy: str = "runtime-overlay"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentSpec":
        agent_id = str(raw.get("id") or "").strip()
        harness = str(raw.get("harness") or "").strip()
        model = str(raw.get("model") or "").strip()
        if not agent_id or not harness or not model:
            raise ValueError("agent.id, agent.harness, and agent.model are required")
        reasoning = raw.get("reasoning") or {}
        metadata = raw.get("metadata") or {}
        if not isinstance(reasoning, dict) or not isinstance(metadata, dict):
            raise ValueError(f"agent {agent_id}: reasoning and metadata must be objects")
        max_steps = int(raw.get("max_steps", 300))
        timeout_seconds = int(raw.get("timeout_seconds", 7200))
        if max_steps < 1 or timeout_seconds < 1:
            raise ValueError(f"agent {agent_id}: max_steps and timeout_seconds must be positive")
        return cls(
            id=agent_id,
            harness=harness,
            model=model,
            provider=str(raw.get("provider") or "unknown"),
            transport=str(raw.get("transport") or "unknown"),
            reasoning=dict(reasoning),
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            runtime_strategy=str(raw.get("runtime_strategy") or "runtime-overlay"),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AGENT_SCHEMA,
            "id": self.id,
            "harness": self.harness,
            "model": self.model,
            "provider": self.provider,
            "transport": self.transport,
            "reasoning": self.reasoning,
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "runtime_strategy": self.runtime_strategy,
            "metadata": self.metadata,
        }


def episode_id(
    *,
    task_version: str,
    image_digest: str,
    agent: Mapping[str, Any],
    seed: int = 0,
    attempt: int = 1,
) -> str:
    return content_digest({
        "task_version": task_version,
        "image_digest": image_digest,
        "agent": dict(agent),
        "seed": seed,
        "attempt": attempt,
    })[:32]


@dataclass(frozen=True)
class StepRecord:
    step: int
    action: dict[str, Any]
    pre_observation: dict[str, Any]
    execution: dict[str, Any]
    post_observation: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    decision_context: dict[str, Any] = field(default_factory=dict)
    observation_quality: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if not isinstance(self.action.get("type"), str) or not self.action.get("type"):
            raise ValueError(f"step {self.step}: action.type is required")
        return {
            "schema_version": STEP_SCHEMA,
            "step": self.step,
            "pre_observation": self.pre_observation,
            "action": self.action,
            "execution": self.execution,
            "post_observation": self.post_observation,
            "provenance": self.provenance,
            "decision_context": self.decision_context,
            "observation_quality": self.observation_quality,
        }
