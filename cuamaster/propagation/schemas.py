"""Task-generation-specific schemas."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Mapping

from cuamaster.contracts import TaskSpec, content_digest


def _normalize_semantic_text(value: str) -> str:
    text = value.casefold().replace("\\", "/")
    text = re.sub(r"(?:[a-z]:)?/(?:[^\s/]+/)+[^\s]+", " <path> ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " <number> ", text)
    text = re.sub(r"[^\w<>:+.-]+", " ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class SourceEvidence:
    task_id: str
    episode_id: str | None = None
    step_start: int | None = None
    step_end: int | None = None
    success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "step_start": self.step_start,
            "step_end": self.step_end,
            "success": self.success,
        }


@dataclass(frozen=True)
class AbilityPrimitive:
    id: str
    application: str
    intent: str
    preconditions: tuple[str, ...] = ()
    inputs: dict[str, str] = field(default_factory=dict)
    effects: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    required_vm_capabilities: tuple[str, ...] = ()
    config_dependencies: tuple[dict[str, Any], ...] = ()
    evaluator_fragments: tuple[dict[str, Any], ...] = ()
    source_evidence: tuple[SourceEvidence, ...] = ()
    failure_modes: tuple[str, ...] = ()
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AbilityPrimitive":
        primitive_id = str(raw.get("id") or "").strip()
        application = str(raw.get("application") or "unknown").strip()
        intent = str(raw.get("intent") or "").strip()
        if not primitive_id or not intent:
            raise ValueError("atomic capability id and intent are required")
        evidence = tuple(
            SourceEvidence(
                task_id=str(item.get("task_id") or ""),
                episode_id=str(item["episode_id"]) if item.get("episode_id") else None,
                step_start=int(item["step_start"]) if item.get("step_start") is not None else None,
                step_end=int(item["step_end"]) if item.get("step_end") is not None else None,
                success=bool(item["success"]) if item.get("success") is not None else None,
            )
            for item in raw.get("source_evidence") or []
            if isinstance(item, Mapping) and item.get("task_id")
        )
        if not evidence:
            raise ValueError(f"capability {primitive_id}: source evidence is required")
        return cls(
            id=primitive_id,
            application=application,
            intent=intent,
            preconditions=tuple(str(item) for item in raw.get("preconditions") or []),
            inputs={str(key): str(value) for key, value in (raw.get("inputs") or {}).items()},
            effects=tuple(str(item) for item in raw.get("effects") or []),
            side_effects=tuple(str(item) for item in raw.get("side_effects") or []),
            required_vm_capabilities=tuple(
                str(item) for item in raw.get("required_vm_capabilities") or []
            ),
            config_dependencies=tuple(dict(item) for item in raw.get("config_dependencies") or []),
            evaluator_fragments=tuple(dict(item) for item in raw.get("evaluator_fragments") or []),
            source_evidence=evidence,
            failure_modes=tuple(str(item) for item in raw.get("failure_modes") or []),
            confidence=float(raw.get("confidence") or 0.0),
        )

    @cached_property
    def semantic_signature(self) -> str:
        return content_digest({
            "application": _normalize_semantic_text(self.application),
            "intent": _normalize_semantic_text(self.intent),
            "preconditions": sorted(
                _normalize_semantic_text(item) for item in self.preconditions
            ),
            "effects": sorted(_normalize_semantic_text(item) for item in self.effects),
            "evaluator_fragments": list(self.evaluator_fragments),
        })

    @cached_property
    def evidence_signature(self) -> str:
        return content_digest([
            item.to_dict()
            for item in sorted(
                self.source_evidence,
                key=lambda value: (
                    value.task_id,
                    value.episode_id or "",
                    value.step_start if value.step_start is not None else -1,
                    value.step_end if value.step_end is not None else -1,
                ),
            )
        ])

    @property
    def signature(self) -> str:
        return self.semantic_signature

    @property
    def evidence_count(self) -> int:
        return len(self.source_evidence)

    @property
    def successful_evidence_count(self) -> int:
        return sum(item.success is True for item in self.source_evidence)

    @property
    def failed_evidence_count(self) -> int:
        return sum(item.success is False for item in self.source_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "propagation.ability_primitive.v1",
            "id": self.id,
            "application": self.application,
            "intent": self.intent,
            "preconditions": list(self.preconditions),
            "inputs": self.inputs,
            "effects": list(self.effects),
            "side_effects": list(self.side_effects),
            "required_vm_capabilities": list(self.required_vm_capabilities),
            "config_dependencies": list(self.config_dependencies),
            "evaluator_fragments": list(self.evaluator_fragments),
            "source_evidence": [item.to_dict() for item in self.source_evidence],
            "failure_modes": list(self.failure_modes),
            "confidence": self.confidence,
            "signature": self.signature,
            "semantic_signature": self.semantic_signature,
            "evidence_signature": self.evidence_signature,
            "evidence_count": self.evidence_count,
            "successful_evidence_count": self.successful_evidence_count,
            "failed_evidence_count": self.failed_evidence_count,
        }


@dataclass(frozen=True)
class SourceTask:
    task: TaskSpec
    source_path: str
    format: str
    application: str
    category: str
    workspace_path: str | None = None
    raw_sections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "source_path": self.source_path,
            "format": self.format,
            "application": self.application,
            "category": self.category,
            "workspace_path": self.workspace_path,
            "raw_sections": self.raw_sections,
        }


@dataclass(frozen=True)
class EnvironmentContext:
    image_id: str
    image_digest: str
    os: dict[str, str]
    applications: dict[str, dict[str, Any]]
    capabilities: tuple[str, ...]
    evaluator_types: tuple[str, ...] = ()
    policy: dict[str, Any] = field(default_factory=dict)

    def supports(self, required: tuple[str, ...] | list[str]) -> bool:
        available = set(self.capabilities)
        return all(item in available for item in required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "propagation.environment_context.v1",
            "image_id": self.image_id,
            "image_digest": self.image_digest,
            "os": self.os,
            "applications": self.applications,
            "capabilities": list(self.capabilities),
            "evaluator_types": list(self.evaluator_types),
            "policy": self.policy,
        }


@dataclass(frozen=True)
class TaskRecipe:
    id: str
    anchor_task_id: str
    primitive_ids: tuple[str, ...]
    application: str
    strategy: str
    expected_effects: tuple[str, ...]
    required_vm_capabilities: tuple[str, ...]
    source_task_ids: tuple[str, ...]
    environment_id: str = ""
    environment_digest: str = ""
    variant_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskRecipe":
        return cls(
            id=str(raw.get("id") or ""),
            anchor_task_id=str(raw.get("anchor_task_id") or ""),
            primitive_ids=tuple(str(item) for item in raw.get("primitive_ids", [])),
            application=str(raw.get("application") or ""),
            strategy=str(raw.get("strategy") or ""),
            expected_effects=tuple(str(item) for item in raw.get("expected_effects", [])),
            required_vm_capabilities=tuple(
                str(item) for item in raw.get("required_vm_capabilities", [])
            ),
            source_task_ids=tuple(str(item) for item in raw.get("source_task_ids", [])),
            environment_id=str(raw.get("environment_id") or ""),
            environment_digest=str(raw.get("environment_digest") or ""),
            variant_index=int(raw.get("variant_index") or 0),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "propagation.recipe.v1",
            "id": self.id,
            "anchor_task_id": self.anchor_task_id,
            "primitive_ids": list(self.primitive_ids),
            "application": self.application,
            "strategy": self.strategy,
            "expected_effects": list(self.expected_effects),
            "required_vm_capabilities": list(self.required_vm_capabilities),
            "source_task_ids": list(self.source_task_ids),
            "environment_id": self.environment_id,
            "environment_digest": self.environment_digest,
            "variant_index": self.variant_index,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GeneratedTask:
    task: TaskSpec
    recipe: TaskRecipe
    evaluator_plugins: tuple[dict[str, Any], ...] = ()
    assets: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GeneratedTask":
        task = raw.get("task")
        recipe = raw.get("recipe")
        if not isinstance(task, Mapping) or not isinstance(recipe, Mapping):
            raise ValueError("generated task requires task and recipe objects")
        return cls(
            task=TaskSpec.from_dict(task),
            recipe=TaskRecipe.from_dict(recipe),
            evaluator_plugins=tuple(
                dict(item)
                for item in raw.get("evaluator_plugins", [])
                if isinstance(item, Mapping)
            ),
            assets=tuple(
                dict(item)
                for item in raw.get("assets", [])
                if isinstance(item, Mapping)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "propagation.generated_task.v1",
            "task": self.task.to_dict(),
            "recipe": self.recipe.to_dict(),
            "evaluator_plugins": list(self.evaluator_plugins),
            "assets": list(self.assets),
        }


@dataclass(frozen=True)
class ValidationReport:
    task_id: str
    valid: bool
    checks: dict[str, bool]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    sandbox: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "propagation.validation.v1",
            "task_id": self.task_id,
            "valid": self.valid,
            "checks": self.checks,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "sandbox": self.sandbox,
        }


@dataclass(frozen=True)
class CandidateQualityScore:
    task_id: str
    total_score: float
    dimensions: dict[str, float]
    hard_gate_passed: bool
    eligible: bool
    rejection_reasons: tuple[str, ...] = ()
    similar_task_id: str | None = None
    similarity: float = 0.0
    coverage_bucket: dict[str, str] = field(default_factory=dict)
    scoring_version: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidateQualityScore":
        return cls(
            task_id=str(raw.get("task_id") or ""),
            total_score=float(raw.get("total_score") or 0.0),
            dimensions={
                str(key): float(value)
                for key, value in dict(raw.get("dimensions") or {}).items()
            },
            hard_gate_passed=bool(raw.get("hard_gate_passed")),
            eligible=bool(raw.get("eligible")),
            rejection_reasons=tuple(
                str(item) for item in raw.get("rejection_reasons") or []
            ),
            similar_task_id=(
                str(raw["similar_task_id"])
                if raw.get("similar_task_id")
                else None
            ),
            similarity=float(raw.get("similarity") or 0.0),
            coverage_bucket={
                str(key): str(value)
                for key, value in dict(raw.get("coverage_bucket") or {}).items()
            },
            scoring_version=int(raw.get("scoring_version") or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "propagation.candidate_quality.v1",
            "task_id": self.task_id,
            "total_score": round(self.total_score, 6),
            "dimensions": {
                key: round(value, 6) for key, value in self.dimensions.items()
            },
            "hard_gate_passed": self.hard_gate_passed,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "similar_task_id": self.similar_task_id,
            "similarity": round(self.similarity, 6),
            "coverage_bucket": self.coverage_bucket,
            "scoring_version": self.scoring_version,
        }
