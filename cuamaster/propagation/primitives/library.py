"""Deduplicated atomic capability library."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..diversity import primitive_similarity
from ..schemas import AbilityPrimitive


class PrimitiveLibrary:
    def __init__(
        self,
        capabilities: Iterable[AbilityPrimitive] = (),
        *,
        similarity_threshold: float = 0.88,
    ) -> None:
        self._by_signature: dict[str, AbilityPrimitive] = {}
        self.similarity_threshold = similarity_threshold
        for capability in capabilities:
            self.add(capability)

    def add(self, capability: AbilityPrimitive) -> AbilityPrimitive:
        signature = capability.semantic_signature
        existing = self._by_signature.get(signature)
        if existing is None:
            primitive_sources = {
                item.task_id for item in capability.source_evidence
            }
            existing = next(
                (
                    item
                    for item in self._by_signature.values()
                    if primitive_sources.isdisjoint(
                        evidence.task_id for evidence in item.source_evidence
                    )
                    and primitive_similarity(item, capability)
                    >= self.similarity_threshold
                ),
                None,
            )
        if existing is None:
            self._by_signature[signature] = capability
            return capability
        merged = AbilityPrimitive(
            id=min(existing.id, capability.id),
            application=existing.application,
            intent=existing.intent,
            preconditions=tuple(sorted(set(existing.preconditions) | set(capability.preconditions))),
            inputs={**existing.inputs, **capability.inputs},
            effects=tuple(sorted(set(existing.effects) | set(capability.effects))),
            side_effects=tuple(sorted(set(existing.side_effects) | set(capability.side_effects))),
            required_vm_capabilities=tuple(sorted(
                set(existing.required_vm_capabilities) | set(capability.required_vm_capabilities)
            )),
            config_dependencies=existing.config_dependencies or capability.config_dependencies,
            evaluator_fragments=existing.evaluator_fragments or capability.evaluator_fragments,
            source_evidence=tuple({
                (item.task_id, item.episode_id, item.step_start, item.step_end, item.success): item
                for item in (*existing.source_evidence, *capability.source_evidence)
            }.values()),
            failure_modes=tuple(sorted(set(existing.failure_modes) | set(capability.failure_modes))),
            confidence=max(existing.confidence, capability.confidence),
        )
        existing_key = next(
            key for key, value in self._by_signature.items() if value is existing
        )
        self._by_signature[existing_key] = merged
        return merged

    def all(self) -> list[AbilityPrimitive]:
        return sorted(self._by_signature.values(), key=lambda item: (item.application, item.id))

    def by_application(self) -> dict[str, list[AbilityPrimitive]]:
        groups: dict[str, list[AbilityPrimitive]] = defaultdict(list)
        for capability in self.all():
            groups[capability.application].append(capability)
        return dict(groups)
