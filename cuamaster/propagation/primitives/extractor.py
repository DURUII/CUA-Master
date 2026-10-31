"""Extract source-grounded semantic capabilities."""
from __future__ import annotations

from dataclasses import replace

from ..generation.backends import GeneratorBackend
from ..ingestion import EpisodeEvidence
from ..schemas import AbilityPrimitive, EnvironmentContext, SourceTask


class PrimitiveExtractor:
    def __init__(self, backend: GeneratorBackend) -> None:
        self.backend = backend

    def extract(
        self,
        source: SourceTask,
        episodes: list[EpisodeEvidence],
        environment: EnvironmentContext,
    ) -> list[AbilityPrimitive]:
        capabilities = self.backend.decompose(source, episodes, environment)
        result: list[AbilityPrimitive] = []
        for capability in capabilities:
            if not capability.source_evidence:
                continue
            if not environment.supports(list(capability.required_vm_capabilities)):
                continue
            namespace = f"{source.task.id}::"
            primitive_id = capability.id
            if not primitive_id.startswith(namespace):
                primitive_id = f"{namespace}{primitive_id}"
            result.append(replace(capability, id=primitive_id))
        return result
