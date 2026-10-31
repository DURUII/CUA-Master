"""Bounded, coverage-oriented composition of compatible capabilities."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from cuamaster.contracts import content_digest

from ..diversity import recipe_similarity
from ..quality import task_class_reference
from ..schemas import AbilityPrimitive, EnvironmentContext, SourceTask, TaskRecipe


class PrimitiveComposer:
    def __init__(
        self,
        *,
        min_primitives: int = 2,
        max_primitives: int = 4,
        variants_per_anchor: int = 3,
        source_complete_variants_per_anchor: int | None = None,
        allow_cross_app: bool = False,
        allow_cross_source: bool = False,
        cross_source_fraction: float = 0.0,
        candidate_pool_multiplier: int = 8,
        beam_width: int = 64,
        recipe_similarity_threshold: float = 0.85,
        random_seed: int = 0,
        anchor_difficulty_targets: dict[str, str] | None = None,
        strategies: tuple[str, ...] = (
            "sequential",
            "parameterized",
            "state_dependent",
            "cross_app_handoff",
        ),
    ) -> None:
        self.min_primitives = min_primitives
        self.max_primitives = max_primitives
        self.variants_per_anchor = variants_per_anchor
        self.source_complete_variants_per_anchor = (
            variants_per_anchor
            if source_complete_variants_per_anchor is None
            else source_complete_variants_per_anchor
        )
        self.allow_cross_app = allow_cross_app
        self.allow_cross_source = allow_cross_source
        self.cross_source_fraction = cross_source_fraction
        self.candidate_pool_multiplier = candidate_pool_multiplier
        self.beam_width = beam_width
        self.recipe_similarity_threshold = recipe_similarity_threshold
        self.random_seed = random_seed
        self.anchor_difficulty_targets = anchor_difficulty_targets or {}
        self.strategies = set(strategies)

    @staticmethod
    def _source_tasks(capabilities: Iterable[AbilityPrimitive]) -> tuple[str, ...]:
        return tuple(sorted({
            evidence.task_id
            for capability in capabilities
            for evidence in capability.source_evidence
        }))

    @staticmethod
    def _belongs_to_source(capability: AbilityPrimitive, task_id: str) -> bool:
        return any(item.task_id == task_id for item in capability.source_evidence)

    @staticmethod
    def _requirements(capability: AbilityPrimitive) -> set[str]:
        return {
            condition.removeprefix("requires:").strip()
            for condition in capability.preconditions
            if condition.startswith("requires:")
            and condition.removeprefix("requires:").strip()
        }

    @classmethod
    def _ordered(
        cls,
        capabilities: Iterable[AbilityPrimitive],
        environment: EnvironmentContext,
    ) -> tuple[AbilityPrimitive, ...]:
        remaining = sorted(capabilities, key=cls._execution_order)
        ordered: list[AbilityPrimitive] = []
        effects = set(environment.capabilities)
        while remaining:
            ready = next(
                (
                    capability
                    for capability in remaining
                    if cls._requirements(capability) <= effects
                ),
                remaining[0],
            )
            remaining.remove(ready)
            ordered.append(ready)
            effects.update(ready.effects)
        return tuple(ordered)

    @classmethod
    def _compatible(
        cls,
        capabilities: tuple[AbilityPrimitive, ...],
        environment: EnvironmentContext,
    ) -> bool:
        required = {
            item
            for capability in capabilities
            for item in capability.required_vm_capabilities
        }
        if not environment.supports(sorted(required)):
            return False
        effects = set(environment.capabilities)
        for capability in cls._ordered(capabilities, environment):
            if not cls._requirements(capability) <= effects:
                return False
            effects.update(capability.effects)
        return True

    @staticmethod
    def _execution_order(capability: AbilityPrimitive) -> tuple[int, str]:
        starts = [
            evidence.step_start
            for evidence in capability.source_evidence
            if evidence.step_start is not None
        ]
        return (min(starts) if starts else 10**9, capability.id)

    @staticmethod
    def _uses_task_class(source: SourceTask) -> bool:
        return task_class_reference(source.task) is not None

    @classmethod
    def _dependency_depth(cls, capabilities: tuple[AbilityPrimitive, ...]) -> int:
        effects: set[str] = set()
        depth = 0
        for capability in capabilities:
            requirements = cls._requirements(capability)
            if requirements & effects:
                depth += 1
            effects.update(capability.effects)
        return depth

    @classmethod
    def _strategy(cls, capabilities: tuple[AbilityPrimitive, ...]) -> str:
        applications = {item.application for item in capabilities}
        dependency_depth = cls._dependency_depth(capabilities)
        if len(applications) > 1:
            return "cross_app_handoff"
        if dependency_depth:
            return "state_dependent"
        intents = [item.intent.casefold() for item in capabilities]
        if len(set(intents)) < len(intents):
            return "parameterized"
        return "sequential"

    @classmethod
    def _has_cross_app_handoff(
        cls,
        capabilities: tuple[AbilityPrimitive, ...],
    ) -> bool:
        if len({item.application for item in capabilities}) <= 1:
            return True
        effects: set[str] = set()
        for capability in capabilities:
            requirements = cls._requirements(capability)
            if requirements & effects:
                return True
            effects.update(capability.effects)
        return False

    def _candidate_score(
        self,
        capabilities: tuple[AbilityPrimitive, ...],
        anchor_task_id: str,
    ) -> float:
        evaluator_coverage = sum(bool(item.evaluator_fragments) for item in capabilities)
        confidence = sum(item.confidence for item in capabilities) / len(capabilities)
        evidence = sum(min(3, item.evidence_count) for item in capabilities) / (
            3 * len(capabilities)
        )
        evidence_outcomes = [
            evidence.success
            for capability in capabilities
            for evidence in capability.source_evidence
            if evidence.success is not None
        ]
        success_rate = (
            sum(evidence_outcomes) / len(evidence_outcomes)
            if evidence_outcomes
            else 0.5
        )
        anchor_coverage = sum(
            self._belongs_to_source(item, anchor_task_id) for item in capabilities
        ) / len(capabilities)
        dependency = min(1.0, self._dependency_depth(capabilities) / 2)
        size = min(1.0, len(capabilities) / max(1, self.max_primitives))
        jitter = int(content_digest({
            "seed": self.random_seed,
            "capabilities": [item.semantic_signature for item in capabilities],
        })[:8], 16) / 0xFFFFFFFF
        difficulty_target = self.anchor_difficulty_targets.get(anchor_task_id, "")
        dependency_weight, size_weight = 0.12, 0.08
        if difficulty_target == "harder":
            dependency_weight, size_weight = 0.2, 0.14
        elif difficulty_target == "easier":
            dependency_weight, size_weight = 0.06, 0.02
        base_weight = 1.0 - dependency_weight - size_weight - 0.01
        return base_weight / 0.79 * (
            0.23 * min(1.0, evaluator_coverage / len(capabilities))
            + 0.17 * confidence
            + 0.13 * evidence
            + 0.12 * success_rate
            + 0.14 * anchor_coverage
        ) + (
            dependency_weight * dependency
            + size_weight * size
            + 0.01 * jitter
        )

    def _recipe(
        self,
        anchor: SourceTask,
        combination: tuple[AbilityPrimitive, ...],
        environment: EnvironmentContext,
        *,
        variant_index: int,
        strategy: str,
        source_ids: tuple[str, ...] | None = None,
        composition_score: float = 0.0,
        selection_novelty: float = 1.0,
    ) -> TaskRecipe:
        ordered = self._ordered(combination, environment)
        recipe_source_ids = source_ids or self._source_tasks(ordered)
        required = tuple(sorted({
            item
            for capability in ordered
            for item in capability.required_vm_capabilities
        }))
        effects = tuple(item for capability in ordered for item in capability.effects)
        payload = {
            "quality_policy_version": 3,
            "anchor": anchor.task.id,
            "capabilities": [item.id for item in ordered],
            "semantic_primitives": [item.semantic_signature for item in ordered],
            "strategy": strategy,
            "environment_id": environment.image_id,
            "environment_digest": environment.image_digest,
            "variant_index": variant_index,
        }
        applications = sorted({item.application for item in ordered})
        difficulty_target = self.anchor_difficulty_targets.get(anchor.task.id, "")
        return TaskRecipe(
            id=content_digest(payload)[:20],
            anchor_task_id=anchor.task.id,
            primitive_ids=tuple(item.id for item in ordered),
            application=anchor.application,
            strategy=strategy,
            expected_effects=effects,
            required_vm_capabilities=required,
            source_task_ids=recipe_source_ids,
            environment_id=environment.image_id,
            environment_digest=environment.image_digest,
            variant_index=variant_index,
            metadata={
                "composition_score": round(composition_score, 6),
                "selection_novelty": round(selection_novelty, 6),
                "semantic_primitive_ids": [
                    item.semantic_signature for item in ordered
                ],
                "application_count": len(applications),
                "applications": applications,
                "dependency_depth": self._dependency_depth(ordered),
                "evidence_count": sum(item.evidence_count for item in ordered),
                "cross_source": len(set(recipe_source_ids)) > 1,
                **({"difficulty_target": difficulty_target} if difficulty_target else {}),
            },
        )

    def _bounded_candidates(
        self,
        anchor: SourceTask,
        pool: list[AbilityPrimitive],
        environment: EnvironmentContext,
        *,
        cross_source: bool | None = None,
        candidate_limit: int | None = None,
    ) -> list[tuple[AbilityPrimitive, ...]]:
        allow_cross_source = (
            self.allow_cross_source if cross_source is None else cross_source
        )
        if not pool:
            return []
        pool = sorted(pool, key=lambda item: (
            -self._candidate_score((item,), anchor.task.id),
            self._execution_order(item),
        ))
        target_candidates = candidate_limit or (
            self.variants_per_anchor * self.candidate_pool_multiplier
        )
        search_pool_size = max(self.max_primitives, target_candidates)
        if len(pool) > search_pool_size:
            anchor_pool = [
                capability
                for capability in pool
                if self._belongs_to_source(capability, anchor.task.id)
            ]
            if allow_cross_source:
                external_pool = [
                    capability
                    for capability in pool
                    if not self._belongs_to_source(capability, anchor.task.id)
                ]
                anchor_limit = min(
                    len(anchor_pool),
                    max(self.max_primitives, search_pool_size // 2),
                )
                pool = [
                    *anchor_pool[:anchor_limit],
                    *external_pool[:search_pool_size],
                ]
                pool.sort(key=lambda item: (
                    -self._candidate_score((item,), anchor.task.id),
                    self._execution_order(item),
                ))
            else:
                pool = anchor_pool[:search_pool_size]
        indexed = list(enumerate(pool))
        states: list[tuple[int, ...]] = [
            (index,)
            for index, capability in indexed
            if self._belongs_to_source(capability, anchor.task.id)
        ][: self.beam_width]
        candidates: dict[tuple[str, ...], tuple[AbilityPrimitive, ...]] = {}
        score_cache: dict[tuple[int, ...], float] = {}

        def state_score(state: tuple[int, ...]) -> float:
            score = score_cache.get(state)
            if score is None:
                score = self._candidate_score(
                    tuple(pool[index] for index in state),
                    anchor.task.id,
                )
                score_cache[state] = score
            return score

        for _ in range(1, min(self.max_primitives, len(pool)) + 1):
            next_states: dict[tuple[int, ...], float] = {}
            for state in states:
                combination = tuple(pool[index] for index in state)
                if len(combination) >= self.min_primitives:
                    ordered = self._ordered(combination, environment)
                    strategy = self._strategy(ordered)
                    if (
                        (
                            allow_cross_source
                            or all(
                                self._belongs_to_source(item, anchor.task.id)
                                for item in ordered
                            )
                        )
                        and (self.allow_cross_app or len({item.application for item in ordered}) == 1)
                        and self._has_cross_app_handoff(ordered)
                        and strategy in self.strategies
                        and self._compatible(ordered, environment)
                    ):
                        key = tuple(item.semantic_signature for item in ordered)
                        candidates[key] = ordered
                if len(state) >= self.max_primitives:
                    continue
                used_signatures = {
                    pool[index].semantic_signature for index in state
                }
                for next_index, capability in indexed:
                    if next_index in state or capability.semantic_signature in used_signatures:
                        continue
                    expanded = tuple(sorted((*state, next_index)))
                    if expanded in next_states:
                        continue
                    expanded_primitives = tuple(pool[index] for index in expanded)
                    if not allow_cross_source and any(
                        not self._belongs_to_source(item, anchor.task.id)
                        for item in expanded_primitives
                    ):
                        continue
                    if not self.allow_cross_app and any(
                        item.application != anchor.application
                        for item in expanded_primitives
                    ):
                        continue
                    next_states[expanded] = state_score(expanded)
            states = [
                state
                for state, _ in sorted(
                    next_states.items(),
                    key=lambda item: (-item[1], item[0]),
                )[: self.beam_width]
            ]
            if len(candidates) >= target_candidates and states:
                break
            if not states:
                break
        return sorted(
            candidates.values(),
            key=lambda items: (
                -self._candidate_score(items, anchor.task.id),
                tuple(item.id for item in items),
            ),
        )[:target_candidates]

    def _select_diverse(
        self,
        anchor: SourceTask,
        candidates: list[tuple[AbilityPrimitive, ...]],
        *,
        limit: int | None = None,
        exclude: list[tuple[AbilityPrimitive, ...]] | None = None,
    ) -> list[tuple[tuple[AbilityPrimitive, ...], float, float]]:
        maximum = self.variants_per_anchor if limit is None else limit
        selected: list[tuple[tuple[AbilityPrimitive, ...], float, float]] = []
        anchored = [(combination, 1.0, 1.0) for combination in exclude or []]
        remaining = list(candidates)
        while remaining and len(selected) < maximum:
            ranked = []
            for combination in remaining:
                quality = self._candidate_score(combination, anchor.task.id)
                maximum_similarity = max(
                    (
                        recipe_similarity(combination, existing[0])
                        for existing in (*anchored, *selected)
                    ),
                    default=0.0,
                )
                novelty = 1.0 - maximum_similarity
                score = 0.75 * quality + 0.25 * novelty
                ranked.append((score, novelty, quality, combination))
            _, novelty, quality, best = max(
                ranked,
                key=lambda item: (item[0], item[1], tuple(value.id for value in item[3])),
            )
            selected.append((best, quality, novelty))
            remaining = [
                item
                for item in remaining
                if recipe_similarity(item, best) < self.recipe_similarity_threshold
            ]
        return selected

    def recipes(
        self,
        sources: list[SourceTask],
        capabilities: list[AbilityPrimitive],
        environment: EnvironmentContext,
    ) -> list[TaskRecipe]:
        sources_by_id = {source.task.id: source for source in sources}
        by_app: dict[str, list[AbilityPrimitive]] = defaultdict(list)
        for capability in capabilities:
            by_app[capability.application].append(capability)
        cross_source_variants = (
            int(self.variants_per_anchor * self.cross_source_fraction)
            if self.allow_cross_source or self.cross_source_fraction > 0
            else 0
        )
        recipes: list[TaskRecipe] = []
        for anchor in sources:
            anchor_pool = [
                capability
                for capability in capabilities
                if self._belongs_to_source(capability, anchor.task.id)
            ]
            if self._uses_task_class(anchor):
                # The trusted task-class evaluator scores the complete source
                # outcome, so every variant keeps the full capability set and
                # differentiation comes from outcome-preserving variation
                # directives at compose time.
                combination = self._ordered(anchor_pool, environment)
                if len(combination) < self.min_primitives:
                    continue
                if not self._compatible(combination, environment):
                    continue
                score = self._candidate_score(combination, anchor.task.id)
                for variant_index in range(self.source_complete_variants_per_anchor):
                    recipes.append(self._recipe(
                        anchor,
                        combination,
                        environment,
                        variant_index=variant_index,
                        strategy="source_complete",
                        source_ids=(anchor.task.id,),
                        composition_score=score,
                    ))
                continue

            pool = capabilities if self.allow_cross_app else by_app.get(anchor.application, [])
            same_source_pool = [
                capability
                for capability in pool
                if self._belongs_to_source(capability, anchor.task.id)
            ]
            candidates = self._bounded_candidates(
                anchor,
                list(same_source_pool),
                environment,
                cross_source=False,
            )
            selected = self._select_diverse(anchor, candidates)
            anchor_recipes: list[TaskRecipe] = []
            selected_combinations = [combination for combination, _, _ in selected]
            for variant_index, (combination, score, novelty) in enumerate(selected):
                source_ids = (anchor.task.id,)
                anchor_recipes.append(self._recipe(
                    anchor,
                    combination,
                    environment,
                    variant_index=variant_index,
                    strategy=self._strategy(combination),
                    source_ids=source_ids,
                    composition_score=score,
                    selection_novelty=novelty,
                ))
            if cross_source_variants and len(pool) > len(same_source_pool):
                cross_candidates = [
                    combination
                    for combination in self._bounded_candidates(
                        anchor,
                        list(pool),
                        environment,
                        cross_source=True,
                        candidate_limit=(
                            cross_source_variants * self.candidate_pool_multiplier
                        ),
                    )
                    if len(self._source_tasks(combination)) > 1
                ]
                cross_selected = self._select_diverse(
                    anchor,
                    cross_candidates,
                    limit=cross_source_variants,
                    exclude=selected_combinations,
                )
                for offset, (combination, score, novelty) in enumerate(cross_selected):
                    source_ids = self._source_tasks(combination)
                    if any(
                        source_id not in sources_by_id for source_id in source_ids
                    ):
                        continue
                    anchor_recipes.append(self._recipe(
                        anchor,
                        combination,
                        environment,
                        variant_index=len(selected) + offset,
                        strategy=self._strategy(combination),
                        source_ids=source_ids,
                        composition_score=score,
                        selection_novelty=novelty,
                    ))
            recipes.extend(anchor_recipes)
        return recipes
