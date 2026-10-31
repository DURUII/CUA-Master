"""End-to-end task decomposition, composition, generation, and validation."""
from __future__ import annotations

import math
import random
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from cuamaster.contracts import content_digest

from .alignment import instruction_alignment_report
from .primitives import PrimitiveExtractor, PrimitiveLibrary
from .config import TaskGenerationConfig
from .diversity import score_candidate
from .environments import select_environment_contexts
from .generation import PrimitiveComposer, create_generator_backend
from .ingestion import load_episodes, load_source_tasks
from .platforms import environment_platform, source_platform_errors
from .quality import solve_plan_quality_errors
from .release import promote_candidates
from .schemas import AbilityPrimitive, EnvironmentContext, GeneratedTask, SourceTask, TaskRecipe, ValidationReport
from .utils import atomic_write_json, read_json, write_jsonl
from .validation import StaticVerifier, TaskValidator, create_verifier


_STAGES = ("index", "decompose", "compose", "validate")
_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class TaskGenerationPipeline:
    def __init__(self, config: TaskGenerationConfig) -> None:
        self.config = config
        self.base_dir = config.path.parent
        self._progress_lock = threading.Lock()

    def _path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (self.base_dir / path).resolve() if not path.is_absolute() else path.resolve()

    def _run_dir(
        self,
        run_id: str | None,
        output_dir: str | Path | None = None,
    ) -> Path:
        if output_dir is not None:
            return Path(output_dir).expanduser().resolve()
        if run_id:
            return self._path(self.config.output_root) / run_id
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = content_digest(self.config.path.read_text(encoding="utf-8"))[:10]
        return self._path(self.config.output_root) / f"{timestamp}-{digest}"

    def run(
        self,
        *,
        stop_after: str = "validate",
        run_id: str | None = None,
        output_dir: str | Path | None = None,
        fresh: bool = False,
    ) -> dict[str, Any]:
        if stop_after not in _STAGES:
            raise ValueError(f"unknown pipeline stage: {stop_after}")
        run_dir = self._run_dir(run_id, output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_run_outputs(run_dir, fresh=fresh)
        self._progress(f"run: start stage={stop_after} run_dir={run_dir}")
        task_roots = [self._path(value) for value in self.config.sources.task_roots]
        episode_roots = [self._path(value) for value in self.config.sources.episode_roots]
        selection_seed = self._environment_seed(run_dir)
        environments = select_environment_contexts(
            self._path(self.config.environment.image_store),
            self.config.environment.image_ids,
            selection=self.config.environment.selection,
            sample_size=self.config.environment.sample_size,
            random_seed=selection_seed,
            allowed_download_domains=self.config.environment.allowed_download_domains,
            include_pending=self.config.environment.include_pending,
            primitive_overrides=self.config.environment.primitive_overrides,
        )
        sources = load_source_tasks(task_roots)
        if self.config.sources.task_ids:
            selected_ids = set(self.config.sources.task_ids)
            sources = [source for source in sources if source.task.id in selected_ids]
        episodes = load_episodes(episode_roots)
        if self.config.sources.require_episodes:
            if not episodes:
                raise ValueError(
                    "sources.require_episodes is true but no episodes were loaded; "
                    "check sources.episodes paths"
                )
            sources = [source for source in sources if episodes.get(source.task.id)]
            if not sources:
                raise ValueError(
                    "sources.require_episodes is true but no source task has matching episodes"
                )
        trajectory_feedback = self._episode_feedback(episodes)
        atomic_write_json(run_dir / "trajectory_feedback.json", trajectory_feedback)
        self._progress(
            "index: loaded "
            f"sources={len(sources)} episode_tasks={len(episodes)} "
            f"environments={len(environments)}"
        )
        environment_payload = {
            "schema_version": "propagation.environment_selection.v1",
            "selection": self.config.environment.selection,
            "sample_size": self.config.environment.sample_size,
            "random_seed": selection_seed,
            "configured_image_ids": list(self.config.environment.image_ids),
            "include_pending": self.config.environment.include_pending,
            "assignment": self.config.environment.assignment,
            "environments_per_task": self.config.environment.environments_per_task,
            "platform_policy": self.config.environment.platform_policy,
            "selected": [environment.to_dict() for environment in environments],
        }
        atomic_write_json(run_dir / "environment_contexts.json", environment_payload)
        if len(environments) == 1:
            atomic_write_json(run_dir / "environment_context.json", environments[0].to_dict())
        write_jsonl(run_dir / "source_tasks.jsonl", (source.to_dict() for source in sources))
        compatible_environment_ids: dict[str, list[str]] = {
            source.task.id: [] for source in sources
        }
        compatibility_rows = []
        for environment in environments:
            for source in sources:
                compatible, missing, target_match = self._source_compatibility(
                    source,
                    environment,
                )
                compatibility_rows.append({
                    "task_id": source.task.id,
                    "environment_id": environment.image_id,
                    "compatible": compatible,
                    "missing_capabilities": missing,
                    "target_environment_match": target_match,
                })
                if compatible:
                    compatible_environment_ids[source.task.id].append(environment.image_id)
        sources_by_environment = self._assign_sources_to_environments(
            sources,
            environments,
            compatible_environment_ids,
            selection_seed,
        )
        assigned_pairs = {
            (source.task.id, environment_id)
            for environment_id, assigned_sources in sources_by_environment.items()
            for source in assigned_sources
        }
        for row in compatibility_rows:
            row["assigned"] = (
                row["task_id"],
                row["environment_id"],
            ) in assigned_pairs
        write_jsonl(run_dir / "source_environment_matrix.jsonl", compatibility_rows)
        compatible_pairs = sum(len(items) for items in compatible_environment_ids.values())
        assignment_counts = {
            environment.image_id: len(sources_by_environment[environment.image_id])
            for environment in environments
        }
        self._progress(
            "index: assigned "
            f"compatible_pairs={compatible_pairs} assigned_pairs={len(assigned_pairs)}"
        )
        if stop_after == "index":
            return self._finish(
                run_dir,
                stop_after,
                sources=len(sources),
                environments=len(environments),
                environment_ids=[environment.image_id for environment in environments],
                compatible_source_environment_pairs=compatible_pairs,
                assigned_source_environment_pairs=len(assigned_pairs),
                assignments_per_environment=assignment_counts,
            )

        generator = create_generator_backend(
            self.config.generation,
            cache_dir=run_dir / ".api_cache",
        )
        extractor = PrimitiveExtractor(generator)
        primitive_records = []
        recipe_records = []
        candidates: list[tuple[GeneratedTask, EnvironmentContext]] = []
        extraction_failures = []
        effective_variants_per_anchor = self.config.generation.variants_per_anchor
        if self.config.release.target_promoted_tasks:
            target_candidates = (
                self.config.release.target_promoted_tasks
                * self.config.generation.search.candidate_pool_multiplier
            )
            effective_variants_per_anchor = max(
                effective_variants_per_anchor,
                math.ceil(target_candidates / max(1, len(assigned_pairs))),
            )
        composer = PrimitiveComposer(
            min_primitives=self.config.generation.min_primitives,
            max_primitives=self.config.generation.max_primitives,
            variants_per_anchor=effective_variants_per_anchor,
            source_complete_variants_per_anchor=(
                self.config.generation.variants_per_anchor
            ),
            allow_cross_app=self.config.generation.allow_cross_app,
            allow_cross_source=self.config.generation.allow_cross_source,
            cross_source_fraction=self.config.generation.cross_source_fraction,
            candidate_pool_multiplier=(
                self.config.generation.search.candidate_pool_multiplier
            ),
            beam_width=self.config.generation.search.beam_width,
            recipe_similarity_threshold=(
                self.config.generation.search.recipe_similarity_threshold
            ),
            random_seed=self.config.generation.search.random_seed,
            anchor_difficulty_targets=self._anchor_difficulty_targets(
                trajectory_feedback
            ),
            strategies=self.config.generation.search.strategies,
        )
        generation_failures = []
        candidates_root = run_dir / "candidates"
        existing_candidates = self._load_existing_candidates(candidates_root)
        reused_candidates = 0
        indexes = {
            environment.image_id: PrimitiveLibrary(
                similarity_threshold=(
                    self.config.generation.search.primitive_similarity_threshold
                )
            )
            for environment in environments
        }
        extraction_jobs = [
            (environment, source)
            for environment in environments
            for source in sources_by_environment[environment.image_id]
        ]

        def extract_job(
            job: tuple[EnvironmentContext, SourceTask],
        ) -> tuple[EnvironmentContext, SourceTask, list[AbilityPrimitive], Exception | None]:
            environment, source = job
            try:
                extracted = extractor.extract(
                    source,
                    episodes.get(source.task.id, []),
                    environment,
                )
                return environment, source, extracted, None
            except Exception as exc:  # noqa: BLE001
                return environment, source, [], exc

        for environment, source, extracted, error in self._concurrent_map(
            extract_job,
            extraction_jobs,
            stage="decompose",
            describe_item=lambda job: (
                f"task={job[1].task.id} environment={job[0].image_id}"
            ),
            result_status=lambda result: "failed" if result[3] is not None else "ok",
        ):
            if error is not None:
                extraction_failures.append({
                    "task_id": source.task.id,
                    "environment_id": environment.image_id,
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            for capability in extracted:
                indexes[environment.image_id].add(capability)

        composition_jobs: list[
            tuple[EnvironmentContext, TaskRecipe, SourceTask, list[AbilityPrimitive]]
        ] = []
        for environment in environments:
            environment_sources = sources_by_environment[environment.image_id]
            capabilities = indexes[environment.image_id].all()
            primitive_records.extend({
                **capability.to_dict(),
                "environment_id": environment.image_id,
                "environment_digest": environment.image_digest,
            } for capability in capabilities)
            if stop_after == "decompose":
                continue
            planning_started = time.monotonic()
            self._progress(
                "compose: planning "
                f"environment={environment.image_id} "
                f"sources={len(environment_sources)} capabilities={len(capabilities)}"
            )
            recipes = composer.recipes(environment_sources, capabilities, environment)
            self._progress(
                "compose: planned "
                f"environment={environment.image_id} recipes={len(recipes)} "
                f"elapsed={_format_duration(time.monotonic() - planning_started)}"
            )
            recipe_records.extend(recipe.to_dict() for recipe in recipes)
            source_by_id = {source.task.id: source for source in environment_sources}
            primitive_by_id: dict[str, AbilityPrimitive] = {}
            for capability in capabilities:
                if capability.id in primitive_by_id:
                    raise ValueError(
                        f"duplicate capability id in environment {environment.image_id}: "
                        f"{capability.id}"
                    )
                primitive_by_id[capability.id] = capability
            for recipe in recipes:
                existing = existing_candidates.get(recipe.id)
                if existing is not None:
                    candidates.append((existing, environment))
                    reused_candidates += 1
                    continue
                anchor = source_by_id[recipe.anchor_task_id]
                selected = [
                    primitive_by_id[primitive_id]
                    for primitive_id in recipe.primitive_ids
                ]
                composition_jobs.append((environment, recipe, anchor, selected))

        if stop_after != "decompose":
            self._progress(
                "compose: prepared "
                f"recipes={len(recipe_records)} jobs={len(composition_jobs)} "
                f"reused={reused_candidates}"
            )

        def compose_job(
            job: tuple[EnvironmentContext, TaskRecipe, SourceTask, list[AbilityPrimitive]],
        ) -> tuple[EnvironmentContext, TaskRecipe, GeneratedTask | None, Exception | None]:
            environment, recipe, anchor, selected = job
            try:
                generated = generator.compose(recipe, anchor, selected, environment)
                return environment, recipe, generated, None
            except Exception as exc:  # noqa: BLE001
                return environment, recipe, None, exc

        def persist_generated_task(
            result: tuple[
                EnvironmentContext,
                TaskRecipe,
                GeneratedTask | None,
                Exception | None,
            ],
        ) -> None:
            _, _, generated, error = result
            if error is not None or generated is None:
                return
            candidate_dir = candidates_root / generated.task.id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                candidate_dir / "generated_task.json",
                generated.to_dict(),
            )

        if stop_after != "decompose":
            for environment, recipe, generated, error in self._concurrent_map(
                compose_job,
                composition_jobs,
                stage="compose",
                describe_item=lambda job: (
                    f"task={job[2].task.id} recipe={job[1].id} "
                    f"environment={job[0].image_id}"
                ),
                result_status=lambda result: (
                    "failed"
                    if result[3] is not None or result[2] is None
                    else "ok"
                ),
                on_result=persist_generated_task,
            ):
                if error is not None:
                    generation_failures.append({
                        "recipe_id": recipe.id,
                        "environment_id": environment.image_id,
                        "error": f"{type(error).__name__}: {error}",
                    })
                    continue
                if generated is None:
                    continue
                candidates.append((generated, environment))
        write_jsonl(run_dir / "ability_primitives.jsonl", primitive_records)
        write_jsonl(run_dir / "extraction_failures.jsonl", extraction_failures)
        if stop_after == "decompose":
            return self._finish(
                run_dir,
                stop_after,
                sources=len(sources),
                environments=len(environments),
                compatible_source_environment_pairs=compatible_pairs,
                assigned_source_environment_pairs=len(assigned_pairs),
                assignments_per_environment=assignment_counts,
                concurrency=self.config.generation.concurrency,
                capabilities=len(primitive_records),
                extraction_failures=len(extraction_failures),
            )
        write_jsonl(run_dir / "recipes.jsonl", recipe_records)
        write_jsonl(run_dir / "generation_failures.jsonl", generation_failures)
        if stop_after == "compose":
            return self._finish(
                run_dir,
                stop_after,
                sources=len(sources),
                environments=len(environments),
                compatible_source_environment_pairs=compatible_pairs,
                assigned_source_environment_pairs=len(assigned_pairs),
                assignments_per_environment=assignment_counts,
                concurrency=self.config.generation.concurrency,
                capabilities=len(primitive_records),
                recipes=len(recipe_records),
                candidates=len(candidates),
                generated_candidates=len(candidates) - reused_candidates,
                reused_candidates=reused_candidates,
                generation_failures=len(generation_failures),
            )

        validation_root = run_dir / "validation"
        quality_root = run_dir / "quality"
        rejected_root = run_dir / "rejected"
        # Sandbox verifiers share one VM per environment and must be
        # serialized; static verification has no shared state, so it runs at
        # full generation concurrency.
        sandbox_sample_rate = self.config.validation.sandbox_sample_rate
        environment_locks = {
            environment.image_id: threading.Lock() for environment in environments
        }
        verifiers = {
            environment.image_id: create_verifier(self.config.validation)
            for environment in environments
        }
        uses_sandbox_verifier = any(
            not isinstance(verifier, StaticVerifier)
            for verifier in verifiers.values()
        )
        static_verifier = StaticVerifier()

        def sampled_for_sandbox(task_id: str) -> bool:
            if not uses_sandbox_verifier:
                return False
            if sandbox_sample_rate <= 0:
                return True
            digest = int(content_digest({
                "sandbox_sample": task_id,
                "seed": self.config.generation.search.random_seed,
            })[:8], 16)
            return digest / 0xFFFFFFFF < sandbox_sample_rate

        def judged_report(
            generated: GeneratedTask,
            environment: EnvironmentContext,
            validator: TaskValidator,
        ) -> tuple[Any, dict[str, Any] | None]:
            """Run static validation plus LLM alignment/solvability judges."""
            report = validator.validate(generated)
            errors = list(report.errors)
            warnings = list(report.warnings)
            checks = dict(report.checks)
            alignment_errors, alignment_warnings = instruction_alignment_report(
                generated,
                generator,
                judgements=self.config.validation.alignment_judgements,
            )
            errors.extend(alignment_errors)
            warnings.extend(alignment_warnings)
            checks["instruction_alignment"] = not alignment_errors
            solve_verdict: dict[str, Any] | None = None
            if self.config.validation.solve_check_enabled:
                try:
                    solve_verdict = generator.solve_check(generated, environment)
                except Exception as exc:  # noqa: BLE001 - backend specific
                    warnings.append(
                        f"solvability judgement failed: {type(exc).__name__}: {exc}"
                    )
                if solve_verdict is not None:
                    checks["solve_check"] = bool(solve_verdict.get("solvable"))
                    blockers = list(solve_verdict.get("blockers") or [])
                    if not solve_verdict.get("solvable") and not blockers:
                        blockers.append("solve checker marked the task unsolvable")
                    for blocker in blockers:
                        errors.append(f"solvability blocker: {blocker}")
                    realism_errors = (
                        solve_plan_quality_errors(solve_verdict.get("plan") or [])
                        if self.config.validation.require_gui_realism
                        else []
                    )
                    checks["solve_plan_gui_realism"] = not realism_errors
                    errors.extend(realism_errors)
            return ValidationReport(
                task_id=report.task_id,
                valid=not errors,
                checks=checks,
                errors=tuple(errors),
                warnings=tuple(warnings),
                sandbox=report.sandbox,
            ), solve_verdict

        def validate_job(
            job: tuple[GeneratedTask, EnvironmentContext],
        ) -> dict[str, Any]:
            generated, environment = job
            run_sandbox = sampled_for_sandbox(generated.task.id)
            lock = (
                environment_locks[environment.image_id]
                if run_sandbox
                else nullcontext()
            )
            with lock:
                validator = TaskValidator(
                    environment,
                    verifiers[environment.image_id]
                    if run_sandbox or not uses_sandbox_verifier
                    else static_verifier,
                    require_sandbox=self.config.validation.require_sandbox
                    and (run_sandbox or not uses_sandbox_verifier),
                    max_instruction_chars=self.config.validation.max_instruction_chars,
                    max_instruction_words=self.config.validation.max_instruction_words,
                    require_platform_consistency=(
                        self.config.validation.require_platform_consistency
                    ),
                    require_supported_evaluator=(
                        self.config.validation.require_supported_evaluator
                    ),
                    require_gui_realism=self.config.validation.require_gui_realism,
                )
                current = generated
                report, solve_verdict = judged_report(current, environment, validator)
                initially_valid = report.valid
                repairs = 0
                seen_error_sets: set[tuple[str, ...]] = set()
                previous_error_count = len(report.errors)
                reject_threshold = self.config.validation.repair_reject_below_total
                while not report.valid and repairs < self.config.validation.max_repairs:
                    error_set = tuple(sorted(report.errors))
                    if error_set in seen_error_sets:
                        break
                    seen_error_sets.add(error_set)
                    if reject_threshold > 0:
                        preview = score_candidate(
                            current,
                            environment,
                            report,
                            minimum_total_score=0.0,
                            minimum_grounding=0.0,
                            minimum_alignment=0.0,
                            minimum_solvability=0.0,
                            weights=self.config.validation.scoring.weights,
                        )
                        if preview.total_score < reject_threshold:
                            break
                    repaired = generator.repair(current, report.to_dict(), environment)
                    if repaired.to_dict() == current.to_dict():
                        break
                    current = repaired
                    repairs += 1
                    candidate_dir = candidates_root / current.task.id
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(
                        candidate_dir / "generated_task.json",
                        current.to_dict(),
                    )
                    report, solve_verdict = judged_report(
                        current,
                        environment,
                        validator,
                    )
                    if len(report.errors) >= previous_error_count and not report.valid:
                        break
                    previous_error_count = len(report.errors)
                scoring = self.config.validation.scoring
                report_value = report.to_dict()
                if solve_verdict is not None:
                    report_value["solve_check"] = solve_verdict
                score = score_candidate(
                    current,
                    environment,
                    report_value,
                    minimum_total_score=(
                        scoring.minimum_total_score if scoring.enabled else 0.0
                    ),
                    minimum_grounding=(
                        scoring.minimum_grounding if scoring.enabled else 0.0
                    ),
                    minimum_alignment=(
                        scoring.minimum_alignment if scoring.enabled else 0.0
                    ),
                    minimum_solvability=(
                        scoring.minimum_solvability if scoring.enabled else 0.0
                    ),
                    weights=scoring.weights,
                )
                atomic_write_json(
                    quality_root / f"{current.task.id}.json",
                    score.to_dict(),
                )
                report_value = report.to_dict()
                if solve_verdict is not None:
                    report_value["solve_check"] = solve_verdict
                final_valid = report.valid and score.eligible
                report_value["valid"] = final_valid
                report_value["checks"]["quality_score"] = score.eligible
                report_value["quality_score"] = score.to_dict()
                if score.rejection_reasons:
                    report_value["errors"] = [
                        *report_value["errors"],
                        *score.rejection_reasons,
                    ]
                report_value["repair_attempts"] = repairs
                report_value["environment_id"] = environment.image_id
                report_value["environment_digest"] = environment.image_digest
                atomic_write_json(
                    validation_root / f"{current.task.id}.json",
                    report_value,
                )
                if final_valid:
                    stale_rejection = rejected_root / current.task.id
                    if stale_rejection.exists():
                        shutil.rmtree(stale_rejection)
                    if repairs:
                        stale_promotion = run_dir / "promoted" / current.task.id
                        if stale_promotion.exists():
                            shutil.rmtree(stale_promotion)
                else:
                    stale_promotion = run_dir / "promoted" / current.task.id
                    if stale_promotion.exists():
                        shutil.rmtree(stale_promotion)
                    source_dir = candidates_root / current.task.id
                    destination = rejected_root / current.task.id
                    if destination.exists():
                        shutil.rmtree(destination)
                    if source_dir.exists():
                        shutil.copytree(source_dir, destination)
                    atomic_write_json(destination / "validation.json", report_value)
                return {
                    "task_id": current.task.id,
                    "final_valid": final_valid,
                    "initially_valid": initially_valid,
                    "repairs": repairs,
                    "score": score,
                    "errors": tuple(report_value["errors"]),
                }

        validation_results = self._concurrent_map(
            validate_job,
            candidates,
            stage="validate",
            describe_item=lambda job: (
                f"task={job[0].task.id} environment={job[1].image_id}"
            ),
            result_status=lambda result: (
                "valid" if result["final_valid"] else "rejected"
            ),
            max_workers=(
                min(self.config.generation.concurrency, len(environments))
                if uses_sandbox_verifier
                else self.config.generation.concurrency
            ),
        )
        valid_count = sum(result["final_valid"] for result in validation_results)
        first_pass_valid = sum(
            result["final_valid"] and result["initially_valid"]
            for result in validation_results
        )
        repaired_valid = sum(
            result["final_valid"]
            and not result["initially_valid"]
            and result["repairs"] > 0
            for result in validation_results
        )
        total_repairs = sum(result["repairs"] for result in validation_results)
        validated_candidate_ids = {
            result["task_id"] for result in validation_results
        }
        quality_scores = [result["score"] for result in validation_results]
        rejection_reasons: Counter[str] = Counter()
        for result in validation_results:
            if not result["final_valid"]:
                rejection_reasons.update(result["errors"])
        promotion = promote_candidates(
            run_dir,
            candidate_ids=validated_candidate_ids,
            config=self.config.release,
        )
        dimension_totals: Counter[str] = Counter()
        for score in quality_scores:
            dimension_totals.update(score.dimensions)
        quality_summary = {
            "scored_candidates": len(quality_scores),
            "eligible_candidates": sum(score.eligible for score in quality_scores),
            "average_total_score": (
                sum(score.total_score for score in quality_scores) / len(quality_scores)
                if quality_scores
                else 0.0
            ),
            "average_dimensions": {
                name: total / len(quality_scores)
                for name, total in sorted(dimension_totals.items())
            } if quality_scores else {},
            "first_pass_valid": first_pass_valid,
            "repaired_valid": repaired_valid,
            "repair_attempts": total_repairs,
        }
        dataset_report = {
            "schema_version": "propagation.dataset_report.v1",
            "generated_candidates": len(candidates),
            "valid_candidates": valid_count,
            "promoted_candidates": len(promotion["promoted"]),
            "quality": quality_summary,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "promotion": promotion,
            "generator_usage": generator.usage_stats(),
        }
        atomic_write_json(run_dir / "dataset_report.json", dataset_report)
        return self._finish(
            run_dir,
            stop_after,
            sources=len(sources),
            environments=len(environments),
            environment_ids=[environment.image_id for environment in environments],
            compatible_source_environment_pairs=compatible_pairs,
            assigned_source_environment_pairs=len(assigned_pairs),
            assignments_per_environment=assignment_counts,
            concurrency=self.config.generation.concurrency,
            effective_variants_per_anchor=effective_variants_per_anchor,
            capabilities=len(primitive_records),
            recipes=len(recipe_records),
            candidates=len(candidates),
            generated_candidates=len(candidates) - reused_candidates,
            reused_candidates=reused_candidates,
            valid_candidates=valid_count,
            rejected_candidates=len(candidates) - valid_count,
            promoted_candidates=len(promotion["promoted"]),
            average_quality_score=quality_summary["average_total_score"],
            diversity_rejections=len(promotion.get("diversity_rejections") or {}),
        )

    def _concurrent_map(
        self,
        function: Callable[[_Input], _Output],
        items: list[_Input],
        *,
        stage: str,
        describe_item: Callable[[_Input], str],
        result_status: Callable[[_Output], str],
        on_result: Callable[[_Output], None] | None = None,
        max_workers: int | None = None,
    ) -> list[_Output]:
        configured_workers = max_workers or self.config.generation.concurrency
        workers = min(configured_workers, len(items))
        total = len(items)
        started = self._progress_stage_start(stage, total, workers=workers)
        counts: dict[str, int] = {}
        if not items:
            self._progress_stage_done(stage, started, counts)
            return []

        def execute(record_index: int, item: _Input) -> tuple[int, str, float, _Output]:
            label = describe_item(item)
            item_started = time.monotonic()
            self._progress_item_start(stage, record_index, total, label)
            return record_index, label, item_started, function(item)

        completed = 0
        results: dict[int, _Output] = {}
        if workers <= 1:
            completed_items = (
                execute(record_index, item)
                for record_index, item in enumerate(items, start=1)
            )
            for record_index, label, item_started, result in completed_items:
                if on_result is not None:
                    on_result(result)
                completed += 1
                results[record_index] = result
                self._progress_item_result(
                    stage,
                    completed,
                    total,
                    record_index,
                    result_status(result),
                    label,
                    counts,
                    started,
                    item_started,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(execute, record_index, item): record_index
                    for record_index, item in enumerate(items, start=1)
                }
                for future in as_completed(futures):
                    record_index, label, item_started, result = future.result()
                    if on_result is not None:
                        on_result(result)
                    completed += 1
                    results[record_index] = result
                    self._progress_item_result(
                        stage,
                        completed,
                        total,
                        record_index,
                        result_status(result),
                        label,
                        counts,
                        started,
                        item_started,
                    )
        self._progress_stage_done(stage, started, counts)
        return [results[index] for index in range(1, total + 1)]

    def _progress(self, message: str) -> None:
        with self._progress_lock:
            print(f"[task-generation] {message}", file=sys.stderr, flush=True)

    def _progress_stage_start(
        self,
        stage: str,
        total: int,
        *,
        workers: int,
    ) -> float:
        self._progress(f"{stage}: start total={total} workers={workers}")
        return time.monotonic()

    def _progress_item_start(
        self,
        stage: str,
        record_index: int,
        total: int,
        label: str,
    ) -> None:
        self._progress(f"[{record_index}/{total}] start stage={stage} {label}")

    def _progress_item_result(
        self,
        stage: str,
        completed: int,
        total: int,
        record_index: int,
        status: str,
        label: str,
        counts: dict[str, int],
        started: float,
        item_started: float,
    ) -> None:
        counts[status] = counts.get(status, 0) + 1
        elapsed = time.monotonic() - started
        item_elapsed = time.monotonic() - item_started
        throughput = completed / elapsed if elapsed > 0 else 0.0
        eta = (total - completed) / throughput if throughput > 0 else 0.0
        self._progress(
            f"[{completed}/{total} record={record_index} stage={stage} "
            f"throughput={throughput * 60:.2f}/min "
            f"task={_format_duration(item_elapsed)} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}] "
            f"{status} {label} counts={counts}"
        )

    def _progress_stage_done(
        self,
        stage: str,
        started: float,
        counts: dict[str, int],
    ) -> None:
        self._progress(
            f"{stage}: done elapsed={_format_duration(time.monotonic() - started)} "
            f"counts={counts}"
        )

    def _anchor_difficulty_targets(
        self,
        trajectory_feedback: dict[str, Any],
    ) -> dict[str, str]:
        bias = self.config.generation.difficulty_bias
        if not bias.enabled:
            return {}
        targets: dict[str, str] = {}
        for task_id, stats in (trajectory_feedback.get("tasks") or {}).items():
            success_rate = float(stats.get("success_rate") or 0.0)
            episodes = int(stats.get("episodes") or 0)
            if not episodes:
                continue
            if success_rate >= bias.harder_success_rate:
                # A single lucky success is too noisy to justify harder
                # variants; require at least two episodes for "harder".
                if episodes >= 2:
                    targets[task_id] = "harder"
            elif success_rate <= bias.easier_success_rate:
                # Extra scaffolding on a failing anchor is safe even when the
                # failure was noise, so "easier" is allowed at any sample size.
                targets[task_id] = "easier"
        return targets

    def _assign_sources_to_environments(
        self,
        sources: list[SourceTask],
        environments: list[EnvironmentContext],
        compatible_environment_ids: dict[str, list[str]],
        random_seed: int,
    ) -> dict[str, list[SourceTask]]:
        assignments = {environment.image_id: [] for environment in environments}
        if self.config.environment.assignment == "all_compatible":
            for source in sources:
                for environment_id in compatible_environment_ids[source.task.id]:
                    assignments[environment_id].append(source)
            return assignments

        rng = random.Random(random_seed)
        ordered_sources = list(sources)
        rng.shuffle(ordered_sources)
        assignment_counts = {environment.image_id: 0 for environment in environments}
        platform_counts: Counter[str] = Counter()
        environments_by_id = {
            environment.image_id: environment for environment in environments
        }
        for source in ordered_sources:
            available = list(compatible_environment_ids[source.task.id])
            selected_platforms: set[str] = set()
            target_count = min(
                self.config.environment.environments_per_task,
                len(available),
            )
            for _ in range(target_count):
                if self.config.environment.assignment == "platform_balanced":
                    distinct_platform_candidates = [
                        item
                        for item in available
                        if environment_platform(environments_by_id[item])
                        not in selected_platforms
                    ]
                    platform_pool = distinct_platform_candidates or available
                    platform_minimum = min(
                        platform_counts[
                            environment_platform(environments_by_id[item])
                        ]
                        for item in platform_pool
                    )
                    least_used_platforms = {
                        environment_platform(environments_by_id[item])
                        for item in platform_pool
                        if platform_counts[
                            environment_platform(environments_by_id[item])
                        ] == platform_minimum
                    }
                    platform_candidates = [
                        item
                        for item in platform_pool
                        if environment_platform(environments_by_id[item])
                        in least_used_platforms
                    ]
                else:
                    platform_candidates = available
                minimum = min(assignment_counts[item] for item in platform_candidates)
                least_used = [
                    item
                    for item in platform_candidates
                    if assignment_counts[item] == minimum
                ]
                selected = rng.choice(least_used)
                assignments[selected].append(source)
                assignment_counts[selected] += 1
                platform_counts[
                    environment_platform(environments_by_id[selected])
                ] += 1
                selected_platforms.add(
                    environment_platform(environments_by_id[selected])
                )
                available.remove(selected)
        return assignments

    def _environment_seed(self, run_dir: Path) -> int:
        if self.config.environment.random_seed is not None:
            return self.config.environment.random_seed
        return int(content_digest({
            "run_id": run_dir.name,
            "image_ids": self.config.environment.image_ids,
        })[:16], 16)

    def _source_compatibility(
        self,
        source: SourceTask,
        environment: EnvironmentContext,
    ) -> tuple[bool, list[str], bool]:
        required = set(source.task.required_capabilities)
        if source.application not in {"", "unknown"}:
            required.add(source.application)
        available = set(environment.capabilities)
        missing = sorted(required - available)
        target_match = self._target_environment_matches(
            source,
            environment,
        )
        portability_errors = []
        if self.config.environment.platform_policy == "primitive_compatible":
            portability_errors = source_platform_errors(source, environment)
        if portability_errors:
            missing.extend(f"platform:{error}" for error in portability_errors)
            target_match = False
        return not missing and target_match, missing, target_match

    def _target_environment_matches(
        self,
        source: SourceTask,
        environment: EnvironmentContext,
    ) -> bool:
        platform = str(source.task.metadata.get("platform") or "").casefold()
        os_type = str(environment.os.get("type") or "").casefold()
        if self.config.environment.platform_policy == "strict":
            if platform in {"linux", "ubuntu"} and os_type not in {"linux", "ubuntu"}:
                return False
            if platform in {"windows", "win32"} and os_type != "windows":
                return False
            if platform in {"darwin", "mac", "macos"} and os_type not in {
                "darwin",
                "mac",
                "macos",
            }:
                return False
        else:
            return True
        targets = source.task.target_environments
        if not targets:
            return True
        version = str(environment.os.get("version") or "")
        architecture = str(environment.os.get("architecture") or "")
        for target in targets:
            if target.os_type.casefold() != os_type:
                continue
            if target.os_versions and version not in target.os_versions:
                continue
            if target.architectures and architecture not in target.architectures:
                continue
            return True
        return False

    def _finish(self, run_dir: Path, stage: str, **stats: Any) -> dict[str, Any]:
        manifest = {
            "schema_version": "propagation.run.v1",
            "run_id": run_dir.name,
            "stage": stage,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": str(self.config.path),
            "stats": stats,
        }
        atomic_write_json(run_dir / "run_manifest.json", manifest)
        self._progress(f"run: complete stage={stage} run_dir={run_dir}")
        return {**manifest, "run_dir": str(run_dir)}

    @staticmethod
    def _prepare_run_outputs(run_dir: Path, *, fresh: bool) -> None:
        names = [
            "ability_primitives.jsonl",
            "environment_context.json",
            "environment_contexts.json",
            "extraction_failures.jsonl",
            "recipes.jsonl",
            "generation_failures.jsonl",
            "dataset_report.json",
            "run_manifest.json",
            "source_tasks.jsonl",
            "source_environment_matrix.jsonl",
            "trajectory_feedback.json",
        ]
        if fresh:
            names.extend(("candidates", "validation", "quality", "rejected", "promoted"))
        for name in names:
            path = run_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    @staticmethod
    def _load_existing_candidates(candidates_root: Path) -> dict[str, GeneratedTask]:
        existing: dict[str, GeneratedTask] = {}
        for path in sorted(candidates_root.rglob("generated_task.json")):
            generated = GeneratedTask.from_dict(read_json(path))
            existing[generated.recipe.id] = generated
        return existing

    @staticmethod
    def _episode_feedback(episodes: dict[str, list[Any]]) -> dict[str, Any]:
        tier_counts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        task_stats: dict[str, dict[str, Any]] = {}
        total_episodes = 0
        successful_episodes = 0
        total_steps = 0
        for task_id, task_episodes in sorted(episodes.items()):
            task_successes = sum(item.success for item in task_episodes)
            task_steps = sum(len(item.steps) for item in task_episodes)
            total_episodes += len(task_episodes)
            successful_episodes += task_successes
            total_steps += task_steps
            for episode in task_episodes:
                tier_counts[episode.training_tier] += 1
                for step in episode.steps:
                    tool_call = step.get("tool_call") or {}
                    action = step.get("action") or {}
                    name = str(
                        step.get("action_type")
                        or step.get("tool_name")
                        or tool_call.get("name")
                        or action.get("name")
                        or "unknown"
                    )
                    action_counts[name] += 1
            task_stats[task_id] = {
                "episodes": len(task_episodes),
                "success_rate": (
                    task_successes / len(task_episodes) if task_episodes else 0.0
                ),
                "average_steps": (
                    task_steps / len(task_episodes) if task_episodes else 0.0
                ),
                "training_tiers": dict(sorted(Counter(
                    item.training_tier for item in task_episodes
                ).items())),
            }
        return {
            "schema_version": "propagation.trajectory_feedback.v1",
            "episodes": total_episodes,
            "success_rate": (
                successful_episodes / total_episodes if total_episodes else 0.0
            ),
            "average_steps": total_steps / total_episodes if total_episodes else 0.0,
            "training_tiers": dict(sorted(tier_counts.items())),
            "action_distribution": dict(sorted(action_counts.items())),
            "tasks": task_stats,
        }
