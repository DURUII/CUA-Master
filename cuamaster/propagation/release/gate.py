"""Quality- and diversity-aware promotion of validated task candidates."""
from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..config import ReleaseConfig
from ..diversity import generated_task_similarity, normalize_semantic_text
from ..platforms import platform_family
from ..schemas import CandidateQualityScore, GeneratedTask
from ..utils import atomic_write_json


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _quality_score(root: Path, task_id: str) -> CandidateQualityScore:
    path = root / "quality" / f"{task_id}.json"
    if path.is_file():
        return CandidateQualityScore.from_dict(_load_json(path))
    return CandidateQualityScore(
        task_id=task_id,
        total_score=1.0,
        dimensions={},
        hard_gate_passed=True,
        eligible=True,
    )


def _quota(total: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, math.ceil(total * fraction))


_MASK_TOKENS = {"<file>", "<number>", "<path>", "<datetime>", "<color>", "<measure>"}


def _load_prior_instructions(roots: tuple[str, ...] | list[str]) -> set[str]:
    """Normalized instructions promoted by earlier runs, for cross-run dedup."""
    seen: set[str] = set()
    for root in roots:
        promoted = Path(root) / "promoted"
        if not promoted.is_dir():
            promoted = Path(root)
        for path in promoted.rglob("generated_task.json"):
            try:
                task = _load_json(path).get("task") or {}
            except Exception:
                continue
            instruction = str(task.get("instruction") or "")
            if instruction:
                seen.add(normalize_semantic_text(instruction))
    return seen


def _select_diverse_candidates(
    records: list[tuple[Path, GeneratedTask, CandidateQualityScore]],
    config: ReleaseConfig,
) -> tuple[
    list[tuple[Path, GeneratedTask, CandidateQualityScore]],
    dict[str, str],
]:
    if not config.diversity_enabled:
        return records, {}
    target = config.target_promoted_tasks or len(records)
    source_limit = _quota(target, config.max_source_fraction)
    primitive_limit = _quota(target, config.max_primitive_fraction)
    application_limit = _quota(target, config.max_application_fraction)
    strategy_limit = _quota(target, config.max_strategy_fraction)
    difficulty_limits = {
        name: _quota(target, fraction)
        for name, fraction in config.difficulty_targets.items()
    }
    platform_limits = {
        platform_family(name): _quota(target, fraction)
        for name, fraction in config.platform_targets.items()
    }
    source_counts: Counter[str] = Counter()
    primitive_counts: Counter[str] = Counter()
    application_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()
    normalized_instructions: set[str] = _load_prior_instructions(config.dedup_against)
    selected: list[tuple[Path, GeneratedTask, CandidateQualityScore]] = []
    token_index: dict[str, set[int]] = defaultdict(set)
    rejected: dict[str, str] = {}

    def _try_select(
        record: tuple[Path, GeneratedTask, CandidateQualityScore],
        *,
        enforce_quotas: bool,
    ) -> str | None:
        candidate, generated, score = record
        normalized_instruction = normalize_semantic_text(generated.task.instruction)
        if normalized_instruction in normalized_instructions:
            return "exact normalized instruction duplicate"
        tokens = {
            token
            for token in normalized_instruction.split()
            if token not in _MASK_TOKENS
        }
        comparison_indices = set().union(
            *(token_index[token] for token in tokens)
        ) if tokens else set()
        for index in comparison_indices:
            existing = selected[index][1]
            similarity = generated_task_similarity(generated, existing)
            if similarity >= config.semantic_similarity_threshold:
                return (
                    f"near duplicate of {existing.task.id} "
                    f"with similarity {similarity:.3f}"
                )
        source_ids = generated.recipe.source_task_ids
        semantic_primitives = tuple(
            str(item)
            for item in generated.recipe.metadata.get("semantic_primitive_ids") or []
        ) or generated.recipe.primitive_ids
        application = score.coverage_bucket.get("application", "unknown")
        strategy = score.coverage_bucket.get("strategy", "unknown")
        difficulty = score.coverage_bucket.get("difficulty", "unknown")
        platform = platform_family(
            score.coverage_bucket.get("platform")
            or generated.task.metadata.get("platform")
        )
        anchor = generated.recipe.anchor_task_id or "unknown"
        if (
            config.max_promotions_per_anchor > 0
            and anchor_counts[anchor] >= config.max_promotions_per_anchor
        ):
            return "source task promotion maximum reached"
        platform_limit = platform_limits.get(platform)
        if platform_limit is not None and platform_counts[platform] >= platform_limit:
            return f"platform quota exceeded: {platform}"
        if enforce_quotas:
            if any(
                source_counts[source_id] >= source_limit for source_id in source_ids
            ):
                return "source task promotion quota exceeded"
            if any(
                primitive_counts[primitive_id] >= primitive_limit
                for primitive_id in semantic_primitives
            ):
                return "semantic capability promotion quota exceeded"
            difficulty_limit = difficulty_limits.get(difficulty)
            if (
                difficulty_limit is not None
                and difficulty_counts[difficulty] >= difficulty_limit
            ):
                return f"difficulty quota exceeded: {difficulty}"
            if application_counts[application] >= application_limit:
                return f"application quota exceeded: {application}"
            if strategy_counts[strategy] >= strategy_limit:
                return f"strategy quota exceeded: {strategy}"
        selected.append(record)
        normalized_instructions.add(normalized_instruction)
        selected_index = len(selected) - 1
        for token in tokens:
            token_index[token].add(selected_index)
        source_counts.update(source_ids)
        primitive_counts.update(semantic_primitives)
        application_counts[application] += 1
        strategy_counts[strategy] += 1
        difficulty_counts[difficulty] += 1
        platform_counts[platform] += 1
        anchor_counts[anchor] += 1
        return None

    def _platform_saturation(
        record: tuple[Path, GeneratedTask, CandidateQualityScore],
    ) -> float:
        if not platform_limits:
            return 0.0
        _, generated, score = record
        platform = platform_family(
            score.coverage_bucket.get("platform")
            or generated.task.metadata.get("platform")
        )
        limit = platform_limits.get(platform)
        if limit is None:
            return 1.0
        if limit == 0:
            return float("inf")
        return platform_counts[platform] / limit

    selected_ids: set[str] = set()
    # Pass 1: guarantee a per-anchor floor before global quotas apply so that
    # coverage of source tasks is never wiped out by difficulty or source
    # quotas. Only dedup checks are enforced here.
    if config.min_promotions_per_anchor > 0:
        anchor_groups: dict[str, list[
            tuple[Path, GeneratedTask, CandidateQualityScore]
        ]] = defaultdict(list)
        for record in records:
            anchor = record[1].recipe.anchor_task_id or "unknown"
            anchor_groups[anchor].append(record)
        for anchor in sorted(anchor_groups):
            group = list(anchor_groups[anchor])
            picked = 0
            while group and picked < config.min_promotions_per_anchor:
                group.sort(key=lambda item: (
                    _platform_saturation(item),
                    -item[2].total_score,
                    item[0].name,
                ))
                record = group.pop(0)
                reason = _try_select(record, enforce_quotas=False)
                if reason is None:
                    picked += 1
                    selected_ids.add(record[1].task.id)
                else:
                    rejected[record[1].task.id] = reason
    # Pass 2: fill the remaining budget with the existing quota-aware greedy
    # walk, with counters seeded by pass-1 selections.
    remaining = list(records)
    while remaining:
        remaining.sort(key=lambda item: (
            _platform_saturation(item),
            -item[2].total_score,
            item[0].name,
        ))
        record = remaining.pop(0)
        task_id = record[1].task.id
        if task_id in selected_ids or task_id in rejected:
            continue
        if len(selected) >= target:
            break
        reason = _try_select(record, enforce_quotas=True)
        if reason is None:
            selected_ids.add(task_id)
        else:
            rejected[task_id] = reason
    return selected, rejected


def promote_candidates(
    run_dir: str | Path,
    candidate_ids: set[str] | None = None,
    *,
    config: ReleaseConfig | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    release_config = config or ReleaseConfig()
    candidates_root = root / "candidates"
    candidate_dirs = [
        (path.parent, path.parent.relative_to(candidates_root).as_posix())
        for path in sorted(candidates_root.rglob("generated_task.json"))
        if candidate_ids is None
        or path.parent.relative_to(candidates_root).as_posix() in candidate_ids
    ]
    validated_candidates = [
        (candidate, task_id)
        for candidate, task_id in candidate_dirs
        if (root / "validation" / f"{task_id}.json").is_file()
        and _load_json(root / "validation" / f"{task_id}.json").get("valid")
        and _quality_score(root, task_id).eligible
    ]
    if not release_config.diversity_enabled:
        promoted_root = root / "promoted"
        promoted_root.mkdir(parents=True, exist_ok=True)
        valid_names = {task_id for _, task_id in validated_candidates}
        if candidate_ids is not None:
            for path in promoted_root.rglob("generated_task.json"):
                task_dir = path.parent
                task_id = task_dir.relative_to(promoted_root).as_posix()
                if task_id not in valid_names:
                    shutil.rmtree(task_dir)
        scores = []
        for candidate, task_id in validated_candidates:
            destination = promoted_root / task_id
            if not destination.exists():
                shutil.copytree(candidate, destination)
            score = _quality_score(root, task_id)
            scores.append(score.total_score)
            if (root / "quality" / f"{task_id}.json").is_file():
                atomic_write_json(destination / "quality_score.json", score.to_dict())
        manifest = {
            "schema_version": "propagation.promotion.v2",
            "policy": "automatic_after_validation",
            "valid_candidates": len(validated_candidates),
            "promoted": [task_id for _, task_id in validated_candidates],
            "diversity_rejections": {},
            "coverage": {},
            "average_quality_score": sum(scores) / len(scores) if scores else 0.0,
        }
        atomic_write_json(promoted_root / "promotion_manifest.json", manifest)
        return manifest
    valid_records: list[tuple[Path, GeneratedTask, CandidateQualityScore]] = []
    for candidate, task_id in validated_candidates:
        generated = GeneratedTask.from_dict(_load_json(candidate / "generated_task.json"))
        score = _quality_score(root, task_id)
        if score.eligible:
            valid_records.append((candidate, generated, score))
    selected, diversity_rejections = _select_diverse_candidates(
        valid_records,
        release_config,
    )
    promoted_root = root / "promoted"
    promoted_root.mkdir(parents=True, exist_ok=True)
    selected_names = {
        candidate.relative_to(candidates_root).as_posix()
        for candidate, _, _ in selected
    }
    if candidate_ids is not None:
        for path in promoted_root.rglob("generated_task.json"):
            task_dir = path.parent
            task_id = task_dir.relative_to(promoted_root).as_posix()
            if task_id not in selected_names:
                shutil.rmtree(task_dir)
    for candidate, generated, score in selected:
        task_id = candidate.relative_to(candidates_root).as_posix()
        destination = promoted_root / task_id
        if not destination.exists():
            shutil.copytree(candidate, destination)
        atomic_write_json(destination / "quality_score.json", score.to_dict())
    application_counts = Counter(
        score.coverage_bucket.get("application", "unknown")
        for _, _, score in selected
    )
    strategy_counts = Counter(
        score.coverage_bucket.get("strategy", "unknown")
        for _, _, score in selected
    )
    difficulty_counts = Counter(
        score.coverage_bucket.get("difficulty", "unknown")
        for _, _, score in selected
    )
    platform_counts = Counter(
        platform_family(
            score.coverage_bucket.get("platform")
            or generated.task.metadata.get("platform")
        )
        for _, generated, score in selected
    )
    target = release_config.target_promoted_tasks or len(valid_records)
    difficulty_targets = {
        name: math.ceil(target * fraction)
        for name, fraction in release_config.difficulty_targets.items()
    }
    platform_targets = {
        platform_family(name): math.ceil(target * fraction)
        for name, fraction in release_config.platform_targets.items()
    }
    manifest = {
        "schema_version": "propagation.promotion.v2",
        "policy": (
            "quality_and_diversity"
            if release_config.diversity_enabled
            else "automatic_after_quality_validation"
        ),
        "valid_candidates": len(valid_records),
        "promoted": [
            candidate.relative_to(candidates_root).as_posix()
            for candidate, _, _ in selected
        ],
        "diversity_rejections": diversity_rejections,
        "coverage": {
            "applications": dict(sorted(application_counts.items())),
            "strategies": dict(sorted(strategy_counts.items())),
            "difficulty": dict(sorted(difficulty_counts.items())),
            "platforms": dict(sorted(platform_counts.items())),
        },
        "average_quality_score": (
            sum(score.total_score for _, _, score in selected) / len(selected)
            if selected
            else 0.0
        ),
        "target_promoted_tasks": target,
        "coverage_gaps": {
            "total": max(0, target - len(selected)),
            "difficulty": {
                name: max(0, expected - difficulty_counts.get(name, 0))
                for name, expected in sorted(difficulty_targets.items())
            },
            "platforms": {
                name: max(0, expected - platform_counts.get(name, 0))
                for name, expected in sorted(platform_targets.items())
            },
        },
    }
    atomic_write_json(promoted_root / "promotion_manifest.json", manifest)
    return manifest
