"""Configuration loader for task-generation runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cuamaster.endpoints import load_endpoint_pool_config, resolve_endpoint_pool_path


@dataclass(frozen=True)
class SourceConfig:
    task_roots: tuple[str, ...]
    episode_roots: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    require_episodes: bool = False


@dataclass(frozen=True)
class EnvironmentConfig:
    image_store: str
    image_id: str = ""
    image_ids: tuple[str, ...] = ()
    selection: str = "fixed"
    sample_size: int = 1
    random_seed: int | None = None
    include_pending: bool = False
    primitive_overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)
    assignment: str = "all_compatible"
    environments_per_task: int = 1
    platform_policy: str = "strict"
    allowed_download_domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchConfig:
    candidate_pool_multiplier: int = 8
    beam_width: int = 64
    primitive_similarity_threshold: float = 0.88
    recipe_similarity_threshold: float = 0.85
    random_seed: int = 0
    strategies: tuple[str, ...] = (
        "sequential",
        "parameterized",
        "state_dependent",
        "cross_app_handoff",
    )


@dataclass(frozen=True)
class DifficultyBiasConfig:
    enabled: bool = True
    harder_success_rate: float = 0.8
    easier_success_rate: float = 0.2


@dataclass(frozen=True)
class GeneratorConfig:
    backend: str = "deterministic"
    model: str = "deterministic"
    command: tuple[str, ...] = ()
    base_url: str | None = None
    base_urls: tuple[str, ...] = ()
    endpoint_pool: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    wire_api: str = "chat_completions"
    store: bool = False
    trust_env: bool = True
    reasoning_effort: str = "none"
    max_tokens: int = 16384
    max_retries: int = 3
    request_delay: float = 0.0
    timeout_seconds: float = 7200.0
    concurrency: int = 1
    variants_per_anchor: int = 3
    min_primitives: int = 2
    max_primitives: int = 4
    allow_cross_app: bool = False
    allow_cross_source: bool = False
    cross_source_fraction: float = 0.0
    allow_local_fixtures: bool = True
    blueprint_first: bool = False
    difficulty_bias: DifficultyBiasConfig = field(default_factory=DifficultyBiasConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


@dataclass(frozen=True)
class ScoringConfig:
    enabled: bool = True
    minimum_total_score: float = 0.75
    minimum_grounding: float = 0.8
    minimum_alignment: float = 0.8
    minimum_solvability: float = 0.8
    weights: dict[str, float] = field(default_factory=lambda: {
        "grounding": 0.2,
        "alignment": 0.2,
        "clarity": 0.1,
        "novelty": 0.15,
        "difficulty": 0.1,
        "robustness": 0.1,
        "solvability": 0.1,
        "trajectory_value": 0.05,
    })


@dataclass(frozen=True)
class ValidationConfig:
    max_repairs: int = 2
    require_sandbox: bool = True
    verifier_command: tuple[str, ...] = ()
    reset_trials: int = 2
    solve_attempts: int = 2
    sandbox_sample_rate: float = 0.0
    max_instruction_chars: int = 4000
    max_instruction_words: int = 700
    require_platform_consistency: bool = True
    require_supported_evaluator: bool = True
    require_gui_realism: bool = True
    alignment_judgements: int = 1
    solve_check_enabled: bool = False
    repair_reject_below_total: float = 0.0
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@dataclass(frozen=True)
class ReleaseConfig:
    diversity_enabled: bool = False
    semantic_similarity_threshold: float = 0.85
    target_promoted_tasks: int = 0
    min_promotions_per_anchor: int = 0
    max_promotions_per_anchor: int = 0
    max_source_fraction: float = 0.02
    max_primitive_fraction: float = 0.05
    max_application_fraction: float = 1.0
    max_strategy_fraction: float = 1.0
    dedup_against: tuple[str, ...] = ()
    difficulty_targets: dict[str, float] = field(default_factory=lambda: {
        "easy": 0.2,
        "medium": 0.5,
        "hard": 0.3,
    })
    platform_targets: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskGenerationConfig:
    path: Path
    sources: SourceConfig
    environment: EnvironmentConfig
    generation: GeneratorConfig = field(default_factory=GeneratorConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)
    output_root: str = "propagation_runs"


def _load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML configs require PyYAML or use JSON") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("task-generation config must be an object")
    return value


def load_config(path: str | Path) -> TaskGenerationConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _load_raw(config_path)
    if raw.get("schema_version") != 1:
        raise ValueError("task-generation config schema_version must be 1")
    sources = raw.get("sources") or {}
    environment = raw.get("environment") or {}
    generation = raw.get("generation") or {}
    validation = raw.get("validation") or {}
    release = raw.get("release") or {}
    search = generation.get("search") or {}
    difficulty_bias = generation.get("difficulty_bias") or {}
    scoring = validation.get("scoring") or {}
    task_roots = tuple(str(item) for item in sources.get("tasks") or [])
    if not task_roots:
        raise ValueError("sources.tasks must contain at least one task root")
    image_store = str(environment.get("image_store") or "")
    image_id = str(environment.get("image_id") or "")
    image_ids = tuple(str(item) for item in environment.get("image_ids") or [])
    if image_id and not image_ids:
        image_ids = (image_id,)
    selection = str(environment.get("selection") or "fixed").lower()
    sample_size = int(environment.get("sample_size", 1))
    random_seed_raw = environment.get("random_seed")
    random_seed = int(random_seed_raw) if random_seed_raw is not None else None
    primitive_overrides_raw = environment.get("primitive_overrides") or {}
    if not isinstance(primitive_overrides_raw, dict):
        raise ValueError("environment.primitive_overrides must be an object")
    primitive_overrides = {
        str(image): tuple(str(item) for item in capabilities or [])
        for image, capabilities in primitive_overrides_raw.items()
    }
    if not image_store or not image_ids:
        raise ValueError(
            "environment.image_store and environment.image_id or image_ids are required"
        )
    if selection not in {"fixed", "random", "all"}:
        raise ValueError("environment.selection must be fixed, random, or all")
    if sample_size < 1:
        raise ValueError("environment.sample_size must be at least 1")
    assignment = str(environment.get("assignment") or "all_compatible").lower()
    environments_per_task = int(environment.get("environments_per_task", 1))
    if assignment not in {"all_compatible", "balanced_random", "platform_balanced"}:
        raise ValueError(
            "environment.assignment must be all_compatible, balanced_random, "
            "or platform_balanced"
        )
    if environments_per_task < 1:
        raise ValueError("environment.environments_per_task must be at least 1")
    platform_policy = str(environment.get("platform_policy") or "strict").lower()
    if platform_policy not in {"strict", "primitive_compatible"}:
        raise ValueError(
            "environment.platform_policy must be strict or primitive_compatible"
        )
    max_instruction_chars = int(validation.get("max_instruction_chars", 4000))
    max_instruction_words = int(validation.get("max_instruction_words", 700))
    if max_instruction_chars < 200 or max_instruction_words < 20:
        raise ValueError("validation instruction limits are too small")
    endpoint_pool_value = generation.get("endpoint_pool")
    if endpoint_pool_value and (
        generation.get("base_url") or generation.get("base_urls")
    ):
        raise ValueError(
            "generation.endpoint_pool cannot be combined with generation.base_url or generation.base_urls"
        )
    endpoint_pool = None
    endpoint_pool_config = None
    if endpoint_pool_value:
        endpoint_pool = str(
            resolve_endpoint_pool_path(endpoint_pool_value, config_path.parent)
        )
        endpoint_pool_config = load_endpoint_pool_config(endpoint_pool)
    generator = GeneratorConfig(
        backend=str(generation.get("backend") or "deterministic"),
        model=str(
            generation.get("model")
            or (endpoint_pool_config.model if endpoint_pool_config else None)
            or "deterministic"
        ),
        command=tuple(str(item) for item in generation.get("command") or []),
        base_url=str(generation["base_url"]) if generation.get("base_url") else None,
        base_urls=(
            tuple(str(item) for item in generation.get("base_urls") or [])
            if endpoint_pool_config is None
            else endpoint_pool_config.base_urls
        ),
        endpoint_pool=endpoint_pool,
        api_key_env=str(
            generation.get("api_key_env")
            or (endpoint_pool_config.api_key_env if endpoint_pool_config else None)
            or "OPENAI_API_KEY"
        ),
        wire_api=str(
            generation.get("wire_api")
            or (endpoint_pool_config.wire_api if endpoint_pool_config else None)
            or "chat_completions"
        ),
        store=bool(
            generation.get(
                "store",
                endpoint_pool_config.store if endpoint_pool_config else False,
            )
        ),
        trust_env=bool(
            generation.get(
                "trust_env",
                endpoint_pool_config.trust_env if endpoint_pool_config else True,
            )
        ),
        reasoning_effort=str(
            generation.get("reasoning_effort")
            or (endpoint_pool_config.reasoning_effort if endpoint_pool_config else None)
            or "none"
        ),
        max_tokens=int(generation.get("max_tokens", 16384)),
        max_retries=int(generation.get("max_retries", 3)),
        request_delay=float(generation.get("request_delay", 0.0)),
        timeout_seconds=float(generation.get("timeout_seconds", 7200.0)),
        concurrency=int(generation.get("concurrency", 1)),
        variants_per_anchor=int(generation.get("variants_per_anchor", 3)),
        min_primitives=int(generation.get("min_primitives", 2)),
        max_primitives=int(generation.get("max_primitives", 4)),
        allow_cross_app=bool(generation.get("allow_cross_app", False)),
        allow_cross_source=bool(generation.get("allow_cross_source", False)),
        cross_source_fraction=float(generation.get("cross_source_fraction", 0.0)),
        allow_local_fixtures=bool(generation.get("allow_local_fixtures", True)),
        blueprint_first=bool(generation.get("blueprint_first", False)),
        difficulty_bias=DifficultyBiasConfig(
            enabled=bool(difficulty_bias.get("enabled", True)),
            harder_success_rate=float(difficulty_bias.get("harder_success_rate", 0.8)),
            easier_success_rate=float(difficulty_bias.get("easier_success_rate", 0.2)),
        ),
        search=SearchConfig(
            candidate_pool_multiplier=int(search.get("candidate_pool_multiplier", 8)),
            beam_width=int(search.get("beam_width", 64)),
            primitive_similarity_threshold=float(
                search.get("primitive_similarity_threshold", 0.88)
            ),
            recipe_similarity_threshold=float(
                search.get("recipe_similarity_threshold", 0.85)
            ),
            random_seed=int(search.get("random_seed", 0)),
            strategies=tuple(
                str(item)
                for item in search.get("strategies")
                or (
                    "sequential",
                    "parameterized",
                    "state_dependent",
                    "cross_app_handoff",
                )
            ),
        ),
    )
    if not 1 <= generator.min_primitives <= generator.max_primitives:
        raise ValueError("generation capability limits are invalid")
    if generator.variants_per_anchor < 1:
        raise ValueError("generation.variants_per_anchor must be at least 1")
    if not 0 <= generator.cross_source_fraction <= 1:
        raise ValueError("generation.cross_source_fraction must be 0..1")
    if not (
        0 <= generator.difficulty_bias.easier_success_rate
        <= generator.difficulty_bias.harder_success_rate
        <= 1
    ):
        raise ValueError(
            "generation.difficulty_bias success-rate thresholds must satisfy "
            "0 <= easier <= harder <= 1"
        )
    if generator.concurrency < 1:
        raise ValueError("generation.concurrency must be at least 1")
    if generator.max_tokens < 1 or generator.max_retries < 1 or generator.timeout_seconds <= 0:
        raise ValueError("generation API limits must be positive")
    if generator.search.candidate_pool_multiplier < 1 or generator.search.beam_width < 1:
        raise ValueError("generation.search limits must be positive")
    if not 0 <= generator.search.primitive_similarity_threshold <= 1:
        raise ValueError("generation.search capability similarity threshold must be 0..1")
    if not 0 <= generator.search.recipe_similarity_threshold <= 1:
        raise ValueError("generation.search recipe similarity threshold must be 0..1")
    allowed_strategies = {
        "sequential",
        "parameterized",
        "state_dependent",
        "cross_app_handoff",
    }
    unknown_strategies = set(generator.search.strategies) - allowed_strategies
    if unknown_strategies:
        raise ValueError(
            "generation.search contains unknown strategies: "
            + ", ".join(sorted(unknown_strategies))
        )
    scoring_config = ScoringConfig(
        enabled=bool(scoring.get("enabled", True)),
        minimum_total_score=float(scoring.get("minimum_total_score", 0.75)),
        minimum_grounding=float(scoring.get("minimum_grounding", 0.8)),
        minimum_alignment=float(scoring.get("minimum_alignment", 0.8)),
        minimum_solvability=float(scoring.get("minimum_solvability", 0.8)),
        weights={
            str(key): float(value)
            for key, value in dict(scoring.get("weights") or ScoringConfig().weights).items()
        },
    )
    for name, value in (
        ("minimum_total_score", scoring_config.minimum_total_score),
        ("minimum_grounding", scoring_config.minimum_grounding),
        ("minimum_alignment", scoring_config.minimum_alignment),
        ("minimum_solvability", scoring_config.minimum_solvability),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"validation.scoring.{name} must be 0..1")
    if any(value < 0 for value in scoring_config.weights.values()):
        raise ValueError("validation.scoring weights must be non-negative")
    if not 0 <= float(validation.get("sandbox_sample_rate", 0.0)) <= 1:
        raise ValueError("validation.sandbox_sample_rate must be 0..1")
    if int(validation.get("alignment_judgements", 1)) < 1:
        raise ValueError("validation.alignment_judgements must be at least 1")
    if not 0 <= float(validation.get("repair_reject_below_total", 0.0)) <= 1:
        raise ValueError("validation.repair_reject_below_total must be 0..1")
    release_config = ReleaseConfig(
        diversity_enabled=bool(release.get("diversity_enabled", False)),
        semantic_similarity_threshold=float(
            release.get("semantic_similarity_threshold", 0.85)
        ),
        target_promoted_tasks=int(release.get("target_promoted_tasks", 0)),
        min_promotions_per_anchor=int(release.get("min_promotions_per_anchor", 0)),
        max_promotions_per_anchor=int(release.get("max_promotions_per_anchor", 0)),
        max_source_fraction=float(release.get("max_source_fraction", 0.02)),
        max_primitive_fraction=float(release.get("max_primitive_fraction", 0.05)),
        max_application_fraction=float(release.get("max_application_fraction", 1.0)),
        max_strategy_fraction=float(release.get("max_strategy_fraction", 1.0)),
        dedup_against=tuple(str(item) for item in release.get("dedup_against") or []),
        difficulty_targets={
            str(key): float(value)
            for key, value in dict(
                release.get("difficulty_targets")
                or {"easy": 0.2, "medium": 0.5, "hard": 0.3}
            ).items()
        },
        platform_targets={
            str(key): float(value)
            for key, value in dict(release.get("platform_targets") or {}).items()
        },
    )
    if release_config.target_promoted_tasks < 0:
        raise ValueError("release.target_promoted_tasks must be non-negative")
    if release_config.min_promotions_per_anchor < 0:
        raise ValueError("release.min_promotions_per_anchor must be non-negative")
    if release_config.max_promotions_per_anchor < 0:
        raise ValueError("release.max_promotions_per_anchor must be non-negative")
    if (
        release_config.max_promotions_per_anchor
        and release_config.min_promotions_per_anchor
        > release_config.max_promotions_per_anchor
    ):
        raise ValueError(
            "release.min_promotions_per_anchor cannot exceed max_promotions_per_anchor"
        )
    for name, value in (
        ("semantic_similarity_threshold", release_config.semantic_similarity_threshold),
        ("max_source_fraction", release_config.max_source_fraction),
        ("max_primitive_fraction", release_config.max_primitive_fraction),
        ("max_application_fraction", release_config.max_application_fraction),
        ("max_strategy_fraction", release_config.max_strategy_fraction),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"release.{name} must be 0..1")
    if any(value < 0 for value in release_config.difficulty_targets.values()):
        raise ValueError("release.difficulty_targets must be non-negative")
    if any(value < 0 for value in release_config.platform_targets.values()):
        raise ValueError("release.platform_targets must be non-negative")
    difficulty_total = sum(release_config.difficulty_targets.values())
    if release_config.diversity_enabled and not 0.999 <= difficulty_total <= 1.001:
        raise ValueError("release.difficulty_targets must sum to 1")
    platform_total = sum(release_config.platform_targets.values())
    if release_config.platform_targets and not 0.999 <= platform_total <= 1.001:
        raise ValueError("release.platform_targets must sum to 1")
    return TaskGenerationConfig(
        path=config_path,
        sources=SourceConfig(
            task_roots=task_roots,
            episode_roots=tuple(str(item) for item in sources.get("episodes") or []),
            task_ids=tuple(str(item) for item in sources.get("task_ids") or []),
            require_episodes=bool(sources.get("require_episodes", False)),
        ),
        environment=EnvironmentConfig(
            image_store=image_store,
            image_id=image_id,
            image_ids=image_ids,
            selection=selection,
            sample_size=sample_size,
            random_seed=random_seed,
            include_pending=bool(environment.get("include_pending", False)),
            primitive_overrides=primitive_overrides,
            assignment=assignment,
            environments_per_task=environments_per_task,
            platform_policy=platform_policy,
            allowed_download_domains=tuple(
                str(item) for item in environment.get("allowed_download_domains") or []
            ),
        ),
        generation=generator,
        validation=ValidationConfig(
            max_repairs=int(validation.get("max_repairs", 2)),
            require_sandbox=bool(validation.get("require_sandbox", True)),
            verifier_command=tuple(str(item) for item in validation.get("verifier_command") or []),
            reset_trials=int(validation.get("reset_trials", 2)),
            solve_attempts=int(validation.get("solve_attempts", 2)),
            sandbox_sample_rate=float(validation.get("sandbox_sample_rate", 0.0)),
            max_instruction_chars=max_instruction_chars,
            max_instruction_words=max_instruction_words,
            require_platform_consistency=bool(
                validation.get("require_platform_consistency", True)
            ),
            require_supported_evaluator=bool(
                validation.get("require_supported_evaluator", True)
            ),
            require_gui_realism=bool(validation.get("require_gui_realism", True)),
            alignment_judgements=int(validation.get("alignment_judgements", 1)),
            solve_check_enabled=bool(validation.get("solve_check_enabled", False)),
            repair_reject_below_total=float(
                validation.get("repair_reject_below_total", 0.0)
            ),
            scoring=scoring_config,
        ),
        release=release_config,
        output_root=str(raw.get("output_root") or "propagation_runs"),
    )
