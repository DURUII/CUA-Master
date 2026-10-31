"""Command-line interface for task generation and taskset publishing."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .compilers import compile_osworld_taskset, compile_weavebench_taskset
from .config import TaskGenerationConfig, load_config
from .ingestion import load_source_tasks
from .pipeline import TaskGenerationPipeline
from .release import promote_candidates


_LEGACY_COMMANDS = {"index", "decompose", "compose", "validate", "run", "promote", "compile"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _resolved_paths(values: list[str]) -> tuple[str, ...]:
    return tuple(str(Path(value).expanduser().resolve()) for value in values)


def _apply_run_overrides(
    config: TaskGenerationConfig,
    args: argparse.Namespace,
) -> TaskGenerationConfig:
    sources = config.sources
    if args.input is not None:
        sources = replace(sources, task_roots=_resolved_paths(args.input))
    if args.trajectory_dir is not None:
        sources = replace(sources, episode_roots=_resolved_paths(args.trajectory_dir))
    if args.task_id is not None:
        sources = replace(sources, task_ids=tuple(args.task_id))
    if args.require_trajectories is not None:
        sources = replace(sources, require_episodes=args.require_trajectories)

    environment = config.environment
    if args.vm_images_dir is not None:
        environment = replace(
            environment,
            image_store=str(Path(args.vm_images_dir).expanduser().resolve()),
        )
    if args.image_id is not None:
        image_ids = tuple(args.image_id)
        environment = replace(
            environment,
            image_id=image_ids[0] if len(image_ids) == 1 else "",
            image_ids=image_ids,
        )
    if args.vm_selection is not None:
        environment = replace(environment, selection=args.vm_selection)
    if args.vm_sample_size is not None:
        environment = replace(environment, sample_size=args.vm_sample_size)
    if args.vm_seed is not None:
        environment = replace(environment, random_seed=args.vm_seed)
    if args.include_pending_vms is not None:
        environment = replace(environment, include_pending=args.include_pending_vms)
    if args.vm_assignment is not None:
        environment = replace(environment, assignment=args.vm_assignment)
    if args.vms_per_task is not None:
        environment = replace(environment, environments_per_task=args.vms_per_task)
    if args.platform_policy is not None:
        environment = replace(environment, platform_policy=args.platform_policy)

    generation_overrides = {}
    if args.concurrency is not None:
        generation_overrides["concurrency"] = args.concurrency
    if args.variants_per_anchor is not None:
        generation_overrides["variants_per_anchor"] = args.variants_per_anchor
    if args.allow_cross_source is not None:
        generation_overrides["allow_cross_source"] = args.allow_cross_source
    if args.cross_source_fraction is not None:
        generation_overrides["cross_source_fraction"] = args.cross_source_fraction
    generation = (
        replace(config.generation, **generation_overrides)
        if generation_overrides
        else config.generation
    )
    release = config.release
    if args.dedup_against is not None:
        release = replace(release, dedup_against=_resolved_paths(args.dedup_against))
    return replace(
        config,
        sources=sources,
        environment=environment,
        generation=generation,
        release=release,
    )


def _apply_generation_overrides(
    config: TaskGenerationConfig,
    args: argparse.Namespace,
) -> TaskGenerationConfig:
    return _apply_run_overrides(config, args)


def _add_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    stage: str | None = None,
) -> None:
    parser.add_argument(
        "--config",
        required=True,
        help="YAML/JSON defaults; explicit CLI flags override config values",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="source task root(s), overriding sources.tasks",
    )
    parser.add_argument(
        "--trajectory-dir",
        nargs="+",
        default=None,
        help="trajectory root(s), overriding sources.episodes",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="exact output directory for this run",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="clear existing generated artifacts instead of resuming this output directory",
    )
    parser.add_argument("--run-id", default=None, help="run directory name under output_root")
    parser.add_argument(
        "--vm-images-dir",
        default=None,
        help="VM image store, overriding environment.image_store",
    )
    parser.add_argument(
        "--image-id",
        nargs="+",
        default=None,
        help="VM image candidate ID(s), overriding environment.image_ids",
    )
    parser.add_argument(
        "--vm-selection",
        choices=("fixed", "random", "all"),
        default=None,
        help="VM selection policy, overriding environment.selection",
    )
    parser.add_argument(
        "--vm-sample-size",
        type=_positive_int,
        default=None,
        help="number of VM images sampled in random mode",
    )
    parser.add_argument(
        "--vm-seed",
        type=int,
        default=None,
        help="reproducible VM sampling seed; defaults to a run-derived seed",
    )
    parser.add_argument(
        "--include-pending-vms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include images with pending inventory when capability overrides are configured",
    )
    parser.add_argument(
        "--vm-assignment",
        choices=("all_compatible", "balanced_random", "platform_balanced"),
        default=None,
        help=(
            "assign every compatible VM, balance individual VMs, or prefer "
            "distinct OS platforms per task"
        ),
    )
    parser.add_argument(
        "--vms-per-task",
        type=_positive_int,
        default=None,
        help="number of compatible VMs assigned per task in balanced assignment modes",
    )
    parser.add_argument(
        "--platform-policy",
        choices=("strict", "primitive_compatible"),
        default=None,
        help="enforce source platform metadata or allow capability-compatible OS assignment",
    )
    parser.add_argument(
        "--task-id",
        nargs="+",
        default=None,
        help="only generate from these source task IDs",
    )
    parser.add_argument(
        "--require-trajectories",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="only use source tasks with matched trajectory episodes",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=None,
        help="number of worker threads for decomposition and task construction",
    )
    parser.add_argument(
        "--variants-per-anchor",
        type=_positive_int,
        default=None,
        help="override generation.variants_per_anchor for this run",
    )
    parser.add_argument(
        "--allow-cross-source",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override generation.allow_cross_source for this run",
    )
    parser.add_argument(
        "--cross-source-fraction",
        type=float,
        default=None,
        help="fraction of extra cross-source variants per anchor (0..1)",
    )
    parser.add_argument(
        "--dedup-against",
        nargs="+",
        default=None,
        help="earlier run directories whose promoted instructions are excluded",
    )
    if stage is None:
        parser.add_argument(
            "--stage",
            choices=("index", "decompose", "compose", "validate"),
            default="validate",
            help="last pipeline stage to execute",
        )
    else:
        parser.set_defaults(stage=stage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuamaster-synthesize",
        description="Generate VM-aware GUI tasks from source tasks and trajectories.",
    )
    _add_run_arguments(parser)
    parser.set_defaults(command="run")
    return parser


def _build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cuamaster-synthesize")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, stage in (
        ("index", "index"),
        ("decompose", "decompose"),
        ("compose", "compose"),
        ("validate", "validate"),
        ("run", "validate"),
    ):
        _add_run_arguments(subparsers.add_parser(name), stage=stage)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--run-dir", required=True)
    promote.add_argument(
        "--config",
        help="optional task-generation config for quality/diversity release policy",
    )
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--config", required=True)
    compile_parser.add_argument("--run-dir", required=True)
    compile_parser.add_argument("--format", choices=("weavebench", "osworld"), required=True)
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument(
        "--pool", choices=("promoted", "candidates"), default="promoted"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    tokens = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_legacy_parser() if tokens and tokens[0] in _LEGACY_COMMANDS else build_parser()
    return parser.parse_args(tokens)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "promote":
        release_config = load_config(args.config).release if args.config else None
        result = promote_candidates(args.run_dir, config=release_config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    config = load_config(args.config)
    if args.command in {"index", "decompose", "compose", "validate", "run"}:
        config = _apply_run_overrides(config, args)
        result = TaskGenerationPipeline(config).run(
            stop_after=args.stage,
            run_id=args.run_id,
            output_dir=args.output_dir,
            fresh=args.fresh,
        )
    else:
        generated_root = Path(args.run_dir) / args.pool
        if args.format == "weavebench":
            base = config.path.parent
            source_roots = [
                (base / value).resolve() if not Path(value).is_absolute() else Path(value)
                for value in config.sources.task_roots
            ]
            result = compile_weavebench_taskset(
                generated_root,
                args.output,
                sources=load_source_tasks(source_roots),
            )
        else:
            result = compile_osworld_taskset(generated_root, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
