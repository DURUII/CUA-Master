#!/usr/bin/env python3
"""Build ms-swift SFT data matching the canonical OSWorld runtime contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cuamaster.action_space import CUAAction, validate_action
from cuamaster.benchmarks import hybrid_system_prompt
from cuamaster.runtime import (
    ACTION_PROTOCOL_CANONICAL,
    DEFAULT_ACTION_PROTOCOL,
    SUPPORTED_ACTION_PROTOCOLS,
    CanonicalActionParser,
    QwenXmlActionParser,
    format_qwen_xml_action,
)
from cuamaster.runtime.core import _gui_command
from cuamaster.collection.dataset import (
    ImagePackager,
    _episode_content_flags,
    _policy_exclusion_reasons,
)
from scripts.scroll_sanitization import (
    DEFAULT_MAX_SCROLL_DELTA,
    sanitize_scroll_action,
)


DEFAULT_OUTPUT = Path(
    "data/traj_collection_swift/traj_collection_canonical_runtime_sft.jsonl"
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield value


def runtime_system_prompt(action_protocol: str) -> str:
    return hybrid_system_prompt(action_protocol)


def transform_action(
    action: dict[str, Any],
    *,
    max_scroll_delta: float = DEFAULT_MAX_SCROLL_DELTA,
) -> dict[str, Any]:
    if action.get("type") == "tool.call":
        name = str((action.get("args") or {}).get("name") or "unknown")
        raise ValueError(f"structured tool forbidden in GUI+CLI collection: {name}")
    transformed = json.loads(json.dumps(action))
    transformed, _ = sanitize_scroll_action(
        transformed, max_scroll_delta=max_scroll_delta
    )
    validate_action(transformed)
    return transformed


def resolve_screenshot(dataset_root: Path, step: dict[str, Any]) -> Path | None:
    screenshot = (step.get("observation") or {}).get("screenshot") or {}
    value = screenshot.get("path")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = dataset_root / "mixed" / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def unchanged_wait(step: dict[str, Any], action: dict[str, Any]) -> bool:
    if action.get("type") != "computer.wait":
        return False
    fingerprints = []
    for key in ("observation", "post_observation"):
        screenshot = (step.get(key) or {}).get("screenshot") or {}
        fingerprint = screenshot.get("sha256") or screenshot.get("path")
        fingerprints.append(str(fingerprint or ""))
    return bool(fingerprints[0] and fingerprints[0] == fingerprints[1])


def runtime_user_prompt(
    action_protocol: str,
    instruction: str,
    history: list[dict[str, Any]],
    *,
    has_image: bool,
) -> str:
    if action_protocol == ACTION_PROTOCOL_CANONICAL:
        recent = history[-12:]
        maximum = 16_000
        prompt_template = (
            "Task: {instruction}\n"
            "Previous canonical actions and observations: {history}\n"
            "The attached image is the latest screenshot. Return the next single action."
        )
    else:
        recent = history[-4:]
        maximum = 8_000
        prompt_template = (
            "<image>\n"
            "Please generate the next move according to the UI screenshot, instruction, "
            "and previous actions.\n\n"
            "Instruction: {instruction}\n\n"
            "Previous canonical actions and observations: {history}\n\n"
            "Return the next single action using the Qwen XML tool-call format."
        )
    history_text = json.dumps(recent, ensure_ascii=False, default=str)
    while recent and len(history_text) > maximum:
        recent = recent[1:]
        history_text = json.dumps(recent, ensure_ascii=False, default=str)
    prompt = prompt_template.format(
        instruction=instruction,
        history=history_text if recent else "None",
    )
    if has_image:
        return prompt
    prompt = prompt.removeprefix("<image>\n").replace(
        "according to the UI screenshot, instruction, and previous actions",
        "according to the instruction and previous actions",
        1,
    )
    return prompt.replace(
        "The attached image is the latest screenshot.",
        "No screenshot is attached for this non-visual action.",
        1,
    )


def _bounded_output(value: Any, maximum: int = 4000) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= maximum else text[:maximum] + "…<truncated>"


def normalize_execution(
    action: dict[str, Any],
    source_execution: Any,
    *,
    screenshot_available: bool,
) -> dict[str, Any]:
    execution = source_execution if isinstance(source_execution, dict) else {}
    ok = bool(execution.get("ok", True))
    action_type = action["type"]
    if action_type.startswith("computer."):
        observation = {
            "category": "gui",
            "command": _gui_command(CUAAction.from_dict(action), 1920, 1080),
            "screen_size": {"width": 1920, "height": 1080},
            "reward": 0,
            "done": False,
            "info": {},
            "screenshot_available": screenshot_available,
        }
    elif action_type == "shell.exec":
        args = action["args"]
        observation = {
            "category": "cli",
            "stdout": _bounded_output(execution.get("output") or execution.get("stdout") or ""),
            "stderr": _bounded_output(execution.get("error") or execution.get("stderr") or ""),
            "returncode": 0 if ok else 1,
            "timed_out": bool(execution.get("timed_out", False)),
            "cwd": str(args.get("cwd") or "/home/user/Desktop"),
            "timeout_sec": int(args.get("timeout_sec") or 30),
        }
    elif action_type == "finish":
        observation = {
            "category": "control",
            "status": action["args"]["status"],
            "answer": action["args"].get("answer"),
            "done": True,
        }
    else:
        raise ValueError(f"unsupported GUI+CLI action type: {action_type}")
    return {"ok": ok, "observation": observation}


def action_modality(action: dict[str, Any]) -> str:
    action_type = str(action.get("type") or "")
    if action_type.startswith("computer."):
        return "visual"
    if action_type == "shell.exec":
        return "shell"
    if action_type == "finish":
        return "control"
    raise ValueError(f"unsupported GUI+CLI action type: {action_type}")


def prepare_episode(
    dataset_root: Path,
    episode: dict[str, Any],
    *,
    max_target_chars: int,
    action_protocol: str = DEFAULT_ACTION_PROTOCOL,
    max_scroll_delta: float = DEFAULT_MAX_SCROLL_DELTA,
    include_empty_think: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    prepared: list[dict[str, Any]] = []
    for source_step_index, step in enumerate(episode.get("steps") or [], 1):
        source_action = step.get("action") or {}
        try:
            action = transform_action(
                source_action, max_scroll_delta=max_scroll_delta
            )
        except (TypeError, ValueError) as exc:
            return [], str(exc)
        if action is None:
            continue
        try:
            target = (
                json.dumps(
                    action,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if action_protocol == ACTION_PROTOCOL_CANONICAL
                else format_qwen_xml_action(
                    action,
                    include_empty_think=include_empty_think,
                    max_scroll_delta=max_scroll_delta,
                )
            )
        except (TypeError, ValueError) as exc:
            return [], f"qwen_xml_target_error:{exc}"
        if len(target) > max_target_chars:
            return [], f"target_too_long:{len(target)}"
        prepared.append(
            {
                "action": action,
                "target": target,
                "source_execution": step.get("execution") or {},
                "screenshot": resolve_screenshot(dataset_root, step),
                "source_step_index": source_step_index,
                "unchanged_wait": unchanged_wait(step, action),
                "scroll_clipped_axes": tuple(
                    axis
                    for axis in ("dx", "dy")
                    if source_action.get("type") == "computer.scroll"
                    and (source_action.get("args") or {}).get(axis)
                    != action["args"].get(axis)
                ),
            }
        )
    return prepared, None


def build_dataset(
    dataset_roots: list[Path],
    output: Path,
    *,
    max_target_chars: int,
    action_protocol: str = DEFAULT_ACTION_PROTOCOL,
    max_scroll_delta: float = DEFAULT_MAX_SCROLL_DELTA,
    include_empty_think: bool = False,
    drop_unchanged_waits: bool = True,
    copy_images: bool = True,
    quality_policy: str = "high_quality",
) -> dict[str, Any]:
    if action_protocol not in SUPPORTED_ACTION_PROTOCOLS - {"auto"}:
        raise ValueError(f"unsupported action protocol: {action_protocol}")
    if quality_policy not in {"structural", "strict", "high_quality"}:
        raise ValueError(f"unsupported quality policy: {quality_policy}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    system_prompt = runtime_system_prompt(action_protocol)
    parser = (
        CanonicalActionParser()
        if action_protocol == ACTION_PROTOCOL_CANONICAL
        else QwenXmlActionParser()
    )
    action_counts: Counter[str] = Counter()
    excluded_reasons: Counter[str] = Counter()
    skipped_without_screenshot: Counter[str] = Counter()
    run_counts: dict[str, dict[str, int]] = {}
    total_rows = 0
    total_episodes = 0
    included_episodes = 0
    clipped_scroll_actions = 0
    clipped_scroll_axes: Counter[str] = Counter()
    unchanged_waits_dropped = 0
    text_only_rows = 0
    quality_warning_counts: Counter[str] = Counter()
    assets_root = output.parent / f"{output.stem}.assets"
    image_packager = ImagePackager(assets_root, copy_images=True) if copy_images else None

    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for dataset_root in dataset_roots:
                source_path = dataset_root / "mixed" / "episodes.jsonl"
                run_episodes = 0
                run_rows = 0
                for episode in read_jsonl(source_path):
                    total_episodes += 1
                    instruction = str(episode.get("task") or "").strip()
                    if not instruction:
                        excluded_reasons["missing_instruction"] += 1
                        continue
                    blockers, warnings = _episode_content_flags(
                        episode.get("steps") or []
                    )
                    quality_warning_counts.update(warnings)
                    policy_reasons = _policy_exclusion_reasons(
                        blockers, warnings, quality_policy
                    )
                    if policy_reasons:
                        for reason in policy_reasons:
                            excluded_reasons[f"quality_policy:{reason}"] += 1
                        continue
                    prepared, exclusion = prepare_episode(
                        dataset_root,
                        episode,
                        max_target_chars=max_target_chars,
                        action_protocol=action_protocol,
                        max_scroll_delta=max_scroll_delta,
                        include_empty_think=include_empty_think,
                    )
                    if exclusion:
                        excluded_reasons[exclusion] += 1
                        continue
                    if not prepared:
                        excluded_reasons["no_supported_steps"] += 1
                        continue
                    included_episodes += 1
                    run_episodes += 1
                    for step in prepared:
                        axes = step["scroll_clipped_axes"]
                        if axes:
                            clipped_scroll_actions += 1
                            clipped_scroll_axes.update(axes)
                    history: list[dict[str, Any]] = []
                    latest_screenshot: Path | None = None
                    episode_rows = 0
                    for step_index, step in enumerate(prepared, 1):
                        if step["screenshot"] is not None:
                            latest_screenshot = step["screenshot"]
                        action = step["action"]
                        parser.parse(step["target"])
                        if drop_unchanged_waits and step["unchanged_wait"]:
                            unchanged_waits_dropped += 1
                        elif latest_screenshot is None and action["type"].startswith("computer."):
                            skipped_without_screenshot[action["type"]] += 1
                        else:
                            image_metadata = []
                            images = []
                            if latest_screenshot is not None:
                                if image_packager is None:
                                    images = [str(latest_screenshot)]
                                else:
                                    image = image_packager.package(latest_screenshot)
                                    image["path"] = (
                                        Path(assets_root.name) / image["path"]
                                    ).as_posix()
                                    image_metadata = [image]
                                    images = [image["path"]]
                            else:
                                text_only_rows += 1
                            user_prompt = runtime_user_prompt(
                                action_protocol,
                                instruction,
                                history,
                                has_image=bool(images),
                            )
                            source_step_index = step["source_step_index"]
                            sample_id = hashlib.sha256(
                                json.dumps(
                                    {
                                        "episode_id": episode.get("episode_id"),
                                        "step_index": source_step_index,
                                        "action": action,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()[:32]
                            destination.write(
                                json.dumps(
                                    {
                                        "schema_version": "canonical_runtime_sft.v2",
                                        "sample_id": sample_id,
                                        "messages": [
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_prompt},
                                            {"role": "assistant", "content": step["target"]},
                                        ],
                                        "images": images,
                                        "image_metadata": image_metadata,
                                        "episode_id": episode.get("episode_id"),
                                        "task_id": episode.get("task_id"),
                                        "source_task_ids": episode.get("source_task_ids") or [],
                                        "split_group_id": (
                                            episode.get("split_group_id")
                                            or episode.get("task_id")
                                        ),
                                        "step_index": source_step_index,
                                        "modality": action_modality(action),
                                        "action_protocol": action_protocol,
                                        "target_action": action,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            action_counts[action["type"]] += 1
                            total_rows += 1
                            run_rows += 1
                            episode_rows += 1
                        history.append(
                            {
                                "step": step_index,
                                "action": action,
                                "observation": normalize_execution(
                                    action,
                                    step["source_execution"],
                                    screenshot_available=latest_screenshot is not None,
                                ),
                            }
                        )
                    if not episode_rows:
                        included_episodes -= 1
                        run_episodes -= 1
                        excluded_reasons["no_exportable_rows"] += 1
                run_counts[dataset_root.parent.name] = {
                    "episodes": run_episodes,
                    "rows": run_rows,
                }
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "output": str(output),
        "dataset_roots": [str(path) for path in dataset_roots],
        "action_protocol": action_protocol,
        "episodes_seen": total_episodes,
        "episodes_included": included_episodes,
        "rows": total_rows,
        "max_target_chars": max_target_chars,
        "max_scroll_delta": max_scroll_delta,
        "include_empty_think": include_empty_think,
        "drop_unchanged_waits": drop_unchanged_waits,
        "unchanged_waits_dropped": unchanged_waits_dropped,
        "copy_images": copy_images,
        "quality_policy": quality_policy,
        "quality_warning_counts": dict(sorted(quality_warning_counts.items())),
        "image_assets_root": str(assets_root) if copy_images else None,
        "unique_images": len(image_packager.references) if image_packager else None,
        "images_copied": image_packager.images_copied if image_packager else None,
        "text_only_rows": text_only_rows,
        "clipped_scroll_actions": clipped_scroll_actions,
        "clipped_scroll_axes": dict(sorted(clipped_scroll_axes.items())),
        "runs": run_counts,
        "action_counts": dict(sorted(action_counts.items())),
        "excluded_episode_reasons": dict(sorted(excluded_reasons.items())),
        "skipped_targets_without_screenshot": dict(
            sorted(skipped_without_screenshot.items())
        ),
    }


def discover_dataset_roots(root: Path) -> list[Path]:
    roots = sorted(path / "dataset" for path in root.iterdir() if path.is_dir())
    return [path for path in roots if (path / "mixed" / "episodes.jsonl").is_file()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="collection output root whose run directories are scanned for datasets",
    )
    parser.add_argument("--dataset-roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-target-chars", type=int, default=4000)
    parser.add_argument(
        "--action-protocol",
        choices=tuple(sorted(SUPPORTED_ACTION_PROTOCOLS - {"auto"})),
        default=DEFAULT_ACTION_PROTOCOL,
    )
    parser.add_argument(
        "--max-scroll-delta", type=float, default=DEFAULT_MAX_SCROLL_DELTA
    )
    parser.add_argument(
        "--include-empty-think",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--drop-unchanged-waits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--quality-policy",
        choices=("structural", "strict", "high_quality"),
        default="high_quality",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_target_chars <= 0:
        raise ValueError("max-target-chars must be positive")
    if not 0 < args.max_scroll_delta < float("inf"):
        raise ValueError("max-scroll-delta must be finite and positive")
    if not args.dataset_roots and not args.root:
        raise ValueError("provide --dataset-roots or --root")
    dataset_roots = (
        [path.expanduser().resolve() for path in args.dataset_roots]
        if args.dataset_roots
        else discover_dataset_roots(args.root.expanduser().resolve())
    )
    if not dataset_roots:
        raise ValueError("no trajectory dataset roots found")
    output = args.output.expanduser().resolve()
    stats = build_dataset(
        dataset_roots,
        output,
        max_target_chars=args.max_target_chars,
        action_protocol=args.action_protocol,
        max_scroll_delta=args.max_scroll_delta,
        include_empty_think=args.include_empty_think,
        drop_unchanged_waits=args.drop_unchanged_waits,
        copy_images=args.copy_images,
        quality_policy=args.quality_policy,
    )
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
