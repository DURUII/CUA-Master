#!/usr/bin/env python3
"""Build ms-swift datasets from exported OSWorld trajectory bundles."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cuamaster.runtime import (
    QWEN_XML_SYSTEM_PROMPT,
    action_to_qwen_xml_calls,
    format_qwen_xml_action,
    qwen_coordinate as _qwen_coordinate,
)
from scripts.scroll_sanitization import (
    DEFAULT_MAX_SCROLL_DELTA,
    constrain_qwen_scroll_tool_schema,
    sanitize_scroll_action,
)


DEFAULT_OUTPUT_DIR = Path("data/osworld_swift")
DEFAULT_SOURCE_NAME = "osworld_trajectories"
COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."


def load_osworld_system_prompt(osworld_root: Path) -> str:
    """Load the Qwen system prompt from a local OSWorld checkout."""
    prompt_path = osworld_root / "mm_agents" / "qwen" / "prompts.py"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"OSWorld Qwen prompt module not found: {prompt_path}")
    spec = importlib.util.spec_from_file_location("osworld_qwen_prompts", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load OSWorld prompt module: {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools = module.build_internal_tools_def(1920, 1080, "relative")
    return constrain_qwen_scroll_tool_schema(
        module.build_internal_system_prompt(tools, COLLAPSED_SCREENSHOT_TEXT)
    )


def instruction_prompt(instruction: str, previous_actions: list[str]) -> str:
    previous = (
        "\n".join(f"Step {index + 1}: {action}" for index, action in enumerate(previous_actions))
        if previous_actions
        else "None"
    )
    return (
        "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n{previous}"
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def load_reference_system_prompt(path: Path) -> str:
    for row in read_jsonl(path):
        messages = row.get("messages")
        if not isinstance(messages, list):
            break
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
        break
    raise ValueError(f"reference JSONL has no system prompt: {path}")


def qwen_coordinate(value: Any) -> int:
    """Compatibility wrapper around the shared Qwen coordinate conversion."""

    return _qwen_coordinate(value)


def action_description(action: dict[str, Any]) -> str:
    action_type = str(action.get("type") or "")
    args = action.get("args") or {}
    if action_type == "computer.click":
        button = str(args.get("button") or "left")
        return f"{button.capitalize()}-click"
    if action_type == "computer.double_click":
        return "Double-click"
    if action_type == "computer.move":
        return "Move the mouse"
    if action_type == "computer.drag":
        return "Drag"
    if action_type == "computer.type":
        return f"Type: {str(args.get('text') or '')!r}"
    if action_type == "computer.press":
        keys = args.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        return "Press " + "+".join(str(key) for key in keys)
    if action_type == "computer.scroll":
        dy = float(args.get("dy") or 0)
        dx = float(args.get("dx") or 0)
        if dy:
            return "Scroll down" if dy > 0 else "Scroll up"
        return "Scroll right" if dx > 0 else "Scroll left"
    if action_type == "computer.wait":
        return f"Wait {float(args.get('seconds') or 1):g} seconds"
    if action_type == "finish":
        return f"Terminate with {str(args.get('status') or 'success')} status"
    raise ValueError(f"unsupported Qwen GUI action: {action_type}")


def action_to_qwen_calls(
    action: dict[str, Any],
    *,
    max_scroll_delta: float = DEFAULT_MAX_SCROLL_DELTA,
) -> list[str]:
    action, _ = sanitize_scroll_action(
        action, max_scroll_delta=max_scroll_delta
    )
    return action_to_qwen_xml_calls(
        action, max_scroll_delta=max_scroll_delta
    )


def format_qwen_assistant(action: dict[str, Any]) -> str:
    return format_qwen_xml_action(
        action,
        description=action_description(action),
    )


def _task_group(task: dict[str, Any]) -> str:
    source_task_ids = task.get("source_task_ids") or []
    if source_task_ids:
        return str(source_task_ids[0])
    return str(task.get("id") or "")


def _absolute_image(dataset_root: Path, screenshot: dict[str, Any]) -> Path:
    value = screenshot.get("path")
    if not value:
        raise ValueError("step screenshot has no path")
    path = Path(str(value))
    if not path.is_absolute():
        path = dataset_root / "canonical" / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_qwen_examples(
    dataset_root: Path,
    *,
    system_prompt: str,
    history_n: int,
    source_name: str = DEFAULT_SOURCE_NAME,
) -> tuple[list[dict[str, Any]], dict[str, str], Counter[str]]:
    system_prompt = constrain_qwen_scroll_tool_schema(system_prompt)
    episodes_path = dataset_root / "canonical" / "episodes.jsonl"
    examples: list[dict[str, Any]] = []
    episode_groups: dict[str, str] = {}
    action_counts: Counter[str] = Counter()

    for episode in read_jsonl(episodes_path):
        task = episode.get("task") or {}
        episode_id = str(episode.get("episode_id") or "")
        task_id = str(task.get("id") or "")
        group_id = _task_group(task)
        category = str((task.get("metadata") or {}).get("category") or "")
        instruction = str(task.get("instruction") or "").strip()
        source_task_ids = [str(value) for value in task.get("source_task_ids") or []]
        if not episode_id or not task_id or not instruction or not group_id:
            raise ValueError(f"incomplete canonical episode metadata: {episode_id or task_id}")
        episode_groups[episode_id] = group_id

        prepared: list[dict[str, Any]] = []
        for source_step in episode.get("steps") or []:
            action = source_step.get("action") or {}
            if action.get("type") == "finish":
                continue
            screenshot = (source_step.get("observation") or {}).get("screenshot")
            if not isinstance(screenshot, dict):
                raise ValueError(
                    f"visual action has no screenshot: {episode_id} step "
                    f"{source_step.get('step_index')}"
                )
            prepared.append(
                {
                    "source_step_index": int(source_step.get("step_index") or len(prepared) + 1),
                    "action": action,
                    "operation": action_description(action),
                    "assistant": format_qwen_assistant(action),
                    "image": _absolute_image(dataset_root, screenshot),
                }
            )
            action_counts[str(action.get("type") or "unknown")] += 1

        for target_index, target in enumerate(prepared):
            start_index = max(0, target_index - history_n)
            prior_summary = [step["operation"] for step in prepared[:start_index]]
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
            images: list[str] = []
            for index in range(start_index, target_index + 1):
                step = prepared[index]
                images.append(str(step["image"]))
                if index == start_index:
                    user_content = (
                        "<image>\n" + instruction_prompt(instruction, prior_summary)
                    )
                else:
                    user_content = "<tool_response>\n<image>\n</tool_response>"
                messages.append({"role": "user", "content": user_content})
                messages.append(
                    {
                        "role": "assistant",
                        "content": step["assistant"],
                        "loss": index == target_index,
                    }
                )
            examples.append(
                {
                    "messages": messages,
                    "images": images,
                    "task_id": task_id,
                    "step_id": f"{episode_id}:step-{target['source_step_index']:04d}",
                    "step_index": target_index,
                    "app": category,
                    "source": source_name,
                    "episode_id": episode_id,
                    "source_task_ids": source_task_ids,
                    "split_group_id": group_id,
                    "source_step_index": target["source_step_index"],
                }
            )
    return examples, episode_groups, action_counts


def split_examples_by_group(
    examples: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    groups = sorted({str(row["split_group_id"]) for row in examples})
    random.Random(seed).shuffle(groups)
    val_count = round(len(groups) * val_ratio)
    if val_ratio and groups:
        val_count = max(1, val_count)
    val_groups = set(groups[:val_count])
    train = [row for row in examples if row["split_group_id"] not in val_groups]
    val = [row for row in examples if row["split_group_id"] in val_groups]
    return train, val, val_groups


def build_hybrid_examples(
    dataset_root: Path, *, source_name: str = DEFAULT_SOURCE_NAME
) -> tuple[list[dict[str, Any]], Counter[str]]:
    source_path = dataset_root / "trajectories.sft.jsonl"
    rows: list[dict[str, Any]] = []
    modality_counts: Counter[str] = Counter()
    for source in read_jsonl(source_path):
        images = []
        for value in source.get("images") or []:
            path = Path(str(value))
            if not path.is_absolute():
                path = dataset_root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append(str(path))
        modality = str(source.get("modality") or "unknown")
        modality_counts[modality] += 1
        rows.append(
            {
                "messages": source.get("messages"),
                "images": images,
                "task_id": source.get("task_id"),
                "step_index": source.get("step_index"),
                "source": source_name,
                "episode_id": source.get("episode_id"),
                "source_task_ids": source.get("source_task_ids") or [],
                "split_group_id": source.get("split_group_id"),
                "modality": modality,
            }
        )
    return rows, modality_counts


def validate_swift_rows(rows: list[dict[str, Any]], *, require_loss_mask: bool) -> dict[str, int]:
    missing_images = 0
    placeholder_mismatches = 0
    invalid_messages = 0
    invalid_loss_masks = 0
    for row in rows:
        messages = row.get("messages")
        images = row.get("images")
        if not isinstance(messages, list) or not isinstance(images, list):
            invalid_messages += 1
            continue
        placeholder_count = sum(
            str(message.get("content") or "").count("<image>")
            for message in messages
            if isinstance(message, dict)
        )
        if placeholder_count != len(images):
            placeholder_mismatches += 1
        missing_images += sum(not Path(str(path)).is_file() for path in images)
        if require_loss_mask:
            assistant_messages = [
                message
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            if (
                not assistant_messages
                or assistant_messages[-1].get("loss") is not True
                or any(message.get("loss") is not False for message in assistant_messages[:-1])
            ):
                invalid_loss_masks += 1
    return {
        "rows": len(rows),
        "missing_images": missing_images,
        "placeholder_mismatches": placeholder_mismatches,
        "invalid_messages": invalid_messages,
        "invalid_loss_masks": invalid_loss_masks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="collected OSWorld trajectory dataset root (contains canonical/episodes.jsonl)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument(
        "--osworld-root",
        type=Path,
        help="local OSWorld checkout; required with --reuse-osworld-system-prompt",
    )
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        help="existing Swift JSONL whose system prompt is reused; "
        "required with --reuse-reference-system-prompt",
    )
    parser.add_argument("--reuse-reference-system-prompt", action="store_true")
    parser.add_argument(
        "--reuse-osworld-system-prompt",
        action="store_true",
        help="Compatibility override; the default is the fixed runtime Qwen XML prompt.",
    )
    parser.add_argument("--history-n", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history_n < 0:
        raise ValueError("history_n must be non-negative")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.reuse_reference_system_prompt and args.reuse_osworld_system_prompt:
        raise ValueError("choose at most one system-prompt compatibility override")
    if args.reuse_reference_system_prompt:
        if not args.reference_jsonl:
            raise ValueError("--reference-jsonl is required with --reuse-reference-system-prompt")
        reference_jsonl = args.reference_jsonl.expanduser().resolve()
        if not reference_jsonl.is_file():
            raise FileNotFoundError(reference_jsonl)
        system_prompt = load_reference_system_prompt(reference_jsonl)
        system_prompt_source = str(reference_jsonl)
    elif args.reuse_osworld_system_prompt:
        if not args.osworld_root:
            raise ValueError("--osworld-root is required with --reuse-osworld-system-prompt")
        system_prompt_source = str(args.osworld_root.expanduser().resolve())
        system_prompt = load_osworld_system_prompt(Path(system_prompt_source))
    else:
        system_prompt_source = "cuamaster.runtime.QWEN_XML_SYSTEM_PROMPT"
        system_prompt = QWEN_XML_SYSTEM_PROMPT

    qwen_rows, episode_groups, action_counts = build_qwen_examples(
        dataset_root,
        system_prompt=system_prompt,
        history_n=args.history_n,
        source_name=args.source_name,
    )
    train_rows, val_rows, val_groups = split_examples_by_group(
        qwen_rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    hybrid_rows, modality_counts = build_hybrid_examples(
        dataset_root, source_name=args.source_name
    )

    outputs = {
        "qwen_all": output_dir / f"{args.source_name}_qwen_gui_sft.jsonl",
        "qwen_train": output_dir / f"{args.source_name}_qwen_gui_sft_train.jsonl",
        "qwen_val": output_dir / f"{args.source_name}_qwen_gui_sft_val.jsonl",
        "hybrid": output_dir / f"{args.source_name}_hybrid_qwen_xml_sft.jsonl",
    }
    write_jsonl(outputs["qwen_all"], qwen_rows)
    write_jsonl(outputs["qwen_train"], train_rows)
    write_jsonl(outputs["qwen_val"], val_rows)
    write_jsonl(outputs["hybrid"], hybrid_rows)

    stats = {
        "schema_version": "osworld_trajectory_swift.build_stats.v1",
        "action_protocol": "qwen_xml",
        "dataset_root": str(dataset_root),
        "system_prompt_source": system_prompt_source,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "qwen_gui": {
            "episodes": len(episode_groups),
            "source_groups": len(set(episode_groups.values())),
            "examples": len(qwen_rows),
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "val_groups": sorted(val_groups),
            "history_n": args.history_n,
            "action_counts": dict(sorted(action_counts.items())),
            "validation": validate_swift_rows(qwen_rows, require_loss_mask=True),
        },
        "hybrid_qwen_xml": {
            "examples": len(hybrid_rows),
            "modality_counts": dict(sorted(modality_counts.items())),
            "validation": validate_swift_rows(hybrid_rows, require_loss_mask=False),
        },
    }
    stats_path = output_dir / f"{args.source_name}.stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
