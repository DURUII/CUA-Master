#!/usr/bin/env python3
"""Merge collected GUI, CLI, and finish trajectories for ms-swift SFT."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cuamaster.benchmarks import hybrid_system_prompt
from cuamaster.runtime import (
    ACTION_PROTOCOL_CANONICAL,
    DEFAULT_ACTION_PROTOCOL,
    SUPPORTED_ACTION_PROTOCOLS,
    format_qwen_xml_action,
)


DEFAULT_OUTPUT = Path("data/traj_collection_swift/traj_collection_mixed_qwen_xml_sft.jsonl")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield value


def resolve_images(dataset_root: Path, values: Any) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("images must be a list")
    images: list[str] = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = dataset_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append(str(path))
    return images


def validate_messages(messages: Any, images: list[str], context: str) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{context}: messages must be a non-empty list")
    if any(
        not isinstance(message, dict)
        or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError(f"{context}: invalid message schema")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{context}: final message must be assistant")
    placeholder_count = sum(message["content"].count("<image>") for message in messages)
    if placeholder_count != len(images):
        raise ValueError(
            f"{context}: {placeholder_count} image placeholders for {len(images)} images"
        )


def _protocol_messages(
    messages: list[dict[str, Any]],
    target_action: dict[str, Any],
    action_protocol: str,
) -> list[dict[str, Any]]:
    converted = [dict(message) for message in messages]
    converted[0]["content"] = hybrid_system_prompt(action_protocol)
    if action_protocol == ACTION_PROTOCOL_CANONICAL:
        converted[-1]["content"] = json.dumps(
            target_action, ensure_ascii=False, separators=(",", ":")
        )
        return converted
    for message in reversed(converted[:-1]):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        message["content"] = content.replace(
            "Return exactly one canonical action JSON object.",
            "Return the next single action using the Qwen XML tool-call format.",
        )
        break
    converted[-1]["content"] = format_qwen_xml_action(target_action)
    return converted


def convert_dataset_roots(
    dataset_roots: list[Path],
    output: Path,
    *,
    action_protocol: str = DEFAULT_ACTION_PROTOCOL,
) -> dict[str, Any]:
    if action_protocol not in SUPPORTED_ACTION_PROTOCOLS - {"auto"}:
        raise ValueError(f"unsupported action protocol: {action_protocol}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    action_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    run_counts: dict[str, int] = {}
    total_rows = 0
    total_image_references = 0

    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for dataset_root in dataset_roots:
                source_path = dataset_root / "trajectories.sft.jsonl"
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                run_rows = 0
                for line_no, source in enumerate(read_jsonl(source_path), 1):
                    context = f"{source_path}:{line_no}"
                    images = resolve_images(dataset_root, source.get("images"))
                    messages = source.get("messages")
                    validate_messages(messages, images, context)
                    target_action = source.get("target_action") or {}
                    if not isinstance(target_action, dict):
                        raise ValueError(f"{context}: target_action must be an object")
                    action_type = str(target_action.get("type") or "")
                    if not (
                        action_type.startswith("computer.")
                        or action_type in {"shell.exec", "finish"}
                    ):
                        raise ValueError(
                            f"{context}: unsupported GUI+CLI action type: {action_type}"
                        )
                    messages = _protocol_messages(
                        messages, target_action, action_protocol
                    )
                    modality = str(source.get("modality") or "unknown")
                    destination.write(
                        json.dumps(
                            {"messages": messages, "images": images},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    action_counts[action_type] += 1
                    modality_counts[modality] += 1
                    run_rows += 1
                    total_rows += 1
                    total_image_references += len(images)
                run_counts[dataset_root.parent.name] = run_rows
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "output": str(output),
        "dataset_roots": [str(path) for path in dataset_roots],
        "rows": total_rows,
        "action_protocol": action_protocol,
        "image_references": total_image_references,
        "runs": run_counts,
        "action_counts": dict(sorted(action_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
    }


def discover_dataset_roots(root: Path) -> list[Path]:
    roots = sorted(path / "dataset" for path in root.iterdir() if path.is_dir())
    return [path for path in roots if (path / "trajectories.sft.jsonl").is_file()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="collection output root whose run directories are scanned for datasets",
    )
    parser.add_argument("--dataset-roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--action-protocol",
        choices=tuple(sorted(SUPPORTED_ACTION_PROTOCOLS - {"auto"})),
        default=DEFAULT_ACTION_PROTOCOL,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    stats = convert_dataset_roots(
        dataset_roots, output, action_protocol=args.action_protocol
    )
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
