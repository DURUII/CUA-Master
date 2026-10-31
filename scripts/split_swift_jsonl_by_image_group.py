#!/usr/bin/env python3
"""Split Swift JSONL by image-derived groups without trajectory leakage."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def group_from_row(row: dict[str, Any]) -> tuple[str, str]:
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("row must contain at least one image")
    parents = {str(Path(str(image)).parent) for image in images}
    if len(parents) != 1:
        raise ValueError("all images in a row must belong to the same trajectory")
    group_path = Path(parents.pop())
    return group_path.parent.name, group_path.name


def split_rows(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_from_row(row)].append(row)

    groups_by_app: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for group in grouped:
        groups_by_app[group[0]].append(group)

    rng = random.Random(seed)
    val_groups: set[tuple[str, str]] = set()
    for app, groups in sorted(groups_by_app.items()):
        rng.shuffle(groups)
        target_rows = max(1, round(sum(len(grouped[group]) for group in groups) * val_ratio))
        selected_rows = 0
        for group in groups:
            if selected_rows >= target_rows and val_groups:
                break
            val_groups.add(group)
            selected_rows += len(grouped[group])

    train_rows = []
    val_rows = []
    train_groups: set[tuple[str, str]] = set()
    for row in rows:
        group = group_from_row(row)
        if group in val_groups:
            val_rows.append(row)
        else:
            train_rows.append(row)
            train_groups.add(group)

    if not train_rows or not val_rows:
        raise ValueError("split produced an empty train or validation dataset")
    overlap = train_groups & val_groups
    if overlap:
        raise AssertionError(f"group leakage detected: {sorted(overlap)[:3]}")

    stats = {
        "seed": seed,
        "requested_val_ratio": val_ratio,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "actual_val_ratio": len(val_rows) / len(rows),
        "groups": len(grouped),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "train_app_rows": dict(sorted(Counter(group_from_row(row)[0] for row in train_rows).items())),
        "val_app_rows": dict(sorted(Counter(group_from_row(row)[0] for row in val_rows).items())),
        "val_group_ids": [group_id for _, group_id in sorted(val_groups)],
    }
    return train_rows, val_rows, stats


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input.expanduser().resolve())
    train_rows, val_rows, stats = split_rows(
        rows, val_ratio=args.val_ratio, seed=args.seed
    )
    train_output = args.train_output.expanduser().resolve()
    val_output = args.val_output.expanduser().resolve()
    write_jsonl(train_output, train_rows)
    write_jsonl(val_output, val_rows)
    stats.update(
        {
            "input": str(args.input.expanduser().resolve()),
            "train_output": str(train_output),
            "val_output": str(val_output),
        }
    )
    stats_path = train_output.with_suffix(".split_stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
