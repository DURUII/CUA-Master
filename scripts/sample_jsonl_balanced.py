#!/usr/bin/env python3
"""Filter a JSONL file and write a balanced sample.

Rows can be filtered by arbitrary fields with repeated ``--where`` predicates.
The remaining rows are sampled across strata such as
``category/subcategory/application`` so the output is not dominated by a few
large apps.

Examples:
    python scripts/sample_jsonl_balanced.py \
        --input data/data_list_labeled_enriched.jsonl \
        --output data/sample_2000.jsonl \
        --sample-size 2000 \
        --where score ge 4 \
        --where is_cua eq true \
        --where has_trajectory eq true \
        --where platform in desktop,website

    python scripts/sample_jsonl_balanced.py \
        --input data/data_list_labeled_enriched.jsonl \
        --output data/mobile_sample.jsonl \
        --sample-size 500 \
        --balance-fields category subcategory application platform \
        --where platform eq mobile
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


Predicate = tuple[str, str, str]


def parse_scalar(raw: str) -> Any:
    text = raw.strip()
    lowered = text.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def field_value(row: dict[str, Any], field: str) -> Any:
    """Read a dotted field path such as ``quality.score`` from a row."""
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def compare_values(actual: Any, op: str, expected_raw: str) -> bool:
    op = op.casefold()
    if op == "exists":
        return actual is not None
    if op == "missing":
        return actual is None

    if op in {"in", "not-in"}:
        expected_values = [parse_scalar(part) for part in expected_raw.split(",")]
        matched = actual in expected_values
        return matched if op == "in" else not matched

    expected = parse_scalar(expected_raw)
    if op in {"eq", "=="}:
        return actual == expected
    if op in {"ne", "!="}:
        return actual != expected

    if actual is None or expected is None:
        return False
    if op in {"contains", "not-contains"}:
        matched = str(expected) in str(actual)
        return matched if op == "contains" else not matched

    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError):
        left = str(actual)
        right = str(expected)

    if op in {"gt", ">"}:
        return left > right
    if op in {"ge", ">="}:
        return left >= right
    if op in {"lt", "<"}:
        return left < right
    if op in {"le", "<="}:
        return left <= right
    raise ValueError(f"unsupported operator: {op}")


def row_matches(row: dict[str, Any], predicates: list[Predicate]) -> bool:
    for field, op, expected in predicates:
        if not compare_values(field_value(row, field), op, expected):
            return False
    return True


def load_filtered(path: Path, predicates: list[Predicate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            if row_matches(row, predicates):
                rows.append(row)
    return rows


def stratum_key(row: dict[str, Any], balance_fields: list[str]) -> tuple[Any, ...]:
    return tuple(field_value(row, field) for field in balance_fields)


def equal_balanced_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    balance_fields: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample by shuffled round-robin over strata for near-equal coverage."""
    if sample_size >= len(rows):
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled

    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[stratum_key(row, balance_fields)].append(row)
    for bucket in strata.values():
        rng.shuffle(bucket)

    keys = list(strata)
    rng.shuffle(keys)
    sampled: list[dict[str, Any]] = []
    cursor = 0
    while len(sampled) < sample_size and keys:
        key = keys[cursor % len(keys)]
        bucket = strata[key]
        sampled.append(bucket.pop())
        if not bucket:
            keys.pop(cursor % len(keys))
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor = (cursor + 1) % len(keys)
    rng.shuffle(sampled)
    return sampled


def ranked_equal_balanced_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    balance_fields: list[str],
) -> list[dict[str, Any]]:
    """Sample ranked rows evenly across strata while preferring earlier rows.

    The input order is treated as descending quality. Within each stratum, rows
    keep that order. Across strata, selection proceeds round-robin over strata
    ordered by their best-ranked row, then the final sample is restored to input
    order so downstream consumers still see a ranked JSONL.
    """
    if sample_size >= len(rows):
        return list(rows)

    strata: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[stratum_key(row, balance_fields)].append((index, row))

    keys = sorted(strata, key=lambda key: strata[key][0][0])
    sampled: list[tuple[int, dict[str, Any]]] = []
    cursor = 0
    while len(sampled) < sample_size and keys:
        key = keys[cursor % len(keys)]
        bucket = strata[key]
        sampled.append(bucket.pop(0))
        if not bucket:
            keys.pop(cursor % len(keys))
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor = (cursor + 1) % len(keys)

    sampled.sort(key=lambda item: item[0])
    return [row for _, row in sampled]


def proportional_sample(
    rows: list[dict[str, Any]], sample_size: int, rng: random.Random
) -> list[dict[str, Any]]:
    if sample_size >= len(rows):
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(rows, sample_size)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_where(values: list[list[str]] | None) -> list[Predicate]:
    predicates: list[Predicate] = []
    for item in values or []:
        if len(item) != 3:
            raise ValueError("--where requires exactly: FIELD OP VALUE")
        predicates.append((item[0], item[1], item[2]))
    return predicates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter JSONL rows and write a balanced fixed-size sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Operators: eq, ne, gt, ge, lt, le, in, not-in, contains, "
            "not-contains, exists, missing. Values true/false/null and numbers "
            "are parsed automatically."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="source JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="sample output JSONL file")
    parser.add_argument("--sample-size", type=int, default=2000, help="maximum rows to write")
    parser.add_argument(
        "--where",
        nargs=3,
        action="append",
        metavar=("FIELD", "OP", "VALUE"),
        help="filter predicate; repeat for AND semantics",
    )
    parser.add_argument(
        "--balance-fields",
        nargs="+",
        default=["category", "subcategory", "application"],
        help="fields used as balanced strata; dotted paths are supported",
    )
    parser.add_argument(
        "--balance-mode",
        choices=["equal", "proportional", "ranked-equal"],
        default="equal",
        help=(
            "equal balances shuffled strata; proportional is plain random sampling; "
            "ranked-equal treats input order as rank and balances strata while "
            "preferring earlier rows"
        ),
    )
    parser.add_argument("--seed", type=int, default=13, help="random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative")

    predicates = parse_where(args.where)
    rng = random.Random(args.seed)
    rows = load_filtered(args.input, predicates)
    if args.balance_mode == "equal":
        sampled = equal_balanced_sample(rows, args.sample_size, args.balance_fields, rng)
    elif args.balance_mode == "ranked-equal":
        sampled = ranked_equal_balanced_sample(rows, args.sample_size, args.balance_fields)
    else:
        sampled = proportional_sample(rows, args.sample_size, rng)
    write_jsonl(args.output, sampled)

    strata_count = len({stratum_key(row, args.balance_fields) for row in rows})
    sampled_strata_count = len({stratum_key(row, args.balance_fields) for row in sampled})
    print(
        f"[sample] input_matched={len(rows)} output={len(sampled)} "
        f"strata={strata_count} sampled_strata={sampled_strata_count} "
        f"mode={args.balance_mode} seed={args.seed} path={args.output}"
    )


if __name__ == "__main__":
    main()
