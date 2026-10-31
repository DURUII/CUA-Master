#!/usr/bin/env python3
"""Backfill verified training trajectories with a canonical success finish step."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cuamaster.seeding.reconstruction.construct import _summarize, ensure_success_finish_step


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.finish.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def backfill(output_dir: Path, *, apply: bool) -> dict[str, int]:
    stats = {
        "trajectory_files": 0,
        "eligible": 0,
        "already_finished": 0,
        "would_update": 0,
        "updated": 0,
        "skipped": 0,
        "unreadable": 0,
    }
    for path in sorted(output_dir.rglob("*.trajectory*.json")):
        stats["trajectory_files"] += 1
        try:
            with path.open(encoding="utf-8") as f:
                trajectory = json.load(f)
        except (OSError, json.JSONDecodeError):
            stats["unreadable"] += 1
            continue

        verification = trajectory.get("verification") or {}
        if verification.get("exportable_for_training") is not True:
            stats["skipped"] += 1
            continue
        stats["eligible"] += 1
        steps = trajectory.get("steps") or []
        if any(
            isinstance(step, dict) and (step.get("action") or {}).get("type") == "finish"
            for step in steps
        ):
            stats["already_finished"] += 1
            continue
        if not ensure_success_finish_step(trajectory):
            stats["skipped"] += 1
            continue
        stats["would_update"] += 1
        if apply:
            _write_json_atomic(path, trajectory)
            stats["updated"] += 1
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="trajectory dataset root to scan for *.trajectory*.json files",
    )
    parser.add_argument("--apply", action="store_true", help="write changes; otherwise perform a dry run")
    parser.add_argument("--no-summarize", action="store_true", help="do not rebuild trajectory JSONL summaries")
    parser.add_argument("--write-audit-jsonl", action="store_true")
    parser.add_argument("--include-long-candidates-in-train", action="store_true")
    parser.add_argument("--long-min-steps", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = backfill(args.output_dir, apply=args.apply)
    if args.apply and not args.no_summarize:
        _summarize(
            args.output_dir,
            write_audit_jsonl=args.write_audit_jsonl,
            include_long_candidates_in_train=args.include_long_candidates_in_train,
            long_min_steps=args.long_min_steps,
        )
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
