#!/usr/bin/env python3
"""Collect full-score OSWorld-V2 trajectories into one deduplicated JSON file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _results_files(source: Path) -> list[Path]:
    try:
        return sorted(source.glob("Run*/summary/results.json"))
    except PermissionError as exc:
        raise PermissionError(f"cannot scan source: {source}") from exc


def _trajectory_path(run_root: Path, task_id: str) -> Path:
    pattern = f"pyautogui/screenshot/*/tasks/{task_id}/traj.jsonl"
    try:
        candidates = sorted(run_root.glob(pattern))
    except PermissionError as exc:
        raise PermissionError(f"cannot scan trajectories in: {run_root}") from exc
    if not candidates:
        raise FileNotFoundError(f"missing traj.jsonl for task {task_id} in {run_root}")
    if len(candidates) > 1:
        raise ValueError(
            f"multiple traj.jsonl files for task {task_id} in {run_root}: {candidates}"
        )
    return candidates[0]


def _collect_source(source: Path) -> list[dict[str, Any]]:
    results_files = _results_files(source)
    if not results_files:
        try:
            next(source.iterdir())
        except PermissionError as exc:
            raise PermissionError(f"cannot read source: {source}") from exc
        except StopIteration:
            pass
        raise FileNotFoundError(f"no Run*/summary/results.json under {source}")

    collected = []
    for results_path in results_files:
        results = _load_json(results_path)
        if not isinstance(results, list):
            raise ValueError(f"expected a JSON array: {results_path}")
        run_root = results_path.parents[1]
        for result in results:
            if not isinstance(result, dict) or float(result.get("score", 0.0)) != 1.0:
                continue
            task_id = str(result["task_id"])
            trajectory_path = _trajectory_path(run_root, task_id)
            trajectory = _load_jsonl(trajectory_path)
            collected.append(
                {
                    "task_id": task_id,
                    "application": result.get("application"),
                    "score": float(result["score"]),
                    "status": result.get("status"),
                    "timestamp": result.get("timestamp"),
                    "source": source.name,
                    "run": run_root.name,
                    "results_path": str(results_path),
                    "trajectory_path": str(trajectory_path),
                    "step_count": len(trajectory),
                    "trajectory": trajectory,
                }
            )
    return collected


def build(source_root: Path, output: Path, allow_incomplete: bool) -> dict[str, Any]:
    sources = sorted(
        path
        for path in source_root.glob("OSWorld-V2-*-Vanilla")
        if path.is_dir()
    )
    if not sources:
        raise FileNotFoundError(f"no OSWorld-V2-*-Vanilla sources under {source_root}")

    candidates = []
    skipped_sources = []
    for source in sources:
        try:
            candidates.extend(_collect_source(source))
        except (FileNotFoundError, PermissionError) as exc:
            skipped_sources.append({"source": str(source), "error": str(exc)})

    if skipped_sources and not allow_incomplete:
        details = "\n".join(
            f"- {item['source']}: {item['error']}" for item in skipped_sources
        )
        raise PermissionError(f"not all sources could be read:\n{details}")

    selected: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        task_id = candidate["task_id"]
        current = selected.get(task_id)
        candidate_key = (
            candidate["step_count"],
            candidate["source"],
            candidate["run"],
            candidate["trajectory_path"],
        )
        if current is None:
            selected[task_id] = candidate
            continue
        current_key = (
            current["step_count"],
            current["source"],
            current["run"],
            current["trajectory_path"],
        )
        if candidate_key < current_key:
            selected[task_id] = candidate

    trajectories = [selected[task_id] for task_id in sorted(selected)]
    payload = {
        "metadata": {
            "benchmark": "OSWorld-V2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "source_glob": "OSWorld-V2-*-Vanilla",
            "success_rule": "score == 1.0",
            "deduplication_rule": "one trajectory per task_id; prefer the fewest JSONL rows, then lexical source order",
            "complete": not skipped_sources,
            "sources_discovered": len(sources),
            "sources_read": len(sources) - len(skipped_sources),
            "full_score_candidates": len(candidates),
            "unique_tasks": len(trajectories),
            "duplicates_removed": len(candidates) - len(trajectories),
            "skipped_sources": skipped_sources,
        },
        "trajectories": trajectories,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    payload = build(
        args.source_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.allow_incomplete,
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
