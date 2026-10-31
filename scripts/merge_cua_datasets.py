#!/usr/bin/env python3
"""Merge two CUA trajectory directories without overwriting logical duplicates."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from cuamaster.seeding.reconstruction.construct import (
    _canonical_action_errors,
    _summarize,
    _summary_row,
    _summary_row_trainable,
)


TRAJECTORY_GLOB = "*.trajectory*.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _quality_score(trajectory: dict[str, Any]) -> float:
    value = (trajectory.get("verification") or {}).get("quality_score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _selection_rank(trajectory: dict[str, Any], path: Path, *, prefer_primary: bool) -> tuple[Any, ...]:
    verification = trajectory.get("verification") or {}
    quality = trajectory.get("quality") or {}
    summary = _summary_row(trajectory, path)
    canonical_valid = not _canonical_action_errors(trajectory)
    trainable = canonical_valid and _summary_row_trainable(summary)
    return (
        trainable,
        verification.get("exportable_for_training") is True,
        _quality_score(trajectory),
        verification.get("trajectory_valid") is True,
        canonical_valid,
        quality.get("has_screenshots") is True,
        -int(quality.get("needs_grounding_steps") or 0),
        -int(quality.get("api_failed_steps") or 0),
        -int(quality.get("action_type_mismatches") or 0),
        int(quality.get("num_steps") or len(trajectory.get("steps") or [])),
        prefer_primary,
    )


def _rewrite_paths(value: Any, source_roots: tuple[Path, Path], output: Path) -> Any:
    if isinstance(value, str):
        for source_root in source_roots:
            source = str(source_root)
            if value == source or value.startswith(source + os.sep):
                return str(output) + value[len(source) :]
        return value
    if isinstance(value, list):
        return [_rewrite_paths(item, source_roots, output) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, source_roots, output) for key, item in value.items()}
    return value


def _referenced_screenshots(trajectory: dict[str, Any]) -> list[Path]:
    screenshots: list[Path] = []
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for key in ("observation", "post_observation"):
            screenshot = (step.get(key) or {}).get("screenshot")
            if screenshot:
                screenshots.append(Path(screenshot))
    return screenshots


def _relative_to_source(path: Path, source_root: Path) -> Path:
    try:
        return path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"referenced file is outside its dataset root: {path}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _inventory(root: Path) -> dict[Path, tuple[Path, dict[str, Any]]]:
    inventory: dict[Path, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.rglob(TRAJECTORY_GLOB)):
        relative = path.relative_to(root)
        if relative in inventory:
            raise ValueError(f"duplicate trajectory path in {root}: {relative}")
        inventory[relative] = (path, _load_json(path))
    return inventory


def _select(
    primary: Path,
    secondary: Path,
) -> tuple[dict[Path, tuple[str, Path, dict[str, Any]]], dict[str, Any]]:
    primary_inventory = _inventory(primary)
    secondary_inventory = _inventory(secondary)
    selected: dict[Path, tuple[str, Path, dict[str, Any]]] = {}
    reasons: Counter[str] = Counter()
    conflicts: list[dict[str, Any]] = []

    for relative in sorted(primary_inventory.keys() | secondary_inventory.keys()):
        primary_item = primary_inventory.get(relative)
        secondary_item = secondary_inventory.get(relative)
        if primary_item is None:
            selected[relative] = ("secondary", *secondary_item)
            reasons["secondary_only"] += 1
            continue
        if secondary_item is None:
            selected[relative] = ("primary", *primary_item)
            reasons["primary_only"] += 1
            continue

        primary_path, primary_obj = primary_item
        secondary_path, secondary_obj = secondary_item
        primary_rank = _selection_rank(primary_obj, primary_path, prefer_primary=True)
        secondary_rank = _selection_rank(secondary_obj, secondary_path, prefer_primary=False)
        if primary_rank >= secondary_rank:
            selected[relative] = ("primary", primary_path, primary_obj)
            winner = "primary"
        else:
            selected[relative] = ("secondary", secondary_path, secondary_obj)
            winner = "secondary"
        reason = "conflict_" + winner
        reasons[reason] += 1
        conflicts.append(
            {
                "relative_path": str(relative),
                "winner": winner,
                "primary_rank": list(primary_rank),
                "secondary_rank": list(secondary_rank),
                "primary_task": primary_obj.get("task"),
                "secondary_task": secondary_obj.get("task"),
            }
        )

    report = {
        "primary_trajectories": len(primary_inventory),
        "secondary_trajectories": len(secondary_inventory),
        "selected_trajectories": len(selected),
        "selection_counts": dict(sorted(reasons.items())),
        "conflicts": conflicts,
    }
    return selected, report


def _validate_selected_assets(
    selected: dict[Path, tuple[str, Path, dict[str, Any]]],
    roots: dict[str, Path],
) -> tuple[dict[Path, Path], list[str]]:
    assets: dict[Path, Path] = {}
    errors: list[str] = []
    for relative, (source_name, source_path, trajectory) in selected.items():
        source_root = roots[source_name]
        for screenshot in _referenced_screenshots(trajectory):
            try:
                asset_relative = _relative_to_source(screenshot, source_root)
            except ValueError as exc:
                errors.append(f"{relative}: {exc}")
                continue
            if not screenshot.is_file():
                errors.append(f"{relative}: missing screenshot {screenshot}")
                continue
            previous = assets.setdefault(asset_relative, screenshot)
            if previous != screenshot and not _same_file_content(previous, screenshot):
                errors.append(
                    f"{relative}: conflicting screenshot sources for {asset_relative}: "
                    f"{previous} vs {screenshot}"
                )
        if not source_path.is_file():
            errors.append(f"missing trajectory {source_path}")
    return assets, errors


def _same_file_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(1024 * 1024)
            right_chunk = right_file.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _materialized_path(path: Path, storage_root: Path, logical_root: Path) -> Path:
    try:
        return storage_root / path.relative_to(logical_root)
    except ValueError:
        return path


def _rewrite_jsonl_paths(path: Path, source_root: Path, output: Path) -> None:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            rows.append(_rewrite_paths(json.loads(line), (source_root, source_root), output))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _validate_output(
    storage_root: Path,
    logical_root: Path,
    expected_trajectories: int,
    expected_assets: int,
) -> dict[str, Any]:
    trajectory_paths = sorted(storage_root.rglob(TRAJECTORY_GLOB))
    image_paths = sorted(storage_root.rglob("*.jpg"))
    errors: list[str] = []
    if len(trajectory_paths) != expected_trajectories:
        errors.append(f"trajectory count {len(trajectory_paths)} != {expected_trajectories}")
    if len(image_paths) != expected_assets:
        errors.append(f"image count {len(image_paths)} != {expected_assets}")

    for path in trajectory_paths:
        trajectory = _load_json(path)
        for screenshot in _referenced_screenshots(trajectory):
            if not _materialized_path(screenshot, storage_root, logical_root).is_file():
                errors.append(f"missing output screenshot referenced by {path}: {screenshot}")

    train_rows: list[dict[str, Any]] = []
    train_path = storage_root / "trajectories.train.jsonl"
    with train_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid train JSONL line {line_number}: {exc}")
                continue
            train_rows.append(row)
    train_keys = [row.get("trajectory_path") for row in train_rows]
    duplicate_train_paths = len(train_keys) - len(set(train_keys))
    if duplicate_train_paths:
        errors.append(f"duplicate train trajectory paths: {duplicate_train_paths}")
    for row in train_rows:
        trajectory_path = Path(row.get("trajectory_path") or "")
        if not row.get("trajectory_path") or not _materialized_path(
            trajectory_path, storage_root, logical_root
        ).is_file():
            errors.append(f"missing train trajectory: {trajectory_path}")
        for step in row.get("steps") or []:
            for key in ("observation", "post_observation"):
                screenshot_value = (step.get(key) or {}).get("screenshot")
                screenshot = Path(screenshot_value or "")
                if not screenshot_value or not _materialized_path(
                    screenshot, storage_root, logical_root
                ).is_file():
                    errors.append(f"missing train screenshot: {screenshot}")

    summary_path = storage_root / "trajectories.jsonl"
    if train_path.read_bytes() != summary_path.read_bytes():
        errors.append("trajectories.jsonl differs from trajectories.train.jsonl")
    return {
        "trajectory_files": len(trajectory_paths),
        "image_files": len(image_paths),
        "train_records": len(train_rows),
        "duplicate_train_paths": duplicate_train_paths,
        "validation_errors": errors,
    }


def merge(primary: Path, secondary: Path, output: Path, *, dry_run: bool) -> dict[str, Any]:
    primary = primary.resolve()
    secondary = secondary.resolve()
    output = output.resolve()
    if primary == secondary:
        raise ValueError("primary and secondary directories must differ")
    if output in {primary, secondary}:
        raise ValueError("output must differ from both source directories")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    selected, report = _select(primary, secondary)
    roots = {"primary": primary, "secondary": secondary}
    assets, asset_errors = _validate_selected_assets(selected, roots)
    report.update(
        {
            "primary": str(primary),
            "secondary": str(secondary),
            "output": str(output),
            "selected_screenshot_files": len(assets),
            "preflight_errors": asset_errors,
            "dry_run": dry_run,
        }
    )
    if asset_errors:
        raise RuntimeError("preflight failed:\n" + "\n".join(asset_errors[:50]))
    if dry_run:
        return report

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    source_roots = (primary, secondary)

    for relative, (_, _, trajectory) in selected.items():
        rewritten = _rewrite_paths(trajectory, source_roots, output)
        _write_json(staging / relative, rewritten)
    for relative, source in assets.items():
        _copy_file(source, staging / relative)

    _summarize(staging, write_audit_jsonl=True)
    for summary_path in staging.glob("trajectories*.jsonl"):
        _rewrite_jsonl_paths(summary_path, staging, output)
    validation = _validate_output(staging, output, len(selected), len(assets))
    report["validation"] = validation
    if validation["validation_errors"]:
        _write_json(staging / "merge_report.json", report)
        raise RuntimeError("output validation failed:\n" + "\n".join(validation["validation_errors"][:50]))

    report["dry_run"] = False
    report_path = staging / "merge_report.json"
    _write_json(report_path, report)
    os.replace(staging, output)
    return report


def verify_existing(primary: Path, secondary: Path, output: Path) -> dict[str, Any]:
    primary = primary.resolve()
    secondary = secondary.resolve()
    output = output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output}")

    selected, selection_report = _select(primary, secondary)
    roots = {"primary": primary, "secondary": secondary}
    assets, asset_errors = _validate_selected_assets(selected, roots)
    errors = list(asset_errors)
    source_roots = (primary, secondary)

    matched_trajectories = 0
    for relative, (_, _, trajectory) in selected.items():
        output_path = output / relative
        if not output_path.is_file():
            errors.append(f"missing merged trajectory: {output_path}")
            continue
        expected = _rewrite_paths(trajectory, source_roots, output)
        if _load_json(output_path) != expected:
            errors.append(f"merged trajectory differs from selected source: {relative}")
            continue
        matched_trajectories += 1

    matched_assets = 0
    for relative, source in assets.items():
        destination = output / relative
        if not destination.is_file():
            errors.append(f"missing merged screenshot: {destination}")
            continue
        if not _same_file_content(source, destination):
            errors.append(f"merged screenshot differs from source: {relative}")
            continue
        matched_assets += 1

    output_validation = _validate_output(output, output, len(selected), len(assets))
    errors.extend(output_validation["validation_errors"])
    return {
        "selection": {
            key: value
            for key, value in selection_report.items()
            if key != "conflicts"
        },
        "matched_trajectories": matched_trajectories,
        "matched_screenshot_files": matched_assets,
        "output_validation": output_validation,
        "verification_errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.verify_existing:
        if args.dry_run:
            raise ValueError("--verify-existing and --dry-run cannot be used together")
        report = verify_existing(args.primary, args.secondary, args.output)
        if report["verification_errors"]:
            raise RuntimeError(
                "verification failed:\n" + "\n".join(report["verification_errors"][:50])
            )
    else:
        report = merge(args.primary, args.secondary, args.output, dry_run=args.dry_run)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
