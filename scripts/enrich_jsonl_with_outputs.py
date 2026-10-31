#!/usr/bin/env python3
"""Add transcript and score metadata to each row in a JSONL file.

The script joins rows by ``video_id`` against files produced by the pipeline:

* ``youtube_transcript/**/*.segments.json`` supplies ``language`` and
  ``duration_sec``.
* ``youtube_scores/scores.jsonl`` (or ``youtube_scores/**/*.score.json`` when no
  summary exists) supplies ``score``, ``is_gui``, and ``has_trajectory``.

Existing values are overwritten only when the matching transcript/score record
has a non-null value. If a row has no matching output, existing values are kept
and absent target fields are written as ``null`` so every row has the same keys.

Usage:
    python scripts/enrich_jsonl_with_outputs.py \
        --input data/data_list_labeled.jsonl \
        --output data/data_list_labeled_enriched.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_TRANSCRIPT_DIR = Path("data/transcripts")
DEFAULT_SCORE_DIR = Path("data/scores")
TARGET_FIELDS = ("language", "duration_sec", "score", "is_gui", "has_trajectory")


def collect_video_ids(input_path: Path) -> set[str]:
    """Read the input once to limit expensive transcript/score parsing."""
    video_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {input_path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"expected JSON object at {input_path}:{line_no}")
            video_id = obj.get("video_id")
            if video_id:
                video_ids.add(str(video_id))
    return video_ids


def _video_id_from_output_name(path: Path, suffix: str) -> str | None:
    """Extract the 11-char YouTube id from ``<app>_<id><suffix>`` filenames."""
    name = path.name
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    if len(stem) < 11:
        return None
    return stem[-11:]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _has_value(obj: dict[str, Any], key: str) -> bool:
    return obj.get(key) is not None


def _choose_more_complete(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    if current is None:
        return candidate
    current_count = sum(_has_value(current, field) for field in fields)
    candidate_count = sum(_has_value(candidate, field) for field in fields)
    return candidate if candidate_count > current_count else current


def load_transcripts(
    transcript_dir: Path,
    wanted_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Return ``video_id -> {language, duration_sec}``, scanned count, duplicate count."""
    records: dict[str, dict[str, Any]] = {}
    scanned = duplicates = 0
    for path in transcript_dir.rglob("*.segments.json"):
        candidate_id = _video_id_from_output_name(path, ".segments.json")
        if wanted_ids is not None and candidate_id not in wanted_ids:
            continue
        scanned += 1
        obj = _load_json(path)
        if not obj:
            continue
        video_id = obj.get("video_id")
        if not video_id:
            continue
        row = {
            "language": obj.get("language"),
            "duration_sec": obj.get("duration_sec"),
        }
        if video_id in records:
            duplicates += 1
        records[str(video_id)] = _choose_more_complete(
            records.get(str(video_id)), row, ("language", "duration_sec")
        )
    return records, scanned, duplicates


def _load_score_summary(
    summary_path: Path,
    wanted_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    records: dict[str, dict[str, Any]] = {}
    scanned = duplicates = 0
    with summary_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = obj.get("video_id")
            if not video_id:
                continue
            if wanted_ids is not None and str(video_id) not in wanted_ids:
                continue
            row = {
                "score": obj.get("score"),
                "is_gui": obj.get("is_gui"),
                "has_trajectory": obj.get("has_trajectory"),
            }
            if video_id in records:
                duplicates += 1
            records[str(video_id)] = _choose_more_complete(
                records.get(str(video_id)), row, ("score", "is_gui", "has_trajectory")
            )
    return records, scanned, duplicates


def _load_score_files(
    score_dir: Path,
    wanted_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    records: dict[str, dict[str, Any]] = {}
    scanned = duplicates = 0
    for path in score_dir.rglob("*.score.json"):
        candidate_id = _video_id_from_output_name(path, ".score.json")
        if wanted_ids is not None and candidate_id not in wanted_ids:
            continue
        scanned += 1
        obj = _load_json(path)
        if not obj:
            continue
        video_id = obj.get("video_id")
        if not video_id:
            continue
        row = {
            "score": obj.get("score"),
            "is_gui": obj.get("is_gui"),
            "has_trajectory": obj.get("has_trajectory"),
        }
        if video_id in records:
            duplicates += 1
        records[str(video_id)] = _choose_more_complete(
            records.get(str(video_id)), row, ("score", "is_gui", "has_trajectory")
        )
    return records, scanned, duplicates


def load_scores(
    score_dir: Path,
    score_summary: Path | None,
    wanted_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], int, int, Path | None]:
    """Load score metadata, preferring the aggregate scores.jsonl when present."""
    summary_path = score_summary if score_summary is not None else score_dir / "scores.jsonl"
    if summary_path.exists():
        records, scanned, duplicates = _load_score_summary(summary_path, wanted_ids)
        return records, scanned, duplicates, summary_path
    records, scanned, duplicates = _load_score_files(score_dir, wanted_ids)
    return records, scanned, duplicates, None


def enrich_jsonl(
    input_path: Path,
    output_path: Path,
    transcripts: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    overwrite: bool,
    fill_missing: bool,
) -> dict[str, int]:
    counts = {
        "rows": 0,
        "matched_transcript": 0,
        "matched_score": 0,
        "missing_video_id": 0,
        "changed_rows": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")

    with input_path.open(encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {input_path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {input_path}:{line_no}")

            before = dict(row)
            counts["rows"] += 1
            video_id = row.get("video_id")
            if not video_id:
                counts["missing_video_id"] += 1
            else:
                video_id = str(video_id)
                transcript = transcripts.get(video_id)
                score = scores.get(video_id)
                if transcript:
                    counts["matched_transcript"] += 1
                    _merge_fields(row, transcript, ("language", "duration_sec"), overwrite)
                if score:
                    counts["matched_score"] += 1
                    _merge_fields(row, score, ("score", "is_gui", "has_trajectory"), overwrite)

            if fill_missing:
                for field in TARGET_FIELDS:
                    row.setdefault(field, None)
            if row != before:
                counts["changed_rows"] += 1
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    os.replace(tmp_path, output_path)
    return counts


def _merge_fields(
    row: dict[str, Any],
    source: dict[str, Any],
    fields: tuple[str, ...],
    overwrite: bool,
) -> None:
    for field in fields:
        value = source.get(field)
        if value is None:
            continue
        if overwrite or row.get(field) is None:
            row[field] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join transcript language/duration and score metadata into a JSONL file."
    )
    parser.add_argument("--input", required=True, type=Path, help="source JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="destination JSONL file")
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument(
        "--score-summary",
        type=Path,
        default=None,
        help="score summary JSONL; defaults to <score-dir>/scores.jsonl when it exists",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="only fill target fields that are currently null or missing",
    )
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="do not add null target fields for rows with no matching output",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wanted_ids = collect_video_ids(args.input)
    transcripts, transcript_scanned, transcript_duplicates = load_transcripts(
        args.transcript_dir, wanted_ids
    )
    scores, score_scanned, score_duplicates, score_source = load_scores(
        args.score_dir, args.score_summary, wanted_ids
    )
    counts = enrich_jsonl(
        args.input,
        args.output,
        transcripts,
        scores,
        overwrite=not args.no_overwrite,
        fill_missing=not args.no_fill_missing,
    )

    score_label = str(score_source) if score_source is not None else f"{args.score_dir}/**/*.score.json"
    print(
        f"[transcripts] scanned={transcript_scanned} indexed={len(transcripts)} "
        f"duplicates={transcript_duplicates} source={args.transcript_dir}"
    )
    print(
        f"[scores] scanned={score_scanned} indexed={len(scores)} "
        f"duplicates={score_duplicates} source={score_label}"
    )
    print(
        "[jsonl] "
        f"rows={counts['rows']} changed={counts['changed_rows']} "
        f"matched_transcript={counts['matched_transcript']} "
        f"matched_score={counts['matched_score']} "
        f"missing_video_id={counts['missing_video_id']} output={args.output}"
    )


if __name__ == "__main__":
    main()
