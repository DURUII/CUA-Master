#!/usr/bin/env python3
"""Rank high-quality videos for long-horizon CUA trajectory construction."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from cuamaster.seeding.reconstruction.paths import load_transcript_segments
from cuamaster.seeding.reconstruction.transcript import mine_semantic_candidates
from cuamaster.seeding.common.core import target_path


DEFAULT_INPUT = "data/sample_list/sample_high_quality.jsonl"
DEFAULT_VIDEO_DIR = "data/videos"
DEFAULT_TRANSCRIPT_DIR = "data/transcripts"
DEFAULT_SCORE_DIR = "data/scores"
DEFAULT_OUTPUT_PREFIX = "data/sample_list/long_horizon_potential"

NEGATIVE_REASON_TERMS = (
    "marketing",
    "animation",
    "slide",
    "slides",
    "presentation",
    "speaker",
    "talking head",
    "bookended",
    "intro",
    "outro",
    "webinar",
    "lecture",
)

POSITIVE_REASON_TERMS = (
    "step-by-step",
    "screen recording",
    "screencast",
    "workflow",
    "clear navigation",
    "input data",
    "demonstrating how to",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("video_id"):
                raise ValueError(f"missing video_id at {path}:{line_no}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _output_row(row: dict[str, Any], *, include_rank_metadata: bool) -> dict[str, Any]:
    if include_rank_metadata:
        return row
    return {
        key: value
        for key, value in row.items()
        if key not in {"long_horizon_potential", "long_horizon_rank", "rank_features"}
    }


def _score_path_for_video(video_path: Path, video_dir: Path, score_dir: Path) -> Path:
    try:
        relative = video_path.relative_to(video_dir)
    except ValueError:
        relative = Path(video_path.name)
    return (score_dir / relative).with_suffix(".score.json")


def _transcript_path_for_video(video_path: Path, video_dir: Path, transcript_dir: Path) -> Path:
    try:
        relative = video_path.relative_to(video_dir)
    except ValueError:
        relative = Path(video_path.name)
    return (transcript_dir / relative).with_suffix(".segments.json")


def _load_score(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"score_file_error": True}
    return obj if isinstance(obj, dict) else {}


def _duration_score(duration_sec: float) -> float:
    """Prefer complete workflows without pushing very long broad tutorials first."""
    minutes = duration_sec / 60.0
    if minutes < 4:
        return 0.0
    if minutes < 8:
        return (minutes - 4) / 4 * 0.65
    if minutes <= 35:
        return 1.0
    if minutes <= 60:
        return 0.8
    if minutes <= 90:
        return 0.55
    if minutes <= 120:
        return 0.3
    return 0.05


def _count_score(value: int, target: int, low: int, high: int) -> float:
    if value <= low or value >= high:
        return 0.0
    if value == target:
        return 1.0
    if value < target:
        return (value - low) / max(target - low, 1)
    return (high - value) / max(high - target, 1)


def _density_score(actions_per_minute: float) -> float:
    if actions_per_minute <= 0.05:
        return 0.0
    if actions_per_minute < 0.4:
        return actions_per_minute / 0.4 * 0.6
    if actions_per_minute <= 4.5:
        return 1.0
    if actions_per_minute <= 8.0:
        return 0.7
    return 0.25


def _longest_chain(candidates: list[Any], max_gap_sec: float = 45.0) -> tuple[int, float]:
    if not candidates:
        return 0, 0.0
    ordered = sorted(candidates, key=lambda c: (c.start, c.end))
    best_count = cur_count = 1
    best_span = cur_start = ordered[0].start
    cur_end = ordered[0].end
    best_span = max(0.0, cur_end - cur_start)
    for candidate in ordered[1:]:
        if candidate.start - cur_end <= max_gap_sec:
            cur_count += 1
            cur_end = max(cur_end, candidate.end)
        else:
            span = max(0.0, cur_end - cur_start)
            if cur_count > best_count or (cur_count == best_count and span > best_span):
                best_count = cur_count
                best_span = span
            cur_count = 1
            cur_start = candidate.start
            cur_end = candidate.end
    span = max(0.0, cur_end - cur_start)
    if cur_count > best_count or (cur_count == best_count and span > best_span):
        best_count = cur_count
        best_span = span
    return best_count, best_span


def _reason_adjustment(reason: str) -> float:
    lowered = reason.lower()
    adjustment = 0.0
    adjustment += 0.03 * sum(1 for term in POSITIVE_REASON_TERMS if term in lowered)
    adjustment -= 0.05 * sum(1 for term in NEGATIVE_REASON_TERMS if term in lowered)
    return max(-0.25, min(0.15, adjustment))


def _platform_score(record: dict[str, Any]) -> float:
    platform = str(record.get("platform") or "").lower()
    os_name = str(record.get("os") or "").lower()
    if platform in {"desktop", "website"} and os_name not in {"android"}:
        return 1.0
    if platform in {"desktop", "website"}:
        return 0.65
    return 0.25


def _language_score(record: dict[str, Any]) -> float:
    language = str(record.get("language") or "").lower()
    if language == "en":
        return 1.0
    if language in {"es", "de", "fr", "it", "pt", "zh", "ja", "ko"}:
        return 0.8
    if language and language != "nn":
        return 0.62
    return 0.35


def _rank_one(
    record: dict[str, Any],
    video_dir: Path,
    transcript_dir: Path,
    score_dir: Path,
    target_actions: int,
) -> dict[str, Any]:
    video_path = target_path(record, video_dir)
    transcript_path = _transcript_path_for_video(video_path, video_dir, transcript_dir)
    score_path = _score_path_for_video(video_path, video_dir, score_dir)
    disk_score = _load_score(score_path)
    duration = float(record.get("duration_sec") or disk_score.get("duration_sec") or 0.0)

    segments = []
    transcript_error = None
    if transcript_path.exists():
        try:
            segments = load_transcript_segments(transcript_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            transcript_error = str(exc)[:200]
    candidates = mine_semantic_candidates(segments) if segments else []
    action_types = Counter(c.action_type for c in candidates if c.action_type)
    longest_chain_count, longest_chain_span = _longest_chain(candidates)
    action_count = len(candidates)
    duration_min = max(duration / 60.0, 1e-6)
    actions_per_minute = action_count / duration_min
    text_chars = sum(len(segment.text) for segment in segments)
    transcript_span = max((segment.end for segment in segments), default=0.0)
    transcript_coverage = min(1.0, transcript_span / duration) if duration > 0 else 0.0

    score_value = disk_score.get("score", record.get("score"))
    clarity = disk_score.get("clarity")
    score_norm = max(0.0, min(1.0, float(score_value or 0) / 5.0))
    clarity_norm = max(0.0, min(1.0, float(clarity or score_value or 0) / 5.0))
    reason = str(disk_score.get("reason") or "")

    potential = (
        0.18 * score_norm
        + 0.10 * clarity_norm
        + 0.18 * _duration_score(duration)
        + 0.20 * _count_score(action_count, target=target_actions, low=6, high=140)
        + 0.12 * _density_score(actions_per_minute)
        + 0.10 * _count_score(longest_chain_count, target=24, low=4, high=100)
        + 0.04 * min(1.0, len(action_types) / 5.0)
        + 0.04 * transcript_coverage
        + 0.02 * _platform_score(record)
        + 0.02 * _language_score(record)
        + _reason_adjustment(reason)
    )
    if not video_path.exists():
        potential -= 0.40
    if not transcript_path.exists() or transcript_error:
        potential -= 0.30
    if disk_score and (disk_score.get("is_gui") is False or disk_score.get("has_trajectory") is False):
        potential -= 0.45
    if score_value is not None and float(score_value) < 4:
        potential -= 0.25
    if duration > 7200:
        potential -= 0.15
    potential = max(0.0, potential)

    return {
        **record,
        "long_horizon_potential": round(potential, 6),
        "rank_features": {
            "video_exists": video_path.exists(),
            "transcript_exists": transcript_path.exists(),
            "score_exists": bool(disk_score),
            "score_file_score": score_value,
            "score_file_clarity": clarity,
            "score_reason": reason,
            "duration_score": round(_duration_score(duration), 6),
            "target_actions": target_actions,
            "semantic_action_count": action_count,
            "semantic_action_types": dict(sorted(action_types.items())),
            "semantic_actions_per_minute": round(actions_per_minute, 6),
            "longest_action_chain_count": longest_chain_count,
            "longest_action_chain_span_sec": round(longest_chain_span, 3),
            "transcript_segments": len(segments),
            "transcript_text_chars": text_chars,
            "transcript_coverage": round(transcript_coverage, 6),
            "video_path": str(video_path),
            "transcript_path": str(transcript_path),
            "score_path": str(score_path),
            "transcript_error": transcript_error,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--score-dir", default=DEFAULT_SCORE_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--target-actions",
        type=int,
        default=100,
        help="semantic action count that receives the highest action-count score",
    )
    parser.add_argument(
        "--include-rank-metadata",
        action="store_true",
        help="include long_horizon_potential, long_horizon_rank, and rank_features in output rows",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    video_dir = Path(args.video_dir)
    transcript_dir = Path(args.transcript_dir)
    score_dir = Path(args.score_dir)
    target_actions = max(1, int(args.target_actions))
    rows = _read_jsonl(input_path)

    ranked: list[dict[str, Any]] = []
    workers = max(1, int(args.workers or 1))
    if workers == 1:
        for index, row in enumerate(rows, 1):
            ranked.append(_rank_one(row, video_dir, transcript_dir, score_dir, target_actions))
            if args.progress_every and index % args.progress_every == 0:
                print(f"ranked {index}/{len(rows)}", file=sys.stderr, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {
                executor.submit(_rank_one, row, video_dir, transcript_dir, score_dir, target_actions)
                for row in rows
            }
            completed = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    ranked.append(future.result())
                    completed += 1
                if args.progress_every and completed // args.progress_every != (completed - len(done)) // args.progress_every:
                    print(f"ranked {completed}/{len(rows)}", file=sys.stderr, flush=True)
    ranked.sort(
        key=lambda row: (
            row["long_horizon_potential"],
            row["rank_features"]["longest_action_chain_count"],
            row["rank_features"]["semantic_action_count"],
            row.get("duration_sec") or 0,
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, 1):
        row["long_horizon_rank"] = index

    prefix = Path(args.output_prefix)
    audit_path = prefix.with_suffix(".ranked.jsonl")
    _write_jsonl(
        audit_path,
        [_output_row(row, include_rank_metadata=args.include_rank_metadata) for row in ranked],
    )

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "ranked_rows": len(ranked),
                "audit_path": str(audit_path),
                "target_actions": target_actions,
                "top_potential": ranked[0]["long_horizon_potential"] if ranked else None,
                "bottom_potential": ranked[-1]["long_horizon_potential"] if ranked else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
