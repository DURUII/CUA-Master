#!/usr/bin/env python3
"""Report transcription, scoring, and platform/OS labeling progress per subcategory.

For each ``<category>/<subcategory>`` group, counts downloaded ``.mp4`` videos,
``.segments.json`` transcripts (Trans), ``.score.json`` ratings (Scored), rows in
the platform/OS label JSONL whose ``platform`` and ``os`` are both populated
(Labeled), and videos that have all three outputs (Both). Each progress column
shows count plus percent.

Usage:
    python scripts/pipeline_progress.py \
        --video-dir data/videos \
        --transcript-dir data/transcripts \
        --score-dir data/scores \
        --label-file data/data_list_labeled.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuamaster.seeding.common.core import sanitize


def ids_by_subcategory(root: Path, pattern: str, suffix: str) -> dict[tuple[str, str], set[str]]:
    """Collect video ids matching ``pattern`` grouped by (category, subcategory)."""
    ids: dict[tuple[str, str], set[str]] = {}
    for path in root.rglob(pattern):
        rel = path.relative_to(root).parts
        if len(rel) < 3:
            continue  # skip stray files not under category/subcategory
        key = (rel[0], rel[1])
        stem = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
        video_id = stem[-11:]
        ids.setdefault(key, set()).add(video_id)
    return ids


def format_progress(count: int, total: int) -> str:
    if not total:
        return f"{count} (n/a)"
    return f"{count} ({count / total * 100:.2f}%)"


def key_from_record(record: dict) -> tuple[str, str, str] | None:
    video_id = record.get("video_id")
    if not video_id:
        return None
    return (
        sanitize(record.get("category", "")),
        sanitize(record.get("subcategory", "")),
        str(video_id),
    )


def labeled_ids_by_subcategory(path: Path) -> dict[tuple[str, str], set[str]]:
    """Collect completed platform/OS labels from a JSONL output file."""
    ids: dict[tuple[str, str], set[str]] = {}
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("platform") or not record.get("os"):
                continue
            key = key_from_record(record)
            if key is None:
                continue
            category, subcategory, video_id = key
            ids.setdefault((category, subcategory), set()).add(video_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--transcript-dir", required=True)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--label-file", default="data/data_list_labeled.jsonl")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    transcript_dir = Path(args.transcript_dir)
    score_dir = Path(args.score_dir)
    label_file = Path(args.label_file)

    videos = ids_by_subcategory(video_dir, "*.mp4", ".mp4")
    transcripts = ids_by_subcategory(transcript_dir, "*.segments.json", ".segments.json")
    scores = ids_by_subcategory(score_dir, "*.score.json", ".score.json")
    labeled = labeled_ids_by_subcategory(label_file)

    keys = sorted(set(videos) | set(transcripts) | set(scores) | set(labeled))
    if not keys:
        print("No videos or transcripts found; check the directory paths.")
        return

    name_w = max(len(f"{c}/{s}") for c, s in keys)
    header = (
        f"{'Category / Subcategory':<{name_w}}  {'Trans':>18}"
        f"  {'Scored':>18}  {'Labeled':>18}  {'Both':>18}  {'Total':>7}"
    )
    print(header)
    print("-" * len(header))

    total_video_ids: set[tuple[str, str, str]] = set()
    total_transcript_ids: set[tuple[str, str, str]] = set()
    total_score_ids: set[tuple[str, str, str]] = set()
    total_labeled_ids: set[tuple[str, str, str]] = set()
    for c, s in keys:
        video_ids = videos.get((c, s), set())
        transcript_ids = transcripts.get((c, s), set())
        score_ids = scores.get((c, s), set())
        labeled_ids = labeled.get((c, s), set())
        total = len(video_ids)
        trans = len(transcript_ids & video_ids)
        scored = len(score_ids & video_ids)
        labeled_count = len(labeled_ids & video_ids)
        both = len(video_ids & transcript_ids & score_ids & labeled_ids)
        total_video_ids.update((c, s, vid) for vid in video_ids)
        total_transcript_ids.update((c, s, vid) for vid in transcript_ids)
        total_score_ids.update((c, s, vid) for vid in score_ids)
        total_labeled_ids.update((c, s, vid) for vid in labeled_ids)
        print(
            f"{c + '/' + s:<{name_w}}  {format_progress(trans, total):>18}"
            f"  {format_progress(scored, total):>18}"
            f"  {format_progress(labeled_count, total):>18}"
            f"  {format_progress(both, total):>18}"
            f"  {total:>7}"
        )

    print("-" * len(header))
    total_all = len(total_video_ids)
    total_trans = len(total_transcript_ids & total_video_ids)
    total_scored = len(total_score_ids & total_video_ids)
    total_labeled = len(total_labeled_ids & total_video_ids)
    total_both = len(total_video_ids & total_transcript_ids & total_score_ids & total_labeled_ids)
    print(
        f"{'TOTAL':<{name_w}}  {format_progress(total_trans, total_all):>18}"
        f"  {format_progress(total_scored, total_all):>18}"
        f"  {format_progress(total_labeled, total_all):>18}"
        f"  {format_progress(total_both, total_all):>18}  {total_all:>7}"
    )


if __name__ == "__main__":
    main()
