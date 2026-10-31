"""Normalize existing mp4/transcript paths to the shared sanitize() layout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..common.core import load_records, target_path


DEFAULT_VIDEO_DIR = "data/videos"
DEFAULT_TRANSCRIPT_DIR = "data/transcripts"
DEFAULT_INPUTS = [
    "data/youtube/video_data1.jsonl",
    "data/youtube/video_data2.jsonl",
    "data/youtube/expand_video_data.jsonl",
]


def _transcript_target(record: dict, video_dir: Path, transcript_dir: Path) -> Path:
    video_path = target_path(record, video_dir)
    return (transcript_dir / video_path.relative_to(video_dir)).with_suffix(".segments.json")


def _existing_by_video_id(root: Path, suffix: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in root.rglob(f"*{suffix}"):
        stem = path.name[: -len(suffix)]
        if len(stem) < 12:
            continue
        video_id = stem[-11:]
        if video_id not in paths:
            paths[video_id] = path
    return paths


def _plan_moves(records: list[dict], video_dir: Path, transcript_dir: Path) -> list[tuple[Path, Path]]:
    videos = _existing_by_video_id(video_dir, ".mp4")
    transcripts = _existing_by_video_id(transcript_dir, ".segments.json")
    moves: list[tuple[Path, Path]] = []
    for record in records:
        video_id = record.get("video_id")
        if not video_id:
            continue
        source_video = videos.get(video_id)
        target_video = target_path(record, video_dir)
        if source_video is not None and source_video != target_video:
            moves.append((source_video, target_video))
        source_transcript = transcripts.get(video_id)
        target_transcript = _transcript_target(record, video_dir, transcript_dir)
        if source_transcript is not None and source_transcript != target_transcript:
            moves.append((source_transcript, target_transcript))
    return moves


def _validate_moves(moves: list[tuple[Path, Path]]) -> None:
    targets: dict[Path, Path] = {}
    sources = {src for src, _ in moves}
    for src, dst in moves:
        other = targets.get(dst)
        if other is not None:
            raise SystemExit(f"target collision: {other} and {src} -> {dst}")
        targets[dst] = src
        if dst.exists() and dst not in sources:
            raise SystemExit(f"target already exists: {dst}")


def _apply_moves(moves: list[tuple[Path, Path]]) -> None:
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--apply", action="store_true", help="rename files; default only reports")
    parser.add_argument("--sample", type=int, default=20, help="number of planned moves to print")
    args = parser.parse_args()

    records = load_records(args.inputs)
    moves = _plan_moves(records, Path(args.video_dir), Path(args.transcript_dir))
    _validate_moves(moves)

    print(json.dumps({"planned_moves": len(moves), "apply": args.apply}, ensure_ascii=False))
    for src, dst in moves[: args.sample]:
        print(f"{src}\t{dst}")
    if args.apply:
        _apply_moves(moves)


if __name__ == "__main__":
    main()
