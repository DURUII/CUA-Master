#!/usr/bin/env python3
"""Purge videos/transcripts/scores listed in the score failure ledger.

The score failure ledger records the source video path for items that could not
be scored. This script maps each failed source video to the corresponding files
in the video, transcript, and score trees, writes a JSONL manifest, and deletes
those files only when --apply is passed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_VIDEO_DIR = Path("data/videos")
DEFAULT_TRANSCRIPT_DIR = Path("data/transcripts")
DEFAULT_SCORE_DIR = Path("data/scores")
DEFAULT_FAILURES = DEFAULT_SCORE_DIR / "_score_failures.jsonl"


@dataclass(frozen=True)
class Victim:
    source_video: Path
    video: Path
    transcript: Path
    score: Path
    error: str | None


def _relative_stem(source_video: Path, video_dir: Path) -> Path | None:
    try:
        rel = source_video.relative_to(video_dir)
    except ValueError:
        return None
    return rel.with_suffix("")


def collect_victims(
    failures_path: Path,
    video_dir: Path,
    transcript_dir: Path,
    score_dir: Path,
) -> tuple[list[Victim], list[dict]]:
    victims: dict[Path, Victim] = {}
    unmatched: list[dict] = []

    with failures_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_source = record.get("source_video")
            if not raw_source:
                unmatched.append({"line": line_no, "reason": "missing source_video", "record": record})
                continue
            source_video = Path(raw_source)
            rel_stem = _relative_stem(source_video, video_dir)
            if rel_stem is None:
                unmatched.append({"line": line_no, "reason": "outside video_dir", "record": record})
                continue
            victims.setdefault(
                rel_stem,
                Victim(
                    source_video=source_video,
                    video=video_dir / rel_stem.parent / f"{rel_stem.name}.mp4",
                    transcript=transcript_dir / rel_stem.parent / f"{rel_stem.name}.segments.json",
                    score=score_dir / rel_stem.parent / f"{rel_stem.name}.score.json",
                    error=record.get("error"),
                ),
            )
    return sorted(victims.values(), key=lambda v: str(v.source_video)), unmatched


def write_manifest(path: Path, victims: list[Victim]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for victim in victims:
            row = {
                "source_video": str(victim.source_video),
                "video": str(victim.video),
                "transcript": str(victim.transcript),
                "score": str(victim.score),
                "error": victim.error,
                "exists": {
                    "video": victim.video.exists(),
                    "transcript": victim.transcript.exists(),
                    "score": victim.score.exists(),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prune_failure_ledger(failures_path: Path, video_dir: Path, victims: list[Victim]) -> int:
    victim_stems = {_relative_stem(v.source_video, video_dir) for v in victims}
    kept: list[str] = []
    removed = 0
    with failures_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            raw_source = record.get("source_video")
            rel_stem = _relative_stem(Path(raw_source), video_dir) if raw_source else None
            if rel_stem in victim_stems:
                removed += 1
            else:
                kept.append(line if line.endswith("\n") else line + "\n")

    backup = failures_path.with_suffix(failures_path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(failures_path.read_text(encoding="utf-8"), encoding="utf-8")
    failures_path.write_text("".join(kept), encoding="utf-8")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--manifest", type=Path, default=Path("data/score_failure_purge_manifest.jsonl"))
    parser.add_argument("--apply", action="store_true", help="delete files and prune the failure ledger")
    args = parser.parse_args()

    victims, unmatched = collect_victims(args.failures, args.video_dir, args.transcript_dir, args.score_dir)
    write_manifest(args.manifest, victims)

    counts = {"video": 0, "transcript": 0, "score": 0}
    for victim in victims:
        counts["video"] += victim.video.exists()
        counts["transcript"] += victim.transcript.exists()
        counts["score"] += victim.score.exists()

    print(f"victims={len(victims)} unmatched_failure_rows={len(unmatched)}")
    print(
        "existing_files "
        f"video={counts['video']} transcript={counts['transcript']} score={counts['score']}"
    )
    print(f"manifest={args.manifest}")

    if not args.apply:
        print("dry_run=true")
        return

    deleted = {"video": 0, "transcript": 0, "score": 0}
    for victim in victims:
        for key, path in (("video", victim.video), ("transcript", victim.transcript), ("score", victim.score)):
            if path.exists():
                path.unlink()
                deleted[key] += 1

    removed_failures = prune_failure_ledger(args.failures, args.video_dir, victims)
    print(
        "deleted_files "
        f"video={deleted['video']} transcript={deleted['transcript']} score={deleted['score']}"
    )
    print(f"removed_failure_rows={removed_failures}")


if __name__ == "__main__":
    main()
