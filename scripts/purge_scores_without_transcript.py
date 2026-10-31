#!/usr/bin/env python3
"""Delete .score.json files scored before their transcript existed.

Background: scoring used to fall back to frame-only rating when a video had no
``.segments.json`` transcript. We now require the transcript file to exist before
scoring (``--require-transcript`` defaults on). This script removes the stale
frame-only ratings so a re-run of ``score`` re-judges them once transcribed.

A ``.score.json`` is deleted only when its matching ``.segments.json`` (same
relative path, same layout as the two trees share) is MISSING. Videos whose
transcript exists but is empty (music / no narration) are kept: they were
correctly scored from frames and will stay that way.

Usage:
    # dry-run: report how many would be deleted, touch nothing
    python scripts/purge_scores_without_transcript.py --dry-run

    # delete, then rebuild scores.jsonl
    python scripts/purge_scores_without_transcript.py --summarize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_SCORE_DIR = "data/scores"
DEFAULT_TRANSCRIPT_DIR = "data/transcripts"


def _transcript_for(score_path: Path, score_dir: Path, transcript_dir: Path) -> Path:
    relative = score_path.relative_to(score_dir)
    stem = relative.name[: -len(".score.json")]
    return transcript_dir / relative.parent / f"{stem}.segments.json"


def purge(score_dir: Path, transcript_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Return (scanned, deleted). Collect victims first, then delete."""
    victims: list[Path] = []
    scanned = 0
    for score_path in score_dir.rglob("*.score.json"):
        scanned += 1
        if not _transcript_for(score_path, score_dir, transcript_dir).exists():
            victims.append(score_path)
    if not dry_run:
        for path in victims:
            path.unlink()
    return scanned, len(victims)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--score-dir", default=DEFAULT_SCORE_DIR)
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--summarize", action="store_true", help="rebuild scores.jsonl after deleting")
    parser.add_argument("--dry-run", action="store_true", help="report counts only, delete nothing")
    args = parser.parse_args()

    score_dir = Path(args.score_dir)
    transcript_dir = Path(args.transcript_dir)

    scanned, deleted = purge(score_dir, transcript_dir, args.dry_run)
    verb = "would delete" if args.dry_run else "deleted"
    print(f"[purge] scanned={scanned} {verb}={deleted} kept={scanned - deleted}")

    if args.summarize and not args.dry_run:
        from cuamaster.seeding.scoring.score import _summarize

        _summarize(score_dir)
        print(f"[summarize] rebuilt {score_dir}/scores.jsonl")
    elif args.summarize and args.dry_run:
        print("[summarize] skipped (dry-run)")


if __name__ == "__main__":
    main()
