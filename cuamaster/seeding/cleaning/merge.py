"""Merge per-category JSONL outputs into one globally deduplicated file.

Reads the category files produced by cli.py (--category runs), deduplicates by
video_id across all of them (first occurrence wins), and writes a single JSONL.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

logger = logging.getLogger("video2traj")


def merge(inputs: list[str], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    total = 0
    duplicates = 0

    with open(output, "w", encoding="utf-8") as out:
        for path in inputs:
            kept = 0
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                total += 1
                video_id = json.loads(line)["video_id"]
                if video_id in seen:
                    duplicates += 1
                    continue
                seen.add(video_id)
                out.write(line + "\n")
                kept += 1
            logger.info("%s -> %d kept", path, kept)

    logger.info(
        "done: %d records read, %d unique, %d duplicates dropped -> %s",
        total, len(seen), duplicates, output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-category JSONL files into one deduplicated JSONL."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="input JSONL files; default: all data/*.jsonl except the output and legacy videos.jsonl",
    )
    parser.add_argument("--output", default="data/merged.jsonl", help="output JSONL path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    inputs = args.inputs
    if not inputs:
        output_name = Path(args.output).name
        inputs = sorted(
            p for p in glob.glob("data/*.jsonl")
            if Path(p).name not in {output_name, "videos.jsonl"}
        )
    if not inputs:
        parser.error("no input files found")

    merge(inputs, args.output)


if __name__ == "__main__":
    main()
