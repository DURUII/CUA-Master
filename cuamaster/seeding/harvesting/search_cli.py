"""CLI entry point for YouTube video source search (pipeline step 1.1)."""
from __future__ import annotations

import argparse
import logging
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from ..common.config import load_config
from ..common.models import VideoMetadata
from .query import build_queries
from .search import create_backend
from ..common.writer import JsonlWriter

logger = logging.getLogger("video2traj")


def _keep(
    video: VideoMetadata,
    min_duration_sec: int | None,
    max_duration_sec: int | None,
    min_upload_year: int | None,
) -> bool:
    """Drop videos outside [min_duration_sec, max_duration_sec] or uploaded before min_upload_year.

    Any bound may be None to disable it. Unknown duration/date are kept.
    """
    if (
        min_duration_sec is not None
        and video.duration_sec is not None
        and video.duration_sec < min_duration_sec
    ):
        return False
    if (
        max_duration_sec is not None
        and video.duration_sec is not None
        and video.duration_sec > max_duration_sec
    ):
        return False
    if min_upload_year is not None:
        year = _upload_year(video.upload_date)
        if year is not None and year < min_upload_year:
            return False
    return True


def _upload_year(upload_date: str | None) -> int | None:
    """Parse the leading 4-digit year from YYYY-MM-DD or YYYYMMDD."""
    if not upload_date or len(upload_date) < 4 or not upload_date[:4].isdigit():
        return None
    return int(upload_date[:4])


def main() -> None:
    parser = argparse.ArgumentParser(description="Search YouTube for video sources (metadata only, no download).")
    parser.add_argument(
        "--config",
        default="configs/seeding/sources.example.yaml",
        help="path to sources.yaml",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="only search this single top-level category (case-insensitive); default: all",
    )
    parser.add_argument(
        "--app",
        default=None,
        help="only search this single leaf application (case-insensitive), reusing "
        "its taxonomy labels; must exist in the taxonomy",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    config = load_config(args.config)
    queries = build_queries(config, category=args.category, app=args.app)
    backend = create_backend(config.backend, config)

    # When scoping to a single app or category, name the output after it so
    # subset runs don't clobber the full dataset (app takes precedence).
    output_path = config.output_path
    scope = args.app or args.category
    if scope:
        p = Path(config.output_path)
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", scope).strip("_").lower()
        output_path = str(p.with_name(f"{slug}{p.suffix}"))

    logger.info("backend=%s  queries=%d  limit/query=%d", config.backend, len(queries), config.max_results_per_query)

    total_hits = 0
    with JsonlWriter(output_path) as writer:
        for i, spec in enumerate(queries, 1):
            videos = backend.search(spec.text, config.max_results_per_query)
            videos = [
                v for v in videos
                if _keep(v, config.min_duration_sec, config.max_duration_sec, config.min_upload_year)
            ]
            for v in videos:
                v.category = spec.category
                v.subcategory = spec.subcategory
                v.application = spec.application
            new = sum(writer.write(v) for v in videos)
            total_hits += len(videos)
            logger.info("[%d/%d] %r -> %d hits (%d new)", i, len(queries), spec.text, len(videos), new)

            if i < len(queries) and config.delay_max_sec > 0:
                delay = random.uniform(config.delay_min_sec, config.delay_max_sec)
                logger.debug("sleeping %.2fs before next query", delay)
                time.sleep(delay)

        logger.info("done: %d hits, %d unique videos -> %s", total_hits, writer.count, output_path)


if __name__ == "__main__":
    main()
