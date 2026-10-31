"""Shared helpers for the bulk download and transcribe steps."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .core import sanitize


YOUTUBE_ID_LEN = 11


def parse_stem(stem: str) -> tuple[str, str]:
    """Split a ``<app>_<video_id>`` filename stem into ``(application, video_id)``.

    YouTube IDs are a fixed 11 chars and may themselves contain ``_``/``-``, so we
    slice by length instead of splitting on ``_`` (which mis-parses IDs whose last
    11 chars contain an underscore).
    """
    if len(stem) > YOUTUBE_ID_LEN:
        return stem[: -(YOUTUBE_ID_LEN + 1)], stem[-YOUTUBE_ID_LEN:]
    return "", stem


def video_id_from_path(path: Path) -> str:
    return parse_stem(path.stem)[1]


def normalized_label(name: str) -> str:
    """Normalize a taxonomy label for comparisons with on-disk path segments."""
    return sanitize(name).casefold()


def label_matches(value: str | None, query: str | None) -> bool:
    """Return whether two labels match as raw text or sanitized path segments."""
    if query is None:
        return True
    value = value or ""
    return value.casefold() == query.casefold() or normalized_label(value) == normalized_label(query)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def read_failure_records(path: Path) -> dict[str, dict]:
    """Read a pending-failure ledger keyed by video_id (falling back to source_video)."""
    failures: dict[str, dict] = {}
    if not path.exists():
        return failures
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("video_id") or obj.get("source_video")
            if key:
                failures[key] = obj
    return failures


def write_failure_records(path: Path, failures: dict[str, dict]) -> None:
    """Atomically rewrite the pending-failure ledger."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in failures.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def filter_records(
    records: list[dict],
    category: str | None = None,
    app: str | None = None,
    limit: int | None = None,
    subcategory: str | None = None,
) -> list[dict]:
    """Filter metadata records by taxonomy labels, then cap to `limit`."""
    if category:
        records = [r for r in records if label_matches(r.get("category"), category)]
    if subcategory:
        records = [r for r in records if label_matches(r.get("subcategory"), subcategory)]
    if app:
        records = [r for r in records if label_matches(r.get("application"), app)]
    if limit is not None:
        records = records[:limit]
    return records
