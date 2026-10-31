"""Load pipeline configuration from a YAML file."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SourceConfig:
    backend: str = "ytdlp"
    # category -> subcategory -> list of application names
    taxonomy: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    max_results_per_query: int = 50
    output_path: str = "data/videos.jsonl"
    # Filters applied before writing. None disables that filter.
    min_duration_sec: int | None = 30
    max_duration_sec: int | None = 18000  # drop videos longer than this (5h)
    min_upload_year: int | None = 2010
    # Random delay between queries (seconds), drawn uniformly from [min, max].
    # 0/0 disables it. Helps avoid tripping YouTube anti-bot throttling.
    delay_min_sec: float = 0.0
    delay_max_sec: float = 0.0
    # ytdlp backend only: pass YouTube login state to bypass anti-bot checks.
    cookiefile: str | None = None
    cookies_from_browser: str | None = None


def load_config(path: str | Path) -> SourceConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    taxonomy = raw.get("taxonomy") or {}
    if not any(apps for subs in taxonomy.values() for apps in subs.values()):
        raise ValueError(
            "config must define a non-empty 'taxonomy' "
            "(category -> subcategory -> application list)"
        )

    return SourceConfig(
        backend=raw.get("backend", "ytdlp"),
        taxonomy=taxonomy,
        keywords=list(raw.get("keywords", [])),
        max_results_per_query=int(raw.get("max_results_per_query", 50)),
        output_path=raw.get("output_path", "data/videos.jsonl"),
        min_duration_sec=_opt_int(raw.get("min_duration_sec", 30)),
        max_duration_sec=_opt_int(raw.get("max_duration_sec", 18000)),
        min_upload_year=_opt_int(raw.get("min_upload_year", 2010)),
        delay_min_sec=float(raw.get("delay_min_sec", 0.0)),
        delay_max_sec=float(raw.get("delay_max_sec", 0.0)),
        cookiefile=raw.get("cookiefile"),
        cookies_from_browser=raw.get("cookies_from_browser"),
    )


def _opt_int(value) -> int | None:
    """None (or missing/null in YAML) disables the filter; otherwise coerce to int."""
    return None if value is None else int(value)
