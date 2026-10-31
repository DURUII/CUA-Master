"""Select a search backend by name."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import SearchBackend

if TYPE_CHECKING:
    from ...common.config import SourceConfig


def create_backend(name: str, config: "SourceConfig | None" = None) -> SearchBackend:
    if name == "ytdlp":
        from .ytdlp import YtDlpBackend

        return YtDlpBackend(
            cookiefile=config.cookiefile if config else None,
            cookies_from_browser=config.cookies_from_browser if config else None,
        )
    if name == "youtube_api":
        from .youtube_api import YouTubeApiBackend

        return YouTubeApiBackend()
    raise ValueError(f"unknown backend: {name!r} (expected 'ytdlp' or 'youtube_api')")
