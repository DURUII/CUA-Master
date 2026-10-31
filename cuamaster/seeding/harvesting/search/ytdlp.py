"""yt-dlp search backend.

Two-stage: a flat search (id + basic fields) then a best-effort per-video
extract to enrich with tags/like_count/categories. The detail extract runs
only when cookies are configured, since YouTube's anti-bot blocks it
otherwise; without cookies the flat metadata is kept as-is (faster, no wasted
per-video requests). Never downloads video files.
"""
from __future__ import annotations

import logging

from yt_dlp import YoutubeDL

from ...common.models import VideoMetadata
from .base import SearchBackend

logger = logging.getLogger(__name__)


class YtDlpBackend(SearchBackend):
    name = "ytdlp"

    def __init__(
        self,
        cookiefile: str | None = None,
        cookies_from_browser: str | None = None,
    ):
        self._cookie_opts: dict = {}
        if cookiefile:
            self._cookie_opts["cookiefile"] = cookiefile
        if cookies_from_browser:
            # yt-dlp expects a tuple: (browser, profile, keyring, container)
            self._cookie_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    def search(self, query: str, limit: int) -> list[VideoMetadata]:
        entries = self._search_entries(query, limit)
        results: list[VideoMetadata] = []
        for entry in entries:
            meta = self._from_flat(entry, query)
            if meta is None:
                continue
            # Detail extract is blocked by YouTube's anti-bot without cookies,
            # so skip it entirely when no cookies are configured.
            if self._cookie_opts:
                self._enrich(meta)
            results.append(meta)
        return results

    def _search_entries(self, query: str, limit: int) -> list[dict]:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            **self._cookie_opts,
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        except Exception as e:  # yt-dlp raises many error types
            logger.warning("flat search failed for %r: %s", query, e)
            return []
        return [e for e in (info or {}).get("entries", []) if e and e.get("id")]

    def _from_flat(self, entry: dict, query: str) -> VideoMetadata | None:
        vid = entry.get("id")
        if not vid:
            return None
        duration = entry.get("duration")
        return VideoMetadata(
            video_id=vid,
            title=entry.get("title", ""),
            channel=entry.get("channel") or entry.get("uploader", ""),
            url=entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
            duration_sec=int(duration) if duration is not None else None,
            view_count=entry.get("view_count"),
            like_count=None,
            upload_date=_ts_to_date(entry.get("timestamp")),
            description=entry.get("description"),
            tags=[],
            categories=[],
            source_query=query,
            backend=self.name,
        )

    def _enrich(self, meta: VideoMetadata) -> None:
        """Best-effort detail fetch. On failure, keep the flat metadata."""
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            **self._cookie_opts,
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={meta.video_id}", download=False
                )
        except Exception as e:
            logger.debug("detail extract failed for %s: %s", meta.video_id, e)
            return
        if not info:
            return

        meta.title = info.get("title") or meta.title
        meta.channel = info.get("channel") or info.get("uploader") or meta.channel
        meta.url = info.get("webpage_url") or meta.url
        if info.get("duration") is not None:
            meta.duration_sec = int(info["duration"])
        if info.get("view_count") is not None:
            meta.view_count = info["view_count"]
        meta.like_count = info.get("like_count")
        meta.upload_date = info.get("upload_date") or meta.upload_date  # YYYYMMDD
        meta.description = info.get("description") or meta.description
        meta.tags = info.get("tags") or []
        meta.categories = info.get("categories") or []


def _ts_to_date(ts) -> str | None:
    if ts is None:
        return None
    try:
        import datetime

        return datetime.datetime.utcfromtimestamp(int(ts)).strftime("%Y%m%d")
    except (TypeError, ValueError, OSError):
        return None
