"""YouTube Data API v3 search backend (two-stage: search.list + videos.list)."""
from __future__ import annotations

import logging
import os

import isodate
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ...common.models import VideoMetadata
from .base import SearchBackend

logger = logging.getLogger(__name__)

_VIDEOS_BATCH = 50  # videos.list accepts up to 50 ids per call


class YouTubeApiBackend(SearchBackend):
    name = "youtube_api"

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError(
                "YOUTUBE_API_KEY is required for the youtube_api backend "
                "(set it in .env or the environment)"
            )
        self._yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    def search(self, query: str, limit: int) -> list[VideoMetadata]:
        video_ids = self._search_ids(query, limit)
        if not video_ids:
            return []
        return self._fetch_details(video_ids, query)

    def _search_ids(self, query: str, limit: int) -> list[str]:
        ids: list[str] = []
        page_token: str | None = None
        try:
            while len(ids) < limit:
                page_size = min(50, limit - len(ids))
                resp = (
                    self._yt.search()
                    .list(
                        part="id",
                        q=query,
                        type="video",
                        maxResults=page_size,
                        pageToken=page_token,
                    )
                    .execute()
                )
                for item in resp.get("items", []):
                    vid = item.get("id", {}).get("videoId")
                    if vid:
                        ids.append(vid)
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as e:
            logger.warning("search.list failed for %r: %s", query, e)
        return ids[:limit]

    def _fetch_details(self, video_ids: list[str], query: str) -> list[VideoMetadata]:
        results: list[VideoMetadata] = []
        for i in range(0, len(video_ids), _VIDEOS_BATCH):
            batch = video_ids[i : i + _VIDEOS_BATCH]
            try:
                resp = (
                    self._yt.videos()
                    .list(part="snippet,contentDetails,statistics", id=",".join(batch))
                    .execute()
                )
            except HttpError as e:
                logger.warning("videos.list failed: %s", e)
                continue
            for item in resp.get("items", []):
                results.append(self._to_metadata(item, query))
        return results

    def _to_metadata(self, item: dict, query: str) -> VideoMetadata:
        snippet = item.get("snippet", {})
        content = item.get("contentDetails", {})
        stats = item.get("statistics", {})
        vid = item["id"]

        duration_sec = None
        iso_dur = content.get("duration")
        if iso_dur:
            try:
                duration_sec = int(isodate.parse_duration(iso_dur).total_seconds())
            except (isodate.ISO8601Error, ValueError):
                duration_sec = None

        upload_date = None
        published = snippet.get("publishedAt")  # e.g. 2023-05-01T12:00:00Z
        if published:
            upload_date = published[:10]

        return VideoMetadata(
            video_id=vid,
            title=snippet.get("title", ""),
            channel=snippet.get("channelTitle", ""),
            url=f"https://www.youtube.com/watch?v={vid}",
            duration_sec=duration_sec,
            view_count=_to_int(stats.get("viewCount")),
            like_count=_to_int(stats.get("likeCount")),
            upload_date=upload_date,
            description=snippet.get("description"),
            tags=snippet.get("tags", []) or [],
            categories=[snippet["categoryId"]] if snippet.get("categoryId") else [],
            source_query=query,
            backend=self.name,
        )


def _to_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
