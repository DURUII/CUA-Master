"""Abstract search backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ...common.models import VideoMetadata


class SearchBackend(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query: str, limit: int) -> list[VideoMetadata]:
        """Search YouTube for `query`, returning up to `limit` videos.

        Metadata only — never downloads the video files.
        """
        raise NotImplementedError
