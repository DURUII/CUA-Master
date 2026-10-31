"""Write video metadata to JSONL, deduplicating by video_id."""
from __future__ import annotations

import json
from pathlib import Path

from .models import VideoMetadata


class JsonlWriter:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        self._fh = open(self.output_path, "w", encoding="utf-8")

    def write(self, video: VideoMetadata) -> bool:
        """Write one record. Returns False if it was a duplicate (skipped)."""
        if video.video_id in self._seen:
            return False
        self._seen.add(video.video_id)
        self._fh.write(json.dumps(video.to_dict(), ensure_ascii=False) + "\n")
        return True

    @property
    def count(self) -> int:
        return len(self._seen)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
