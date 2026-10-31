"""Video metadata model shared across search backends."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    channel: str
    url: str
    duration_sec: int | None = None
    view_count: int | None = None
    like_count: int | None = None
    upload_date: str | None = None  # YYYY-MM-DD or YYYYMMDD depending on backend
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    source_query: str = ""
    category: str = ""  # taxonomy major category (大类)
    subcategory: str = ""  # taxonomy subcategory (子类)
    application: str = ""  # taxonomy leaf app (app)
    backend: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
