from __future__ import annotations

import sys
import types
from pathlib import Path

from cuamaster.seeding.harvesting.download import DEFAULT_FAILURE_CLASSES, classify_failure
from cuamaster.seeding.common import core


class _FakeYoutubeDL:
    def __init__(self, opts: dict) -> None:
        self.opts = opts

    def __enter__(self) -> "_FakeYoutubeDL":
        return self

    def __exit__(self, *args) -> None:
        return None

    def download(self, urls: list[str]) -> None:
        outtmpl = self.opts["outtmpl"]
        Path(outtmpl.replace("%(ext)s", "mp4")).write_bytes(b"new broken mp4")


def test_overwrite_keeps_existing_file_when_repair_download_is_not_decodable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_module = types.SimpleNamespace(YoutubeDL=_FakeYoutubeDL)
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
    monkeypatch.setattr(core, "validate_video_file", lambda path: "downloaded file is not decodable: test")

    record = {
        "video_id": "abc123def45",
        "url": "https://www.youtube.com/watch?v=abc123def45",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "App",
    }
    existing = core.target_path(record, tmp_path)
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old file")

    vid, error = core.download_one(record, tmp_path, overwrite=True, validate_output=True)

    assert vid == "abc123def45"
    assert error == "downloaded file is not decodable: test"
    assert existing.read_bytes() == b"old file"
    assert not existing.with_name(f".{existing.stem}.repair{existing.suffix}").exists()


def test_bad_download_is_recoverable_failure_class() -> None:
    assert classify_failure("downloaded file is not decodable: ffmpeg produced no frames") == "bad_download"
    assert "bad_download" in DEFAULT_FAILURE_CLASSES
