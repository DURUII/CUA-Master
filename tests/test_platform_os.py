from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuamaster.seeding.scoring.base import parse_platform_os
from cuamaster.seeding.scoring import platform_os


def test_parse_platform_os_normalizes_aliases() -> None:
    parsed = parse_platform_os('{"platform":"browser","os":"macOS","reason":"Safari UI"}')

    assert parsed == {
        "platform": "website",
        "os": "MacOS&iOS",
        "reason": "Safari UI",
    }


def test_classify_one_writes_empty_labels_for_missing_video(tmp_path: Path) -> None:
    record = {
        "video_id": "abc123def45",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
        "source": "data1",
    }
    args = argparse.Namespace(require_transcript=False, include_diagnostics=False)

    row, status = platform_os._classify_one(
        record,
        args,
        tmp_path / "videos",
        tmp_path / "transcripts",
    )

    assert status == "missing"
    assert row["platform"] is None
    assert row["os"] is None
    assert "platform_os_error" not in row
    assert row["video_id"] == record["video_id"]


def test_main_resumes_existing_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    row = {
        "video_id": "abc123def45",
        "duration_sec": 10,
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
        "source": "data1",
    }
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    existing = dict(row, platform="desktop", os="Windows")
    output_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    platform_os.main([
        "--input", str(input_path),
        "--output", str(output_path),
        "--video-dir", str(tmp_path / "videos"),
        "--limit", "1",
    ])

    assert output_path.read_text(encoding="utf-8").count("\n") == 1


def test_require_transcript_defers_without_writing(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    row = {
        "video_id": "abc123def45",
        "duration_sec": 10,
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
        "source": "data1",
    }
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    platform_os.main([
        "--input", str(input_path),
        "--output", str(output_path),
        "--video-dir", str(tmp_path / "videos"),
        "--require-transcript",
        "--limit", "1",
    ])

    assert not output_path.exists() or output_path.read_text(encoding="utf-8") == ""


def test_empty_existing_transcript_is_classified_from_frames(tmp_path: Path, monkeypatch) -> None:
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    record = {
        "video_id": "abc123def45",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
        "source": "data1",
    }
    video_path = video_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.mp4"
    transcript_path = transcript_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.segments.json"
    video_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake video")
    transcript_path.write_text('{"segments": []}', encoding="utf-8")

    class FakeBackend:
        def classify_platform_os(self, frames, transcript, meta):
            assert transcript is None
            return {"platform": "website", "os": "Windows", "reason": "test"}

    monkeypatch.setattr(platform_os, "probe_duration", lambda path: 12.0)
    monkeypatch.setattr(platform_os, "extract_frames", lambda video, out, n, width, duration=None: [out / "frame.jpg"])
    monkeypatch.setattr(platform_os, "make_backend", lambda *args, **kwargs: FakeBackend())
    platform_os._worker_state.__dict__.clear()
    args = argparse.Namespace(
        require_transcript=True,
        include_diagnostics=False,
        backend="anthropic",
        model=None,
        base_url=None,
        num_frames=1,
        frame_width=64,
    )

    row, status = platform_os._classify_one(record, args, video_dir, transcript_dir)

    assert status == "ok"
    assert row["platform"] == "website"
    assert row["os"] == "Windows"
