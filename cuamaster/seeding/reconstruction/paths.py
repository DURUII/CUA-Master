"""Path resolution helpers for action construction."""
from __future__ import annotations

import json
from pathlib import Path

from ..common.core import target_path
from .schema import TranscriptSegment, VideoTask


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("video_id"):
                raise ValueError(f"missing video_id at {path}:{line_no}")
            records.append(obj)
    return records


def transcript_path_for_video(video_path: Path, video_dir: Path, transcript_dir: Path) -> Path:
    try:
        relative = video_path.relative_to(video_dir)
    except ValueError:
        relative = Path(video_path.name)
    return (transcript_dir / relative).with_suffix(".segments.json")


def resolve_video_task(record: dict, video_dir: Path, transcript_dir: Path) -> VideoTask:
    video_path = target_path(record, video_dir)
    transcript_path = transcript_path_for_video(video_path, video_dir, transcript_dir)
    try:
        relative = video_path.relative_to(video_dir)
    except ValueError:
        relative = Path(video_path.name)
    return VideoTask(
        record=record,
        video_path=video_path,
        transcript_path=transcript_path,
        relative_path=relative,
    )


def load_transcript_segments(path: Path) -> list[TranscriptSegment]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segments: list[TranscriptSegment] = []
    for raw in data.get("segments") or []:
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                id=int(raw.get("id") or len(segments) + 1),
                start=float(raw.get("start") or 0.0),
                end=float(raw.get("end") or 0.0),
                text=text,
            )
        )
    return segments
