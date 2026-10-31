"""Frame extraction and transcript loading for CUA-suitability scoring.

Given a downloaded mp4, sample a handful of evenly spaced frames with ffmpeg and
(optionally) load its faster-whisper transcript so the scoring backend can judge
the video from both picture and narration.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path


def probe_duration(video_path: Path) -> float:
    """Return the video duration in seconds via ffprobe (0.0 if unknown)."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    raw = result.stdout.strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def extract_frames(
    video_path: Path,
    out_dir: Path,
    num_frames: int,
    frame_width: int,
    duration: float | None = None,
) -> list[Path]:
    """Sample `num_frames` evenly spaced JPEG frames into `out_dir`.

    Timestamps are placed at the midpoints of `num_frames` equal slices so we
    avoid the black intro/outro frames at exactly 0 and duration.
    """
    if duration is None:
        duration = probe_duration(video_path)
    if duration <= 0:
        raise RuntimeError("could not determine duration")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for i in range(num_frames):
        ts = duration * (i + 0.5) / num_frames
        dest = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={frame_width}:-2",
                str(dest),
            ],
            capture_output=True,
            text=True,
        )
        if dest.exists():
            frames.append(dest)
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    return frames


def encode_frame(frame_path: Path) -> str:
    """Base64-encode a JPEG frame for API image blocks."""
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def load_transcript_text(transcript_path: Path, max_chars: int = 6000) -> str | None:
    """Load a `.segments.json` transcript as a timestamped script, or None."""
    if not transcript_path.exists():
        return None
    try:
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    segments = data.get("segments") or []
    lines: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{seg.get('start', 0):.0f}s] {text}")
    if not lines:
        return None
    script = "\n".join(lines)
    if len(script) > max_chars:
        script = script[:max_chars] + "\n...(truncated)"
    return script
