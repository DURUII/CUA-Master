"""Screenshot extraction for draft CUA trajectory steps."""
from __future__ import annotations

import subprocess
from pathlib import Path


def extract_screenshot(
    video_path: Path,
    dest: Path,
    timestamp: float,
    *,
    frame_width: int = 1280,
) -> Path:
    """Extract one JPEG frame at ``timestamp``.

    The caller owns timestamp selection. This helper keeps ffmpeg invocation
    deterministic and raises if no frame is produced.
    """
    timestamp = max(0.0, float(timestamp))
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{timestamp:.3f}",
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
    if not dest.exists():
        message = result.stderr.strip() or "ffmpeg produced no frame"
        raise RuntimeError(message)
    return dest


def frame_dir_for_trajectory(trajectory_path: Path) -> Path:
    return trajectory_path.with_suffix(".frames")


def step_frame_path(frame_dir: Path, step_index: int, kind: str) -> Path:
    if kind != "state":
        raise ValueError(f"unsupported frame kind: {kind}")
    return frame_dir / f"step_{step_index:04d}_state.jpg"
