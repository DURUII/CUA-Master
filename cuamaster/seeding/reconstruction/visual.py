"""Lightweight visual event mining.

This module intentionally produces broad state-change candidates, not final
action labels. A model backend can later refine these windows into grounded CUA
actions.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .schema import ActionCandidate


_SCENE_RE = re.compile(r"pts_time:(?P<time>[0-9.]+)")


def mine_scene_change_candidates(
    video_path: Path,
    *,
    threshold: float = 0.18,
    pre_roll: float = 0.8,
    post_roll: float = 1.2,
    max_candidates: int = 200,
) -> list[ActionCandidate]:
    """Return visual candidates around ffmpeg scene-change events."""
    cmd = [
        "ffmpeg",
        "-v",
        "info",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    raw = result.stderr or ""
    candidates: list[ActionCandidate] = []
    seen_times: set[int] = set()
    for match in _SCENE_RE.finditer(raw):
        timestamp = float(match.group("time"))
        bucket = int(timestamp * 2)
        if bucket in seen_times:
            continue
        seen_times.add(bucket)
        candidates.append(
            ActionCandidate(
                start=max(0.0, timestamp - pre_roll),
                end=timestamp + post_roll,
                source="visual",
                visual_score=0.55,
                confidence=0.55,
                evidence={"visual_event": "scene_change", "timestamp": timestamp},
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates
