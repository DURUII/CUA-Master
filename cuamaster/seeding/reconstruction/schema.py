"""Data structures for video-to-CUA action construction.

This package keeps an intermediate schema separate from the final canonical CUA
action schema because draft video labels often know the action type and intent
before visual grounding has supplied exact coordinates or typed text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


CandidateSource = Literal["semantic", "visual", "api", "merged"]


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ActionCandidate:
    """One possible GUI, CLI, or structured-tool action in a video window."""

    start: float
    end: float
    source: CandidateSource
    action_type: str | None = None
    text: str = ""
    target: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    semantic_score: float = 0.0
    visual_score: float = 0.0
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoTask:
    """Resolved paths and metadata for one video/transcript pair."""

    record: dict[str, Any]
    video_path: Path
    transcript_path: Path
    relative_path: Path


@dataclass(frozen=True)
class TrajectoryStep:
    index: int
    start: float
    end: float
    intent: str
    action: dict[str, Any]
    needs_grounding: bool
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DraftTrajectory:
    video_id: str
    application: str
    category: str
    subcategory: str
    source_video: str
    source_transcript: str
    task: str
    steps: list[TrajectoryStep]
    quality: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        obj = asdict(self)
        obj["steps"] = [step.to_dict() for step in self.steps]
        return obj
