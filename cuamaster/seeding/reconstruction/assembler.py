"""Build draft CUA trajectories from action candidates."""
from __future__ import annotations

import re
from statistics import mean

from .schema import ActionCandidate, DraftTrajectory, TrajectoryStep, VideoTask


_COORD_ACTIONS = {
    "computer.click",
    "computer.double_click",
    "computer.move",
    "computer.drag",
}


def build_draft_trajectory(
    task: VideoTask,
    candidates: list[ActionCandidate],
    *,
    min_confidence: float = 0.45,
) -> DraftTrajectory:
    kept = [c for c in candidates if c.action_type and c.confidence >= min_confidence]
    kept = _reduce_repetitions(sorted(kept, key=lambda c: (c.start, c.end)))
    steps = [_step_from_candidate(i + 1, candidate) for i, candidate in enumerate(kept)]
    confidences = [step.confidence for step in steps]
    record = task.record
    return DraftTrajectory(
        video_id=str(record.get("video_id") or ""),
        application=str(record.get("application") or ""),
        category=str(record.get("category") or ""),
        subcategory=str(record.get("subcategory") or ""),
        source_video=str(task.video_path),
        source_transcript=str(task.transcript_path),
        task=infer_task(record, candidates),
        steps=steps,
        quality={
            "trajectory_confidence": round(mean(confidences), 4) if confidences else 0.0,
            "num_steps": len(steps),
            "needs_grounding_steps": sum(1 for step in steps if step.needs_grounding),
            "candidate_count": len(candidates),
            "kept_candidate_count": len(kept),
            "has_state_verification": False,
            "draft": True,
        },
    )


def infer_task(record: dict, candidates: list[ActionCandidate]) -> str:
    title = str(record.get("title") or "").strip()
    if title:
        return title[:180]
    for candidate in candidates:
        if candidate.text:
            return _clean_task_text(candidate.text)
    app = record.get("application") or "the application"
    return f"Operate {app} from the tutorial video"


def _step_from_candidate(index: int, candidate: ActionCandidate) -> TrajectoryStep:
    action = {"type": candidate.action_type, "args": dict(candidate.args)}
    needs_grounding = _needs_grounding(candidate)
    evidence = dict(candidate.evidence)
    evidence.update(
        {
            "source": candidate.source,
            "target": candidate.target,
            "semantic_score": candidate.semantic_score,
            "visual_score": candidate.visual_score,
        }
    )
    return TrajectoryStep(
        index=index,
        start=round(candidate.start, 3),
        end=round(candidate.end, 3),
        intent=_intent(candidate),
        action=action,
        needs_grounding=needs_grounding,
        confidence=round(candidate.confidence, 4),
        evidence=evidence,
    )


def _needs_grounding(candidate: ActionCandidate) -> bool:
    if candidate.action_type in _COORD_ACTIONS:
        args = candidate.args
        if candidate.action_type == "computer.drag":
            return not all(key in args for key in ("x1", "y1", "x2", "y2"))
        return not all(key in args for key in ("x", "y"))
    if candidate.action_type == "computer.type":
        return "text" not in candidate.args
    if candidate.action_type == "computer.press":
        return "keys" not in candidate.args
    if candidate.action_type == "shell.exec":
        command = candidate.args.get("command")
        return not isinstance(command, str) or not command.strip()
    if candidate.action_type == "tool.call":
        name = candidate.args.get("name")
        return (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(candidate.args.get("args"), dict)
        )
    return False


def _intent(candidate: ActionCandidate) -> str:
    if candidate.target:
        verb = candidate.action_type.rsplit(".", 1)[-1] if candidate.action_type else "act"
        return f"{verb.replace('_', ' ')} {candidate.target}".strip()
    if candidate.text:
        return _clean_task_text(candidate.text)
    return candidate.action_type or "perform CUA action"


def _clean_task_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        return "Operate the visible computer environment"
    return text[:1].upper() + text[1:180]


def _reduce_repetitions(candidates: list[ActionCandidate]) -> list[ActionCandidate]:
    reduced: list[ActionCandidate] = []
    for candidate in candidates:
        if reduced and _looks_duplicate(reduced[-1], candidate):
            if candidate.confidence > reduced[-1].confidence:
                reduced[-1] = candidate
            continue
        reduced.append(candidate)
    return reduced


def _looks_duplicate(left: ActionCandidate, right: ActionCandidate) -> bool:
    if left.action_type != right.action_type:
        return False
    if right.start - left.end > 2.0:
        return False
    left_text = left.target or left.text
    right_text = right.target or right.text
    return bool(left_text and right_text and left_text.lower() == right_text.lower())
