"""Merge semantic and visual action candidates into a boundary lattice."""
from __future__ import annotations

from .schema import ActionCandidate


def merge_candidates(
    candidates: list[ActionCandidate],
    *,
    overlap_tolerance: float = 1.5,
) -> list[ActionCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (c.start, c.end, c.source))
    merged: list[ActionCandidate] = []
    current = ordered[0]
    for candidate in ordered[1:]:
        if _should_merge(current, candidate, overlap_tolerance):
            current = _merge_pair(current, candidate)
        else:
            merged.append(current)
            current = candidate
    merged.append(current)
    return merged


def _should_merge(left: ActionCandidate, right: ActionCandidate, tolerance: float) -> bool:
    if right.start > left.end + tolerance:
        return False
    if {left.source, right.source} == {"semantic", "visual"}:
        return True
    if left.source == right.source == "semantic":
        return _same_semantic_target(left, right)
    if left.action_type is None or right.action_type is None or left.action_type == right.action_type:
        return True
    return False


def _same_semantic_target(left: ActionCandidate, right: ActionCandidate) -> bool:
    if left.action_type != right.action_type:
        return False
    left_target = _normalized_target(left)
    right_target = _normalized_target(right)
    return bool(left_target and left_target == right_target)


def _normalized_target(candidate: ActionCandidate) -> str:
    return " ".join((candidate.target or candidate.text).lower().split())


def _merge_pair(left: ActionCandidate, right: ActionCandidate) -> ActionCandidate:
    semantic = left if left.source == "semantic" else right if right.source == "semantic" else None
    preferred = semantic or (left if left.confidence >= right.confidence else right)
    semantic_score = max(left.semantic_score, right.semantic_score)
    visual_score = max(left.visual_score, right.visual_score)
    alignment_bonus = 0.12 if semantic_score > 0 and visual_score > 0 else 0.0
    confidence = min(1.0, max(left.confidence, right.confidence, semantic_score, visual_score) + alignment_bonus)
    evidence = {
        "merged_from": [left.to_dict(), right.to_dict()],
        "semantic_visual_aligned": semantic_score > 0 and visual_score > 0,
    }
    return ActionCandidate(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
        source="merged",
        action_type=preferred.action_type,
        text=preferred.text or left.text or right.text,
        target=preferred.target or left.target or right.target,
        args=preferred.args or left.args or right.args,
        semantic_score=semantic_score,
        visual_score=visual_score,
        confidence=confidence,
        evidence=evidence,
    )
