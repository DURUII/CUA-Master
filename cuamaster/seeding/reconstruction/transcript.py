"""Semantic action-candidate mining from timestamped transcripts."""
from __future__ import annotations

import re
from collections.abc import Iterable

from .schema import ActionCandidate, TranscriptSegment


_SPLIT_RE = re.compile(r"\b(?:and then|then|after that|next|and)\b|[,.;]", re.IGNORECASE)

_ACTION_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (
        "shell.exec",
        re.compile(
            r"\b(?:run|execute|issue)\s+(?:(?:the|this|following)\s+)?"
            r"(?:(?:shell|terminal|command[- ]line)\s+)?command\b",
            re.I,
        ),
        0.88,
    ),
    ("computer.double_click", re.compile(r"\bdouble[- ]?click(?:ed|ing)?\b", re.I), 0.92),
    ("computer.click", re.compile(r"\b(?:click|tap|select|choose|check|uncheck|open)\b", re.I), 0.84),
    ("computer.drag", re.compile(r"\b(?:drag|drop|dragging|resize|move the slider|slide)\b", re.I), 0.82),
    ("computer.scroll", re.compile(r"\b(?:scroll|swipe)\b", re.I), 0.80),
    ("computer.press", re.compile(r"\b(?:press|hit)\b", re.I), 0.76),
    ("computer.type", re.compile(r"\b(?:type|enter|input|write|paste)\b", re.I), 0.78),
    ("computer.wait", re.compile(r"\b(?:wait|let it load|loading|takes a moment)\b", re.I), 0.62),
]

_NON_ACTION_RE = re.compile(
    r"\b(?:important|remember|notice|you can see|as you can see|this means|because|for example|talk about)\b",
    re.I,
)
_FALSE_CHOICE_RE = re.compile(
    r"\b(?:why\s+)?(?:people|users|customers|companies|they|we)\s+"
    r"(?:continue\s+to\s+|often\s+|usually\s+)?choose\b",
    re.I,
)
_QUOTE_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{1,120})[\"'“”‘’]")
_KEY_RE = re.compile(
    r"\b(?:press|hit)\s+(?:the\s+)?(?P<keys>enter|return|tab|escape|esc|delete|backspace|space|ctrl\s*\+\s*s|cmd\s*\+\s*s|control\s*\+\s*s)\b",
    re.I,
)


def mine_semantic_candidates(
    segments: Iterable[TranscriptSegment],
    *,
    pre_roll: float = 1.0,
    post_roll: float = 1.5,
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    for segment in segments:
        clauses = _split_clauses(segment.text)
        if not clauses:
            continue
        width = max(segment.end - segment.start, 0.1) / len(clauses)
        for clause_index, clause in enumerate(clauses):
            candidate = _candidate_from_clause(segment, clause, clause_index, width, pre_roll, post_roll)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _split_clauses(text: str) -> list[str]:
    return [part.strip(" ,") for part in _SPLIT_RE.split(text) if part.strip(" ,")]


def _candidate_from_clause(
    segment: TranscriptSegment,
    clause: str,
    clause_index: int,
    clause_width: float,
    pre_roll: float,
    post_roll: float,
) -> ActionCandidate | None:
    lowered = clause.lower()
    match_info: tuple[str, re.Match[str], float] | None = None
    for action_type, pattern, base_score in _ACTION_PATTERNS:
        match = pattern.search(clause)
        if match:
            match_info = (action_type, match, base_score)
            break
    if match_info is None:
        return None

    action_type, match, base_score = match_info
    if action_type == "computer.press" and "mouse" in lowered:
        return None
    if action_type == "computer.press" and _KEY_RE.search(clause) is None:
        action_type = "computer.click"
        base_score = 0.82
    if action_type == "computer.type" and _looks_like_type_noun(clause, match):
        return None
    if match.group(0).lower().startswith("choose") and _FALSE_CHOICE_RE.search(clause):
        return None

    offset_start = segment.start + clause_width * clause_index
    offset_end = min(segment.end, offset_start + clause_width)
    start = max(0.0, offset_start - pre_roll)
    end = max(start + 0.1, offset_end + post_roll)
    args = _draft_args(action_type, clause)
    target = _target_phrase(clause, match.end())
    score = base_score
    if _NON_ACTION_RE.search(clause):
        score -= 0.45
    if not target and action_type in {"computer.click", "computer.double_click", "computer.drag"}:
        score -= 0.10
    score = max(0.05, min(1.0, score))
    return ActionCandidate(
        start=start,
        end=end,
        source="semantic",
        action_type=action_type,
        text=clause,
        target=target,
        args=args,
        semantic_score=score,
        confidence=score,
        evidence={"transcript_segment_id": segment.id, "transcript": segment.text},
    )


def _target_phrase(clause: str, action_end: int) -> str:
    phrase = clause[action_end:].strip(" ,")
    phrase = re.sub(r"^(?:(?:on|to|into|in|at|the|a|an)\s+)+", "", phrase, flags=re.I)
    phrase = re.sub(r"\b(?:please|right here|here|there)$", "", phrase, flags=re.I).strip(" ,")
    return phrase[:160]


def _draft_args(action_type: str, clause: str) -> dict:
    if action_type == "shell.exec":
        quoted = re.search(r"[`\"'“”‘’]([^`\"'“”‘’]{1,500})[`\"'“”‘’]", clause)
        return {"command": quoted.group(1)} if quoted else {}
    if action_type == "computer.scroll":
        lowered = clause.lower()
        dy = -5 if "down" in lowered else 5 if "up" in lowered else 0
        dx = -5 if "right" in lowered else 5 if "left" in lowered else 0
        if dx == 0 and dy == 0:
            dy = -5
        return {"dx": dx, "dy": dy}
    if action_type == "computer.type":
        quoted = _QUOTE_RE.search(clause)
        if quoted:
            return {"text": quoted.group(1)}
        match = re.search(r"\b(?:type|enter|input|write|paste)\s+(.+)$", clause, re.I)
        if match:
            text = match.group(1).strip(" .")
            if 0 < len(text) <= 80 and not re.search(r"\b(?:field|box|column|button|menu)\b", text, re.I):
                return {"text": text}
        return {}
    if action_type == "computer.press":
        match = _KEY_RE.search(clause)
        if match:
            return {"keys": _normalize_keys(match.group("keys"))}
    if action_type == "computer.wait":
        return {"seconds": 1.0}
    return {}


def _looks_like_type_noun(clause: str, match: re.Match[str]) -> bool:
    if match.group(0).lower() != "type":
        return False
    prefix = clause[: match.start()].lower()
    suffix = clause[match.end() :].lower()
    return bool(re.search(r"\b(?:comment|field|data|record|payment|vacation)\s+$", prefix)) or suffix.lstrip().startswith("of ")


def _normalize_keys(raw: str) -> list[str]:
    key = re.sub(r"\s+", "", raw.lower())
    aliases = {
        "return": ["enter"],
        "esc": ["escape"],
        "control+s": ["ctrl", "s"],
        "ctrl+s": ["ctrl", "s"],
        "cmd+s": ["cmd", "s"],
    }
    if "+" in key:
        return aliases.get(key, key.split("+"))
    return aliases.get(key, [key])
