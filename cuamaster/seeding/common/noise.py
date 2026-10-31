"""Shared title-based noise regexes for the GUI-relevance filters.

These classify obviously non-GUI content (game playthroughs, music, movie/TV
clips, pronunciation lessons) by title, plus an EXEMPT rule that rescues real
software tutorials. Both the conservative `noise_filter` pass and the extended
`clean` pass build on these.
"""
from __future__ import annotations

import re

GAMEPLAY = re.compile(
    r"(?i)(?:\bgameplay\b|no commentary|let'?s play|\bplaythrough\b|\bspeedrun\b"
    r"|full game|all achievements|100% walkthrough|boss fight"
    r"|\bpart\s*\d+\b.*walkthrough|walkthrough.*\bpart\s*\d+\b|通关|游戏实况)"
)
MUSIC = re.compile(
    r"(?i)(cover song|piano cover|guitar cover|drum cover|\blive (?:at|in|from)\b"
    r"|in concert|full album|official (?:music )?video|official lyric|\blyrics?\b"
    r"|\bremix\b|grade \d+ (?:piano|keyboard|guitar|drum|violin|rock|pop))"
)
MOVIETV = re.compile(
    r"(?i)(official trailer|teaser trailer|\bfull movie\b|full episode|movie clip)"
)
PRONOUNCE = re.compile(r"(?i)(how to pronounce|pronunciation of|\bpronounce\b)")

# Only used to rescue game-titled records that are real software tutorials.
EXEMPT = re.compile(
    r"(?i)(how to (?:use|install|create|set ?up|configure|add|make|build|edit|record|export|import)"
    r"|\btutorial\b|step[- ]by[- ]step|getting started)"
)


def basic_reason(title: str) -> str | None:
    """Return why a title is obvious non-GUI noise, or None to keep it."""
    if GAMEPLAY.search(title):
        if EXEMPT.search(title):
            return None
        return "gameplay"
    if MUSIC.search(title):
        return "music"
    if MOVIETV.search(title):
        return "movietv"
    if PRONOUNCE.search(title):
        return "pronounce"
    return None
