"""Scoring backends for CUA-suitability rating.

A backend takes sampled frames + optional transcript + taxonomy metadata and
returns a structured rating. All backends share the same rubric so scores from
the cloud (`anthropic`) and local (`qwen_vl`) models are comparable.
"""
from __future__ import annotations

import json
import re

RUBRIC = """You are rating how suitable a video is as training data for a \
Computer-Use Agent (CUA). CUA data requires real computer-use screen recordings \
that show a step-by-step operation trajectory, such as clicks, typing, menus, \
and dialogs, that an agent could learn to reproduce.

You are given a few evenly spaced frames sampled across the video and, \
optionally, its transcript. Judge the whole video from this evidence.

Score the overall CUA suitability using an integer from 0 to 5:
- 5: A clear, high-resolution software CUA screencast with an explicit \
step-by-step operation trajectory; ideal CUA training data.
- 3-4: Mostly CUA operation, but with minor issues such as talking-head \
segments, slides, incomplete trajectories, or moderate visual clarity.
- 1-2: Only incidental CUA content, predominantly non-operational content, \
or poor visual clarity.
- 0: No software CUA trajectory, such as game footage, real-world video, \
music videos, movies, or pure talking-head content.

Additionally assess:
- is_cua: whether the video primarily presents a real software computer-use \
interaction.
- has_trajectory: whether it contains a reproducible step-by-step operation.
- clarity: an integer from 0 to 5 indicating UI legibility.

Respond with only the following JSON object:
{"score": <0-5 integer>, "is_cua": <boolean>, "has_trajectory": <boolean>, \
"clarity": <0-5 integer>, "reason": "<one concise sentence>"}"""

PLATFORM_OS_RUBRIC = """You classify a YouTube software-operation video by the platform and operating system shown in the sampled frames and optional transcript.

Use exactly one platform label:
- desktop: native desktop software or a desktop browser UI shown on a PC/laptop screen.
- mobile: a phone/tablet app or mobile browser UI, including Android emulator/iOS simulator when the operated UI is mobile.
- website: a web application/browser page where the browser/URL bar or web-page context is clearly the relevant operated surface and the OS is not the main signal.

Use exactly one os label:
- Windows
- MacOS&iOS
- Android
- Linux

Guidance:
- Choose MacOS&iOS for either macOS desktop or iPhone/iPad/iOS UI.
- For website videos, still infer the visible host OS when clear from window chrome, taskbar/dock, browser style, transcript, or context. If the OS is not visible, use the most likely OS from evidence and explain uncertainty.
- Prefer visual evidence over transcript claims when they conflict.

Respond with ONLY a JSON object, no other text:
{"platform": "desktop|mobile|website", "os": "Windows|MacOS&iOS|Android|Linux", "reason": "<one concise sentence>"}"""


def build_user_text(meta: dict, transcript: str | None) -> str:
    """Compose the per-video text block: taxonomy context + transcript."""
    app = meta.get("application") or "unknown"
    category = meta.get("category") or ""
    subcategory = meta.get("subcategory") or ""
    parts = [
        f"Target application: {app}",
        f"Taxonomy: {category} / {subcategory}",
    ]
    if transcript:
        parts.append("Transcript (timestamped):\n" + transcript)
    else:
        parts.append("Transcript: (none available — judge from frames only)")
    return "\n".join(parts)


def build_platform_os_user_text(meta: dict, transcript: str | None) -> str:
    """Compose context for platform/OS classification."""
    parts = [
        f"Video ID: {meta.get('video_id') or 'unknown'}",
        f"Target application: {meta.get('application') or 'unknown'}",
        f"Taxonomy: {meta.get('category') or ''} / {meta.get('subcategory') or ''}",
    ]
    if transcript:
        parts.append("Transcript (timestamped):\n" + transcript)
    else:
        parts.append("Transcript: (none available — classify from frames only)")
    return "\n".join(parts)


def parse_rating(raw: str) -> dict:
    """Extract the JSON rating object from a model response, tolerating fences."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    obj = json.loads(match.group(0))
    score = int(obj.get("score"))
    clarity = int(obj.get("clarity", 0))
    return {
        "score": max(0, min(5, score)),
        "is_cua": bool(obj.get("is_cua", obj.get("is_gui", False))),
        "has_trajectory": bool(obj.get("has_trajectory", False)),
        "clarity": max(0, min(5, clarity)),
        "reason": str(obj.get("reason", "")).strip(),
    }


def parse_platform_os(raw: str) -> dict:
    """Extract and normalize platform/OS labels from a model response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    obj = json.loads(match.group(0))
    platform = str(obj.get("platform", "")).strip().lower()
    os_label = str(obj.get("os", "")).strip()

    platform_aliases = {
        "desktop": "desktop",
        "pc": "desktop",
        "laptop": "desktop",
        "mobile": "mobile",
        "phone": "mobile",
        "tablet": "mobile",
        "website": "website",
        "web": "website",
        "browser": "website",
    }
    os_aliases = {
        "windows": "Windows",
        "win": "Windows",
        "macos&ios": "MacOS&iOS",
        "macos": "MacOS&iOS",
        "mac os": "MacOS&iOS",
        "ios": "MacOS&iOS",
        "ipad os": "MacOS&iOS",
        "ipados": "MacOS&iOS",
        "android": "Android",
        "linux": "Linux",
        "ubuntu": "Linux",
    }
    normalized_platform = platform_aliases.get(platform)
    normalized_os = os_aliases.get(os_label.lower())
    if normalized_platform is None:
        raise ValueError(f"invalid platform label: {platform!r}")
    if normalized_os is None:
        raise ValueError(f"invalid os label: {os_label!r}")
    return {
        "platform": normalized_platform,
        "os": normalized_os,
        "reason": str(obj.get("reason", "")).strip(),
    }
