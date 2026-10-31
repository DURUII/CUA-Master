from __future__ import annotations

import copy
import json
import math
import re
from typing import Any


DEFAULT_MAX_SCROLL_DELTA = 50.0
_TOOLS_BLOCK_RE = re.compile(r"(<tools>\s*)(.*?)(\s*</tools>)", re.DOTALL)


def _validated_limit(max_scroll_delta: float) -> float:
    if not isinstance(max_scroll_delta, (int, float)) or isinstance(
        max_scroll_delta, bool
    ):
        raise TypeError("max_scroll_delta must be a number")
    if not math.isfinite(max_scroll_delta) or max_scroll_delta <= 0:
        raise ValueError("max_scroll_delta must be finite and > 0")
    return float(max_scroll_delta)


def sanitize_scroll_action(
    action: dict[str, Any],
    *,
    max_scroll_delta: float = DEFAULT_MAX_SCROLL_DELTA,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    max_scroll_delta = _validated_limit(max_scroll_delta)

    sanitized = copy.deepcopy(action)
    if sanitized.get("type") != "computer.scroll":
        return sanitized, ()

    args = sanitized.get("args") or {}
    clipped_axes: list[str] = []
    for axis in ("dx", "dy"):
        value = args.get(axis)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"scroll {axis} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"scroll {axis} must be finite")
        if abs(value) > max_scroll_delta:
            args[axis] = math.copysign(max_scroll_delta, value)
            clipped_axes.append(axis)
    sanitized["args"] = args
    return sanitized, tuple(clipped_axes)


def constrain_qwen_scroll_tool_schema(
    system_prompt: str,
    *,
    max_scroll_delta: float = DEFAULT_MAX_SCROLL_DELTA,
) -> str:
    max_scroll_delta = _validated_limit(max_scroll_delta)
    match = _TOOLS_BLOCK_RE.search(system_prompt)
    if match is None:
        return system_prompt
    try:
        tools = json.loads(match.group(2))
    except json.JSONDecodeError:
        return system_prompt

    entries = tools if isinstance(tools, list) else [tools]
    updated = False
    limit: int | float = (
        int(max_scroll_delta)
        if max_scroll_delta.is_integer()
        else max_scroll_delta
    )
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict) or function.get("name") != "computer_use":
            continue
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        pixels = properties.get("pixels") if isinstance(properties, dict) else None
        if not isinstance(pixels, dict):
            continue
        pixels["minimum"] = -limit
        pixels["maximum"] = limit
        pixels["description"] = (
            f"Scroll amount. Required only by action=scroll or action=hscroll. "
            f"Must be between {-limit:g} and {limit:g}."
        )
        updated = True
    if not updated:
        return system_prompt

    replacement = (
        match.group(1)
        + json.dumps(tools, ensure_ascii=False)
        + match.group(3)
    )
    return system_prompt[: match.start()] + replacement + system_prompt[match.end() :]
