from __future__ import annotations

import math

import pytest

from scripts.scroll_sanitization import (
    constrain_qwen_scroll_tool_schema,
    sanitize_scroll_action,
)


def test_sanitize_scroll_action_clips_each_axis_without_mutating_input() -> None:
    action = {
        "type": "computer.scroll",
        "args": {"dx": -500, "dy": 1200.0, "x": 500, "y": 500},
    }

    sanitized, clipped_axes = sanitize_scroll_action(action)

    assert sanitized["args"] == {"dx": -50.0, "dy": 50.0, "x": 500, "y": 500}
    assert clipped_axes == ("dx", "dy")
    assert action["args"] == {"dx": -500, "dy": 1200.0, "x": 500, "y": 500}


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_sanitize_scroll_action_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        sanitize_scroll_action(
            {"type": "computer.scroll", "args": {"dx": 0, "dy": value}}
        )


def test_constrain_qwen_scroll_tool_schema_sets_numeric_bounds() -> None:
    prompt = (
        '<tools>\n{"type":"function","function":{"name":"computer_use",'
        '"parameters":{"type":"object","properties":{"pixels":'
        '{"type":"number","description":"Scroll amount."}}}}}\n</tools>'
    )

    constrained = constrain_qwen_scroll_tool_schema(prompt)

    assert '"minimum": -50' in constrained
    assert '"maximum": 50' in constrained
    assert "Must be between -50 and 50" in constrained
