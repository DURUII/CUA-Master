from __future__ import annotations

from scripts.prepare_osworld_trajectory_swift_sft import (
    action_to_qwen_calls,
    load_reference_system_prompt,
    qwen_coordinate,
    split_examples_by_group,
)


def test_qwen_coordinate_converts_1000_grid_to_999_grid() -> None:
    assert qwen_coordinate(0) == 0
    assert qwen_coordinate(500) == 500
    assert qwen_coordinate(1000) == 999


def test_action_to_qwen_calls_maps_scroll_direction() -> None:
    calls = action_to_qwen_calls(
        {
            "type": "computer.scroll",
            "args": {"dx": 0, "dy": 700, "x": 500, "y": 500},
        }
    )

    assert len(calls) == 1
    assert "<parameter=action>\nscroll" in calls[0]
    assert "<parameter=pixels>\n-50" in calls[0]


def test_action_to_qwen_calls_maps_drag_as_move_then_drag() -> None:
    calls = action_to_qwen_calls(
        {
            "type": "computer.drag",
            "args": {"x1": 100, "y1": 200, "x2": 800, "y2": 900},
        }
    )

    assert len(calls) == 2
    assert "<parameter=action>\nmouse_move" in calls[0]
    assert "[100, 200]" in calls[0]
    assert "<parameter=action>\nleft_click_drag" in calls[1]
    assert "[799, 899]" in calls[1]


def test_group_split_keeps_source_variants_together() -> None:
    rows = [
        {"task_id": "a1", "split_group_id": "a"},
        {"task_id": "a2", "split_group_id": "a"},
        {"task_id": "b1", "split_group_id": "b"},
        {"task_id": "c1", "split_group_id": "c"},
    ]

    train, val, val_groups = split_examples_by_group(rows, val_ratio=0.34, seed=7)

    assert {row["split_group_id"] for row in train}.isdisjoint(val_groups)
    assert {row["split_group_id"] for row in val} == val_groups
    assert len(train) + len(val) == len(rows)


def test_load_reference_system_prompt(tmp_path) -> None:
    path = tmp_path / "reference.jsonl"
    path.write_text(
        '{"messages":[{"role":"system","content":"REFERENCE"}]}\n',
        encoding="utf-8",
    )

    assert load_reference_system_prompt(path) == "REFERENCE"
