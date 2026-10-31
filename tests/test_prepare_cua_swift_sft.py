from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.prepare_cua_swift_sft import (
    DEFAULT_SYSTEM,
    action_to_code,
    convert_dataset,
    format_assistant,
    parse_args,
)


def test_finish_action_uses_opencua_termination_format() -> None:
    action = {"type": "finish", "args": {"status": "success"}}

    code = action_to_code(action)

    assert code == "computer.terminate(status='success')"
    assert format_assistant({"action": action}, code).splitlines()[-1] == (
        "Code: computer.terminate(status='success')"
    )


def test_non_gui_actions_use_canonical_code_format() -> None:
    assert action_to_code({
        "type": "shell.exec",
        "args": {"command": "pwd", "cwd": "/tmp", "timeout_sec": 5},
    }) == "shell.exec(command='pwd', cwd='/tmp', timeout_sec=5)"
    assert action_to_code({
        "type": "tool.call",
        "args": {"name": "search", "args": {"query": "CUA"}},
    }) == "tool.call(name='search', args={'query': 'CUA'})"


def test_gui_code_preserves_canonical_drag_and_scroll_args() -> None:
    assert action_to_code({
        "type": "computer.drag",
        "args": {"x1": 10, "y1": 20, "x2": 30, "y2": 40, "duration_ms": 250},
    }) == (
        "pyautogui.dragTo(x1=10, y1=20, x2=30, y2=40, "
        "button='left', duration_ms=250)"
    )
    assert action_to_code({
        "type": "computer.scroll",
        "args": {"dx": -2, "dy": 5, "x": 500, "y": 400},
    }) == "pyautogui.scroll(dx=-2, dy=5, x=500, y=400)"


def test_action_to_code_rejects_noncanonical_coordinates() -> None:
    assert action_to_code({
        "type": "computer.click",
        "args": {"x": 1200, "y": 10},
    }) is None


def test_cli_and_tool_minimums_default_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_cua_swift_sft.py",
            "--cua-root",
            "data/cua",
            "--rank-paths",
            "data/rank.jsonl",
        ],
    )

    args = parse_args()

    assert args.min_shell_actions == 0
    assert args.min_tool_actions == 0


def test_convert_dataset_keeps_shell_actions(tmp_path: Path) -> None:
    cua_root = tmp_path / "cua"
    cua_root.mkdir()
    image = tmp_path / "terminal.jpg"
    image.write_bytes(b"jpg")
    rank_path = tmp_path / "rank.jsonl"
    rank_path.write_text(json.dumps({"video_id": "video-shell"}) + "\n", encoding="utf-8")
    source = cua_root / "trajectories.train.jsonl"
    source.write_text(json.dumps({
        "video_id": "video-shell",
        "trajectory_path": str(tmp_path / "video-shell.trajectory.json"),
        "task": "Inspect the working directory",
        "steps": [{
            "index": 1,
            "intent": "print the working directory",
            "action": {"type": "shell.exec", "args": {"command": "pwd"}},
            "observation": {"screenshot": str(image)},
        }],
    }) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        rank_paths=[rank_path],
        cua_root=cua_root,
        source_name=source.name,
        limit_examples=None,
        max_history_actions=3,
        system_prompt=DEFAULT_SYSTEM,
        min_shell_actions=1,
        min_tool_actions=0,
    )
    output = tmp_path / "sft.jsonl"

    stats = convert_dataset(args, output)

    row = json.loads(output.read_text())
    assert stats.examples_written == 1
    assert stats.skipped_bad_action == 0
    assert stats.action_types_written == {"shell.exec": 1}
    assert stats.action_categories_written == {"cli": 1}
    assert "shell.exec(command='pwd')" in row["messages"][-1]["content"]


def test_convert_dataset_writes_one_terminal_example_per_trajectory(tmp_path: Path) -> None:
    cua_root = tmp_path / "cua"
    cua_root.mkdir()
    image = tmp_path / "final.jpg"
    image.write_bytes(b"jpg")
    rank_path = tmp_path / "rank.jsonl"
    rank_path.write_text(json.dumps({"video_id": "video-1"}) + "\n", encoding="utf-8")
    source = cua_root / "trajectories.train.jsonl"
    source.write_text(
        json.dumps(
            {
                "video_id": "video-1",
                "trajectory_path": str(tmp_path / "video-1.trajectory.json"),
                "task": "Complete the task",
                "steps": [
                    {
                        "index": 1,
                        "intent": "finish",
                        "action": {"type": "finish", "args": {"status": "success"}},
                        "observation": {"screenshot": str(image)},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        rank_paths=[rank_path],
        cua_root=cua_root,
        source_name=source.name,
        limit_examples=None,
        max_history_actions=3,
        system_prompt=DEFAULT_SYSTEM,
        min_shell_actions=0,
        min_tool_actions=0,
    )
    output = tmp_path / "sft.jsonl"

    stats = convert_dataset(args, output)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert stats.examples_written == 1
    assert stats.termination_examples_written == 1
    assert row["images"] == [str(image)]
    assert "computer.terminate(status='success')" in row["messages"][-1]["content"]


def test_convert_dataset_fails_action_minimums_without_replacing_output(tmp_path: Path) -> None:
    cua_root = tmp_path / "cua"
    cua_root.mkdir()
    image = tmp_path / "screen.jpg"
    image.write_bytes(b"jpg")
    rank_path = tmp_path / "rank.jsonl"
    rank_path.write_text(json.dumps({"video_id": "video-gui"}) + "\n", encoding="utf-8")
    source = cua_root / "trajectories.train.jsonl"
    source.write_text(json.dumps({
        "video_id": "video-gui",
        "task": "Click",
        "steps": [{
            "index": 1,
            "action": {"type": "computer.click", "args": {"x": 500, "y": 500}},
            "observation": {"screenshot": str(image)},
        }],
    }) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        rank_paths=[rank_path],
        cua_root=cua_root,
        source_name=source.name,
        limit_examples=None,
        max_history_actions=3,
        system_prompt=DEFAULT_SYSTEM,
        min_shell_actions=1,
        min_tool_actions=1,
    )
    output = tmp_path / "sft.jsonl"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="shell.exec=0 < 1; tool.call=0 < 1"):
        convert_dataset(args, output)

    assert output.read_text(encoding="utf-8") == "existing\n"
    stats = json.loads((tmp_path / "sft.stats.json").read_text(encoding="utf-8"))
    assert stats["passed"] is False
    assert stats["stats"]["action_types_written"] == {"computer.click": 1}


def test_convert_dataset_counts_and_validates_individual_tools(tmp_path: Path) -> None:
    cua_root = tmp_path / "cua"
    cua_root.mkdir()
    image = tmp_path / "screen.jpg"
    image.write_bytes(b"jpg")
    rank_path = tmp_path / "rank.jsonl"
    rank_path.write_text(json.dumps({"video_id": "video-tool"}) + "\n", encoding="utf-8")
    source = cua_root / "trajectories.train.jsonl"
    source.write_text(
        json.dumps(
            {
                "video_id": "video-tool",
                "task": "Inspect the UI",
                "steps": [
                    {
                        "index": 1,
                        "action": {
                            "type": "tool.call",
                            "args": {"name": "ui.inspect", "args": {"query": "Save"}},
                        },
                        "observation": {"screenshot": str(image)},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        rank_paths=[rank_path],
        cua_root=cua_root,
        source_name=source.name,
        limit_examples=None,
        max_history_actions=3,
        system_prompt=DEFAULT_SYSTEM,
        min_shell_actions=0,
        min_tool_actions=1,
        min_tool_counts=["ui.inspect=1"],
    )
    output = tmp_path / "sft.jsonl"

    stats = convert_dataset(args, output)

    assert stats.tool_names_written == {"ui.inspect": 1}
    report = json.loads((tmp_path / "sft.stats.json").read_text(encoding="utf-8"))
    assert report["minimum_tool_counts"] == {"ui.inspect": 1}
