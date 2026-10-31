from __future__ import annotations

import json

import pytest

from scripts.prepare_traj_collection_mixed_swift_sft import convert_dataset_roots


def test_convert_preserves_gui_cli_and_finish_targets(tmp_path) -> None:
    dataset_root = tmp_path / "run" / "dataset"
    image_path = dataset_root / "mixed" / "images" / "screen.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    source_path = dataset_root / "trajectories.sft.jsonl"
    actions = [
        {"type": "computer.click", "args": {"x": 100, "y": 200}},
        {"type": "shell.exec", "args": {"command": "pwd"}},
        {"type": "finish", "args": {"status": "success"}},
    ]
    with source_path.open("w", encoding="utf-8") as stream:
        for index, action in enumerate(actions):
            action_type = action["type"]
            images = ["mixed/images/screen.png"] if index == 0 else []
            user_content = "<image>\nTask" if images else "Task"
            stream.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "SYSTEM"},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": action_type},
                        ],
                        "images": images,
                        "target_action": action,
                        "modality": "visual" if index == 0 else "control",
                    }
                )
                + "\n"
            )

    output = tmp_path / "mixed.jsonl"
    stats = convert_dataset_roots([dataset_root], output)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert all(set(row) == {"messages", "images"} for row in rows)
    assert rows[0]["images"] == [str(image_path.resolve())]
    assert all("<tool_call>" in row["messages"][-1]["content"] for row in rows)
    assert stats["action_protocol"] == "qwen_xml"
    assert stats["action_counts"] == {action["type"]: 1 for action in actions}


def test_convert_rejects_structured_tool_targets(tmp_path) -> None:
    dataset_root = tmp_path / "run" / "dataset"
    dataset_root.mkdir(parents=True)
    source_path = dataset_root / "trajectories.sft.jsonl"
    source_path.write_text(
        json.dumps({
            "messages": [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "Task"},
                {"role": "assistant", "content": "tool"},
            ],
            "images": [],
            "target_action": {
                "type": "tool.call",
                "args": {"name": "lookup", "args": {}},
            },
            "modality": "tool",
        })
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported GUI\\+CLI action type"):
        convert_dataset_roots([dataset_root], tmp_path / "mixed.jsonl")
