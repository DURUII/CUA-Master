from __future__ import annotations

import json
from pathlib import Path

from cuamaster.runtime import CanonicalEpisodeRunner, QwenXmlActionParser, ToolRegistry
from scripts.prepare_traj_collection_canonical_runtime_sft import (
    build_dataset,
    prepare_episode,
    runtime_system_prompt,
)


def test_build_dataset_matches_qwen_xml_runtime_prompts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "run" / "dataset"
    image_path = dataset_root / "mixed" / "images" / "screen.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    episodes = dataset_root / "mixed" / "episodes.jsonl"
    episode = {
        "episode_id": "episode-1",
        "task": "Create a file and finish.",
        "steps": [
            {
                "action": {
                    "type": "computer.click",
                    "args": {"x": 500, "y": 500, "button": "left"},
                },
                "observation": {"screenshot": {"path": "images/screen.png"}},
                "execution": {"ok": True},
            },
            {
                "action": {
                    "type": "computer.scroll",
                    "args": {"dx": 0, "dy": 500, "x": 500, "y": 500},
                },
                "observation": {"screenshot": {"path": "images/screen.png"}},
                "execution": {"ok": True},
            },
            {
                "action": {
                    "type": "shell.exec",
                    "args": {"command": "printf done > /home/user/Desktop/result.txt"},
                },
                "observation": {},
                "execution": {"ok": True, "output": "created"},
            },
            {
                "action": {"type": "finish", "args": {"status": "success"}},
                "observation": {},
                "execution": {"ok": True},
            },
        ],
    }
    episodes.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"

    stats = build_dataset([dataset_root], output, max_target_chars=4000)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    actions = [
        QwenXmlActionParser().parse(row["messages"][-1]["content"])
        for row in rows
    ]
    assert [action.type for action in actions] == [
        "computer.click",
        "computer.scroll",
        "shell.exec",
        "finish",
    ]
    assert actions[1].args["dy"] == 50
    history = rows[2]["messages"][1]["content"]
    assert '"dy": 50.0' in history
    assert "pyautogui.scroll(50.0)" in history
    assert all(
        row["images"] == [
            f"output.assets/images/{row['image_metadata'][0]['sha256']}.png"
        ]
        for row in rows
    )
    assert all((tmp_path / row["images"][0]).is_file() for row in rows)
    assert all(row["episode_id"] == "episode-1" for row in rows)
    assert [row["step_index"] for row in rows] == [1, 2, 3, 4]
    assert all(
        row["messages"][0]["content"]
        == runtime_system_prompt("qwen_xml")
        for row in rows
    )
    runner = CanonicalEpisodeRunner(  # type: ignore[arg-type]
        model=object(), tool_registry=ToolRegistry(), action_protocol="qwen_xml"
    )
    assert rows[0]["messages"][1]["content"] == (
        runner._model_user_prompt("Create a file and finish.", [])
    )
    assert rows[0]["messages"][-1]["content"].startswith("Action:")
    assert "<think>" not in rows[0]["messages"][-1]["content"]
    assert stats["action_protocol"] == "qwen_xml"
    assert "tool_counts" not in stats
    assert stats["max_scroll_delta"] == 50
    assert stats["clipped_scroll_actions"] == 1
    assert stats["clipped_scroll_axes"] == {"dy": 1}
    assert stats["copy_images"] is True


def test_build_dataset_keeps_initial_shell_as_text_only_row(tmp_path: Path) -> None:
    dataset_root = tmp_path / "run" / "dataset"
    episodes = dataset_root / "mixed" / "episodes.jsonl"
    episodes.parent.mkdir(parents=True)
    episodes.write_text(json.dumps({
        "episode_id": "shell-first",
        "task_id": "task-1",
        "split_group_id": "source-1",
        "task": "Inspect the workspace.",
        "steps": [{
            "action": {"type": "shell.exec", "args": {"command": "pwd"}},
            "observation": {},
            "execution": {"ok": True, "output": "/home/user/Desktop"},
        }],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"

    stats = build_dataset([dataset_root], output, max_target_chars=4000)

    row = json.loads(output.read_text())
    assert row["images"] == []
    assert not row["messages"][1]["content"].startswith("<image>")
    assert row["split_group_id"] == "source-1"
    assert stats["text_only_rows"] == 1


def test_build_dataset_drops_wait_without_visual_change(tmp_path: Path) -> None:
    dataset_root = tmp_path / "run" / "dataset"
    image_path = dataset_root / "mixed" / "images" / "screen.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    episodes = dataset_root / "mixed" / "episodes.jsonl"
    episodes.write_text(json.dumps({
        "episode_id": "wait-episode",
        "task": "Wait and finish.",
        "steps": [
            {
                "action": {"type": "computer.wait", "args": {"seconds": 1}},
                "observation": {"screenshot": {"path": "images/screen.png", "sha256": "same"}},
                "post_observation": {"screenshot": {"path": "images/screen.png", "sha256": "same"}},
                "execution": {"ok": True},
            },
            {
                "action": {"type": "finish", "args": {"status": "success"}},
                "observation": {"screenshot": {"path": "images/screen.png"}},
                "execution": {"ok": True},
            },
        ],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"

    stats = build_dataset([dataset_root], output, max_target_chars=4000)

    rows = output.read_text().splitlines()
    assert len(rows) == 1
    assert stats["unchanged_waits_dropped"] == 1


def test_build_dataset_excludes_quoted_sudo_password(tmp_path: Path) -> None:
    dataset_root = tmp_path / "run" / "dataset"
    episodes = dataset_root / "mixed" / "episodes.jsonl"
    episodes.parent.mkdir(parents=True)
    episodes.write_text(json.dumps({
        "episode_id": "unsafe-shell",
        "task": "Inspect the system.",
        "steps": [{
            "action": {
                "type": "shell.exec",
                "args": {"command": 'echo "password" | sudo -S true'},
            },
            "observation": {},
            "execution": {"ok": True},
        }],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"

    stats = build_dataset([dataset_root], output, max_target_chars=4000)

    assert output.read_text() == ""
    assert stats["excluded_episode_reasons"] == {
        "quality_policy:hardcoded_sudo_password": 1
    }


def test_prepare_episode_rejects_empty_qwen_type_without_aborting_build(
    tmp_path: Path,
) -> None:
    prepared, reason = prepare_episode(
        tmp_path,
        {
            "steps": [
                {
                    "action": {"type": "computer.type", "args": {"text": ""}},
                    "observation": {},
                }
            ]
        },
        max_target_chars=4000,
    )

    assert prepared == []
    # Empty type text is rejected by action validation before the Qwen XML
    # renderer ever sees it; the episode is skipped without aborting the build.
    assert reason == "text must not be empty"


def test_prepare_episode_rejects_structured_tools(tmp_path: Path) -> None:
    prepared, reason = prepare_episode(
        tmp_path,
        {
            "steps": [{
                "action": {
                    "type": "tool.call",
                    "args": {"name": "Write", "args": {"file_path": "/tmp/a"}},
                },
                "observation": {},
            }]
        },
        max_target_chars=4000,
    )

    assert prepared == []
    assert reason == "structured tool forbidden in GUI+CLI collection: Write"
