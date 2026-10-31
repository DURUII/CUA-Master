from __future__ import annotations

import json
import sqlite3
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cuamaster.runtime import QwenXmlActionParser
from cuamaster.collection.dataset import (
    _episode_content_flags,
    _policy_exclusion_reasons,
    _restore_source_shell_timeout,
    build_dataset,
)
from cuamaster.collection.export import export_result_tree
from cuamaster.collection.ledger import EpisodeLedger
from cuamaster.collection.profiles import VMProfile
from cuamaster.collection.quality import validate_episode
from cuamaster.collection.sampling import FrontierSampler
from cuamaster.collection.schemas import EpisodeStatus, TaskSpec
from cuamaster.collection.task_specs import verify_task_spec


def _profile(tmp_path):
    return VMProfile(
        id="ubuntu-codex",
        backend="command",
        os_type="Ubuntu",
        os_version="22.04",
        architecture="x86_64",
        provider="docker",
        image_path=str(tmp_path / "disk.qcow2"),
        image_id="ubuntu-test",
        snapshot="clean-v1",
        num_envs=1,
        image_metadata={"id": "ubuntu-test", "sha256": "abc123"},
        backend_options={"max_steps": 20, "image_strategy": "runtime-overlay"},
    )


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _png_header(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", width, height)


def test_dataset_restores_openclaw_timeout_from_source_event(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    _write_jsonl(episode / "trajectory.jsonl", [{
        "tool_name": "exec",
        "request": {"command": "sleep 1", "timeout": 1200},
    }])
    step = {"provenance": {"raw_turn": 1, "tool_name": "exec"}}
    action = {
        "type": "shell.exec",
        "args": {"command": "sleep 1", "timeout_sec": 1.2},
    }

    restored = _restore_source_shell_timeout(episode, step, action)

    assert restored["args"]["timeout_sec"] == 1200
    assert action["args"]["timeout_sec"] == 1.2


def test_task_spec_version_and_negative_control():
    task = TaskSpec.from_dict({
        "id": "writer-save",
        "instruction": "Save the document.",
        "snapshot": "clean-v1",
        "config": [],
        "evaluator": {"func": "file_exists"},
        "required_capabilities": ["libreoffice"],
    })
    assert task.resolved_version == TaskSpec.from_dict(task.to_dict()).resolved_version
    valid = verify_task_spec(
        task,
        available_capabilities=["libreoffice"],
        judge_score=0.0,
        negative_control_scores=[0.0],
    )
    invalid = verify_task_spec(
        task,
        available_capabilities=["libreoffice"],
        negative_control_scores=[1.0],
    )
    assert valid["valid"] is True
    assert invalid["valid"] is False


def test_ledger_recovers_stale_episode(tmp_path):
    ledger = EpisodeLedger(tmp_path / "ledger.sqlite3")
    ledger.create_run("run-1", config_path="config.json")
    ledger.register_episode("episode-1", run_id="run-1", profile_id="ubuntu")
    ledger.mark_running("episode-1", worker_id="worker-1")
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE episodes SET heartbeat_at=?, updated_at=? WHERE episode_id='episode-1'",
            (stale, stale),
        )
    assert ledger.recover_stale(60) == ["episode-1"]
    assert ledger.rows(run_id="run-1")[0]["status"] == EpisodeStatus.PENDING.value


def test_episode_quality_and_dataset_export(tmp_path):
    task = tmp_path / "results" / "gui" / "model" / "DOC" / "save-document"
    screenshots = task / "screenshots"
    screenshots.mkdir(parents=True)
    initial = screenshots / "initial.png"
    final = screenshots / "final.png"
    initial.write_bytes(_png_header(1280, 720))
    final.write_bytes(_png_header(1280, 720))
    _write_jsonl(task / "computer_events.jsonl", [{
        "actions": [{"type": "click", "x": 10, "y": 20}],
        "executions": [{"ok": True, "screenshot_path": str(final)}],
        "pre_screenshot_path": str(initial),
        "screenshot_path": str(final),
        "ok": True,
        "decision_context": {
            "source": "chat.jsonl",
            "visibility": "observable_assistant_output",
            "intent": "Save the document using the toolbar.",
            "text": "The document is ready. Save the document using the toolbar.",
            "chat_indices": [1, 2],
            "blocks": [
                {"chat_index": 1, "text": "The document is ready."},
                {"chat_index": 2, "text": "Save the document using the toolbar."},
            ],
        },
    }])
    source_task = {
        "id": "save-document",
        "instruction": "Save the current document.",
        "snapshot": "clean-v1",
        "config": [],
        "evaluator": {"func": "file_exists"},
        "required_capabilities": [],
        "target_environments": [{
            "os_type": "Ubuntu",
            "os_versions": ["22.04"],
            "architectures": ["x86_64"],
        }],
    }
    _write_json(task / "task.json", source_task)
    _write_json(task / "score.json", {
        "task_id": "save-document",
        "category": "DOC",
        "mode": "gui",
        "score": 1.0,
        "success": True,
        "model": "test-model",
        "harness": "codex",
    })
    _write_json(task / "execution_status.json", {
        "agent_done": True,
        "model": "test-model",
        "harness": "codex",
    })

    assert export_result_tree(tmp_path / "results", _profile(tmp_path), run_id="run-1") == 1
    quality = json.loads((task / "quality.json").read_text(encoding="utf-8"))
    steps = [
        json.loads(line)
        for line in (task / "canonical" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert quality["training_tier"] == "sft_accepted"
    assert json.loads((task / "task.json").read_text(encoding="utf-8")) == source_task
    canonical_task = json.loads(
        (task / "canonical" / "task.json").read_text(encoding="utf-8")
    )
    assert canonical_task["target_environments"][0]["os_type"] == "Ubuntu"
    assert steps[0]["pre_observation"]["screenshot"] == "screenshots/initial.png"
    assert steps[0]["decision_context"]["intent"] == (
        "Save the document using the toolbar."
    )
    assert steps[-1]["action"] == {"type": "finish", "args": {"status": "success"}}

    stats = build_dataset(tmp_path / "results", tmp_path / "dataset")
    assert stats["episodes_exported"] == 1
    assert stats["sft_examples_exported"] == 2
    canonical_stats = stats["canonical_bundle"]
    assert canonical_stats["episodes_exported"] == 1
    assert canonical_stats["steps_exported"] == 2
    assert canonical_stats["unique_images"] == 1
    assert canonical_stats["action_types"] == {"computer.click": 1, "finish": 1}

    step_rows = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "canonical" / "steps.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    episode_rows = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "canonical" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["action"]["type"] for row in step_rows] == ["computer.click", "finish"]
    assert step_rows[0]["history"] == {"actions": []}
    assert step_rows[0]["decision_context"]["visibility"] == (
        "observable_assistant_output"
    )
    assert step_rows[1]["history"]["actions"] == [step_rows[0]["action"]]
    screenshot = step_rows[0]["observation"]["screenshot"]
    assert screenshot["path"].startswith("images/")
    assert not Path(screenshot["path"]).is_absolute()
    assert screenshot["width"] == 1280
    assert screenshot["height"] == 720
    assert (tmp_path / "dataset" / "canonical" / screenshot["path"]).is_file()
    assert episode_rows[0]["complete"] is True
    assert episode_rows[0]["steps"][0]["observation"]["screenshot"] == screenshot

    sft_row = json.loads(
        (tmp_path / "dataset" / "trajectories.sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert sft_row["decision_context"]["intent"] == (
        "Save the document using the toolbar."
    )


def test_hybrid_bundle_rejects_incomplete_episode(tmp_path):
    episode = tmp_path / "results" / "episode"
    screenshots = episode / "screenshots"
    canonical = episode / "canonical"
    screenshots.mkdir(parents=True)
    canonical.mkdir()
    screenshot = screenshots / "state.png"
    screenshot.write_bytes(_png_header(800, 600))
    _write_json(episode / "episode_manifest.json", {
        "episode_id": "episode-1",
        "status": "succeeded",
        "task": {"id": "task-1", "instruction": "Complete the task.", "metadata": {}},
        "agent": {"id": "agent-1", "harness": "test", "model": "test"},
        "vm": {"id": "ubuntu", "os": {"type": "Ubuntu"}, "image": {"id": "image"}},
        "evaluation": {"success": True, "score": 1.0},
    })
    _write_json(episode / "quality.json", {
        "valid": True,
        "training_tier": "sft_accepted",
    })
    _write_jsonl(canonical / "steps.jsonl", [
        {
            "step": 1,
            "pre_observation": {"screenshot": "screenshots/state.png"},
            "action": {"type": "shell.exec", "args": {"command": "false"}},
            "execution": {"ok": False},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "partial",
        },
        {
            "step": 2,
            "pre_observation": {},
            "action": {"type": "computer.wait", "args": {"seconds": 1}},
            "execution": {"ok": True},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "partial",
        },
        {
            "step": 3,
            "pre_observation": {"screenshot": "screenshots/state.png"},
            "action": {"type": "finish", "args": {"status": "success"}},
            "execution": {"ok": True},
            "post_observation": {"screenshot": "screenshots/state.png"},
            "provenance": {},
            "observation_quality": "complete",
        },
    ])

    stats = build_dataset(tmp_path / "results", tmp_path / "dataset")
    hybrid_stats = stats["hybrid_bundle"]
    assert hybrid_stats["steps_exported"] == 0
    assert hybrid_stats["skipped_failed_action"] == 1
    assert hybrid_stats["skipped_missing_pre_screenshot"] == 1
    assert hybrid_stats["episodes_incomplete"] == 1
    assert hybrid_stats["episodes_rejected_incomplete"] == 1
    assert not (tmp_path / "dataset" / "hybrid" / "steps.jsonl").read_text()


def test_pure_shell_episode_uses_shell_schema_without_screenshots(tmp_path):
    episode = tmp_path / "results" / "episode"
    canonical = episode / "canonical"
    canonical.mkdir(parents=True)
    _write_json(episode / "episode_manifest.json", {
        "episode_id": "shell-episode",
        "status": "succeeded",
        "task": {"id": "shell-task", "instruction": "Create a file.", "metadata": {}},
        "agent": {"id": "agent-1", "harness": "test", "model": "test"},
        "vm": {"id": "ubuntu", "os": {"type": "Ubuntu"}, "image": {"id": "image"}},
        "evaluation": {"success": True, "score": 1.0},
    })
    _write_json(episode / "quality.json", {
        "valid": True,
        "training_tier": "sft_accepted",
    })
    _write_jsonl(canonical / "steps.jsonl", [
        {
            "step": 1,
            "pre_observation": {},
            "action": {"type": "shell.exec", "args": {"command": "touch demo"}},
            "execution": {"ok": True, "returncode": 0},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "complete",
        },
        {
            "step": 2,
            "pre_observation": {},
            "action": {"type": "finish", "args": {"status": "success"}},
            "execution": {"ok": True, "synthetic": True},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "complete",
        },
    ])

    stats = build_dataset(tmp_path / "results", tmp_path / "dataset")

    assert stats["episodes_routed_by_bundle"] == {
        "canonical": 0,
        "hybrid": 0,
        "shell": 1,
    }
    assert stats["shell_bundle"]["episodes_exported"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "shell" / "steps.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[0]["schema_version"] == "shell_sft.step.v1"
    assert rows[0]["modality"] == "shell"
    assert rows[0]["observation"] == {"modality": "shell", "screenshot": None}
    assert rows[0]["execution"]["returncode"] == 0


def test_mixed_episode_uses_hybrid_schema_and_visual_only_screenshot_policy(tmp_path):
    episode = tmp_path / "results" / "episode"
    screenshots = episode / "screenshots"
    canonical = episode / "canonical"
    screenshots.mkdir(parents=True)
    canonical.mkdir()
    screenshot = screenshots / "state.png"
    screenshot.write_bytes(_png_header(800, 600))
    _write_json(episode / "episode_manifest.json", {
        "episode_id": "hybrid-episode",
        "status": "succeeded",
        "task": {"id": "hybrid-task", "instruction": "Inspect and edit.", "metadata": {}},
        "agent": {"id": "agent-1", "harness": "test", "model": "test"},
        "vm": {"id": "ubuntu", "os": {"type": "Ubuntu"}, "image": {"id": "image"}},
        "evaluation": {"success": True, "score": 1.0},
    })
    _write_json(episode / "quality.json", {
        "valid": True,
        "training_tier": "sft_accepted",
    })
    _write_jsonl(canonical / "steps.jsonl", [
        {
            "step": 1,
            "pre_observation": {"screenshot": "screenshots/state.png"},
            "action": {"type": "computer.click", "args": {"x": 10, "y": 20}},
            "execution": {"ok": True},
            "post_observation": {"screenshot": "screenshots/state.png"},
            "provenance": {},
            "observation_quality": "complete",
        },
        {
            "step": 2,
            "pre_observation": {},
            "action": {"type": "shell.exec", "args": {"command": "cat demo"}},
            "execution": {"ok": True, "returncode": 0, "stdout": "done"},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "complete",
        },
        {
            "step": 3,
            "pre_observation": {},
            "action": {"type": "finish", "args": {"status": "success"}},
            "execution": {"ok": True, "synthetic": True},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "complete",
        },
    ])

    stats = build_dataset(tmp_path / "results", tmp_path / "dataset")

    assert stats["hybrid_bundle"]["episodes_exported"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "hybrid" / "steps.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["modality"] for row in rows] == ["visual", "shell", "control"]
    assert rows[0]["observation"]["screenshot"]["width"] == 800
    assert rows[1]["observation"] == {"modality": "shell", "screenshot": None}
    assert rows[1]["history"]["steps"][0]["execution"]["ok"] is True


def test_mixed_sft_preserves_execution_context_and_portable_images(tmp_path):
    episode = tmp_path / "results" / "episode"
    screenshots = episode / "screenshots"
    canonical = episode / "canonical"
    screenshots.mkdir(parents=True)
    canonical.mkdir()
    screenshot = screenshots / "state.png"
    screenshot.write_bytes(_png_header(1024, 768))
    _write_json(episode / "episode_manifest.json", {
        "episode_id": "mixed-context-episode",
        "status": "succeeded",
        "task": {
            "id": "mixed-context-task",
            "instruction": "Inspect the file, then click Save.",
            "metadata": {},
        },
        "agent": {"id": "agent-1", "harness": "test", "model": "test"},
        "vm": {"id": "ubuntu", "os": {"type": "Ubuntu"}, "image": {"id": "image"}},
        "evaluation": {"success": True, "score": 1.0},
    })
    _write_json(episode / "quality.json", {
        "valid": True,
        "training_tier": "sft_accepted",
    })
    _write_jsonl(canonical / "steps.jsonl", [
        {
            "step": 1,
            "pre_observation": {},
            "action": {"type": "shell.exec", "args": {"command": "cat demo.txt"}},
            "execution": {"ok": True, "returncode": 0, "stdout": "important value\n"},
            "post_observation": {},
            "decision_context": {
                "source": "chat.jsonl",
                "visibility": "observable_assistant_output",
                "intent": "[thinking] private reasoning",
                "text": "[thinking] private reasoning",
                "blocks": [{"text": "[thinking] private reasoning"}],
            },
            "provenance": {},
            "observation_quality": "complete",
        },
        {
            "step": 2,
            "pre_observation": {"screenshot": "screenshots/state.png"},
            "action": {"type": "computer.click", "args": {"x": 10, "y": 20}},
            "execution": {"ok": True},
            "post_observation": {"screenshot": "screenshots/state.png"},
            "provenance": {},
            "observation_quality": "complete",
        },
        {
            "step": 3,
            "pre_observation": {},
            "action": {"type": "finish", "args": {"status": "success"}},
            "execution": {"ok": True, "synthetic": True},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "complete",
        },
    ])
    taskset = tmp_path / "taskset"
    (taskset / "mixed").mkdir(parents=True)
    _write_json(taskset / "taskset_manifest.json", {
        "tasks": [{
            "id": "mixed-context-task",
            "markdown": "mixed/mixed-context-task.md",
        }],
    })
    (taskset / "mixed" / "mixed-context-task.md").write_text(
        "## Source\n\nGenerated from: source-task-1\n",
        encoding="utf-8",
    )

    stats = build_dataset(
        tmp_path / "results",
        tmp_path / "dataset",
        taskset_root=taskset,
    )

    assert stats["mixed_bundle"]["episodes_exported"] == 1
    assert stats["action_protocol"] == "qwen_xml"
    assert stats["mixed_bundle"]["action_protocol"] == "qwen_xml"
    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "mixed" / "trajectories.sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    click_row = rows[1]
    parser = QwenXmlActionParser()
    assert [
        parser.parse(row["messages"][-1]["content"]).to_dict()
        for row in rows
    ] == [row["target_action"] for row in rows]
    assert click_row["schema_version"] == "hybrid_cua_sft.example.v2"
    assert click_row["source_task_ids"] == ["source-task-1"]
    assert click_row["split_group_id"] == "source-task-1"
    assert "important value" in click_row["messages"][1]["content"]
    assert "cat demo.txt" in click_row["messages"][1]["content"]
    assert "[thinking]" not in json.dumps(click_row)
    assert click_row["action_protocol"] == "qwen_xml"
    assert click_row["messages"][0]["content"].startswith(
        "You are a computer-use agent"
    )
    assert click_row["messages"][-1]["content"].startswith("Action:")
    assert "<think>" not in click_row["messages"][-1]["content"]
    assert "<function=computer_use>" in click_row["messages"][-1]["content"]
    assert click_row["images"][0].startswith("images/")
    assert (tmp_path / "dataset" / "mixed" / click_row["images"][0]).is_file()

    top_level_row = json.loads(
        (tmp_path / "dataset" / "trajectories.sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[1]
    )
    assert top_level_row["images"][0].startswith("mixed/images/")
    assert not Path(top_level_row["images"][0]).is_absolute()
    trajectory = json.loads(
        (tmp_path / "dataset" / "trajectories.train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert trajectory["trajectory_path"] == "mixed/episodes.jsonl"
    assert trajectory["steps"][1]["observation"]["screenshot"]["path"].startswith(
        "mixed/images/"
    )


def test_strict_mixed_sft_excludes_infrastructure_tampering_and_credentials(tmp_path):
    episode = tmp_path / "results" / "episode"
    canonical = episode / "canonical"
    canonical.mkdir(parents=True)
    _write_json(episode / "episode_manifest.json", {
        "episode_id": "unsafe-episode",
        "status": "succeeded",
        "task": {"id": "unsafe-task", "instruction": "Open the site.", "metadata": {}},
        "agent": {"id": "agent-1"},
        "vm": {"id": "ubuntu"},
        "evaluation": {"success": True, "score": 1.0},
    })
    _write_json(episode / "quality.json", {
        "valid": True,
        "training_tier": "sft_accepted",
    })
    _write_jsonl(canonical / "steps.jsonl", [
        {
            "step": 1,
            "pre_observation": {},
            "action": {
                "type": "shell.exec",
                "args": {
                    "command": (
                        "printf stub > /tmp_workspace/_trajectory/"
                        "claudecode_shell_screenshot_hook.py; "
                        "curl http://user:real-secret@proxy.example"
                    ),
                },
            },
            "execution": {"ok": True, "stdout": "http://user:real-secret@proxy.example"},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "complete",
        },
        {
            "step": 2,
            "pre_observation": {},
            "action": {"type": "finish", "args": {"status": "success"}},
            "execution": {"ok": True, "synthetic": True},
            "post_observation": {},
            "provenance": {},
            "observation_quality": "complete",
        },
    ])

    strict_stats = build_dataset(tmp_path / "results", tmp_path / "strict")
    assert strict_stats["mixed_bundle"]["episodes_exported"] == 0
    assert strict_stats["mixed_bundle"]["exclusion_counts"] == {
        "collector_hook_tampering": 1,
        "embedded_url_credentials": 1,
    }

    structural_stats = build_dataset(
        tmp_path / "results",
        tmp_path / "structural",
        quality_policy="structural",
    )
    assert structural_stats["mixed_bundle"]["episodes_exported"] == 1
    text = (tmp_path / "structural" / "trajectories.sft.jsonl").read_text()
    assert "real-secret" not in text
    assert "http://<redacted>@proxy.example" in text


def test_high_quality_policy_keeps_package_install_and_office_mutation_warnings():
    warnings = ["package_install", "direct_office_package_mutation"]

    assert _policy_exclusion_reasons([], warnings, "high_quality") == []
    assert _policy_exclusion_reasons([], warnings, "strict") == []
    assert _policy_exclusion_reasons([], warnings, "structural") == []


def test_content_flags_detect_quoted_sudo_password() -> None:
    blockers, _ = _episode_content_flags([{
        "action": {
            "type": "shell.exec",
            "args": {"command": 'echo "password" | sudo -S true'},
        }
    }])

    assert blockers == ["hardcoded_sudo_password"]


def test_mixed_sft_drops_wait_without_visual_change(tmp_path):
    episode = tmp_path / "results" / "episode"
    screenshots = episode / "screenshots"
    canonical = episode / "canonical"
    screenshots.mkdir(parents=True)
    canonical.mkdir()
    screenshot = screenshots / "state.png"
    screenshot.write_bytes(_png_header(800, 600))
    _write_json(episode / "episode_manifest.json", {
        "episode_id": "wait-episode",
        "status": "succeeded",
        "task": {"id": "wait-task", "instruction": "Wait, then finish.", "metadata": {}},
        "agent": {"id": "agent-1"},
        "vm": {"id": "ubuntu"},
        "evaluation": {"success": True, "score": 1.0},
    })
    _write_json(episode / "quality.json", {
        "valid": True,
        "training_tier": "sft_accepted",
    })
    _write_jsonl(canonical / "steps.jsonl", [
        {
            "step": 1,
            "pre_observation": {"screenshot": "screenshots/state.png"},
            "action": {"type": "computer.wait", "args": {"seconds": 1}},
            "execution": {"ok": True},
            "post_observation": {"screenshot": "screenshots/state.png"},
            "provenance": {},
            "observation_quality": "complete",
        },
        {
            "step": 2,
            "pre_observation": {"screenshot": "screenshots/state.png"},
            "action": {"type": "finish", "args": {"status": "success"}},
            "execution": {"ok": True, "synthetic": True},
            "post_observation": {"screenshot": "screenshots/state.png"},
            "provenance": {},
            "observation_quality": "complete",
        },
    ])

    stats = build_dataset(tmp_path / "results", tmp_path / "dataset")
    rows = (tmp_path / "dataset" / "trajectories.sft.jsonl").read_text().splitlines()

    assert len(rows) == 1
    assert stats["mixed_bundle"]["unchanged_waits_dropped"] == 1


def test_successful_episode_with_unverified_execution_is_not_sft_accepted(tmp_path):
    episode = tmp_path / "episode"
    canonical = episode / "canonical"
    canonical.mkdir(parents=True)
    _write_json(episode / "episode_manifest.json", {
        "status": "succeeded",
        "task": {"instruction": "Run a command.", "evaluator": {"func": "test"}},
        "evaluation": {"success": True},
    })
    _write_jsonl(canonical / "steps.jsonl", [{
        "step": 1,
        "pre_observation": {},
        "action": {"type": "shell.exec", "args": {"command": "pwd"}},
        "execution": {"ok": None},
        "post_observation": {},
        "observation_quality": "partial",
    }])

    quality = validate_episode(episode)

    assert quality["valid"] is True
    assert quality["training_tier"] == "preference_candidate"
    assert quality["missing_execution_steps"] == 1


def test_episode_quality_blocks_non_trainable_actions_and_counts_noop_wait(tmp_path):
    episode = tmp_path / "episode"
    screenshots = episode / "screenshots"
    canonical = episode / "canonical"
    screenshots.mkdir(parents=True)
    canonical.mkdir()
    screenshot = screenshots / "state.png"
    screenshot.write_bytes(_png_header(800, 600))
    _write_json(episode / "episode_manifest.json", {
        "status": "succeeded",
        "task": {"instruction": "Complete the task.", "evaluator": {"func": "test"}},
        "evaluation": {"success": True},
    })
    _write_jsonl(canonical / "steps.jsonl", [
        {
            "step": 1,
            "pre_observation": {"screenshot": "screenshots/state.png"},
            "action": {"type": "computer.wait", "args": {"seconds": 1}},
            "execution": {"ok": True},
            "post_observation": {"screenshot": "screenshots/state.png"},
            "observation_quality": "complete",
        },
        {
            "step": 2,
            "pre_observation": {"screenshot": "screenshots/state.png"},
            "action": {
                "type": "shell.exec",
                "args": {"command": 'echo "password" | sudo -S true'},
            },
            "execution": {"ok": True},
            "post_observation": {},
            "observation_quality": "complete",
        },
        {
            "step": 3,
            "pre_observation": {},
            "action": {
                "type": "tool.call",
                "args": {
                    "name": "Read",
                    "args": {"file_path": "/home/user/.config/google-chrome/Default/Preferences"},
                },
            },
            "execution": {"ok": True},
            "post_observation": {},
            "observation_quality": "complete",
        },
        {
            "step": 4,
            "pre_observation": {},
            "action": {
                "type": "shell.exec",
                "args": {"command": "python3 -m http.server 8080"},
            },
            "execution": {"ok": True},
            "post_observation": {},
            "observation_quality": "complete",
        },
        {
            "step": 5,
            "pre_observation": {},
            "action": {
                "type": "tool.call",
                "args": {
                    "name": "Write",
                    "args": {"file_path": "/tmp_workspace/large.txt", "content": "x" * 3100},
                },
            },
            "execution": {"ok": True},
            "post_observation": {},
            "observation_quality": "complete",
        },
        {
            "step": 6,
            "pre_observation": {},
            "action": {
                "type": "tool.call",
                "args": {"name": "TaskOutput", "args": {"task_id": "1"}},
            },
            "execution": {"ok": True},
            "post_observation": {},
            "observation_quality": "complete",
        },
    ])

    quality = validate_episode(episode)

    # tool.call is a first-class action (paper Table 7), so the episode is
    # structurally valid; the policy violations block it from SFT instead.
    assert quality["valid"] is True
    assert quality["training_tier"] == "preference_candidate"
    assert any(
        blocker.startswith("action policy violations:")
        for blocker in quality["training_blockers"]
    )
    assert quality["unchanged_wait_steps"] == 1
    assert {finding["reason"] for finding in quality["action_policy_findings"]} == {
        "hardcoded_sudo_password",
        "hidden_application_state_access",
        "network_workaround",
        "structured_tool_forbidden",
        "oversized_action",
    }
    assert sum(
        finding["reason"] == "structured_tool_forbidden"
        for finding in quality["action_policy_findings"]
    ) == 3


def test_episode_quality_rejects_empty_type_action(tmp_path):
    episode = tmp_path / "episode"
    canonical = episode / "canonical"
    canonical.mkdir(parents=True)
    _write_json(episode / "episode_manifest.json", {
        "status": "succeeded",
        "task": {"instruction": "Type text.", "evaluator": {"func": "test"}},
        "evaluation": {"success": True},
    })
    _write_jsonl(canonical / "steps.jsonl", [{
        "step": 1,
        "pre_observation": {},
        "action": {"type": "computer.type", "args": {"text": ""}},
        "execution": {"ok": True},
        "post_observation": {},
        "observation_quality": "complete",
    }])

    quality = validate_episode(episode)

    assert quality["valid"] is False
    assert quality["training_tier"] == "rejected_corrupt"
    assert any("text must not be empty" in error for error in quality["errors"])


def test_frontier_sampler_favors_target_difficulty():
    sampler = FrontierSampler(random_sample_ratio=0, seed=7)
    sampler.update("easy", 1.0)
    sampler.update("frontier", 0.5)
    sampler.update("hard", 0.0)
    assert sampler.weight("frontier") > sampler.weight("easy")
    assert sampler.weight("frontier") > sampler.weight("hard")
