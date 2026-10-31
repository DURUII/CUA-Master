from __future__ import annotations

import json
from pathlib import Path

from cuamaster.collection import judge


def _write_task(task_dir: Path, *, agent_status: str = "ok") -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "id": task_dir.name,
                "instruction": "Create output/results.json containing an answer.",
                "evaluator": {"type": "task_evaluator"},
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "score.json").write_text(
        json.dumps(
            {
                "task_id": task_dir.name,
                "score": 0.0,
                "agent_done": True,
                "agent_exit": 0,
                "agent_status": agent_status,
                "error": "judge_failed: missing host judge",
                "scores": {"judge_method": "agent_as_judge"},
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "trajectory.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "actions": [
                    {
                        "type": "shell.exec",
                        "args": {
                            "command": "mkdir -p output && echo '{}' > output/results.json"
                        },
                    }
                ],
                "executions": [{"ok": True, "output": ""}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "chat.jsonl").write_text(
        json.dumps({"role": "assistant", "content": "Completed the task."}) + "\n",
        encoding="utf-8",
    )


def test_parse_judgment_accepts_fenced_json() -> None:
    parsed = judge._parse_judgment(
        """```json
{"score": 0.8, "task_success": true, "trajectory_usable": true,
 "confidence": 0.9, "reason": "done", "evidence": ["file created"],
 "failure_category": "none"}
```"""
    )

    assert parsed["score"] == 0.8
    assert parsed["task_success"] is True
    assert parsed["trajectory_usable"] is True


def test_score_collection_replaces_failed_host_judge_score(monkeypatch, tmp_path):
    pool_path = tmp_path / "model.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: judge-model\nbase_urls:\n"
        "  - http://judge.example/v1\napi:\n  trust_env: false\n",
        encoding="utf-8",
    )
    task_dir = tmp_path / "collection" / "profile" / "gui" / "model" / "task-a"
    _write_task(task_dir)

    monkeypatch.setattr(
        judge,
        "_request_judgment",
        lambda **kwargs: (
            {
                "score": 0.85,
                "task_success": True,
                "trajectory_usable": True,
                "confidence": 0.9,
                "reason": "deliverable creation is visible in the trace",
                "evidence": ["results.json was written"],
                "failure_category": "none",
            },
            '{"score":0.85}',
        ),
    )

    summary = judge.score_collection(
        tmp_path / "collection",
        endpoint_pool_path=pool_path,
        allow_unauthenticated=True,
        threshold=0.6,
    )

    assert summary["candidates"] == 1
    assert summary["accepted"] == 1
    score = json.loads((task_dir / "score.json").read_text())
    assert score["score"] == 0.85
    assert score["success"] is True
    assert score["error"] is None
    assert score["scores"]["judge_method"] == "trajectory_model_judge"
    assert (task_dir / "score.collection.json").is_file()
    assert json.loads((task_dir / "model_judge.json").read_text())["accepted"] is True


def test_score_collection_skips_failed_agent_and_retry_archives(monkeypatch, tmp_path):
    pool_path = tmp_path / "model.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: judge-model\nbase_urls:\n"
        "  - http://judge.example/v1\n",
        encoding="utf-8",
    )
    failed = tmp_path / "collection" / "profile" / "gui" / "model" / "failed"
    _write_task(failed, agent_status="empty_run")
    archived = (
        tmp_path
        / "collection"
        / ".collection_state"
        / "retry_attempts"
        / "round-01"
        / "profile"
        / "gui"
        / "model"
        / "archived"
    )
    _write_task(archived)
    monkeypatch.setattr(
        judge,
        "_request_judgment",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not call judge")),
    )

    summary = judge.score_collection(
        tmp_path / "collection",
        endpoint_pool_path=pool_path,
        allow_unauthenticated=True,
    )

    assert summary["candidates"] == 1
    assert summary["accepted"] == 0
    assert summary["counts"] == {"skipped_collection_failure": 1}
