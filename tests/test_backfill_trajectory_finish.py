from __future__ import annotations

import json
from pathlib import Path

from scripts.backfill_trajectory_finish import backfill


def _write_trajectory(path: Path, *, exportable: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "video_id": path.stem,
                "steps": [
                    {
                        "index": 1,
                        "start": 1.0,
                        "end": 2.0,
                        "action": {"type": "computer.click", "args": {"x": 1, "y": 2}},
                        "post_observation": {"screenshot": "/tmp/final.jpg", "timestamp": 2.0},
                    }
                ],
                "quality": {"num_steps": 1},
                "verification": {"exportable_for_training": exportable},
            }
        ),
        encoding="utf-8",
    )


def test_backfill_supports_dry_run_apply_and_idempotency(tmp_path: Path) -> None:
    eligible = tmp_path / "eligible.trajectory.json"
    rejected = tmp_path / "rejected.trajectory.json"
    _write_trajectory(eligible, exportable=True)
    _write_trajectory(rejected, exportable=False)

    dry_run = backfill(tmp_path, apply=False)
    assert dry_run["would_update"] == 1
    assert len(json.loads(eligible.read_text())["steps"]) == 1

    applied = backfill(tmp_path, apply=True)
    assert applied["updated"] == 1
    assert json.loads(eligible.read_text())["steps"][-1]["action"]["type"] == "finish"
    assert len(json.loads(rejected.read_text())["steps"]) == 1

    repeated = backfill(tmp_path, apply=True)
    assert repeated["updated"] == 0
    assert repeated["already_finished"] == 1
