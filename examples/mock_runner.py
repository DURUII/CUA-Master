#!/usr/bin/env python3
"""Minimal external-runner contract example for the command backend.

Writes one fake single-step GUI episode in the layout `cuamaster-collect`
expects (screenshots, computer_events.jsonl, execution_status.json, task.json,
score.json) without talking to any VM or model. Pair it with
`configs/collection/mock_command.example.json` to smoke-test the collector
end to end:

    python3 -m cuamaster.collection.collector \
        --config configs/collection/mock_command.example.json \
        --output-root /tmp/cuamaster-mock-run
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    task = Path(args.output) / "gui" / "mock-model" / "MOCK" / "open-settings"
    screenshots = task / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    initial = screenshots / "screenshot_0001_pre.png"
    shot = screenshots / "screenshot_0002_click.png"
    initial.write_bytes(PNG_1X1)
    shot.write_bytes(PNG_1X1)
    event = {
        "schema_version": "gui_trajectory.computer_event.v1",
        "step": 1,
        "actions": [{"type": "click", "x": 100, "y": 100}],
        "executions": [{
            "ok": True,
            "screenshot_path": f"/guest/screenshots/{shot.name}",
        }],
        "ok": True,
        "pre_screenshot_path": f"/guest/screenshots/{initial.name}",
        "screenshot_path": f"/guest/screenshots/{shot.name}",
    }
    (task / "computer_events.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    (task / "execution_status.json").write_text(json.dumps({
        "agent_done": True,
        "model": "mock-model",
        "harness": "mock-command-runner",
        "profile": args.profile,
        "image": args.image,
    }), encoding="utf-8")
    (task / "task.json").write_text(json.dumps({
        "id": "open-settings",
        "instruction": "Open the system settings application.",
        "snapshot": "clean-v1",
        "config": [],
        "evaluator": {"func": "mock_settings_open"},
        "required_capabilities": ["computer"],
        "metadata": {"category": "MOCK", "source": "mock"},
    }), encoding="utf-8")
    (task / "score.json").write_text(json.dumps({
        "task_id": "open-settings",
        "category": "MOCK",
        "mode": "gui",
        "score": 1.0,
        "success": True,
        "evaluator": "mock_settings_open",
    }), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
