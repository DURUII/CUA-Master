from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from cuamaster.seeding.rewriting.backends import _completion_token_param, _should_send_reasoning_effort
from cuamaster.seeding.rewriting.rewrite import (
    _merge_config_args,
    _output_complete,
    _write_trajectory_atomic,
    build_parser,
    rewrite_trajectory,
)
from cuamaster.seeding.rewriting.schema import RewriteTool, load_tools_config, validate_json_schema


class FakeRewriteBackend:
    model = "fake-rewriter"

    def __init__(self, spans: list[dict], rewrites: dict[tuple[int, int], dict]) -> None:
        self.spans = spans
        self.rewrites = rewrites
        self.calls: list[tuple[int, int]] = []

    def propose_spans(self, trajectory: dict, tools: list[RewriteTool]) -> list[dict]:
        return self.spans

    def rewrite_span(self, trajectory, span, source_steps, tools, image_paths):
        key = (span.start_index, span.end_index)
        self.calls.append(key)
        return self.rewrites[key]


def _trajectory(tmp_path: Path | None = None) -> dict:
    screenshots = []
    if tmp_path is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            path = tmp_path / f"source_{index}.jpg"
            path.write_bytes(f"frame-{index}".encode())
            screenshots.append(str(path))

    steps = []
    actions = [
        {"type": "computer.click", "args": {"x": 100, "y": 200}},
        {"type": "computer.type", "args": {"text": "report.csv"}},
        {"type": "computer.press", "args": {"keys": ["enter"]}},
    ]
    for index, action in enumerate(actions, start=1):
        step = {
            "index": index,
            "start": float(index),
            "end": float(index) + 0.5,
            "intent": f"step {index}",
            "action": action,
            "needs_grounding": False,
            "confidence": 0.95,
            "evidence": {"transcript": f"step {index}"},
        }
        if screenshots:
            step["observation"] = {"screenshot": screenshots[index - 1], "timestamp": float(index)}
            step["post_observation"] = {"screenshot": screenshots[index], "timestamp": float(index) + 0.5}
        steps.append(step)
    return {
        "video_id": "demo",
        "application": "Files",
        "category": "Productivity",
        "subcategory": "File management",
        "source_video": "/tmp/demo.mp4",
        "task": "create a report file",
        "steps": steps,
        "quality": {"num_steps": 3, "has_screenshots": bool(screenshots)},
        "verification": {"trajectory_valid": True, "exportable_for_training": True},
    }


def test_rewrite_trajectory_merges_gui_span_into_cli() -> None:
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 2,
                "target_type": "cli",
                "confidence": 0.93,
                "reason": "the same file can be created directly",
                "improves_success_rate": True,
                "improves_efficiency": True,
                "benefit_reason": "one exact command replaces coordinate-sensitive GUI input",
            }
        ],
        rewrites={
            (1, 2): {
                "equivalent": True,
                "action": {"type": "shell.exec", "args": {"command": "touch report.csv"}},
                "intent": "create report.csv",
                "confidence": 0.91,
                "reason": "touch creates the same empty file",
            }
        },
    )

    rewritten, audit = rewrite_trajectory(_trajectory(), backend, [])

    assert [step["action"]["type"] for step in rewritten["steps"]] == ["shell.exec", "computer.press"]
    assert rewritten["steps"][0]["start"] == 1.0
    assert rewritten["steps"][0]["end"] == 2.5
    assert rewritten["steps"][0]["evidence"]["action_rewrite"]["source_step_indices"] == [1, 2]
    assert rewritten["quality"]["num_steps"] == 2
    assert rewritten["quality"]["action_categories"] == {"cli": 1, "gui": 1}
    assert rewritten["rewrite"]["status"] == "rewritten"
    assert rewritten["rewrite"]["observation_policy"] == "preserve_source_observations_without_regeneration"
    rewrite_evidence = rewritten["steps"][0]["evidence"]["action_rewrite"]
    assert rewrite_evidence["improves_success_rate"] is True
    assert rewrite_evidence["improves_efficiency"] is True
    assert audit[-1]["status"] == "accepted"


def test_rewrite_trajectory_accepts_registered_tool_with_valid_args() -> None:
    tool = RewriteTool(
        name="files.create",
        description="Create a file",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 3,
                "target_type": "tool",
                "tool_name": "files.create",
                "confidence": 0.94,
                "reason": "registered file tool performs the workflow",
            }
        ],
        rewrites={
            (1, 3): {
                "equivalent": True,
                "action": {
                    "type": "tool.call",
                    "args": {"name": "files.create", "args": {"path": "report.csv"}},
                },
                "intent": "create report.csv",
                "confidence": 0.92,
                "reason": "the tool creates the requested file",
            }
        },
    )

    rewritten, _ = rewrite_trajectory(_trajectory(), backend, [tool])

    assert len(rewritten["steps"]) == 1
    assert rewritten["steps"][0]["action"]["type"] == "tool.call"
    assert rewritten["quality"]["action_categories"] == {"tool": 1}


def test_rewrite_rejects_invalid_tool_args_and_keeps_gui_steps() -> None:
    tool = RewriteTool(
        name="files.create",
        description="Create a file",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 3,
                "target_type": "tool",
                "tool_name": "files.create",
                "confidence": 0.95,
                "reason": "use the file tool",
            }
        ],
        rewrites={
            (1, 3): {
                "equivalent": True,
                "action": {"type": "tool.call", "args": {"name": "files.create", "args": {}}},
                "intent": "create report.csv",
                "confidence": 0.95,
                "reason": "tool call",
            }
        },
    )

    rewritten, audit = rewrite_trajectory(_trajectory(), backend, [tool])

    assert [step["action"]["type"] for step in rewritten["steps"]] == [
        "computer.click",
        "computer.type",
        "computer.press",
    ]
    assert rewritten["rewrite"]["status"] == "no_change"
    assert "missing required fields: path" in audit[-1]["error"]


def test_rewrite_rejects_low_confidence_and_overlapping_proposals() -> None:
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 2,
                "target_type": "cli",
                "confidence": 0.7,
                "reason": "too uncertain",
            },
            {
                "start_index": 1,
                "end_index": 2,
                "target_type": "cli",
                "confidence": 0.9,
                "reason": "valid range",
            },
            {
                "start_index": 2,
                "end_index": 3,
                "target_type": "cli",
                "confidence": 0.95,
                "reason": "overlaps",
            },
        ],
        rewrites={
            (1, 2): {
                "equivalent": True,
                "action": {"type": "shell.exec", "args": {"command": "touch report.csv"}},
                "intent": "create report.csv",
                "confidence": 0.9,
                "reason": "same result",
            }
        },
    )

    rewritten, audit = rewrite_trajectory(_trajectory(), backend, [])

    assert backend.calls == [(1, 2)]
    assert rewritten["rewrite"]["accepted_spans"] == 1
    assert any("below threshold" in item.get("error", "") for item in audit)
    assert any("overlaps" in item.get("error", "") for item in audit)


def test_rewrite_rejects_shell_placeholders() -> None:
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 1,
                "target_type": "cli",
                "confidence": 0.9,
                "reason": "possible command",
                "improves_success_rate": True,
                "improves_efficiency": False,
                "benefit_reason": "the command would avoid coordinate-sensitive interaction",
            }
        ],
        rewrites={
            (1, 1): {
                "equivalent": True,
                "action": {"type": "shell.exec", "args": {"command": "touch <filename>"}},
                "intent": "create a file",
                "confidence": 0.9,
                "reason": "placeholder command",
            }
        },
    )

    rewritten, audit = rewrite_trajectory(_trajectory(), backend, [])

    assert rewritten["rewrite"]["status"] == "no_change"
    assert "placeholder" in audit[-1]["error"]


def test_rewrite_rejects_passive_wait_span_without_calling_rewriter() -> None:
    trajectory = _trajectory()
    trajectory["steps"][0]["action"] = {"type": "computer.wait", "args": {"seconds": 4}}
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 1,
                "target_type": "cli",
                "confidence": 0.95,
                "reason": "replace wait with sleep",
                "improves_success_rate": False,
                "improves_efficiency": True,
                "benefit_reason": "claimed shorter control path",
            }
        ],
        rewrites={},
    )

    rewritten, audit = rewrite_trajectory(trajectory, backend, [])

    assert rewritten["rewrite"]["status"] == "no_change"
    assert backend.calls == []
    assert "passive wait" in audit[-1]["error"]


def test_rewrite_rejects_proposal_without_success_or_efficiency_benefit() -> None:
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 1,
                "target_type": "cli",
                "confidence": 0.95,
                "reason": "semantically possible but not better",
                "improves_success_rate": False,
                "improves_efficiency": False,
                "benefit_reason": "no measurable benefit",
            }
        ],
        rewrites={},
    )

    rewritten, audit = rewrite_trajectory(_trajectory(), backend, [])

    assert rewritten["rewrite"]["status"] == "no_change"
    assert backend.calls == []
    assert "improve expected success rate or efficiency" in audit[-1]["error"]


def test_rewrite_rejects_delay_only_shell_command() -> None:
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 2,
                "target_type": "cli",
                "confidence": 0.95,
                "reason": "candidate command",
            }
        ],
        rewrites={
            (1, 2): {
                "equivalent": True,
                "action": {"type": "shell.exec", "args": {"command": "sleep 4"}},
                "intent": "wait",
                "confidence": 0.95,
                "reason": "only waits",
            }
        },
    )

    rewritten, audit = rewrite_trajectory(_trajectory(), backend, [])

    assert rewritten["rewrite"]["status"] == "no_change"
    assert "delay-only" in audit[-1]["error"]


def test_rewrite_rejects_effective_confidence_below_threshold() -> None:
    trajectory = _trajectory()
    trajectory["steps"][0]["confidence"] = 0.7
    backend = FakeRewriteBackend(
        spans=[
            {
                "start_index": 1,
                "end_index": 2,
                "target_type": "cli",
                "confidence": 0.95,
                "reason": "candidate command",
            }
        ],
        rewrites={
            (1, 2): {
                "equivalent": True,
                "action": {"type": "shell.exec", "args": {"command": "touch report.csv"}},
                "intent": "create report.csv",
                "confidence": 0.95,
                "reason": "same result",
            }
        },
    )

    rewritten, audit = rewrite_trajectory(trajectory, backend, [])

    assert rewritten["rewrite"]["status"] == "no_change"
    assert "effective confidence" in audit[-1]["error"]


def test_rewrite_rejects_invalid_source_action_before_api_call() -> None:
    trajectory = _trajectory()
    trajectory["steps"][0]["action"] = {
        "type": "computer.click",
        "args": {"x": 100, "y": 200, "keys": ["shift"]},
    }
    backend = FakeRewriteBackend([], {})

    with pytest.raises(ValueError, match="source trajectory contains invalid canonical actions"):
        rewrite_trajectory(trajectory, backend, [])

    assert backend.calls == []


def test_write_trajectory_copies_self_contained_state_chain(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path / "source")
    destination = tmp_path / "output" / "demo.trajectory.json"

    _write_trajectory_atomic(trajectory, destination, frame_mode="copy", overwrite=False)

    written = json.loads(destination.read_text(encoding="utf-8"))
    frame_dir = destination.with_suffix(".frames")
    assert written["steps"][0]["observation"]["screenshot"] == str(frame_dir / "step_0000_state.jpg")
    assert written["steps"][1]["observation"] == written["steps"][0]["post_observation"]
    assert len(list(frame_dir.glob("*.jpg"))) == 4
    assert _output_complete(destination) is False
    written["rewrite"] = {"status": "no_change"}
    destination.write_text(json.dumps(written), encoding="utf-8")
    assert _output_complete(destination) is True


def test_load_tools_config_and_json_schema_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "tools.yaml"
    config_path.write_text(
        """
tools:
  - name: records.lookup
    description: Look up a record
    input_schema:
      type: object
      properties:
        id: {type: integer, minimum: 1}
      required: [id]
      additionalProperties: false
""".strip(),
        encoding="utf-8",
    )

    tools = load_tools_config(config_path)

    assert tools[0].name == "records.lookup"
    validate_json_schema({"id": 1}, tools[0].input_schema)
    with pytest.raises(TypeError, match="must be integer"):
        validate_json_schema({"id": "1"}, tools[0].input_schema)


def test_load_tools_config_from_builtin_runtime_profile() -> None:
    tools = load_tools_config(Path("builtin:cua-core-v1"))

    assert [tool.name for tool in tools] == [
        "ui.inspect",
        "filesystem.list",
        "filesystem.read_text",
        "filesystem.write_text",
        "document.extract_text",
        "archive.extract",
    ]


def test_config_defaults_do_not_override_explicit_cli_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "rewrite.yaml"
    config_path.write_text("concurrency: 8\nmin_rewrite_confidence: 0.9\n", encoding="utf-8")
    parser = build_parser()
    argv = ["--config", str(config_path), "--concurrency", "2"]
    args = parser.parse_args(argv)

    merged = _merge_config_args(parser, args, argv)

    assert isinstance(merged, argparse.Namespace)
    assert merged.concurrency == 2
    assert merged.min_rewrite_confidence == 0.9


def test_nested_api_config_matches_construction_style(tmp_path: Path) -> None:
    config_path = tmp_path / "rewrite.yaml"
    config_path.write_text(
        """
backend: openai_chat
concurrency: 4
api:
  model: kimi-k3
  base_urls:
    - http://10.0.0.1:30000/v1
    - http://10.0.0.2:30000/v1
  key: EMPTY
  reasoning_effort: none
  max_tokens: 16384
  max_retries: 20
  request_delay: 0.1
""".strip(),
        encoding="utf-8",
    )
    parser = build_parser()
    argv = ["--config", str(config_path), "--api-model", "cli-override"]
    args = parser.parse_args(argv)

    merged = _merge_config_args(parser, args, argv)

    assert merged.api_model == "cli-override"
    assert merged.api_base_urls == [
        "http://10.0.0.1:30000/v1",
        "http://10.0.0.2:30000/v1",
    ]
    assert merged.api_key == "EMPTY"
    assert merged.api_max_tokens == 16384
    assert merged.api_max_retries == 20
    assert merged.api_request_delay == 0.1


def test_endpoint_pool_supplies_model_and_urls(tmp_path: Path) -> None:
    pool_path = tmp_path / "kimi.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: kimi-k3\nbase_urls:\n  - model-a:30000\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "rewrite.yaml"
    config_path.write_text(
        "backend: openai_chat\napi:\n  endpoint_pool: kimi.yaml\n",
        encoding="utf-8",
    )
    parser = build_parser()
    argv = ["--config", str(config_path)]

    merged = _merge_config_args(parser, parser.parse_args(argv), argv)

    assert merged.api_model == "kimi-k3"
    assert merged.api_base_urls == ["http://model-a:30000/v1"]
    assert merged.api_endpoint_pool == str(pool_path.resolve())


def test_endpoint_pool_supplies_responses_api_defaults(tmp_path: Path) -> None:
    pool_path = tmp_path / "gpt.yaml"
    pool_path.write_text(
        """
schema_version: 1
model: GPT-5.6-sol
base_urls:
  - https://api.example.com/v1
api:
  wire_api: responses
  api_key_env: EXAMPLE_API_KEY
  reasoning_effort: xhigh
  store: false
  trust_env: false
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "rewrite.yaml"
    config_path.write_text(
        "backend: openai_chat\napi:\n  endpoint_pool: gpt.yaml\n",
        encoding="utf-8",
    )
    parser = build_parser()
    argv = ["--config", str(config_path)]

    merged = _merge_config_args(parser, parser.parse_args(argv), argv)

    assert merged.api_model == "GPT-5.6-sol"
    assert merged.api_wire_api == "responses"
    assert merged.api_key_env == "EXAMPLE_API_KEY"
    assert merged.api_reasoning_effort == "xhigh"
    assert merged.api_store is False
    assert merged.api_trust_env is False


def test_nested_config_rejects_unknown_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "rewrite.yaml"
    config_path.write_text("api:\n  unsupported: true\n", encoding="utf-8")
    parser = build_parser()
    argv = ["--config", str(config_path)]
    args = parser.parse_args(argv)

    with pytest.raises(ValueError, match="api_unsupported"):
        _merge_config_args(parser, args, argv)


def test_gpt5_compatible_request_parameters() -> None:
    base_url = "https://your-model-endpoint.example.com/v1"

    assert _completion_token_param(base_url, "GPT-5.5") == "max_completion_tokens"
    assert _completion_token_param(base_url, "qwen-vl") == "max_tokens"
    assert _should_send_reasoning_effort("high", base_url, "GPT-5.5") is True
    assert _should_send_reasoning_effort("none", base_url, "GPT-5.5") is False
