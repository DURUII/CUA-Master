from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuamaster.seeding.reconstruction.assembler import build_draft_trajectory
from cuamaster.seeding.reconstruction import construct
from cuamaster.seeding.reconstruction.construct import (
    _attach_step_screenshots,
    _build_run_analysis,
    _construct_one,
    _construct_records,
    _enforce_quality_integrity_gate,
    _fallback_candidates_after_empty_filter,
    _filter_candidates_by_plan,
    _filter_candidates_to_primary_episode,
    _extend_short_trajectory_selection,
    _split_candidates_for_continuity,
    _split_selected_candidates,
    _summarize,
    build_parser,
    ensure_success_finish_step,
)
from cuamaster.seeding.reconstruction.backends import (
    OpenAIChatBackend,
    PassthroughBackend,
    _allows_empty_api_key,
    _apply_step_refinement,
    _completion_token_param,
    _candidates_from_boundary_payload,
    _parse_json_object,
    _normalize_base_urls,
    _should_bypass_proxy,
    _should_send_reasoning_effort,
    _supports_response_format,
    _state_chain_image_paths,
    _step_batch_payload_by_index,
    _step_batch_payload_needs_fallback,
)
from cuamaster.seeding.reconstruction.lattice import merge_candidates
from cuamaster.seeding.reconstruction.paths import resolve_video_task
from cuamaster.seeding.reconstruction.screenshots import frame_dir_for_trajectory, step_frame_path
from cuamaster.seeding.reconstruction.schema import ActionCandidate, TranscriptSegment
from cuamaster.seeding.reconstruction.transcript import mine_semantic_candidates


def test_mine_semantic_candidates_splits_multi_action_segment() -> None:
    segments = [
        TranscriptSegment(
            id=1,
            start=9.0,
            end=21.0,
            text="Click into the pay code column, scroll down and click vacation prepay.",
        )
    ]

    candidates = mine_semantic_candidates(segments)

    assert [c.action_type for c in candidates] == [
        "computer.click",
        "computer.scroll",
        "computer.click",
    ]
    assert candidates[0].target == "pay code column"
    assert candidates[1].args == {"dx": 0, "dy": -5}
    assert candidates[2].target == "vacation prepay"


def test_mine_semantic_candidates_extracts_type_and_press_args() -> None:
    segments = [
        TranscriptSegment(id=1, start=0.0, end=2.0, text='Type "hello world" and press Enter.'),
    ]

    candidates = mine_semantic_candidates(segments)

    assert candidates[0].action_type == "computer.type"
    assert candidates[0].args == {"text": "hello world"}
    assert candidates[1].action_type == "computer.press"
    assert candidates[1].args == {"keys": ["enter"]}


def test_mine_semantic_candidates_extracts_explicit_shell_command() -> None:
    segments = [
        TranscriptSegment(
            id=1,
            start=0.0,
            end=3.0,
            text='Run the terminal command "mkdir -p output".',
        ),
    ]

    candidates = mine_semantic_candidates(segments)

    assert len(candidates) == 1
    assert candidates[0].action_type == "shell.exec"
    assert candidates[0].args == {"command": "mkdir -p output"}


def test_mine_semantic_candidates_ignores_non_gui_choice_language() -> None:
    segments = [
        TranscriptSegment(id=1, start=0.0, end=3.0, text="Let's talk about why people choose ADP."),
        TranscriptSegment(id=2, start=3.0, end=6.0, text="They continue to choose ADP because it is designed with people in mind."),
    ]

    assert mine_semantic_candidates(segments) == []


def test_summarize_writes_long_short_training_rows_and_candidates(tmp_path: Path) -> None:
    def write_trajectory(
        name: str,
        *,
        num_steps: int,
        screenshots: bool,
        valid: bool | None,
        exportable: bool | None,
        quality_score: int | None = 4,
    ) -> None:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "video_id": name,
                    "application": "Demo",
                    "category": "Business & Administration",
                    "subcategory": "Career",
                    "source_video": "/tmp/demo.mp4",
                    "task": f"task {name}",
                    "steps": [
                        {
                            "index": index + 1,
                            "intent": f"step {index + 1}",
                            "action": {"type": "computer.click", "args": {"x": 500, "y": 500}},
                        }
                        for index in range(num_steps)
                    ],
                    "quality": {
                        "num_steps": num_steps,
                        "trajectory_confidence": 0.8,
                        "needs_grounding_steps": 0,
                        "has_screenshots": screenshots,
                        "num_screenshots": num_steps + 1 if screenshots else 0,
                        "api_refined_steps": num_steps,
                        "api_failed_steps": 0,
                        "draft": True,
                    },
                    "candidate_filter": {"reason": "partially_filtered"},
                    "verification": {
                        "trajectory_valid": valid,
                        "exportable_for_training": exportable,
                        "quality_score": quality_score,
                    },
                }
            ),
            encoding="utf-8",
        )

    write_trajectory("short.trajectory.json", num_steps=3, screenshots=True, valid=True, exportable=True, quality_score=3)
    write_trajectory("quality_short.trajectory.json", num_steps=8, screenshots=True, valid=True, exportable=True, quality_score=4)
    write_trajectory("long.trajectory.json", num_steps=20, screenshots=True, valid=True, exportable=True, quality_score=3)
    write_trajectory("long40.trajectory.json", num_steps=40, screenshots=True, valid=True, exportable=True, quality_score=4)
    write_trajectory("candidate.trajectory.json", num_steps=22, screenshots=True, valid=True, exportable=False, quality_score=3)
    write_trajectory("fragment.trajectory.json", num_steps=3, screenshots=True, valid=True, exportable=False)
    write_trajectory("zero.trajectory.json", num_steps=0, screenshots=False, valid=None, exportable=None)
    write_trajectory("no_screens.trajectory.json", num_steps=3, screenshots=False, valid=True, exportable=True)
    write_trajectory("not_exportable.trajectory.json", num_steps=3, screenshots=True, valid=False, exportable=False)

    summary_path = _summarize(tmp_path)

    train_rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines()]
    train_file_rows = [json.loads(line) for line in (tmp_path / "trajectories.train.jsonl").read_text(encoding="utf-8").splitlines()]
    long_candidate_rows = [
        json.loads(line) for line in (tmp_path / "trajectories.long_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert train_file_rows == train_rows
    assert [row["video_id"] for row in train_rows] == [
        "long.trajectory.json",
        "long40.trajectory.json",
        "quality_short.trajectory.json",
        "short.trajectory.json",
    ]
    assert [row["training_tier"] for row in train_rows] == ["long", "long", "short", "short"]
    assert [row["recommended_train_weight"] for row in train_rows] == [1.0, 1.2, 0.35, 0.25]
    assert all("steps" in row and "trajectory_path" in row for row in train_rows)
    assert [row["video_id"] for row in long_candidate_rows] == ["candidate.trajectory.json"]
    assert long_candidate_rows[0]["training_tier"] == "long"
    assert long_candidate_rows[0]["recommended_train_weight"] == 0.45
    action_stats = json.loads((tmp_path / "trajectories.action_stats.json").read_text(encoding="utf-8"))
    assert action_stats["action_types"] == {"computer.click": 71}
    assert not (tmp_path / "trajectories.audit.jsonl").exists()
    assert not (tmp_path / "trajectories.rejected.jsonl").exists()


def test_summarize_can_include_long_candidates_and_write_audit(tmp_path: Path) -> None:
    trajectory = {
        "video_id": "candidate",
        "application": "Demo",
        "category": "Business & Administration",
        "subcategory": "Career",
        "source_video": "/tmp/demo.mp4",
        "task": "candidate task",
        "steps": [
            {"index": index + 1, "action": {"type": "computer.click", "args": {"x": 500, "y": 500}}}
            for index in range(20)
        ],
        "quality": {"num_steps": 20, "has_screenshots": True, "num_screenshots": 21},
        "candidate_filter": {"reason": "partially_filtered"},
        "verification": {"trajectory_valid": True, "exportable_for_training": False, "quality_score": 3},
    }
    (tmp_path / "candidate.trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    summary_path = _summarize(tmp_path, include_long_candidates_in_train=True, write_audit_jsonl=True)

    train_rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines()]
    audit_rows = [json.loads(line) for line in (tmp_path / "trajectories.audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["video_id"] for row in train_rows] == ["candidate"]
    assert train_rows[0]["recommended_train_weight"] == 0.45
    assert audit_rows[0]["retention_reason"] == "valid_long_fragment_quality_ok"


def test_summarize_rejects_noncanonical_training_actions(tmp_path: Path) -> None:
    trajectory = {
        "video_id": "invalid",
        "application": "Demo",
        "category": "Developer",
        "subcategory": "CLI",
        "source_video": "/tmp/demo.mp4",
        "task": "invalid action",
        "steps": [{"index": 1, "action": {"type": "shell.exec", "args": {}}}],
        "quality": {"num_steps": 1, "has_screenshots": True, "num_screenshots": 2},
        "verification": {"trajectory_valid": True, "exportable_for_training": True, "quality_score": 4},
    }
    (tmp_path / "invalid.trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    summary_path = _summarize(tmp_path, write_audit_jsonl=True)

    assert summary_path.read_text(encoding="utf-8") == ""
    rejected = [
        json.loads(line)
        for line in (tmp_path / "trajectories.rejected.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rejected[0]["retention_reason"] == "invalid_canonical_action"
    assert "missing required args: command" in rejected[0]["canonical_action_errors"][0]


def test_summarize_rejects_trajectories_with_quality_integrity_failures(tmp_path: Path) -> None:
    for field in ("needs_grounding_steps", "api_failed_steps", "action_type_mismatches"):
        trajectory = {
            "video_id": field,
            "application": "Demo",
            "category": "Developer",
            "subcategory": "CLI",
            "source_video": "/tmp/demo.mp4",
            "task": field,
            "steps": [{"index": 1, "action": {"type": "computer.click", "args": {"x": 500, "y": 500}}}],
            "quality": {
                "num_steps": 1,
                "has_screenshots": True,
                field: 1,
            },
            "verification": {"trajectory_valid": True, "exportable_for_training": True, "quality_score": 4},
        }
        (tmp_path / f"{field}.trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    summary_path = _summarize(tmp_path, write_audit_jsonl=True)

    assert summary_path.read_text(encoding="utf-8") == ""
    rejected = [
        json.loads(line)
        for line in (tmp_path / "trajectories.rejected.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rejected) == 3
    assert {row["retention_reason"] for row in rejected} == {"quality_integrity_failed"}
    assert all(row["quality_clean"] is False for row in rejected)


def test_mine_semantic_candidates_treats_hit_button_as_click() -> None:
    segments = [TranscriptSegment(id=1, start=0.0, end=2.0, text="Hit okay and then click save.")]

    candidates = mine_semantic_candidates(segments)

    assert [c.action_type for c in candidates] == ["computer.click", "computer.click"]
    assert candidates[0].target == "okay"
    assert candidates[1].target == "save"


def test_mine_semantic_candidates_ignores_type_as_noun() -> None:
    segments = [
        TranscriptSegment(
            id=1,
            start=0.0,
            end=2.0,
            text="Add a comment type of vacation prepay and list the dates.",
        )
    ]

    assert mine_semantic_candidates(segments) == []


def test_merge_candidates_aligns_semantic_and_visual_windows() -> None:
    semantic = ActionCandidate(
        start=9.0,
        end=11.0,
        source="semantic",
        action_type="computer.click",
        target="save button",
        semantic_score=0.8,
        confidence=0.8,
    )
    visual = ActionCandidate(
        start=10.2,
        end=11.5,
        source="visual",
        visual_score=0.55,
        confidence=0.55,
    )

    merged = merge_candidates([semantic, visual])

    assert len(merged) == 1
    assert merged[0].source == "merged"
    assert merged[0].action_type == "computer.click"
    assert merged[0].target == "save button"
    assert merged[0].confidence > semantic.confidence
    assert merged[0].evidence["semantic_visual_aligned"] is True


def test_merge_candidates_keeps_distinct_semantic_clicks_separate() -> None:
    okay = ActionCandidate(
        start=1.0,
        end=2.0,
        source="semantic",
        action_type="computer.click",
        target="okay",
        semantic_score=0.82,
        confidence=0.82,
    )
    save = ActionCandidate(
        start=2.1,
        end=3.0,
        source="semantic",
        action_type="computer.click",
        target="save",
        semantic_score=0.84,
        confidence=0.84,
    )

    merged = merge_candidates([okay, save])

    assert [c.target for c in merged] == ["okay", "save"]


def test_plan_filter_uses_ignore_overlap_ratio_and_high_value_is_advisory() -> None:
    candidates = [
        ActionCandidate(start=9.0, end=11.0, source="semantic", action_type="computer.click", confidence=0.8),
        ActionCandidate(start=20.0, end=22.0, source="semantic", action_type="computer.click", confidence=0.7),
    ]
    plan = {
        "ignore_ranges": [{"start": 8.0, "end": 9.5, "reason": "intro"}],
        "high_value_ranges": [{"start": 100.0, "end": 110.0, "reason": "later GUI"}],
    }

    filtered = _filter_candidates_by_plan(candidates, plan, ignore_overlap_ratio=0.8)

    assert filtered == candidates


def test_empty_filter_fallback_keeps_raw_candidates_without_no_gui_verdict() -> None:
    candidates = [
        ActionCandidate(start=1.0, end=2.0, source="semantic", action_type="computer.click", confidence=0.5),
        ActionCandidate(start=3.0, end=4.0, source="visual", confidence=0.9),
    ]

    fallback = _fallback_candidates_after_empty_filter(candidates, {"enabled": True, "confidence": 0.7}, 1)

    assert fallback == [candidates[1]]


def test_primary_episode_filter_keeps_one_continuous_workflow() -> None:
    candidates = [
        ActionCandidate(start=1.0, end=2.0, source="semantic", action_type="computer.click", confidence=0.8),
        ActionCandidate(start=11.0, end=12.0, source="semantic", action_type="computer.click", confidence=0.8),
        ActionCandidate(start=21.0, end=22.0, source="semantic", action_type="computer.click", confidence=0.8),
    ]
    plan = {"primary_episode": {"start": 10.0, "end": 13.0, "task": "submit form", "confidence": 0.9}}

    filtered = _filter_candidates_to_primary_episode(candidates, plan, enabled=True, tolerance=1.0)

    assert filtered == [candidates[1]]


def test_split_selected_candidates_preserves_tail_steps() -> None:
    candidates = [
        ActionCandidate(start=float(index), end=float(index) + 0.5, source="api", action_type="computer.click", confidence=0.9)
        for index in range(5)
    ]

    parts = _split_selected_candidates(candidates, 2)

    assert [len(part) for part in parts] == [2, 2, 1]
    assert [candidate.start for part in parts for candidate in part] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_extend_short_trajectory_selection_adds_adjacent_candidates() -> None:
    candidates = [
        ActionCandidate(start=1.0, end=2.0, source="api", action_type="computer.click", target="open", confidence=0.8),
        ActionCandidate(start=3.0, end=4.0, source="api", action_type="computer.type", target="name", confidence=0.7),
        ActionCandidate(start=5.0, end=6.0, source="api", action_type="computer.click", target="save", confidence=0.9),
        ActionCandidate(start=120.0, end=121.0, source="api", action_type="computer.click", target="later", confidence=0.95),
    ]

    extended, info = _extend_short_trajectory_selection(
        selected=[candidates[2]],
        candidate_pool=candidates,
        min_steps=3,
        max_gap_seconds=10.0,
        min_confidence=0.35,
    )

    assert [candidate.target for candidate in extended] == ["open", "name", "save"]
    assert info["added_steps"] == 2
    assert info["reason"] == "target_reached"


def test_split_candidates_for_continuity_uses_large_time_gaps() -> None:
    candidates = [
        ActionCandidate(start=0.0, end=1.0, source="api", action_type="computer.click", confidence=0.9),
        ActionCandidate(start=2.0, end=3.0, source="api", action_type="computer.click", confidence=0.9),
        ActionCandidate(start=20.0, end=21.0, source="api", action_type="computer.click", confidence=0.9),
        ActionCandidate(start=22.0, end=23.0, source="api", action_type="computer.click", confidence=0.9),
    ]

    parts = _split_candidates_for_continuity(candidates, enabled=True, max_gap_seconds=5.0, min_steps=2)

    assert [[candidate.start for candidate in part] for part in parts] == [[0.0, 2.0], [20.0, 22.0]]


def test_screenshot_paths_are_stable() -> None:
    trajectory_path = Path("out/App_video.trajectory.json")
    frame_dir = frame_dir_for_trajectory(trajectory_path)

    assert frame_dir == Path("out/App_video.trajectory.frames")
    assert step_frame_path(frame_dir, 0, "state") == Path("out/App_video.trajectory.frames/step_0000_state.jpg")
    assert step_frame_path(frame_dir, 12, "state") == Path("out/App_video.trajectory.frames/step_0012_state.jpg")


def test_build_draft_trajectory_marks_coordinate_actions_for_grounding(tmp_path: Path) -> None:
    record = {
        "video_id": "abc123def45",
        "title": "ADP vacation prepay tutorial",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
    }
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    task = resolve_video_task(record, video_dir, transcript_dir)
    candidates = [
        ActionCandidate(
            start=1.0,
            end=2.0,
            source="semantic",
            action_type="computer.click",
            target="plus sign",
            semantic_score=0.8,
            confidence=0.8,
        ),
        ActionCandidate(
            start=3.0,
            end=4.0,
            source="semantic",
            action_type="computer.scroll",
            args={"dx": 0, "dy": -5},
            semantic_score=0.8,
            confidence=0.8,
        ),
    ]

    trajectory = build_draft_trajectory(task, candidates).to_dict()

    assert trajectory["video_id"] == "abc123def45"
    assert trajectory["quality"]["num_steps"] == 2
    assert trajectory["steps"][0]["needs_grounding"] is True
    assert trajectory["steps"][1]["needs_grounding"] is False


def test_build_draft_trajectory_marks_incomplete_shell_action_for_grounding(tmp_path: Path) -> None:
    task = resolve_video_task(
        {
            "video_id": "shell-video",
            "application": "Terminal",
            "category": "Developer",
            "subcategory": "CLI",
        },
        tmp_path / "videos",
        tmp_path / "transcripts",
    )
    candidates = [
        ActionCandidate(
            start=1.0,
            end=2.0,
            source="semantic",
            action_type="shell.exec",
            confidence=0.9,
        ),
        ActionCandidate(
            start=3.0,
            end=4.0,
            source="semantic",
            action_type="shell.exec",
            args={"command": "pwd"},
            confidence=0.9,
        ),
    ]

    trajectory = build_draft_trajectory(task, candidates).to_dict()

    assert [step["needs_grounding"] for step in trajectory["steps"]] == [True, False]


def test_step_batch_payload_maps_by_step_index() -> None:
    payload = {"steps": [{"index": 2, "is_action": True}, {"index": "4", "is_action": False}]}

    mapped = _step_batch_payload_by_index(payload)

    assert sorted(mapped) == [2, 4]
    assert mapped[2]["is_action"] is True


def test_step_batch_payload_falls_back_for_missing_click_coordinates() -> None:
    step = {
        "index": 1,
        "action": {"type": "computer.click", "args": {}},
        "evidence": {"api_boundary": {"action_type": "computer.click"}},
    }
    payload = {
        "index": 1,
        "is_action": True,
        "action": {"type": "computer.click", "args": {"button": "left"}},
        "confidence": 0.9,
    }

    assert _step_batch_payload_needs_fallback(step, payload) is True


def test_step_batch_payload_accepts_grounded_consistent_click() -> None:
    step = {
        "index": 1,
        "action": {"type": "computer.click", "args": {}},
        "evidence": {"api_boundary": {"action_type": "computer.click"}},
    }
    payload = {
        "index": 1,
        "is_action": True,
        "action": {"type": "computer.click", "args": {"x": 500, "y": 400, "button": "left"}},
        "confidence": 0.9,
    }

    assert _step_batch_payload_needs_fallback(step, payload) is False


def test_step_batch_payload_requires_grounded_shell_command() -> None:
    step = {
        "index": 1,
        "action": {"type": "shell.exec", "args": {}},
        "evidence": {"api_boundary": {"action_type": "shell.exec"}},
    }
    missing = {
        "index": 1,
        "is_action": True,
        "action": {"type": "shell.exec", "args": {}},
        "confidence": 0.9,
    }
    grounded = {
        **missing,
        "action": {"type": "shell.exec", "args": {"command": "pwd"}},
    }

    assert _step_batch_payload_needs_fallback(step, missing) is True
    assert _step_batch_payload_needs_fallback(step, grounded) is False


def test_boundary_refinement_preserves_shell_command_args() -> None:
    candidates = [
        ActionCandidate(
            start=1.0,
            end=2.0,
            source="semantic",
            action_type="shell.exec",
            args={"command": "pwd"},
            confidence=0.8,
        )
    ]
    payload = {
        "actions": [{
            "start": 1.0,
            "end": 2.0,
            "action_type": "shell.exec",
            "args": {"command": "pwd", "cwd": "/tmp"},
            "source_ids": ["c1"],
            "confidence": 0.95,
        }]
    }

    refined = _candidates_from_boundary_payload(candidates, payload)

    assert refined[0].action_type == "shell.exec"
    assert refined[0].args == {"command": "pwd", "cwd": "/tmp"}


def test_state_chain_image_paths_reuses_contiguous_states(tmp_path: Path) -> None:
    state0 = tmp_path / "state0.jpg"
    state1 = tmp_path / "state1.jpg"
    state2 = tmp_path / "state2.jpg"
    for path in (state0, state1, state2):
        path.write_bytes(b"jpg")
    steps = [
        {"observation": {"screenshot": str(state0)}, "post_observation": {"screenshot": str(state1)}},
        {"observation": {"screenshot": str(state1)}, "post_observation": {"screenshot": str(state2)}},
    ]

    assert _state_chain_image_paths(steps) == [state0, state1, state2]


def test_construct_one_writes_draft_trajectory_from_transcript(tmp_path: Path, monkeypatch) -> None:
    record = {
        "video_id": "abc123def45",
        "title": "Vacation prepay",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
    }
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "out"
    video_path = video_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.mp4"
    transcript_path = transcript_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.segments.json"
    video_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": 1,
                        "start": 1.0,
                        "end": 8.0,
                        "text": "Click a plus sign underneath the final row.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        video_dir=str(video_dir),
        transcript_dir=str(transcript_dir),
        output_dir=str(output_dir),
        overwrite=False,
        require_transcript=True,
        semantic_pre_roll=1.0,
        semantic_post_roll=1.5,
        enable_visual=False,
        scene_threshold=0.18,
        max_visual_candidates=200,
        merge_tolerance=1.5,
        min_confidence=0.45,
        extract_screenshots=True,
        screenshot_width=640,
        observation_offset=0.0,
        post_observation_offset=0.0,
    )

    def fake_extract(video_path: Path, dest: Path, timestamp: float, *, frame_width: int = 1280) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake jpg")
        return dest

    monkeypatch.setattr(construct, "extract_screenshot", fake_extract)

    video_id, status = _construct_one(record, args, PassthroughBackend())

    assert (video_id, status) == ("abc123def45", "ok")
    out_path = output_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.trajectory.json"
    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["candidate_stats"]["semantic"] == 1
    assert obj["steps"][0]["action"]["type"] == "computer.click"
    assert obj["steps"][0]["needs_grounding"] is True
    assert obj["steps"][0]["observation"]["screenshot"].endswith("step_0000_state.jpg")
    assert obj["steps"][0]["post_observation"]["screenshot"].endswith("step_0001_state.jpg")
    assert obj["quality"]["has_screenshots"] is True
    assert obj["quality"]["num_screenshots"] == 2


def test_attach_step_screenshots_reuses_previous_state(tmp_path: Path, monkeypatch) -> None:
    trajectory = {
        "steps": [
            {"index": 1, "start": 10.0, "end": 12.0},
            {"index": 2, "start": 20.0, "end": 22.0},
        ],
        "quality": {},
    }
    calls = []

    def fake_extract(video_path: Path, dest: Path, timestamp: float, *, frame_width: int = 1280) -> Path:
        calls.append((dest.name, timestamp, frame_width))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake jpg")
        return dest

    monkeypatch.setattr(construct, "extract_screenshot", fake_extract)

    _attach_step_screenshots(
        trajectory,
        tmp_path / "video.mp4",
        tmp_path / "out" / "App_video.trajectory.json",
        frame_width=640,
        pre_offset=-1.0,
        post_offset=0.5,
    )

    assert calls == [
        ("step_0000_state.jpg", 9.0, 640),
        ("step_0001_state.jpg", 12.5, 640),
        ("step_0002_state.jpg", 22.5, 640),
    ]
    first, second = trajectory["steps"]
    assert first["observation"]["screenshot"].endswith("step_0000_state.jpg")
    assert first["post_observation"]["screenshot"].endswith("step_0001_state.jpg")
    assert second["observation"] == first["post_observation"]
    assert second["post_observation"]["screenshot"].endswith("step_0002_state.jpg")
    assert trajectory["quality"]["has_screenshots"] is True
    assert trajectory["quality"]["num_screenshots"] == 3


def test_ensure_success_finish_step_is_exportable_only_and_idempotent() -> None:
    final_state = {"screenshot": "/tmp/final.jpg", "timestamp": 12.5}
    trajectory = {
        "steps": [
            {
                "index": 1,
                "start": 10.0,
                "end": 12.0,
                "action": {"type": "computer.click", "args": {"x": 1, "y": 2}},
                "needs_grounding": False,
                "post_observation": final_state,
            }
        ],
        "quality": {"num_steps": 1, "needs_grounding_steps": 0},
        "verification": {"exportable_for_training": True},
    }

    assert ensure_success_finish_step(trajectory) is True
    assert ensure_success_finish_step(trajectory) is False
    assert len(trajectory["steps"]) == 2
    finish = trajectory["steps"][-1]
    assert finish["action"] == {"type": "finish", "args": {"status": "success"}}
    assert finish["observation"] == final_state
    assert finish["post_observation"] == final_state
    assert finish["index"] == 2
    assert trajectory["quality"]["num_steps"] == 2
    assert trajectory["quality"]["has_finish_step"] is True

    rejected = {
        "steps": trajectory["steps"][:1],
        "quality": {"num_steps": 1},
        "verification": {"exportable_for_training": False},
    }
    assert ensure_success_finish_step(rejected) is False
    assert len(rejected["steps"]) == 1


def test_action_construction_parser_exposes_concurrency() -> None:
    args = build_parser().parse_args([])

    assert args.concurrency == 1


def test_kimi_k3_backend_uses_generic_chat_parameters() -> None:
    base_url = "http://192.0.2.10:15551/v1"

    assert _allows_empty_api_key(base_url) is True
    assert _should_bypass_proxy(base_url) is True
    assert _completion_token_param(base_url, "Kimi-K3") == "max_tokens"
    assert _should_send_reasoning_effort("none", base_url, "Kimi-K3") is False
    assert _should_send_reasoning_effort("medium", base_url, "Kimi-K3") is False


def test_normalize_base_urls_adds_scheme_and_v1_suffix() -> None:
    assert _normalize_base_urls("10.0.0.65:30000, http://10.0.0.29:30000/v1") == [
        "http://10.0.0.65:30000/v1",
        "http://10.0.0.29:30000/v1",
    ]


def test_construct_records_runs_with_configured_concurrency(monkeypatch) -> None:
    records = [{"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}]
    args = argparse.Namespace(concurrency=2)
    counts = {"ok": 0, "skipped": 0, "missing_video": 0, "missing_transcript": 0, "failed": 0}
    calls = []

    def fake_construct_record(record, args, backend):
        calls.append(record["video_id"])
        status = "skipped" if record["video_id"] == "b" else "ok"
        return record["video_id"], status

    monkeypatch.setattr(construct, "_construct_record", fake_construct_record)

    _construct_records(records, args, PassthroughBackend(), counts, started_at=0.0)

    assert sorted(calls) == ["a", "b", "c"]
    assert counts["ok"] == 2
    assert counts["skipped"] == 1


def _video_cleanup_args(tmp_path: Path, *, overwrite: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(tmp_path / "output"),
        video_dir=str(tmp_path / "videos"),
        transcript_dir=str(tmp_path / "transcripts"),
        overwrite=overwrite,
    )


def _video_cleanup_record(video_id: str, *, application: str = "Demo") -> dict:
    return {
        "video_id": video_id,
        "category": "Category",
        "subcategory": "Subcategory",
        "application": application,
    }


def _write_cleanup_trajectory(path: Path, video_id: str, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "steps": [],
        "quality": {"num_steps": 0, "has_screenshots": False},
        "verification": {"trajectory_valid": False, "exportable_for_training": False},
    }
    if not valid:
        payload.pop("quality")
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_video_ids_preserves_complete_legacy_video_group(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    record = _video_cleanup_record("legacy-video")
    base_path = construct._output_path(
        resolve_video_task(record, Path(args.video_dir), Path(args.transcript_dir)).relative_path,
        Path(args.output_dir),
    )
    first_path = construct._episode_output_path(base_path, 1)
    second_path = construct._episode_output_path(base_path, 2)
    _write_cleanup_trajectory(first_path, "legacy-video")
    _write_cleanup_trajectory(second_path, "legacy-video")
    first_frames = frame_dir_for_trajectory(first_path)
    first_frames.mkdir()

    assert construct._prepare_video_ids_for_run([record], args) == {"legacy-video"}
    assert first_path.exists()
    assert second_path.exists()
    assert first_frames.exists()


def test_prepare_video_ids_treats_completion_marker_as_authoritative(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    record = _video_cleanup_record("completed-video")
    trajectory_path = Path(args.output_dir) / "old" / "completed.trajectory.json"
    _write_cleanup_trajectory(trajectory_path, "completed-video")
    construct._write_completion_manifest(Path(args.output_dir), "completed-video", [trajectory_path])
    trajectory_path.write_text("corrupted after completion", encoding="utf-8")
    frame_dir = frame_dir_for_trajectory(trajectory_path)
    frame_dir.mkdir()

    assert construct._prepare_video_ids_for_run([record], args) == {"completed-video"}
    assert trajectory_path.exists()
    assert frame_dir.exists()
    assert construct._completion_manifest_path(Path(args.output_dir), "completed-video").exists()


def test_prepare_video_ids_accepts_completed_video_without_outputs(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    record = _video_cleanup_record("no-output-video")
    construct._write_completion_manifest(Path(args.output_dir), "no-output-video", [])

    assert construct._prepare_video_ids_for_run([record], args) == {"no-output-video"}
    assert construct._completion_manifest_path(Path(args.output_dir), "no-output-video").exists()


def test_prepare_video_ids_removes_every_artifact_for_unfinished_video(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    record = _video_cleanup_record("unfinished-video")
    current_base = construct._output_path(
        resolve_video_task(record, Path(args.video_dir), Path(args.transcript_dir)).relative_path,
        Path(args.output_dir),
    )
    current_path = construct._episode_output_path(current_base, 1)
    old_path = Path(args.output_dir) / "old_category" / "old_app_unfinished-video.trajectory.episode_02.json"
    _write_cleanup_trajectory(current_path, "unfinished-video")
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_text("partial", encoding="utf-8")
    current_frames = frame_dir_for_trajectory(current_path)
    old_frames = frame_dir_for_trajectory(old_path)
    current_frames.mkdir()
    old_frames.mkdir()
    manifest_path = construct._completion_manifest_path(Path(args.output_dir), "unfinished-video")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("partial", encoding="utf-8")

    assert construct._prepare_video_ids_for_run([record], args) == set()
    assert not current_path.exists()
    assert not old_path.exists()
    assert not current_frames.exists()
    assert not old_frames.exists()
    assert not manifest_path.exists()


def test_prepare_video_ids_does_not_delete_similar_or_unrelated_ids(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    record = _video_cleanup_record("abc")
    target_base = construct._output_path(
        resolve_video_task(record, Path(args.video_dir), Path(args.transcript_dir)).relative_path,
        Path(args.output_dir),
    )
    target_path = construct._episode_output_path(target_base, 1)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("partial", encoding="utf-8")
    similar_prefix_path = Path(args.output_dir) / "other" / "Demo_xabc.trajectory.episode_01.json"
    similar_suffix_path = Path(args.output_dir) / "other" / "Demo_abc-extra.trajectory.episode_01.json"
    unrelated_path = Path(args.output_dir) / "other" / "Demo_other.trajectory.episode_01.json"
    for path in (similar_prefix_path, similar_suffix_path, unrelated_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unrelated", encoding="utf-8")
    foreign_named_path = Path(args.output_dir) / "old" / "Demo_abc.trajectory.episode_02.json"
    _write_cleanup_trajectory(foreign_named_path, "different-video")
    foreign_named_frames = frame_dir_for_trajectory(foreign_named_path)
    foreign_named_frames.mkdir()

    assert construct._prepare_video_ids_for_run([record], args) == set()
    assert not target_path.exists()
    assert similar_prefix_path.exists()
    assert similar_suffix_path.exists()
    assert unrelated_path.exists()
    assert foreign_named_path.exists()
    assert foreign_named_frames.exists()


def test_prepare_video_ids_refuses_output_path_escape(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    record = _video_cleanup_record("escape-video")
    record["_dest_relpath"] = "../escaped.mp4"
    escaped_path = tmp_path / "escaped.trajectory.json"
    escaped_path.write_text("must survive", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes output directory"):
        construct._prepare_video_ids_for_run([record], args)

    assert escaped_path.read_text(encoding="utf-8") == "must survive"


def test_prepare_video_ids_refuses_symlink_escape(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True)
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (output_dir / "linked").symlink_to(external_dir, target_is_directory=True)
    record = _video_cleanup_record("escape-video")
    record["_dest_relpath"] = "linked/escaped.mp4"
    escaped_path = external_dir / "escaped.trajectory.json"
    escaped_path.write_text("must survive", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes output directory"):
        construct._prepare_video_ids_for_run([record], args)

    assert escaped_path.read_text(encoding="utf-8") == "must survive"


def test_prepare_video_ids_refuses_foreign_video_at_current_output_path(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    record = _video_cleanup_record("target-video")
    base_path = construct._output_path(
        resolve_video_task(record, Path(args.video_dir), Path(args.transcript_dir)).relative_path,
        Path(args.output_dir),
    )
    foreign_path = construct._episode_output_path(base_path, 1)
    _write_cleanup_trajectory(foreign_path, "different-video")

    with pytest.raises(ValueError, match="owned by video_id=different-video"):
        construct._prepare_video_ids_for_run([record], args)

    assert foreign_path.exists()


def test_prepare_video_ids_only_resets_unfinished_video(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path)
    complete_record = _video_cleanup_record("complete-video", application="Complete")
    unfinished_record = _video_cleanup_record("unfinished-video", application="Unfinished")
    complete_path = Path(args.output_dir) / "old" / "complete.trajectory.json"
    unfinished_path = Path(args.output_dir) / "old" / "unfinished.trajectory.json"
    _write_cleanup_trajectory(complete_path, "complete-video")
    _write_cleanup_trajectory(unfinished_path, "unfinished-video", valid=False)
    construct._write_completion_manifest(Path(args.output_dir), "complete-video", [complete_path])

    assert construct._prepare_video_ids_for_run([complete_record, unfinished_record], args) == {"complete-video"}
    assert complete_path.exists()
    assert construct._completion_manifest_path(Path(args.output_dir), "complete-video").exists()
    assert not unfinished_path.exists()


def test_prepare_video_ids_overwrite_resets_completed_video_group(tmp_path: Path) -> None:
    args = _video_cleanup_args(tmp_path, overwrite=True)
    record = _video_cleanup_record("completed-video")
    first_path = Path(args.output_dir) / "first" / "completed.trajectory.episode_01.json"
    second_path = Path(args.output_dir) / "second" / "completed.trajectory.episode_02.json"
    _write_cleanup_trajectory(first_path, "completed-video")
    _write_cleanup_trajectory(second_path, "completed-video")
    construct._write_completion_manifest(Path(args.output_dir), "completed-video", [first_path, second_path])

    assert construct._prepare_video_ids_for_run([record], args) == set()
    assert not first_path.exists()
    assert not second_path.exists()
    assert not construct._completion_manifest_path(Path(args.output_dir), "completed-video").exists()


def test_construct_records_skips_duplicate_video_ids(monkeypatch) -> None:
    records = [
        {"video_id": "existing"},
        {"video_id": "new"},
        {"video_id": "new"},
    ]
    args = argparse.Namespace(concurrency=1)
    counts = {"ok": 0, "skipped": 0, "missing_video": 0, "missing_transcript": 0, "failed": 0}
    calls = []

    def fake_construct_record(record, args, backend):
        calls.append(record["video_id"])
        return record["video_id"], "ok"

    monkeypatch.setattr(construct, "_construct_record", fake_construct_record)

    _construct_records(
        records,
        args,
        PassthroughBackend(),
        counts,
        started_at=0.0,
        seen_video_ids={"existing"},
    )

    assert calls == ["new"]
    assert counts["ok"] == 1
    assert counts["skipped"] == 2


def test_construct_record_failure_removes_partial_video_artifacts(tmp_path: Path, monkeypatch) -> None:
    record = {
        "video_id": "failed-video",
        "title": "Failed video",
        "category": "Developer",
        "subcategory": "CLI",
        "application": "Terminal",
    }
    args = argparse.Namespace(
        video_dir=str(tmp_path / "videos"),
        transcript_dir=str(tmp_path / "transcripts"),
        output_dir=str(tmp_path / "output"),
    )
    task = resolve_video_task(record, Path(args.video_dir), Path(args.transcript_dir))
    base_path = construct._output_path(task.relative_path, Path(args.output_dir))
    part_path = construct._part_output_path(base_path, 1)
    frame_dir = frame_dir_for_trajectory(part_path)
    manifest_path = construct._completion_manifest_path(Path(args.output_dir), "failed-video")

    def fail_after_partial_write(record, args, backend):
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_text("partial", encoding="utf-8")
        frame_dir.mkdir()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("backend failed")

    monkeypatch.setattr(construct, "_construct_one", fail_after_partial_write)

    assert construct._construct_record(record, args, PassthroughBackend()) == ("failed-video", "failed")
    assert not part_path.exists()
    assert not frame_dir.exists()
    assert not manifest_path.exists()


def test_construct_record_failure_preserves_foreign_video_artifact(tmp_path: Path, monkeypatch) -> None:
    record = _video_cleanup_record("failed-video")
    args = argparse.Namespace(
        video_dir=str(tmp_path / "videos"),
        transcript_dir=str(tmp_path / "transcripts"),
        output_dir=str(tmp_path / "output"),
    )
    task = resolve_video_task(record, Path(args.video_dir), Path(args.transcript_dir))
    base_path = construct._output_path(task.relative_path, Path(args.output_dir))
    foreign_path = construct._part_output_path(base_path, 1)

    def fail_with_foreign_output(record, args, backend):
        _write_cleanup_trajectory(foreign_path, "different-video")
        raise RuntimeError("backend failed")

    monkeypatch.setattr(construct, "_construct_one", fail_with_foreign_output)

    assert construct._construct_record(record, args, PassthroughBackend()) == ("failed-video", "failed")
    assert foreign_path.exists()


def test_construct_records_reaches_configured_in_flight_peak(monkeypatch) -> None:
    records = [{"video_id": f"video-{index}"} for index in range(16)]
    args = argparse.Namespace(concurrency=8)
    counts = {"ok": 0, "skipped": 0, "missing_video": 0, "missing_transcript": 0, "failed": 0}
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    max_active = 0

    def fake_construct_record(record, args, backend):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if max_active >= args.concurrency:
                release.set()
        release.wait(timeout=2.0)
        with lock:
            active -= 1
        return record["video_id"], "ok"

    monkeypatch.setattr(construct, "_construct_record", fake_construct_record)

    analysis = _construct_records(records, args, PassthroughBackend(), counts, started_at=0.0)

    assert max_active == 8
    assert analysis["in_flight_peak"] == 8
    assert counts["ok"] == 16


def test_build_run_analysis_reports_timing_and_rates() -> None:
    counts = {"ok": 2, "skipped": 1, "missing_video": 1, "missing_transcript": 0, "failed": 1}

    analysis = _build_run_analysis(
        total=5,
        workers=2,
        counts=counts,
        started_at=10.0,
        ended_at=20.0,
        item_durations=[1.0, 2.0, 4.0, 1.5, 1.5],
    )

    assert analysis["records_total"] == 5
    assert analysis["records_completed"] == 5
    assert analysis["workers"] == 2
    assert analysis["elapsed_seconds"] == 10.0
    assert analysis["throughput_per_minute"] == 30.0
    assert analysis["success_rate"] == 0.4
    assert analysis["failure_rate"] == 0.2
    assert analysis["missing_rate"] == 0.2
    assert analysis["average_record_seconds"] == 2.0
    assert analysis["fastest_record_seconds"] == 1.0
    assert analysis["slowest_record_seconds"] == 4.0
    assert analysis["counts"] == counts


def test_api_refinement_applies_grounded_action() -> None:
    step = {
        "action": {"type": "computer.click", "args": {}},
        "needs_grounding": True,
        "confidence": 0.5,
        "evidence": {},
    }
    payload = {
        "is_action": True,
        "action": {"type": "computer.click", "args": {"x": 250, "y": 750, "button": "left"}},
        "intent": "click the Save button",
        "confidence": 0.93,
        "target": "Save button",
        "reason": "The post-observation shows the save action took effect.",
    }

    _apply_step_refinement(step, payload)

    assert step["needs_grounding"] is False
    assert step["action"]["args"] == {"x": 250, "y": 750, "button": "left"}
    assert step["intent"] == "click the Save button"
    assert step["confidence"] == 0.93
    assert step["evidence"]["api_target"] == "Save button"


def test_api_refinement_marks_unexplained_action_type_mismatch() -> None:
    step = {
        "action": {"type": "computer.type", "args": {}},
        "needs_grounding": True,
        "confidence": 0.5,
        "evidence": {"api_boundary": {"action_type": "computer.type"}},
    }

    _apply_step_refinement(
        step,
        {
            "is_action": True,
            "action": {"type": "computer.click", "args": {"x": 250, "y": 750}},
            "confidence": 0.8,
        },
    )

    assert step["needs_review"] is True
    assert step["evidence"]["action_type_consistent"] is False
    assert step["evidence"]["action_type_mismatch"]["expected"] == "computer.type"


def test_api_refinement_rejects_actions_outside_action_space() -> None:
    step = {
        "action": {"type": "computer.click", "args": {"x": 100, "y": 100}},
        "needs_grounding": False,
        "confidence": 0.7,
        "evidence": {},
    }

    _apply_step_refinement(
        step,
        {
            "is_action": True,
            "action": {"type": "computer.long_press", "args": {"x": 250, "y": 750}},
            "confidence": 0.8,
        },
    )

    assert step["action"]["type"] == "computer.click"
    assert step["rejected_by_api"] is True
    assert step["needs_review"] is True


def test_quality_integrity_gate_blocks_exportable_trajectory_with_api_failures() -> None:
    trajectory = {
        "steps": [
            {
                "index": 1,
                "action": {"type": "computer.click", "args": {"x": 100, "y": 200, "button": "left"}},
                "needs_grounding": False,
            }
        ],
        "quality": {
            "needs_grounding_steps": 0,
            "api_failed_steps": 1,
            "action_type_mismatches": 0,
        },
        "verification": {
            "trajectory_valid": True,
            "exportable_for_training": True,
        },
    }

    errors = _enforce_quality_integrity_gate(trajectory)

    assert errors == ["api_failed_steps:1"]
    assert trajectory["verification"]["quality_integrity_passed"] is False
    assert trajectory["verification"]["exportable_for_training"] is False


def test_quality_integrity_gate_blocks_noncanonical_actions() -> None:
    trajectory = {
        "steps": [
            {
                "index": 1,
                "action": {"type": "computer.click", "args": {}},
                "needs_grounding": False,
            }
        ],
        "quality": {
            "needs_grounding_steps": 0,
            "api_failed_steps": 0,
            "action_type_mismatches": 0,
        },
        "verification": {
            "trajectory_valid": True,
            "exportable_for_training": True,
        },
    }

    errors = _enforce_quality_integrity_gate(trajectory)

    assert errors == ["canonical_action:step_1:missing required args: x, y"]
    assert trajectory["quality"]["canonical_action_error_count"] == 1
    assert trajectory["verification"]["exportable_for_training"] is False


def test_step_batch_payload_falls_back_for_actions_outside_action_space() -> None:
    step = {"action": {"type": "computer.click", "args": {}}, "evidence": {}}
    payload = {
        "is_action": True,
        "action": {"type": "computer.long_press", "args": {"x": 250, "y": 750}},
        "confidence": 0.9,
        "changed_action_type_reason": "Model ignored the allowed action list.",
    }

    assert _step_batch_payload_needs_fallback(step, payload) is True


def test_parse_json_object_tolerates_markdown_fence() -> None:
    assert _parse_json_object('```json\n{"is_action": true}\n```') == {"is_action": True}


def test_parse_json_object_uses_balanced_object_before_trailing_text() -> None:
    assert _parse_json_object('prefix {"is_action": true, "args": {"x": 1}} trailing {bad') == {
        "is_action": True,
        "args": {"x": 1},
    }


def test_parse_json_object_repairs_trailing_commas() -> None:
    raw = """
    {
      "actions": [
        {
          "id": "a1",
          "evidence": "keep this comma, inside the string",
        },
      ],
    }
    """

    assert _parse_json_object(raw) == {
        "actions": [
            {
                "id": "a1",
                "evidence": "keep this comma, inside the string",
            }
        ]
    }


def test_chat_json_evicts_invalid_cache_and_retries_invalid_response(tmp_path: Path, monkeypatch) -> None:
    responses = iter(['{"invalid":', '{"valid": true}'])
    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

    backend = OpenAIChatBackend.__new__(OpenAIChatBackend)
    backend.model = "test-model"
    backend.base_urls = ["http://127.0.0.1:30000/v1"]
    backend.max_tokens = 128
    backend.max_retries = 2
    backend.reasoning_effort = "none"
    backend.stage_reasoning_effort = {}
    backend.cache_dir = tmp_path / ".api_cache"
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    backend._next_client = lambda: ("http://127.0.0.1:30000/v1", client)
    monkeypatch.setattr("cuamaster.seeding.reconstruction.backends.time.sleep", lambda _: None)

    content = [{"type": "text", "text": "return JSON"}]
    cache_path = backend._cache_path("verify", "video-id", content, 128)
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"raw": '{"cached":', "stage": "verify"}), encoding="utf-8")

    result = backend._chat_json(content, max_tokens=128, cache_stage="verify", cache_key="video-id")

    assert result == {"valid": True}
    assert len(requests) == 2
    assert json.loads(cache_path.read_text(encoding="utf-8"))["raw"] == '{"valid": true}'


def test_chat_json_does_not_cache_exhausted_invalid_responses(tmp_path: Path, monkeypatch) -> None:
    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"invalid":'))]
            )

    backend = OpenAIChatBackend.__new__(OpenAIChatBackend)
    backend.model = "test-model"
    backend.base_urls = ["http://127.0.0.1:30000/v1"]
    backend.max_tokens = 128
    backend.max_retries = 2
    backend.reasoning_effort = "none"
    backend.stage_reasoning_effort = {}
    backend.cache_dir = tmp_path / ".api_cache"
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    backend._next_client = lambda: ("http://127.0.0.1:30000/v1", client)
    monkeypatch.setattr("cuamaster.seeding.reconstruction.backends.time.sleep", lambda _: None)

    content = [{"type": "text", "text": "return JSON"}]
    cache_path = backend._cache_path("verify", "video-id", content, 128)
    assert cache_path is not None

    with pytest.raises(RuntimeError, match="OpenAI-compatible chat request failed"):
        backend._chat_json(content, max_tokens=128, cache_stage="verify", cache_key="video-id")

    assert len(requests) == 2
    assert not cache_path.exists()


def test_construct_one_uses_backend_trajectory_refinement(tmp_path: Path, monkeypatch) -> None:
    record = {
        "video_id": "abc123def45",
        "title": "Vacation prepay",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
    }
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "out"
    video_path = video_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.mp4"
    transcript_path = transcript_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.segments.json"
    video_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    transcript_path.write_text(
        json.dumps({"segments": [{"id": 1, "start": 1.0, "end": 2.0, "text": "Click save."}]}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        video_dir=str(video_dir),
        transcript_dir=str(transcript_dir),
        output_dir=str(output_dir),
        overwrite=False,
        require_transcript=True,
        semantic_pre_roll=0.0,
        semantic_post_roll=0.0,
        enable_visual=False,
        scene_threshold=0.18,
        max_visual_candidates=200,
        merge_tolerance=1.5,
        min_confidence=0.45,
        extract_screenshots=True,
        screenshot_width=640,
        observation_offset=0.0,
        post_observation_offset=0.0,
    )

    def fake_extract(video_path: Path, dest: Path, timestamp: float, *, frame_width: int = 1280) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake jpg")
        return dest

    class FakeBackend(PassthroughBackend):
        def refine_trajectory(self, task, trajectory):
            for step in trajectory["steps"]:
                _apply_step_refinement(
                    step,
                    {
                        "is_action": True,
                        "action": {"type": "computer.click", "args": {"x": 500, "y": 600, "button": "left"}},
                        "intent": "click save",
                        "confidence": 0.9,
                        "target": "save",
                    },
                )
            trajectory["quality"]["api_refined_steps"] = len(trajectory["steps"])
            trajectory["quality"]["needs_grounding_steps"] = sum(1 for step in trajectory["steps"] if step["needs_grounding"])
            return trajectory

        def verify_trajectory(self, task, trajectory):
            return {"verified": True, "backend": "fake"}

    monkeypatch.setattr(construct, "extract_screenshot", fake_extract)

    video_id, status = _construct_one(record, args, FakeBackend())

    assert (video_id, status) == ("abc123def45", "ok")
    out_path = output_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.trajectory.json"
    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["steps"][0]["needs_grounding"] is False
    assert obj["steps"][0]["action"]["args"] == {"x": 500, "y": 600, "button": "left"}
    assert obj["quality"]["api_refined_steps"] == 1
    assert obj["verification"]["verified"] is True


def test_construct_one_uses_api_heavy_decision_stages(tmp_path: Path, monkeypatch) -> None:
    record = {
        "video_id": "abc123def45",
        "title": "Vacation prepay",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
    }
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "out"
    video_path = video_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.mp4"
    transcript_path = transcript_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.segments.json"
    video_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": 1, "start": 0.0, "end": 1.0, "text": "Intro."},
                    {"id": 2, "start": 2.0, "end": 4.0, "text": "Click save and click submit."},
                    {"id": 3, "start": 5.0, "end": 6.0, "text": "Outro."},
                ]
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        video_dir=str(video_dir),
        transcript_dir=str(transcript_dir),
        output_dir=str(output_dir),
        overwrite=False,
        require_transcript=True,
        semantic_pre_roll=0.0,
        semantic_post_roll=0.0,
        enable_visual=False,
        scene_threshold=0.18,
        max_visual_candidates=200,
        merge_tolerance=1.5,
        min_confidence=0.45,
        extract_screenshots=True,
        screenshot_width=640,
        observation_offset=0.0,
        post_observation_offset=0.0,
        api_video_plan=True,
        api_overview_frames=2,
        api_boundary_refine=True,
        api_boundary_frames=2,
        api_trajectory_select=True,
        api_ground_steps=True,
        api_verify_trajectory=True,
        api_frame_width=64,
    )

    def fake_extract(video_path: Path, dest: Path, timestamp: float, *, frame_width: int = 1280) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake jpg")
        return dest

    class HeavyFakeBackend(PassthroughBackend):
        def plan_video(self, task, transcript, overview_frames):
            return {
                "enabled": True,
                "task_title": "Save and submit payroll",
                "ignore_ranges": [{"start": 0.0, "end": 1.5, "reason": "intro"}],
                "high_value_ranges": [{"start": 1.5, "end": 5.0, "reason": "GUI operation"}],
                "episode_segments": [{"start": 1.5, "end": 5.0, "task": "Save and submit", "summary": "main workflow"}],
            }

        def refine_boundaries(self, task, transcript, candidates, frames, video_plan):
            return [
                ActionCandidate(
                    start=2.0,
                    end=3.0,
                    source="api",
                    action_type="computer.click",
                    target="save",
                    text="click save",
                    confidence=0.95,
                ),
                ActionCandidate(
                    start=3.0,
                    end=4.0,
                    source="api",
                    action_type="computer.click",
                    target="submit",
                    text="click submit",
                    confidence=0.94,
                ),
            ], {"enabled": True, "output_candidates": 2}

        def select_trajectory(self, task, candidates, video_plan, boundary_info):
            return candidates[1:], {"enabled": True, "selected_count": 1, "rejected_action_ids": ["a1"]}

        def refine_trajectory(self, task, trajectory):
            for step in trajectory["steps"]:
                _apply_step_refinement(
                    step,
                    {
                        "is_action": True,
                        "action": {"type": "computer.click", "args": {"x": 700, "y": 800, "button": "left"}},
                        "intent": "click submit",
                        "confidence": 0.96,
                        "target": "submit",
                    },
                )
            trajectory["quality"]["api_refined_steps"] = len(trajectory["steps"])
            trajectory["quality"]["needs_grounding_steps"] = 0
            return trajectory

        def verify_trajectory(self, task, trajectory):
            assert trajectory.get("_api_whole_verify") is True
            return {"trajectory_valid": True, "exportable_for_training": True, "quality_score": 5}

    monkeypatch.setattr(construct, "extract_screenshot", fake_extract)
    monkeypatch.setattr(construct, "_extract_even_frames", lambda *args, **kwargs: [tmp_path / "overview.jpg"])
    monkeypatch.setattr(construct, "_extract_candidate_frames", lambda *args, **kwargs: [tmp_path / "boundary.jpg"])

    video_id, status = _construct_one(record, args, HeavyFakeBackend())

    assert (video_id, status) == ("abc123def45", "ok")
    out_path = output_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.trajectory.json"
    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["task"] == "Save and submit payroll"
    assert obj["video_plan"]["enabled"] is True
    assert obj["api_boundary"]["enabled"] is True
    assert obj["selection"]["selected_count"] == 1
    assert obj["steps"][0]["intent"] == "click submit"
    assert obj["steps"][0]["action"]["args"]["x"] == 700
    assert obj["task_parts"][0]["parent_task"] == "Save and submit payroll"
    assert obj["task_parts"][0]["step_indices"] == [1]
    assert obj["verification"]["exportable_for_training"] is True


def test_construct_one_can_split_video_into_episode_trajectories(tmp_path: Path, monkeypatch) -> None:
    record = {
        "video_id": "abc123def45",
        "title": "Two workflows",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
    }
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "out"
    video_path = video_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.mp4"
    transcript_path = transcript_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.segments.json"
    video_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": 1, "start": 2.0, "end": 3.0, "text": "Click save."},
                    {"id": 2, "start": 12.0, "end": 13.0, "text": "Click submit."},
                ]
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        video_dir=str(video_dir),
        transcript_dir=str(transcript_dir),
        output_dir=str(output_dir),
        overwrite=False,
        split_episodes=True,
        require_transcript=True,
        semantic_pre_roll=0.0,
        semantic_post_roll=0.0,
        enable_visual=False,
        scene_threshold=0.18,
        max_visual_candidates=200,
        merge_tolerance=1.5,
        min_confidence=0.35,
        extract_screenshots=True,
        screenshot_width=640,
        observation_offset=0.0,
        post_observation_offset=0.0,
        api_video_plan=True,
        api_overview_frames=2,
        api_boundary_refine=False,
        api_boundary_frames=2,
        api_trajectory_select=False,
        api_ground_steps=True,
        api_verify_trajectory=True,
        api_frame_width=64,
        plan_ignore_overlap_ratio=0.8,
        plan_high_value_hard_filter=False,
        plan_primary_episode_filter=True,
        plan_primary_episode_tolerance=0.5,
        plan_empty_fallback_top_k=12,
        max_boundary_candidates=40,
        max_selected_steps=20,
    )

    def fake_extract(video_path: Path, dest: Path, timestamp: float, *, frame_width: int = 1280) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake jpg")
        return dest

    class SplitFakeBackend(PassthroughBackend):
        def plan_video(self, task, transcript, overview_frames):
            return {
                "enabled": True,
                "task_title": "Two workflows",
                "primary_episode": {"start": 1.5, "end": 3.5, "task": "save record", "confidence": 0.9},
                "episode_segments": [
                    {"start": 1.5, "end": 3.5, "task": "save record", "summary": "first workflow", "confidence": 0.9},
                    {"start": 11.5, "end": 13.5, "task": "submit record", "summary": "second workflow", "confidence": 0.9},
                ],
            }

        def refine_trajectory(self, task, trajectory):
            for step in trajectory["steps"]:
                _apply_step_refinement(
                    step,
                    {
                        "is_action": True,
                        "action": {"type": "computer.click", "args": {"x": 500, "y": 500, "button": "left"}},
                        "confidence": 0.9,
                    },
                )
            trajectory["quality"]["api_refined_steps"] = len(trajectory["steps"])
            trajectory["quality"]["needs_grounding_steps"] = 0
            return trajectory

        def verify_trajectory(self, task, trajectory):
            return {"trajectory_valid": True, "exportable_for_training": True}

    monkeypatch.setattr(construct, "extract_screenshot", fake_extract)
    monkeypatch.setattr(construct, "_extract_even_frames", lambda *args, **kwargs: [tmp_path / "overview.jpg"])

    video_id, status = _construct_one(record, args, SplitFakeBackend())

    assert (video_id, status) == ("abc123def45", "ok")
    first = output_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.trajectory.episode_01.json"
    second = output_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.trajectory.episode_02.json"
    first_obj = json.loads(first.read_text(encoding="utf-8"))
    second_obj = json.loads(second.read_text(encoding="utf-8"))
    assert first_obj["task"] == "save record"
    assert second_obj["task"] == "submit record"
    assert first_obj["episode"]["index"] == 1
    assert second_obj["episode"]["index"] == 2
    assert first_obj["steps"][0]["start"] == 2.0
    assert second_obj["steps"][0]["start"] == 12.0


def test_construct_one_keeps_long_episode_complete_and_marks_task_parts(tmp_path: Path, monkeypatch) -> None:
    record = {
        "video_id": "abc123def45",
        "title": "Long workflow",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
    }
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "out"
    video_path = video_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.mp4"
    transcript_path = transcript_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.segments.json"
    video_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    transcript_path.write_text(json.dumps({"segments": []}), encoding="utf-8")
    args = argparse.Namespace(
        video_dir=str(video_dir),
        transcript_dir=str(transcript_dir),
        output_dir=str(output_dir),
        overwrite=False,
        split_episodes=True,
        require_transcript=True,
        semantic_pre_roll=0.0,
        semantic_post_roll=0.0,
        enable_visual=False,
        scene_threshold=0.18,
        max_visual_candidates=200,
        merge_tolerance=1.5,
        min_confidence=0.35,
        extract_screenshots=True,
        screenshot_width=640,
        observation_offset=0.0,
        post_observation_offset=0.0,
        api_video_plan=True,
        api_overview_frames=2,
        api_boundary_refine=True,
        api_boundary_frames=2,
        api_trajectory_select=False,
        api_ground_steps=True,
        api_verify_trajectory=True,
        api_frame_width=64,
        plan_ignore_overlap_ratio=0.8,
        plan_high_value_hard_filter=False,
        plan_primary_episode_filter=True,
        plan_primary_episode_tolerance=0.5,
        plan_empty_fallback_top_k=12,
        max_boundary_candidates=10,
        max_selected_steps=2,
    )

    def fake_extract(video_path: Path, dest: Path, timestamp: float, *, frame_width: int = 1280) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake jpg")
        return dest

    class LongFakeBackend(PassthroughBackend):
        def plan_video(self, task, transcript, overview_frames):
            return {
                "enabled": True,
                "episode_segments": [
                    {"start": 0.0, "end": 10.0, "task": "complete long workflow", "summary": "one workflow", "confidence": 0.9},
                ],
            }

        def refine_boundaries(self, task, transcript, candidates, frames, video_plan):
            return [
                ActionCandidate(
                    start=float(index),
                    end=float(index) + 0.5,
                    source="api",
                    action_type="computer.click",
                    target=f"button {index}",
                    text=f"click button {index}",
                    confidence=0.9,
                )
                for index in range(5)
            ], {"enabled": True, "output_candidates": 5}

        def refine_trajectory(self, task, trajectory):
            for step in trajectory["steps"]:
                _apply_step_refinement(
                    step,
                    {
                        "is_action": True,
                        "action": {"type": "computer.click", "args": {"x": 500, "y": 500, "button": "left"}},
                        "confidence": 0.9,
                    },
                )
            trajectory["quality"]["api_refined_steps"] = len(trajectory["steps"])
            trajectory["quality"]["needs_grounding_steps"] = 0
            return trajectory

        def verify_trajectory(self, task, trajectory):
            return {"trajectory_valid": True, "exportable_for_training": True}

    monkeypatch.setattr(construct, "extract_screenshot", fake_extract)
    monkeypatch.setattr(construct, "_extract_even_frames", lambda *args, **kwargs: [tmp_path / "overview.jpg"])
    monkeypatch.setattr(construct, "_extract_candidate_frames", lambda *args, **kwargs: [tmp_path / "boundary.jpg"])

    video_id, status = _construct_one(record, args, LongFakeBackend())

    assert (video_id, status) == ("abc123def45", "ok")
    base = output_dir / "Business_and_Administration" / "Career"
    path = base / "ADP_Workforce_Now_abc123def45.trajectory.episode_01.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert len(obj["steps"]) == 6
    assert [step["start"] for step in obj["steps"]] == [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]
    assert obj["steps"][-1]["action"] == {"type": "finish", "args": {"status": "success"}}
    assert [part["step_indices"] for part in obj["task_parts"]] == [[1, 2], [3, 4], [5]]
    assert obj["task_parts"][0]["parent_task"] == "complete long workflow"
    assert obj["task_parts"][0]["task"].startswith("complete long workflow - subtask 1 of 3:")
    assert "click button 0" in obj["task_parts"][0]["task"]


def test_construct_one_splits_large_jumps_into_continuous_part_files(tmp_path: Path, monkeypatch) -> None:
    record = {
        "video_id": "abc123def45",
        "title": "Two distant workflows",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
    }
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "out"
    video_path = video_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.mp4"
    transcript_path = transcript_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.segments.json"
    video_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    transcript_path.write_text(json.dumps({"segments": []}), encoding="utf-8")
    args = argparse.Namespace(
        video_dir=str(video_dir),
        transcript_dir=str(transcript_dir),
        output_dir=str(output_dir),
        overwrite=False,
        split_episodes=True,
        split_long_trajectories=True,
        trajectory_split_gap_seconds=5.0,
        trajectory_split_min_steps=2,
        require_transcript=True,
        semantic_pre_roll=0.0,
        semantic_post_roll=0.0,
        enable_visual=False,
        scene_threshold=0.18,
        max_visual_candidates=200,
        merge_tolerance=1.5,
        min_confidence=0.35,
        extract_screenshots=True,
        screenshot_width=640,
        observation_offset=0.0,
        post_observation_offset=0.0,
        api_video_plan=True,
        api_overview_frames=2,
        api_boundary_refine=True,
        api_boundary_frames=2,
        api_trajectory_select=False,
        api_ground_steps=True,
        api_verify_trajectory=True,
        api_frame_width=64,
        plan_ignore_overlap_ratio=0.8,
        plan_high_value_hard_filter=False,
        plan_primary_episode_filter=True,
        plan_primary_episode_tolerance=0.5,
        plan_empty_fallback_top_k=12,
        max_boundary_candidates=10,
        max_selected_steps=20,
    )

    def fake_extract(video_path: Path, dest: Path, timestamp: float, *, frame_width: int = 1280) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake jpg")
        return dest

    class JumpFakeBackend(PassthroughBackend):
        def plan_video(self, task, transcript, overview_frames):
            return {
                "enabled": True,
                "episode_segments": [
                    {"start": 0.0, "end": 40.0, "task": "complete visible workflows", "summary": "contains a jump", "confidence": 0.9},
                ],
            }

        def refine_boundaries(self, task, transcript, candidates, frames, video_plan):
            starts = [0.0, 2.0, 20.0, 22.0]
            return [
                ActionCandidate(
                    start=start,
                    end=start + 1.0,
                    source="api",
                    action_type="computer.click",
                    target=f"button {index}",
                    text=f"click button {index}",
                    confidence=0.9,
                )
                for index, start in enumerate(starts, 1)
            ], {"enabled": True, "output_candidates": 4}

        def refine_trajectory(self, task, trajectory):
            for step in trajectory["steps"]:
                _apply_step_refinement(
                    step,
                    {
                        "is_action": True,
                        "action": {"type": "computer.click", "args": {"x": 500, "y": 500, "button": "left"}},
                        "confidence": 0.9,
                    },
                )
            trajectory["quality"]["api_refined_steps"] = len(trajectory["steps"])
            trajectory["quality"]["needs_grounding_steps"] = 0
            return trajectory

        def verify_trajectory(self, task, trajectory):
            return {"trajectory_valid": True, "exportable_for_training": True}

    monkeypatch.setattr(construct, "extract_screenshot", fake_extract)
    monkeypatch.setattr(construct, "_extract_even_frames", lambda *args, **kwargs: [tmp_path / "overview.jpg"])
    monkeypatch.setattr(construct, "_extract_candidate_frames", lambda *args, **kwargs: [tmp_path / "boundary.jpg"])

    video_id, status = _construct_one(record, args, JumpFakeBackend())

    assert (video_id, status) == ("abc123def45", "ok")
    base = output_dir / "Business_and_Administration" / "Career"
    first = base / "ADP_Workforce_Now_abc123def45.trajectory.episode_01.part_01.json"
    second = base / "ADP_Workforce_Now_abc123def45.trajectory.episode_01.part_02.json"
    assert not (base / "ADP_Workforce_Now_abc123def45.trajectory.episode_01.json").exists()
    first_obj = json.loads(first.read_text(encoding="utf-8"))
    second_obj = json.loads(second.read_text(encoding="utf-8"))
    assert [step["start"] for step in first_obj["steps"]] == [0.0, 2.0, 3.0]
    assert [step["start"] for step in second_obj["steps"]] == [20.0, 22.0, 23.0]
    assert first_obj["steps"][-1]["action"]["type"] == "finish"
    assert second_obj["steps"][-1]["action"]["type"] == "finish"
    assert first_obj["continuity_part"] == {
        "index": 1,
        "count": 2,
        "split_reason": "time_gap",
        "max_gap_seconds": 5.0,
    }
    assert second_obj["continuity_part"]["index"] == 2


def test_split_candidates_for_continuity_splits_obvious_context_shift() -> None:
    candidates = [
        ActionCandidate(
            start=0.0,
            end=1.0,
            source="api",
            action_type="computer.click",
            target="Fiverr gig title field",
            text="focus the Fiverr gig title field",
            confidence=0.9,
        ),
        ActionCandidate(
            start=2.0,
            end=3.0,
            source="api",
            action_type="computer.type",
            target="Fiverr gig title field",
            text="type the gig title",
            confidence=0.9,
        ),
        ActionCandidate(
            start=4.0,
            end=5.0,
            source="api",
            action_type="computer.type",
            target="Google/Chrome search or address field",
            text="search Google for ChatGPT",
            confidence=0.9,
        ),
        ActionCandidate(
            start=6.0,
            end=7.0,
            source="api",
            action_type="computer.click",
            target="Google search result for ChatGPT",
            text="open the ChatGPT result",
            confidence=0.9,
        ),
    ]

    chunks = _split_candidates_for_continuity(
        candidates,
        enabled=True,
        max_gap_seconds=20.0,
        min_steps=2,
    )

    assert [[candidate.target for candidate in chunk] for chunk in chunks] == [
        ["Fiverr gig title field", "Fiverr gig title field"],
        ["Google/Chrome search or address field", "Google search result for ChatGPT"],
    ]


def test_write_trajectories_removes_partial_parts_when_any_part_fails(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / "video.trajectory.json"
    args = argparse.Namespace(
        split_long_trajectories=True,
        trajectory_split_gap_seconds=5.0,
        trajectory_split_min_steps=1,
        overwrite=False,
    )
    chunks = [
        [ActionCandidate(start=0.0, end=1.0, source="api", action_type="computer.click")],
        [ActionCandidate(start=10.0, end=11.0, source="api", action_type="computer.click")],
    ]
    monkeypatch.setattr(construct, "_split_candidates_for_continuity", lambda *args, **kwargs: chunks)

    def fake_write_trajectory(*args, out_path: Path, **kwargs):
        if out_path.name.endswith("part_01.json"):
            out_path.write_text(
                json.dumps(
                    {
                        "video_id": "video",
                        "steps": [],
                        "quality": {"num_steps": 0, "has_screenshots": False},
                        "verification": {"trajectory_valid": False, "exportable_for_training": False},
                    }
                ),
                encoding="utf-8",
            )
            return True
        return False

    monkeypatch.setattr(construct, "_write_trajectory", fake_write_trajectory)

    written = construct._write_trajectories(
        argparse.Namespace(record={"video_id": "video"}),
        args,
        None,
        out_path=out_path,
        video_plan={},
        episode=None,
        episode_index=None,
        episode_count=None,
        refined=[],
        semantic_candidates=[],
        visual_candidates=[],
        lattice=[],
        boundary_info={},
        selection_info={},
        filter_diagnostics={},
    )

    assert written == 0
    assert not list(tmp_path.glob("*.part_*.json"))


def test_construct_one_skips_low_quality_continuity_parts(tmp_path: Path, monkeypatch) -> None:
    record = {
        "video_id": "abc123def45",
        "title": "Two distant workflows",
        "category": "Business & Administration",
        "subcategory": "Career",
        "application": "ADP Workforce Now",
    }
    video_dir = tmp_path / "videos"
    transcript_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "out"
    video_path = video_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.mp4"
    transcript_path = transcript_dir / "Business_and_Administration" / "Career" / "ADP_Workforce_Now_abc123def45.segments.json"
    video_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    transcript_path.write_text(json.dumps({"segments": []}), encoding="utf-8")
    args = argparse.Namespace(
        video_dir=str(video_dir),
        transcript_dir=str(transcript_dir),
        output_dir=str(output_dir),
        overwrite=False,
        split_episodes=True,
        split_long_trajectories=True,
        trajectory_split_gap_seconds=5.0,
        trajectory_split_min_steps=2,
        require_transcript=True,
        semantic_pre_roll=0.0,
        semantic_post_roll=0.0,
        enable_visual=False,
        scene_threshold=0.18,
        max_visual_candidates=200,
        merge_tolerance=1.5,
        min_confidence=0.35,
        extract_screenshots=False,
        screenshot_width=640,
        observation_offset=0.0,
        post_observation_offset=0.0,
        api_video_plan=True,
        api_overview_frames=2,
        api_boundary_refine=True,
        api_boundary_frames=2,
        api_trajectory_select=False,
        api_ground_steps=True,
        api_verify_trajectory=True,
        api_frame_width=64,
        plan_ignore_overlap_ratio=0.8,
        plan_high_value_hard_filter=False,
        plan_primary_episode_filter=True,
        plan_primary_episode_tolerance=0.5,
        plan_empty_fallback_top_k=12,
        max_boundary_candidates=10,
        max_selected_steps=20,
    )

    class LowQualityPartBackend(PassthroughBackend):
        def plan_video(self, task, transcript, overview_frames):
            return {
                "enabled": True,
                "episode_segments": [
                    {"start": 0.0, "end": 40.0, "task": "complete visible workflows", "summary": "contains a jump", "confidence": 0.9},
                ],
            }

        def refine_boundaries(self, task, transcript, candidates, frames, video_plan):
            starts = [0.0, 2.0, 20.0, 22.0]
            return [
                ActionCandidate(
                    start=start,
                    end=start + 1.0,
                    source="api",
                    action_type="computer.click",
                    target=f"button {index}",
                    text=f"click button {index}",
                    confidence=0.9,
                )
                for index, start in enumerate(starts, 1)
            ], {"enabled": True, "output_candidates": 4}

        def verify_trajectory(self, task, trajectory):
            return {"trajectory_valid": None, "exportable_for_training": None}

    monkeypatch.setattr(construct, "_extract_even_frames", lambda *args, **kwargs: [tmp_path / "overview.jpg"])
    monkeypatch.setattr(construct, "_extract_candidate_frames", lambda *args, **kwargs: [tmp_path / "boundary.jpg"])

    video_id, status = _construct_one(record, args, LowQualityPartBackend())

    assert (video_id, status) == ("abc123def45", "skipped")
    base = output_dir / "Business_and_Administration" / "Career"
    assert not list(base.glob("*.part_*.json"))
