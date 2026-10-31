"""Construct draft CUA action trajectories from tutorial videos."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from cuamaster.action_space import validate_action
from cuamaster.endpoints import (
    load_endpoint_pool_config,
    load_yaml_mapping,
    resolve_endpoint_pool_path,
)

from .assembler import build_draft_trajectory
from .backends import make_backend
from .lattice import merge_candidates
from .paths import load_jsonl, load_transcript_segments, resolve_video_task
from .screenshots import extract_screenshot, frame_dir_for_trajectory, step_frame_path
from .transcript import mine_semantic_candidates
from .visual import mine_scene_change_candidates
from ..common.utils import format_duration

logger = logging.getLogger("video2traj.construct_actions")

DEFAULT_INPUT = "data/data_list_labeled.jsonl"
DEFAULT_VIDEO_DIR = "data/videos"
DEFAULT_TRANSCRIPT_DIR = "data/transcripts"
DEFAULT_OUTPUT_DIR = "data/action_trajectories"
COMPLETION_SCHEMA_VERSION = 2
LEGACY_COMPLETION_SCHEMA_VERSION = 1
_OUTPUT_VARIANT_RE = re.compile(r"(?:\.episode_\d+)?(?:\.part_\d+)?")


def _output_path(relative_video_path: Path, output_dir: Path) -> Path:
    return (output_dir / relative_video_path).with_suffix(".trajectory.json")


def _episode_output_path(base_path: Path, episode_index: int) -> Path:
    return base_path.with_name(f"{base_path.stem}.episode_{episode_index:02d}{base_path.suffix}")


def _part_output_path(base_path: Path, part_index: int) -> Path:
    return base_path.with_name(f"{base_path.stem}.part_{part_index:02d}{base_path.suffix}")


def _summary_row(obj: dict, trajectory_path: Path | None = None, *, long_min_steps: int = 20) -> dict:
    quality = obj.get("quality") or {}
    verification = obj.get("verification") or {}
    num_steps = quality.get("num_steps")
    training_labels = _training_labels(obj, long_min_steps=long_min_steps)
    return {
        "video_id": obj.get("video_id"),
        "application": obj.get("application"),
        "category": obj.get("category"),
        "subcategory": obj.get("subcategory"),
        "source_video": obj.get("source_video"),
        "trajectory_path": str(trajectory_path) if trajectory_path else None,
        "task": obj.get("task"),
        "num_steps": num_steps,
        "step_count": num_steps,
        "trajectory_confidence": quality.get("trajectory_confidence"),
        "needs_grounding_steps": quality.get("needs_grounding_steps"),
        "has_screenshots": quality.get("has_screenshots"),
        "num_screenshots": quality.get("num_screenshots"),
        "api_refined_steps": quality.get("api_refined_steps"),
        "api_failed_steps": quality.get("api_failed_steps"),
        "zero_step_reason": quality.get("zero_step_reason"),
        "action_type_mismatches": quality.get("action_type_mismatches"),
        "candidate_filter_reason": (obj.get("candidate_filter") or {}).get("reason"),
        "trajectory_valid": verification.get("trajectory_valid"),
        "verified": verification.get("verified"),
        "exportable_for_training": verification.get("exportable_for_training"),
        "quality_score": verification.get("quality_score"),
        "draft": quality.get("draft"),
        **training_labels,
    }


def _training_row(obj: dict, trajectory_path: Path, *, long_min_steps: int = 20) -> dict:
    quality = obj.get("quality") or {}
    verification = obj.get("verification") or {}
    num_steps = int(quality.get("num_steps") or 0)
    labels = _training_labels(obj, long_min_steps=long_min_steps)
    return {
        "video_id": obj.get("video_id"),
        "application": obj.get("application"),
        "category": obj.get("category"),
        "subcategory": obj.get("subcategory"),
        "source_video": obj.get("source_video"),
        "trajectory_path": str(trajectory_path),
        "task": obj.get("task"),
        "num_steps": num_steps,
        "step_count": num_steps,
        "horizon_label": labels["horizon_label"],
        "training_tier": labels["training_tier"],
        "recommended_train_weight": labels["recommended_train_weight"],
        "trajectory_valid": verification.get("trajectory_valid"),
        "exportable_for_training": verification.get("exportable_for_training"),
        "quality_score": verification.get("quality_score"),
        "steps": obj.get("steps") or [],
    }


def ensure_success_finish_step(trajectory: dict) -> bool:
    """Append one synthetic success finish step to an exportable trajectory.

    The terminal decision is made from the final post-action state, so no new
    video frame needs to be extracted. Returns True only when a step is added.
    """
    verification = trajectory.get("verification") or {}
    if not isinstance(verification, dict):
        return False
    if verification.get("exportable_for_training") is not True:
        return False

    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    if any(
        isinstance(step, dict) and (step.get("action") or {}).get("type") == "finish"
        for step in steps
    ):
        return False

    last_step = steps[-1]
    if not isinstance(last_step, dict):
        return False
    final_observation = last_step.get("post_observation")
    if not isinstance(final_observation, dict) or not final_observation.get("screenshot"):
        return False

    try:
        terminal_time = round(float(last_step.get("end") or last_step.get("start") or 0.0), 3)
    except (TypeError, ValueError):
        terminal_time = 0.0
    finish_step = {
        "index": len(steps) + 1,
        "start": terminal_time,
        "end": terminal_time,
        "intent": "terminate the successfully completed task",
        "action": {"type": "finish", "args": {"status": "success"}},
        "needs_grounding": False,
        "confidence": 1.0,
        "evidence": {
            "source": "synthetic",
            "reason": "trajectory verified as exportable for training",
        },
        "observation": dict(final_observation),
        "post_observation": dict(final_observation),
    }
    steps.append(finish_step)

    quality = trajectory.get("quality")
    if not isinstance(quality, dict):
        quality = {}
        trajectory["quality"] = quality
    quality["num_steps"] = len(steps)
    quality["needs_grounding_steps"] = sum(
        1 for step in steps if isinstance(step, dict) and step.get("needs_grounding")
    )
    quality["has_finish_step"] = True
    return True


def _quality_score_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _training_labels(obj: dict, *, long_min_steps: int = 20) -> dict[str, Any]:
    quality = obj.get("quality") or {}
    verification = obj.get("verification") or {}
    num_steps = int(quality.get("num_steps") or 0)
    has_screenshots = quality.get("has_screenshots") is True
    trajectory_valid = verification.get("trajectory_valid") is True
    exportable = verification.get("exportable_for_training") is True
    quality_score = _quality_score_value(verification.get("quality_score"))
    quality_ok = quality_score is not None and quality_score >= 3
    quality_clean = all(
        _quality_issue_count(quality.get(key)) == 0
        for key in ("needs_grounding_steps", "api_failed_steps", "action_type_mismatches")
    )
    long_threshold = max(1, int(long_min_steps or 20))

    if num_steps >= long_threshold:
        horizon_label = "long"
    elif num_steps >= 1:
        horizon_label = "short"
    else:
        horizon_label = "empty"

    if exportable:
        completeness_label = "complete_workflow" if horizon_label == "long" else "complete_subtask"
    elif trajectory_valid:
        completeness_label = "valid_fragment"
    else:
        completeness_label = "incomplete"

    if exportable and trajectory_valid and has_screenshots and quality_ok and quality_clean and num_steps >= long_threshold:
        training_tier = "long"
        weight = 1.2 if num_steps >= 40 else 1.0
        reason = "exportable_long"
    elif trajectory_valid and has_screenshots and quality_ok and quality_clean and num_steps >= long_threshold:
        training_tier = "long"
        weight = 0.45
        reason = "valid_long_fragment_quality_ok"
    elif exportable and trajectory_valid and has_screenshots and quality_ok and quality_clean and num_steps >= 1:
        training_tier = "short"
        weight = 0.35 if quality_score is not None and quality_score >= 4 and num_steps >= 8 else 0.25
        reason = "exportable_short"
    else:
        training_tier = "reject"
        weight = 0.0
        if not quality_clean:
            reason = "quality_integrity_failed"
        elif trajectory_valid and has_screenshots and quality_ok and num_steps >= 1:
            reason = "valid_short_fragment_quality_ok"
        else:
            reason = "below_training_threshold"

    return {
        "training_tier": training_tier,
        "horizon_label": horizon_label,
        "completeness_label": completeness_label,
        "recommended_train_weight": weight,
        "retention_reason": reason,
        "quality_clean": quality_clean,
    }


def _quality_issue_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 1


def _summary_row_trainable(row: dict) -> bool:
    return (
        row.get("training_tier") in {"long", "short"}
        and row.get("retention_reason") in {"exportable_long", "exportable_short"}
    )


def _summary_row_long_candidate(row: dict) -> bool:
    return row.get("retention_reason") == "valid_long_fragment_quality_ok"


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _canonical_action_errors(obj: dict) -> list[str]:
    errors: list[str] = []
    for position, step in enumerate(obj.get("steps") or [], 1):
        if not isinstance(step, dict):
            errors.append(f"step_{position}:step_not_object")
            continue
        index = step.get("index", position)
        if step.get("needs_grounding"):
            errors.append(f"step_{index}:needs_grounding")
        try:
            validate_action(step.get("action"))
        except (TypeError, ValueError) as exc:
            errors.append(f"step_{index}:{exc}")
    return errors


def _enforce_quality_integrity_gate(trajectory: dict) -> list[str]:
    quality = trajectory.get("quality")
    if not isinstance(quality, dict):
        quality = {}
        trajectory["quality"] = quality
    verification = trajectory.get("verification")
    if not isinstance(verification, dict):
        verification = {}
        trajectory["verification"] = verification

    errors = []
    for key in ("needs_grounding_steps", "api_failed_steps", "action_type_mismatches"):
        count = _quality_issue_count(quality.get(key))
        if count:
            errors.append(f"{key}:{count}")
    canonical_errors = _canonical_action_errors(trajectory)
    quality["canonical_action_error_count"] = len(canonical_errors)
    quality["canonical_action_errors"] = canonical_errors[:20]
    errors.extend(f"canonical_action:{error}" for error in canonical_errors[:20])

    verification["quality_integrity_passed"] = not errors
    verification["quality_integrity_errors"] = errors[:20]
    if errors:
        verification["exportable_for_training"] = False
    return errors


def _summarize(
    output_dir: Path,
    *,
    write_audit_jsonl: bool = False,
    include_long_candidates_in_train: bool = False,
    long_min_steps: int = 20,
) -> Path:
    train_path = output_dir / "trajectories.train.jsonl"
    summary_path = output_dir / "trajectories.jsonl"
    long_candidate_summary_path = output_dir / "trajectories.long_candidates.jsonl"
    audit_summary_path = output_dir / "trajectories.audit.jsonl"
    rejected_summary_path = output_dir / "trajectories.rejected.jsonl"
    audit_rows = []
    train_rows = []
    long_candidate_rows = []
    rejected_rows = []
    rejected_canonical = 0
    for path in sorted(output_dir.rglob("*.trajectory*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        audit_row = _summary_row(obj, path, long_min_steps=long_min_steps)
        training_row = _training_row(obj, path, long_min_steps=long_min_steps)
        canonical_errors = _canonical_action_errors(obj)
        audit_row["canonical_action_valid"] = not canonical_errors
        audit_row["canonical_action_errors"] = canonical_errors[:20]
        audit_rows.append(audit_row)
        if canonical_errors:
            rejected_canonical += 1
            if (audit_row.get("num_steps") or 0) > 0:
                audit_row["retention_reason"] = "invalid_canonical_action"
                rejected_rows.append(audit_row)
        elif _summary_row_trainable(audit_row):
            train_rows.append(training_row)
        elif _summary_row_long_candidate(audit_row):
            long_candidate_rows.append(training_row)
        elif (audit_row.get("num_steps") or 0) > 0:
            rejected_rows.append(audit_row)
    if include_long_candidates_in_train:
        train_rows = [*train_rows, *long_candidate_rows]
    action_types = Counter(
        str(action.get("type") or "unknown")
        for row in train_rows
        for step in row.get("steps") or []
        if isinstance(step, dict)
        for action in [step.get("action")]
        if isinstance(action, dict)
    )
    tool_names = Counter(
        str((action.get("args") or {}).get("name") or "unknown")
        for row in train_rows
        for step in row.get("steps") or []
        if isinstance(step, dict)
        for action in [step.get("action")]
        if isinstance(action, dict) and action.get("type") == "tool.call"
    )
    _write_jsonl_atomic(train_path, train_rows)
    _write_jsonl_atomic(summary_path, train_rows)
    _write_jsonl_atomic(long_candidate_summary_path, long_candidate_rows)
    if write_audit_jsonl:
        _write_jsonl_atomic(audit_summary_path, audit_rows)
        _write_jsonl_atomic(rejected_summary_path, rejected_rows)
    stats_path = output_dir / "trajectories.action_stats.json"
    _write_json_atomic(
        stats_path,
        {
            "train_trajectories": len(train_rows),
            "train_actions": sum(action_types.values()),
            "action_types": dict(sorted(action_types.items())),
            "tool_names": dict(sorted(tool_names.items())),
            "rejected_canonical_trajectories": rejected_canonical,
        },
    )
    logger.info(
        "summarized %d train, %d long candidates, %d rejected, %d audit trajectories -> %s",
        len(train_rows),
        len(long_candidate_rows),
        len(rejected_rows),
        len(audit_rows),
        train_path,
    )
    return summary_path


def _trajectory_integrity_errors(path: Path) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable_json:{exc.__class__.__name__}"]

    errors = []
    video_id = str(obj.get("video_id") or "")
    if not video_id:
        errors.append("missing_video_id")
    steps_value = obj.get("steps")
    if not isinstance(steps_value, list):
        errors.append("missing_steps")
        steps = []
    else:
        steps = steps_value
    quality = obj.get("quality")
    if not isinstance(quality, dict):
        errors.append("missing_quality")
        quality = {}
    if not isinstance(obj.get("verification"), dict):
        errors.append("missing_verification")

    expected_steps = quality.get("num_steps")
    if isinstance(expected_steps, int) and expected_steps != len(steps):
        errors.append("step_count_mismatch")
    if quality.get("has_screenshots") is True:
        for step in steps:
            index = step.get("index", "?")
            for key in ("observation", "post_observation"):
                screenshot = ((step.get(key) or {}).get("screenshot"))
                if not screenshot:
                    errors.append(f"missing_{key}_screenshot:{index}")
                    continue
                if not Path(screenshot).exists():
                    errors.append(f"missing_screenshot_file:{index}:{key}")
    return errors


def _remove_trajectory_artifacts(path: Path) -> None:
    frame_dir = frame_dir_for_trajectory(path)
    tmp_path = path.with_name(f".{path.name}.tmp")
    for candidate in (path, tmp_path):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    if frame_dir.exists():
        shutil.rmtree(frame_dir)


def _is_complete_trajectory(path: Path) -> bool:
    return not _trajectory_integrity_errors(path)


def _completion_manifest_path(output_dir: Path, video_id: str) -> Path:
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:24]
    return output_dir / ".completions" / f"{digest}.json"


def _legacy_completion_manifest_path(output_dir: Path, base_path: Path) -> Path:
    relative_path = base_path.relative_to(output_dir).as_posix()
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return output_dir / ".completions" / f"{digest}.json"


def _output_relative_path(output_dir: Path, path: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _safe_output_path(output_dir: Path, relative_path: Any) -> Path | None:
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        return None
    candidate = output_dir / relative_path
    try:
        candidate.resolve().relative_to(output_dir.resolve())
    except ValueError:
        return None
    return candidate


def _trajectory_output_paths(base_path: Path) -> list[Path]:
    paths = []
    for path in base_path.parent.glob(f"{base_path.stem}*{base_path.suffix}"):
        if _trajectory_path_belongs_to_base(base_path, path):
            paths.append(path)
    return sorted(paths)


def _trajectory_path_belongs_to_base(base_path: Path, path: Path) -> bool:
    if path.parent != base_path.parent or not path.name.startswith(base_path.stem) or path.suffix != base_path.suffix:
        return False
    variant = path.name[len(base_path.stem) : -len(base_path.suffix)]
    return _OUTPUT_VARIANT_RE.fullmatch(variant) is not None


def _remove_base_trajectory_artifacts(base_path: Path) -> None:
    for path in _trajectory_output_paths(base_path):
        _remove_trajectory_artifacts(path)
    for tmp_path in base_path.parent.glob(f".{base_path.stem}*{base_path.suffix}.tmp"):
        name = tmp_path.name[1:-4]
        variant = name[len(base_path.stem) : -len(base_path.suffix)]
        if _OUTPUT_VARIANT_RE.fullmatch(variant):
            tmp_path.unlink(missing_ok=True)
    for frame_dir in base_path.parent.glob(f"{base_path.stem}*.frames"):
        variant = frame_dir.name[len(base_path.stem) : -len(".frames")]
        if _OUTPUT_VARIANT_RE.fullmatch(variant):
            shutil.rmtree(frame_dir, ignore_errors=True)


def _trajectory_video_id(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    return str(obj.get("video_id") or "")


def _completion_manifest_errors(output_dir: Path, path: Path, manifest: dict) -> tuple[list[str], list[Path]]:
    errors: list[str] = []
    schema_version = manifest.get("schema_version")
    video_id = str(manifest.get("video_id") or "")
    if not video_id:
        errors.append("missing_video_id")
    status = manifest.get("status")
    if schema_version == COMPLETION_SCHEMA_VERSION:
        if status != "complete":
            errors.append("invalid_status")
        if video_id and path != _completion_manifest_path(output_dir, video_id):
            errors.append("unexpected_manifest_path")
    elif schema_version == LEGACY_COMPLETION_SCHEMA_VERSION:
        if status not in {"complete", "no_output"}:
            errors.append("invalid_status")
        base_path = _safe_output_path(output_dir, manifest.get("base_trajectory_path"))
        if base_path is None:
            errors.append("invalid_base_trajectory_path")
        elif path != _legacy_completion_manifest_path(output_dir, base_path):
            errors.append("unexpected_manifest_path")
    else:
        errors.append("unsupported_schema_version")

    output_values = manifest.get("outputs")
    if not isinstance(output_values, list):
        errors.append("invalid_outputs")
        output_values = []
    output_paths: list[Path] = []
    for value in output_values:
        output_path = _safe_output_path(output_dir, value)
        if output_path is None:
            errors.append("invalid_output_path")
            continue
        output_paths.append(output_path)
    if schema_version == LEGACY_COMPLETION_SCHEMA_VERSION and status == "complete" and not output_paths:
        errors.append("complete_without_outputs")
    if schema_version == LEGACY_COMPLETION_SCHEMA_VERSION and status == "no_output" and output_paths:
        errors.append("no_output_with_outputs")
    if len(set(output_paths)) != len(output_paths):
        errors.append("duplicate_outputs")
    return errors, output_paths


def _read_completion_manifest(output_dir: Path, path: Path) -> tuple[dict | None, list[str], list[Path]]:
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"unreadable_manifest:{exc.__class__.__name__}"], []
    if not isinstance(manifest, dict):
        return None, ["manifest_not_object"], []
    errors, output_paths = _completion_manifest_errors(output_dir, path, manifest)
    return manifest, errors, output_paths


def _write_completion_manifest(output_dir: Path, video_id: str, output_paths: list[Path]) -> None:
    manifest_path = _completion_manifest_path(output_dir, video_id)
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "video_id": video_id,
            "status": "complete",
            "outputs": [_output_relative_path(output_dir, path) for path in output_paths],
        },
    )


def _scan_video_output_state(
    output_dir: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]], set[str]]:
    trajectories_by_video_id: dict[str, list[Path]] = {}
    manifests_by_video_id: dict[str, list[Path]] = {}
    completed_video_ids: set[str] = set()

    for path in sorted(output_dir.rglob("*.trajectory*.json")):
        video_id = _trajectory_video_id(path)
        if video_id:
            trajectories_by_video_id.setdefault(video_id, []).append(path)

    completion_dir = output_dir / ".completions"
    if completion_dir.exists():
        for path in sorted(completion_dir.glob("*.json")):
            manifest, errors, _ = _read_completion_manifest(output_dir, path)
            if manifest is None:
                continue
            video_id = str(manifest.get("video_id") or "")
            if not video_id:
                continue
            manifests_by_video_id.setdefault(video_id, []).append(path)
            if not errors:
                completed_video_ids.add(video_id)
    return trajectories_by_video_id, manifests_by_video_id, completed_video_ids


def _record_base_paths_by_video_id(records: list[dict], args: argparse.Namespace) -> dict[str, list[Path]]:
    output_dir = Path(args.output_dir)
    video_dir = Path(args.video_dir)
    transcript_dir = Path(args.transcript_dir)
    base_paths_by_video_id: dict[str, list[Path]] = {}
    for record in records:
        video_id = str(record.get("video_id") or "")
        if not video_id:
            continue
        task = resolve_video_task(record, video_dir, transcript_dir)
        base_path = _output_path(task.relative_path, output_dir)
        if not _path_is_within_output_dir(output_dir, base_path):
            raise ValueError(f"trajectory output path escapes output directory: {base_path}")
        paths = base_paths_by_video_id.setdefault(video_id, [])
        if base_path not in paths:
            paths.append(base_path)
    return base_paths_by_video_id


def _path_is_within_output_dir(output_dir: Path, path: Path) -> bool:
    normalized_output_dir = output_dir.resolve()
    normalized_path = path.resolve()
    try:
        normalized_path.relative_to(normalized_output_dir)
    except ValueError:
        return False
    return True


def _require_output_artifact_path(output_dir: Path, path: Path) -> None:
    if not _path_is_within_output_dir(output_dir, path):
        raise ValueError(f"refusing to remove path outside output directory: {path}")


def _require_no_foreign_base_outputs(base_paths: list[Path], video_id: str) -> None:
    for base_path in base_paths:
        for path in _trajectory_output_paths(base_path):
            owner_video_id = _trajectory_video_id(path)
            if owner_video_id and owner_video_id != video_id:
                raise ValueError(
                    f"refusing to replace trajectory owned by video_id={owner_video_id}: {path}"
                )


def _remove_base_trajectory_artifacts_for_video(base_path: Path, video_id: str) -> None:
    _require_no_foreign_base_outputs([base_path], video_id)
    _remove_base_trajectory_artifacts(base_path)


def _require_trajectory_owned_by_video(path: Path, video_id: str) -> None:
    owner_video_id = _trajectory_video_id(path)
    if owner_video_id and owner_video_id != video_id:
        raise ValueError(f"refusing to replace trajectory owned by video_id={owner_video_id}: {path}")


def _remove_trajectory_artifacts_for_video(path: Path, video_id: str) -> None:
    _require_trajectory_owned_by_video(path, video_id)
    _remove_trajectory_artifacts(path)


def _remove_video_id_artifacts(
    output_dir: Path,
    video_id: str,
    base_paths: list[Path],
    trajectory_paths: list[Path],
    tmp_paths: list[Path],
    frame_dirs: list[Path],
    manifest_paths: list[Path],
) -> None:
    for path in set(trajectory_paths):
        _require_output_artifact_path(output_dir, path)
        _remove_trajectory_artifacts_for_video(path, video_id)
    for tmp_path in tmp_paths:
        _require_output_artifact_path(output_dir, tmp_path)
        tmp_path.unlink(missing_ok=True)
    for frame_dir in frame_dirs:
        _require_output_artifact_path(output_dir, frame_dir)
        shutil.rmtree(frame_dir, ignore_errors=True)
    for base_path in base_paths:
        _require_output_artifact_path(output_dir, base_path)
        _remove_base_trajectory_artifacts_for_video(base_path, video_id)
        legacy_manifest_path = _legacy_completion_manifest_path(output_dir, base_path)
        legacy_manifest_path.unlink(missing_ok=True)
        legacy_manifest_path.with_name(f".{legacy_manifest_path.name}.tmp").unlink(missing_ok=True)
    for path in manifest_paths:
        path.unlink(missing_ok=True)
        path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
    manifest_path = _completion_manifest_path(output_dir, video_id)
    manifest_path.unlink(missing_ok=True)
    manifest_path.with_name(f".{manifest_path.name}.tmp").unlink(missing_ok=True)


def _artifact_name_belongs_to_video_id(path: Path, video_id: str) -> bool:
    name = path.name.removeprefix(".")
    marker = f"_{video_id}.trajectory"
    marker_index = name.rfind(marker)
    if marker_index < 0:
        return False
    suffix = name[marker_index + len(marker) :]
    for terminal_suffix in (".json.tmp", ".json", ".frames"):
        if not suffix.endswith(terminal_suffix):
            continue
        variant = suffix[: -len(terminal_suffix)]
        return _OUTPUT_VARIANT_RE.fullmatch(variant) is not None
    return False


def _named_artifacts_by_video_id(
    output_dir: Path,
    video_ids: set[str],
) -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]]]:
    ordered_video_ids = sorted(video_ids, key=lambda value: (-len(value), value))
    trajectories: dict[str, list[Path]] = {}
    tmp_paths: dict[str, list[Path]] = {}
    frame_dirs: dict[str, list[Path]] = {}
    for paths, pattern in (
        (trajectories, "*.trajectory*.json"),
        (tmp_paths, ".*.trajectory*.json.tmp"),
        (frame_dirs, "*.trajectory*.frames"),
    ):
        for path in output_dir.rglob(pattern):
            for video_id in ordered_video_ids:
                if _artifact_name_belongs_to_video_id(path, video_id):
                    if paths is trajectories:
                        owner_video_id = _trajectory_video_id(path)
                        if owner_video_id and owner_video_id != video_id:
                            continue
                    if paths is frame_dirs:
                        owner_video_id = _trajectory_video_id(path.with_suffix(".json"))
                        if owner_video_id and owner_video_id != video_id:
                            continue
                    paths.setdefault(video_id, []).append(path)
                    break
    return trajectories, tmp_paths, frame_dirs


def _prepare_video_ids_for_run(records: list[dict], args: argparse.Namespace) -> set[str]:
    output_dir = Path(args.output_dir)
    base_paths_by_video_id = _record_base_paths_by_video_id(records, args)
    trajectories_by_video_id, manifests_by_video_id, manifest_completed_ids = _scan_video_output_state(output_dir)
    named_trajectories, named_tmp_paths, named_frame_dirs = _named_artifacts_by_video_id(
        output_dir,
        set(base_paths_by_video_id),
    )
    completed_video_ids: set[str] = set()
    reset_video_ids = 0

    for video_id, base_paths in base_paths_by_video_id.items():
        trajectory_paths = trajectories_by_video_id.get(video_id, [])
        all_trajectory_paths = list({*trajectory_paths, *named_trajectories.get(video_id, [])})
        manifest_paths = manifests_by_video_id.get(video_id, [])
        expected_manifest_paths = {
            _completion_manifest_path(output_dir, video_id),
            *(_legacy_completion_manifest_path(output_dir, base_path) for base_path in base_paths),
        }
        has_unreadable_expected_manifest = any(
            path.exists() and path not in manifest_paths for path in expected_manifest_paths
        )
        legacy_complete = (
            not manifest_paths
            and not has_unreadable_expected_manifest
            and bool(all_trajectory_paths)
            and all(_is_complete_trajectory(path) for path in all_trajectory_paths)
        )
        if not args.overwrite and (video_id in manifest_completed_ids or legacy_complete):
            completed_video_ids.add(video_id)
            continue
        _require_no_foreign_base_outputs(base_paths, video_id)
        _remove_video_id_artifacts(
            output_dir,
            video_id,
            base_paths,
            all_trajectory_paths,
            named_tmp_paths.get(video_id, []),
            named_frame_dirs.get(video_id, []),
            manifest_paths,
        )
        reset_video_ids += 1
    logger.info(
        "prepared video output state completed=%d reset=%d",
        len(completed_video_ids),
        reset_video_ids,
    )
    return completed_video_ids


def _claim_video_id_or_skip(record: dict, seen_video_ids: set[str] | None) -> bool:
    if seen_video_ids is None:
        return False
    video_id = str(record.get("video_id") or "")
    if not video_id:
        return False
    if video_id in seen_video_ids:
        return True
    seen_video_ids.add(video_id)
    return False


def _construct_one(record: dict, args: argparse.Namespace, backend) -> tuple[str, str]:
    video_dir = Path(args.video_dir)
    transcript_dir = Path(args.transcript_dir)
    output_dir = Path(args.output_dir)
    task = resolve_video_task(record, video_dir, transcript_dir)
    out_path = _output_path(task.relative_path, output_dir)
    video_id = str(record.get("video_id") or "")
    manifest_path = _completion_manifest_path(output_dir, video_id)

    if manifest_path.exists() and not args.overwrite:
        manifest, errors, _ = _read_completion_manifest(output_dir, manifest_path)
        if manifest is not None and not errors and str(manifest.get("video_id") or "") == video_id:
            return video_id, "skipped"
    manifest_path.unlink(missing_ok=True)
    _remove_base_trajectory_artifacts_for_video(out_path, video_id)

    if not task.video_path.exists():
        return video_id, "missing_video"
    if args.require_transcript and not task.transcript_path.exists():
        return video_id, "missing_transcript"

    segments = []
    transcript_rows: list[dict] = []
    if task.transcript_path.exists():
        segments = load_transcript_segments(task.transcript_path)
        transcript_rows = [segment.__dict__ for segment in segments]

    with tempfile.TemporaryDirectory(prefix="action_construct_api_frames_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        video_plan = {"enabled": False}
        if getattr(args, "api_video_plan", False):
            overview_frames = _extract_even_frames(
                task.video_path,
                tmp_path / "overview",
                getattr(args, "api_overview_frames", 12),
                getattr(args, "api_frame_width", 512),
            )
            video_plan = backend.plan_video(task, transcript_rows, overview_frames)

        episodes = _planned_episodes(video_plan, enabled=getattr(args, "split_episodes", False))
        if not episodes:
            episodes = [None]
        output_paths = [
            _episode_output_path(out_path, index) if episode is not None else out_path
            for index, episode in enumerate(episodes, 1)
        ]
        if (
            not getattr(args, "split_long_trajectories", False)
            and output_paths
            and all(_is_complete_trajectory(path) for path in output_paths)
            and not args.overwrite
        ):
            return video_id, "skipped"

        if segments:
            semantic_raw = mine_semantic_candidates(
                segments,
                pre_roll=args.semantic_pre_roll,
                post_roll=args.semantic_post_roll,
            )
        else:
            semantic_raw = []

        if args.enable_visual:
            visual_raw = mine_scene_change_candidates(
                task.video_path,
                threshold=args.scene_threshold,
                max_candidates=args.max_visual_candidates,
            )
        else:
            visual_raw = []

        written = 0
        all_episodes_complete = True
        for index, episode in enumerate(episodes, 1):
            episode_out_path = _episode_output_path(out_path, index) if episode is not None else out_path
            if episode_out_path.exists() and not args.overwrite:
                _require_trajectory_owned_by_video(episode_out_path, video_id)
                if _is_complete_trajectory(episode_out_path):
                    continue
                _remove_trajectory_artifacts_for_video(episode_out_path, video_id)
            wrote = _construct_episode_trajectory(
                task,
                args,
                backend,
                tmp_path,
                out_path=episode_out_path,
                video_plan=video_plan,
                episode=episode,
                episode_index=index if episode is not None else None,
                episode_count=len(episodes) if episode is not None else None,
                transcript_rows=transcript_rows,
                semantic_raw=semantic_raw,
                visual_raw=visual_raw,
            )
            written += wrote
            if wrote == 0:
                all_episodes_complete = False
        if not all_episodes_complete:
            _remove_base_trajectory_artifacts_for_video(out_path, video_id)
            written = 0
    output_paths = _trajectory_output_paths(out_path)
    invalid_outputs = [
        path
        for path in output_paths
        if _trajectory_integrity_errors(path) or _trajectory_video_id(path) != video_id
    ]
    if invalid_outputs:
        _remove_base_trajectory_artifacts_for_video(out_path, video_id)
        raise RuntimeError(f"trajectory output validation failed: {invalid_outputs[0]}")
    _write_completion_manifest(output_dir, video_id, output_paths)
    return video_id, "ok" if written and output_paths else "skipped"


def _construct_episode_trajectory(
    task,
    args: argparse.Namespace,
    backend,
    tmp_path: Path,
    *,
    out_path: Path,
    video_plan: dict,
    episode: dict | None,
    episode_index: int | None,
    episode_count: int | None,
    transcript_rows: list[dict],
    semantic_raw,
    visual_raw,
) -> int:
    episode_plan = _episode_video_plan(video_plan, episode)
    semantic_candidates = _filter_candidates_by_plan(
        semantic_raw,
        episode_plan,
        ignore_overlap_ratio=getattr(args, "plan_ignore_overlap_ratio", 0.8),
        high_value_hard_filter=getattr(args, "plan_high_value_hard_filter", False),
    )
    semantic_candidates = _filter_candidates_to_primary_episode(
        semantic_candidates,
        episode_plan,
        enabled=getattr(args, "plan_primary_episode_filter", False) or episode is not None,
        tolerance=getattr(args, "plan_primary_episode_tolerance", 10.0),
    )
    visual_candidates = _filter_candidates_by_plan(
        visual_raw,
        episode_plan,
        ignore_overlap_ratio=getattr(args, "plan_ignore_overlap_ratio", 0.8),
        high_value_hard_filter=getattr(args, "plan_high_value_hard_filter", False),
    )
    visual_candidates = _filter_candidates_to_primary_episode(
        visual_candidates,
        episode_plan,
        enabled=getattr(args, "plan_primary_episode_filter", False) or episode is not None,
        tolerance=getattr(args, "plan_primary_episode_tolerance", 10.0),
    )

    filter_diagnostics = _candidate_filter_diagnostics(
        semantic_raw,
        semantic_candidates,
        visual_raw,
        visual_candidates,
        episode_plan,
    )
    if episode is not None:
        filter_diagnostics["episode_index"] = episode_index
        filter_diagnostics["episode_count"] = episode_count
    if not semantic_candidates and not visual_candidates:
        fallback = _fallback_candidates_after_empty_filter(
            [*semantic_raw, *visual_raw],
            episode_plan,
            getattr(args, "plan_empty_fallback_top_k", 12),
        )
        if fallback:
            semantic_candidates = [candidate for candidate in fallback if candidate.source == "semantic"]
            visual_candidates = [candidate for candidate in fallback if candidate.source != "semantic"]
            filter_diagnostics["fallback_used"] = True
            filter_diagnostics["fallback_count"] = len(fallback)
        else:
            filter_diagnostics["fallback_used"] = False
            filter_diagnostics["zero_step_reason"] = _zero_step_reason(episode_plan, semantic_raw, visual_raw)

    lattice = merge_candidates(
        [*semantic_candidates, *visual_candidates],
        overlap_tolerance=args.merge_tolerance,
    )
    lattice = _limit_candidates(lattice, getattr(args, "max_boundary_candidates", 40))
    refined = backend.refine_candidates(task, lattice)
    skip_expensive_stages = filter_diagnostics.get("zero_step_reason") == "no_cua_operation" and not refined

    boundary_info = {"enabled": False}
    if getattr(args, "api_boundary_refine", False):
        if skip_expensive_stages:
            boundary_info = {"enabled": False, "skipped": True, "reason": "no_cua_operation"}
        else:
            boundary_frames = _extract_candidate_frames(
                task.video_path,
                tmp_path / f"boundary_{episode_index or 0}",
                refined,
                getattr(args, "api_boundary_frames", 24),
                getattr(args, "api_frame_width", 512),
            )
            refined, boundary_info = backend.refine_boundaries(
                task,
                transcript_rows,
                refined,
                boundary_frames,
                episode_plan,
            )

    selection_info = {"enabled": False, "selected_count": len(refined)}
    pre_selection_refined = list(refined)
    if getattr(args, "api_trajectory_select", False):
        if skip_expensive_stages:
            selection_info = {"enabled": False, "skipped": True, "reason": "no_cua_operation", "selected_count": 0}
        else:
            refined, selection_info = backend.select_trajectory(task, refined, episode_plan, boundary_info)
    if getattr(args, "extend_short_trajectories", False):
        refined, extension_info = _extend_short_trajectory_selection(
            selected=refined,
            candidate_pool=pre_selection_refined,
            min_steps=getattr(args, "long_target_steps", getattr(args, "extension_min_steps", 40)),
            max_gap_seconds=getattr(args, "extension_max_gap_seconds", 120.0),
            min_confidence=getattr(args, "extension_min_confidence", 0.28),
        )
        selection_info = dict(selection_info)
        selection_info["extension"] = extension_info
        selection_info["selected_count_after_extension"] = len(refined)
    if out_path.exists() and not args.overwrite:
        video_id = str(task.record.get("video_id") or "")
        _require_trajectory_owned_by_video(out_path, video_id)
        if _is_complete_trajectory(out_path):
            return 0
        _remove_trajectory_artifacts_for_video(out_path, video_id)
    if not getattr(args, "split_long_trajectories", True):
        refined = _limit_selected_candidates(refined, getattr(args, "max_selected_steps", 20))
    return _write_trajectories(
        task,
        args,
        backend,
        out_path=out_path,
        video_plan=video_plan,
        episode=episode,
        episode_index=episode_index,
        episode_count=episode_count,
        refined=refined,
        semantic_candidates=semantic_candidates,
        visual_candidates=visual_candidates,
        lattice=lattice,
        boundary_info=boundary_info,
        selection_info=selection_info,
        filter_diagnostics=filter_diagnostics,
    )


def _write_trajectories(
    task,
    args: argparse.Namespace,
    backend,
    *,
    out_path: Path,
    video_plan: dict,
    episode: dict | None,
    episode_index: int | None,
    episode_count: int | None,
    refined,
    semantic_candidates,
    visual_candidates,
    lattice,
    boundary_info: dict,
    selection_info: dict,
    filter_diagnostics: dict,
) -> int:
    chunks = _split_candidates_for_continuity(
        refined,
        enabled=getattr(args, "split_long_trajectories", True),
        max_gap_seconds=getattr(args, "trajectory_split_gap_seconds", 20.0),
        min_steps=getattr(args, "trajectory_split_min_steps", 2),
    )
    if len(chunks) == 1:
        wrote = _write_trajectory(
            task,
            args,
            backend,
            out_path=out_path,
            video_plan=video_plan,
            episode=episode,
            episode_index=episode_index,
            episode_count=episode_count,
            refined=chunks[0],
            semantic_candidates=semantic_candidates,
            visual_candidates=visual_candidates,
            lattice=lattice,
            boundary_info=boundary_info,
            selection_info=selection_info,
            filter_diagnostics=filter_diagnostics,
            continuity_part=None,
        )
        return int(wrote)

    written = 0
    all_parts_complete = True
    min_part_steps = max(1, int(getattr(args, "trajectory_split_min_steps", 2) or 1))
    split_reason = _continuity_split_reason(
        chunks,
        max_gap_seconds=getattr(args, "trajectory_split_gap_seconds", 20.0),
    )
    for part_index, chunk in enumerate(chunks, 1):
        if len(chunk) < min_part_steps:
            logger.info("skip short trajectory part steps=%d min=%d path=%s", len(chunk), min_part_steps, out_path)
            all_parts_complete = False
            continue
        part_path = _part_output_path(out_path, part_index)
        if part_path.exists() and not args.overwrite:
            video_id = str(task.record.get("video_id") or "")
            _require_trajectory_owned_by_video(part_path, video_id)
            if _is_complete_trajectory(part_path):
                written += 1
                continue
            _remove_trajectory_artifacts_for_video(part_path, video_id)
        part_info = {
            "index": part_index,
            "count": len(chunks),
            "split_reason": split_reason,
            "max_gap_seconds": getattr(args, "trajectory_split_gap_seconds", 20.0),
        }
        if _write_trajectory(
            task,
            args,
            backend,
            out_path=part_path,
            video_plan=video_plan,
            episode=episode,
            episode_index=episode_index,
            episode_count=episode_count,
            refined=chunk,
            semantic_candidates=semantic_candidates,
            visual_candidates=visual_candidates,
            lattice=lattice,
            boundary_info=boundary_info,
            selection_info=selection_info,
            filter_diagnostics=filter_diagnostics,
            continuity_part=part_info,
        ):
            written += 1
        else:
            all_parts_complete = False
    if not all_parts_complete:
        _remove_base_trajectory_artifacts_for_video(out_path, str(task.record.get("video_id") or ""))
        return 0
    return written


def _write_trajectory(
    task,
    args: argparse.Namespace,
    backend,
    *,
    out_path: Path,
    video_plan: dict,
    episode: dict | None,
    episode_index: int | None,
    episode_count: int | None,
    refined,
    semantic_candidates,
    visual_candidates,
    lattice,
    boundary_info: dict,
    selection_info: dict,
    filter_diagnostics: dict,
    continuity_part: dict | None = None,
) -> bool:
    trajectory = build_draft_trajectory(task, refined, min_confidence=args.min_confidence).to_dict()
    if episode and episode.get("task"):
        trajectory["task"] = str(episode["task"])
    elif video_plan.get("task_title"):
        trajectory["task"] = str(video_plan["task_title"])
    trajectory["video_plan"] = video_plan
    trajectory["api_boundary"] = boundary_info
    trajectory["selection"] = dict(selection_info)
    trajectory["episodes"] = video_plan.get("episode_segments") or []
    if episode is not None:
        trajectory["episode"] = {"index": episode_index, "count": episode_count, **episode}
    if continuity_part is not None:
        trajectory["parent_task"] = trajectory.get("task")
        trajectory["task"] = _local_task_title(trajectory.get("steps") or [], fallback=trajectory.get("task"))
        trajectory["continuity_part"] = continuity_part
    trajectory["task_parts"] = _task_parts(
        trajectory.get("task"),
        trajectory.get("steps") or [],
        max_steps=getattr(args, "max_selected_steps", 20),
    )
    trajectory["selection"]["task_part_count"] = len(trajectory["task_parts"])
    trajectory["candidate_filter"] = dict(filter_diagnostics)
    if args.extract_screenshots:
        _attach_step_screenshots(
            trajectory,
            task.video_path,
            out_path,
            frame_width=args.screenshot_width,
            pre_offset=args.observation_offset,
            post_offset=args.post_observation_offset,
        )
    if getattr(args, "api_ground_steps", True):
        trajectory = backend.refine_trajectory(task, trajectory)
    trajectory["_api_whole_verify"] = bool(getattr(args, "api_verify_trajectory", False))
    trajectory["verification"] = backend.verify_trajectory(task, trajectory)
    trajectory.pop("_api_whole_verify", None)
    trajectory["quality"]["action_type_mismatches"] = _count_action_type_mismatches(trajectory.get("steps") or [])
    _enforce_quality_integrity_gate(trajectory)
    ensure_success_finish_step(trajectory)
    trajectory["candidate_stats"] = {
        "semantic_raw": filter_diagnostics.get("semantic_raw", len(semantic_candidates)),
        "semantic": len(semantic_candidates),
        "visual_raw": filter_diagnostics.get("visual_raw", len(visual_candidates)),
        "visual": len(visual_candidates),
        "lattice": len(lattice),
        "refined": len(refined),
        "selected": len(refined),
        "terminal": sum(
            1
            for step in trajectory.get("steps") or []
            if (step.get("action") or {}).get("type") == "finish"
        ),
        "continuity_parts": continuity_part.get("count") if continuity_part else 1,
    }
    trajectory["quality"].setdefault("zero_step_reason", filter_diagnostics.get("zero_step_reason") if not trajectory.get("steps") else None)
    if continuity_part is not None and not _trajectory_part_has_minimum_quality(trajectory):
        logger.info(
            "skip low-quality trajectory part steps=%s screenshots=%s exportable=%s path=%s",
            trajectory.get("quality", {}).get("num_steps"),
            trajectory.get("quality", {}).get("has_screenshots"),
            (trajectory.get("verification") or {}).get("exportable_for_training"),
            out_path,
        )
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, out_path)
    return True


def _trajectory_part_has_minimum_quality(trajectory: dict) -> bool:
    quality = trajectory.get("quality") or {}
    verification = trajectory.get("verification") or {}
    quality_score = verification.get("quality_score")
    if isinstance(quality_score, str) and quality_score.isdigit():
        quality_score = int(quality_score)
    verified_enough = verification.get("exportable_for_training") is True or (
        verification.get("trajectory_valid") is True
        and isinstance(quality_score, (int, float))
        and quality_score >= 3
    )
    return (
        (quality.get("num_steps") or 0) > 0
        and quality.get("has_screenshots") is True
        and verified_enough
    )


def _task_parts(parent_task: str | None, steps: list[dict], *, max_steps: int) -> list[dict]:
    if not steps:
        return []
    chunks = _split_steps_for_task_parts(steps, max_steps=max_steps)
    return [
        {
            "index": index,
            "count": len(chunks),
            "parent_task": parent_task,
            "task": _task_part_title(parent_task, chunk, index, len(chunks)),
            "step_indices": [step.get("index") for step in chunk],
            "start": chunk[0].get("start"),
            "end": chunk[-1].get("end"),
            "num_steps": len(chunk),
        }
        for index, chunk in enumerate(chunks, 1)
    ]


def _local_task_title(steps: list[dict], *, fallback: str | None) -> str:
    intents = [str(step.get("intent") or "").strip() for step in steps if str(step.get("intent") or "").strip()]
    if not intents:
        return str(fallback or "Complete the visible CUA subtask")[:240]
    if len(intents) == 1 or intents[0] == intents[-1]:
        return intents[0][:240]
    return f"{intents[0]}; then {intents[-1]}"[:240]


def _split_steps_for_task_parts(steps: list[dict], *, max_steps: int) -> list[list[dict]]:
    limit = max_steps if max_steps and max_steps > 0 else len(steps)
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for step in steps:
        if current and len(current) >= limit:
            chunks.append(current)
            current = []
        current.append(step)
        if _is_semantic_part_boundary(step) and len(current) >= 3:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks or [steps]


def _is_semantic_part_boundary(step: dict) -> bool:
    action_type = (step.get("action") or {}).get("type")
    intent = str(step.get("intent") or "").lower()
    if action_type == "computer.wait":
        return True
    return any(marker in intent for marker in ("save", "submit", "confirm", "finish", "complete", "open", "navigate", "search"))


def _task_part_title(parent_task: str | None, steps: list[dict], part_index: int, part_count: int) -> str:
    intents = [str(step.get("intent") or "").strip() for step in steps if str(step.get("intent") or "").strip()]
    if not intents:
        subgoal = "continue the visible workflow"
    elif intents[0] == intents[-1]:
        subgoal = intents[0]
    else:
        subgoal = f"{intents[0]}; then {intents[-1]}"
    prefix = str(parent_task or "Complete the visible workflow").strip() or "Complete the visible workflow"
    return f"{prefix} - subtask {part_index} of {part_count}: {subgoal}"[:240]


def _extract_even_frames(video_path: Path, out_dir: Path, num_frames: int, frame_width: int) -> list[Path]:
    if num_frames <= 0:
        return []
    from ..scoring.frames import extract_frames, probe_duration

    duration = probe_duration(video_path)
    if duration <= 0:
        return []
    return extract_frames(video_path, out_dir, num_frames, frame_width, duration=duration)


def _extract_candidate_frames(
    video_path: Path,
    out_dir: Path,
    candidates,
    max_frames: int,
    frame_width: int,
) -> list[Path]:
    if max_frames <= 0 or not candidates:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    if len(candidates) <= max_frames:
        selected = candidates
    else:
        step = max(1, len(candidates) // max_frames)
        selected = candidates[::step][:max_frames]
    for index, candidate in enumerate(selected, 1):
        timestamp = max(0.0, (float(candidate.start) + float(candidate.end)) / 2)
        dest = out_dir / f"candidate_{index:04d}.jpg"
        try:
            extract_screenshot(video_path, dest, timestamp, frame_width=frame_width)
        except RuntimeError:
            continue
        frames.append(dest)
    return frames


def _filter_candidates_by_plan(
    candidates,
    video_plan: dict,
    *,
    ignore_overlap_ratio: float = 0.8,
    high_value_hard_filter: bool = False,
):
    ignore_ranges = video_plan.get("ignore_ranges") if isinstance(video_plan, dict) else None
    high_value_ranges = video_plan.get("high_value_ranges") if isinstance(video_plan, dict) else None
    if not ignore_ranges and not (high_value_hard_filter and high_value_ranges):
        return candidates

    def overlap_ratio(candidate, ranges) -> float:
        candidate_start = float(candidate.start)
        candidate_end = float(candidate.end)
        duration = max(0.001, candidate_end - candidate_start)
        total = 0.0
        for item in ranges or []:
            try:
                start = float(item.get("start"))
                end = float(item.get("end"))
            except (AttributeError, TypeError, ValueError):
                continue
            total += max(0.0, min(candidate_end, end) - max(candidate_start, start))
        return min(1.0, total / duration)

    def overlaps(candidate, ranges) -> bool:
        return overlap_ratio(candidate, ranges) > 0.0

    filtered = []
    for candidate in candidates:
        if ignore_ranges and overlap_ratio(candidate, ignore_ranges) >= ignore_overlap_ratio:
            continue
        if high_value_hard_filter and high_value_ranges and not overlaps(candidate, high_value_ranges):
            continue
        filtered.append(candidate)
    return filtered


def _filter_candidates_to_primary_episode(candidates, video_plan: dict, *, enabled: bool, tolerance: float):
    if not enabled or not candidates:
        return candidates
    episode = _primary_episode(video_plan)
    if not episode:
        return candidates
    try:
        start = float(episode.get("start")) - max(0.0, tolerance)
        end = float(episode.get("end")) + max(0.0, tolerance)
    except (AttributeError, TypeError, ValueError):
        return candidates
    filtered = [candidate for candidate in candidates if float(candidate.end) >= start and float(candidate.start) <= end]
    return filtered or candidates


def _planned_episodes(video_plan: dict, *, enabled: bool) -> list[dict]:
    if not enabled or not isinstance(video_plan, dict):
        return []
    episodes = [item for item in video_plan.get("episode_segments") or [] if isinstance(item, dict)]
    valid = []
    for episode in episodes:
        try:
            start = float(episode.get("start"))
            end = float(episode.get("end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        valid.append({**episode, "start": start, "end": end})
    max_episodes = int(video_plan.get("max_episodes") or 0) if str(video_plan.get("max_episodes") or "").isdigit() else 0
    valid = sorted(valid, key=lambda item: (float(item["start"]), float(item["end"])))
    return valid[:max_episodes] if max_episodes > 0 else valid


def _episode_video_plan(video_plan: dict, episode: dict | None) -> dict:
    if episode is None:
        return video_plan
    plan = dict(video_plan) if isinstance(video_plan, dict) else {"enabled": True}
    plan["primary_episode"] = episode
    plan["episode_segments"] = [episode]
    plan["high_value_ranges"] = [
        {
            "start": episode.get("start"),
            "end": episode.get("end"),
            "reason": episode.get("summary") or episode.get("task") or "episode CUA operation",
        }
    ]
    return plan


def _primary_episode(video_plan: dict) -> dict | None:
    if not isinstance(video_plan, dict):
        return None
    primary = video_plan.get("primary_episode")
    if isinstance(primary, dict):
        return primary
    episodes = [item for item in video_plan.get("episode_segments") or [] if isinstance(item, dict)]
    if not episodes:
        return None
    return max(episodes, key=_episode_score)


def _episode_score(episode: dict) -> tuple[float, float]:
    try:
        duration = max(0.0, float(episode.get("end")) - float(episode.get("start")))
    except (TypeError, ValueError):
        duration = 0.0
    try:
        confidence = float(episode.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence, duration


def _candidate_filter_diagnostics(
    semantic_raw,
    semantic_filtered,
    visual_raw,
    visual_filtered,
    video_plan: dict,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "semantic_raw": len(semantic_raw),
        "semantic_filtered": len(semantic_filtered),
        "visual_raw": len(visual_raw),
        "visual_filtered": len(visual_filtered),
        "ignore_ranges": len(video_plan.get("ignore_ranges") or []) if isinstance(video_plan, dict) else 0,
        "high_value_ranges": len(video_plan.get("high_value_ranges") or []) if isinstance(video_plan, dict) else 0,
    }
    primary = _primary_episode(video_plan)
    if primary:
        diagnostics["primary_episode"] = {
            "start": primary.get("start"),
            "end": primary.get("end"),
            "task": primary.get("task"),
        }
    raw_total = len(semantic_raw) + len(visual_raw)
    filtered_total = len(semantic_filtered) + len(visual_filtered)
    if raw_total == 0:
        diagnostics["reason"] = "candidate_mining_empty"
    elif filtered_total == 0:
        diagnostics["reason"] = "filtered_to_empty"
    elif filtered_total < raw_total:
        diagnostics["reason"] = "partially_filtered"
    else:
        diagnostics["reason"] = "not_filtered"
    return diagnostics


def _fallback_candidates_after_empty_filter(candidates, video_plan: dict, top_k: int):
    if top_k <= 0 or not candidates:
        return []
    if _plan_declares_no_cua(video_plan):
        return []
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)[:top_k]


def _plan_declares_no_cua(video_plan: dict) -> bool:
    if not isinstance(video_plan, dict) or not video_plan.get("enabled"):
        return False
    if video_plan.get("high_value_ranges"):
        return False
    confidence = video_plan.get("confidence")
    try:
        high_confidence = float(confidence) >= 0.85
    except (TypeError, ValueError):
        high_confidence = False
    text = " ".join(
        str(item.get("reason") or "")
        for item in video_plan.get("ignore_ranges") or []
        if isinstance(item, dict)
    ).lower()
    return high_confidence and any(
        marker in text
        for marker in (
            "no real cua",
            "no computer operation",
            "no software operation",
            "marketing",
            "slide",
            "testimonial",
            "talking-head",
        )
    )


def _zero_step_reason(video_plan: dict, semantic_raw, visual_raw) -> str:
    if _plan_declares_no_cua(video_plan):
        return "no_cua_operation"
    if not semantic_raw and not visual_raw:
        return "candidate_mining_empty"
    return "filtered_to_empty"


def _limit_candidates(candidates, max_candidates: int):
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates
    return sorted(candidates, key=lambda c: (c.confidence, c.semantic_score, c.visual_score), reverse=True)[:max_candidates]


def _limit_selected_candidates(candidates, max_steps: int):
    if max_steps <= 0 or len(candidates) <= max_steps:
        return candidates
    return sorted(candidates, key=lambda c: (c.start, c.end))[:max_steps]


def _split_selected_candidates(candidates, max_steps: int):
    ordered = sorted(candidates, key=lambda c: (c.start, c.end))
    if max_steps <= 0 or len(ordered) <= max_steps:
        return [ordered]
    return [ordered[index : index + max_steps] for index in range(0, len(ordered), max_steps)]


def _split_candidates_for_continuity(candidates, *, enabled: bool, max_gap_seconds: float, min_steps: int):
    ordered = sorted(candidates, key=lambda c: (c.start, c.end))
    if not enabled or len(ordered) <= 1:
        return [ordered]

    raw_chunks = []
    current = [ordered[0]]
    min_len = max(1, int(min_steps or 1))
    for candidate in ordered[1:]:
        previous = current[-1]
        gap = float(candidate.start) - float(previous.end)
        if (max_gap_seconds > 0 and gap > max_gap_seconds) or (
            len(current) >= min_len and _candidate_context_shift(previous, candidate)
        ):
            raw_chunks.append(current)
            current = []
        current.append(candidate)
    if current:
        raw_chunks.append(current)

    chunks = [chunk for chunk in raw_chunks if len(chunk) >= min_len]
    if chunks:
        return chunks
    return [max(raw_chunks, key=len)] if raw_chunks else [ordered]


def _extend_short_trajectory_selection(
    *,
    selected,
    candidate_pool,
    min_steps: int,
    max_gap_seconds: float,
    min_confidence: float,
):
    target = max(1, int(min_steps or 1))
    ordered_selected = sorted(selected, key=lambda c: (c.start, c.end))
    if len(ordered_selected) >= target:
        return ordered_selected, {"enabled": True, "added_steps": 0, "reason": "already_long_enough", "target_steps": target}
    if not ordered_selected:
        return ordered_selected, {"enabled": True, "added_steps": 0, "reason": "no_selected_seed", "target_steps": target}

    chain = list(ordered_selected)
    selected_keys = {_candidate_identity(candidate) for candidate in chain}
    pool = [
        candidate
        for candidate in sorted(candidate_pool, key=lambda c: (c.start, c.end))
        if _candidate_identity(candidate) not in selected_keys
        and candidate.action_type
        and float(candidate.confidence or 0.0) >= min_confidence
    ]
    added = 0
    while len(chain) < target:
        best = _best_extension_candidate(chain, pool, max_gap_seconds=max_gap_seconds)
        if best is None:
            break
        pool.remove(best)
        chain.append(best)
        chain.sort(key=lambda c: (c.start, c.end))
        added += 1
    reason = "target_reached" if len(chain) >= target else "no_continuous_candidates"
    return chain, {
        "enabled": True,
        "target_steps": target,
        "added_steps": added,
        "final_steps": len(chain),
        "max_gap_seconds": max_gap_seconds,
        "min_confidence": min_confidence,
        "reason": reason,
    }


def _best_extension_candidate(chain, pool, *, max_gap_seconds: float):
    best = None
    best_score = None
    for candidate in pool:
        trial = sorted([*chain, candidate], key=lambda c: (c.start, c.end))
        if not _candidate_chain_is_continuous(trial, max_gap_seconds=max_gap_seconds):
            continue
        score = _extension_candidate_score(chain, candidate)
        if best_score is None or score > best_score:
            best = candidate
            best_score = score
    return best


def _candidate_chain_is_continuous(candidates, *, max_gap_seconds: float) -> bool:
    if max_gap_seconds <= 0:
        return True
    for left, right in zip(candidates, candidates[1:]):
        if float(right.start) - float(left.end) > max_gap_seconds:
            return False
    return True


def _extension_candidate_score(chain, candidate) -> tuple[float, float, float]:
    start = min(float(item.start) for item in chain)
    end = max(float(item.end) for item in chain)
    inside_span = start <= float(candidate.start) <= end
    if inside_span:
        distance = 0.0
    else:
        distance = min(abs(float(candidate.end) - start), abs(float(candidate.start) - end))
    return (1.0 if inside_span else 0.0, -distance, float(candidate.confidence or 0.0))


def _candidate_identity(candidate) -> tuple:
    return (
        round(float(candidate.start), 3),
        round(float(candidate.end), 3),
        candidate.action_type,
        candidate.target,
        candidate.text,
    )


def _continuity_split_reason(chunks, *, max_gap_seconds: float) -> str:
    if len(chunks) <= 1:
        return "none"
    for left, right in zip(chunks, chunks[1:]):
        if not left or not right:
            continue
        gap = float(right[0].start) - float(left[-1].end)
        if max_gap_seconds > 0 and gap > max_gap_seconds:
            return "time_gap"
    return "semantic_context_shift"


_CONTEXT_LABELS = {
    "fiverr": ("fiverr",),
    "upwork": ("upwork",),
    "chatgpt": ("chatgpt", "chat gpt", "openai chat"),
    "google": ("google", "chrome search", "address bar", "search results"),
    "gmail": ("gmail",),
    "basecamp": ("basecamp",),
    "suitecrm": ("suitecrm", "suite crm"),
    "github": ("github",),
    "youtube": ("youtube",),
}


def _candidate_context_shift(previous, current) -> bool:
    previous_labels = _candidate_context_labels(previous)
    current_labels = _candidate_context_labels(current)
    if not previous_labels or not current_labels:
        return False
    return previous_labels.isdisjoint(current_labels)


def _candidate_context_labels(candidate) -> set[str]:
    text = " ".join(
        str(value or "")
        for value in (
            getattr(candidate, "target", None),
            getattr(candidate, "text", None),
            getattr(candidate, "action_type", None),
        )
    ).lower()
    labels = set()
    for label, markers in _CONTEXT_LABELS.items():
        if any(marker in text for marker in markers):
            labels.add(label)
    return labels


def _count_action_type_mismatches(steps: list[dict]) -> int:
    mismatches = 0
    for step in steps:
        expected = ((step.get("evidence") or {}).get("api_boundary") or {}).get("action_type")
        actual = (step.get("action") or {}).get("type")
        if expected and actual and expected != actual:
            mismatches += 1
    return mismatches


def _attach_step_screenshots(
    trajectory: dict,
    video_path: Path,
    trajectory_path: Path,
    *,
    frame_width: int,
    pre_offset: float,
    post_offset: float,
) -> None:
    steps = trajectory.get("steps") or []
    if not steps:
        trajectory.setdefault("quality", {})["has_screenshots"] = False
        trajectory.setdefault("quality", {})["num_screenshots"] = 0
        return

    frame_dir = frame_dir_for_trajectory(trajectory_path)
    extracted_paths: list[Path] = []

    first_step = steps[0]
    initial_ts = max(0.0, float(first_step["start"]) + pre_offset)
    previous_path = step_frame_path(frame_dir, 0, "state")
    extract_screenshot(video_path, previous_path, initial_ts, frame_width=frame_width)
    previous_observation = {
        "screenshot": str(previous_path),
        "timestamp": round(initial_ts, 3),
    }
    extracted_paths.append(previous_path)

    for step in steps:
        index = int(step["index"])
        post_ts = max(0.0, float(step["end"]) + post_offset)
        post_path = step_frame_path(frame_dir, index, "state")
        extract_screenshot(video_path, post_path, post_ts, frame_width=frame_width)
        step["observation"] = dict(previous_observation)
        step["post_observation"] = {
            "screenshot": str(post_path),
            "timestamp": round(post_ts, 3),
        }
        previous_observation = dict(step["post_observation"])
        extracted_paths.append(post_path)
    trajectory.setdefault("quality", {})["has_screenshots"] = bool(extracted_paths)
    trajectory.setdefault("quality", {})["num_screenshots"] = len(extracted_paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct draft CUA action trajectories from filtered video metadata."
    )
    parser.add_argument("--config", default=None, help="YAML file with action construction defaults; CLI flags override config values")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="filtered video metadata JSONL")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="downloaded mp4 root")
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR, help="root of .segments.json transcripts")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="trajectory output root")
    parser.add_argument("--limit", type=int, default=None, help="construct at most N records")
    parser.add_argument("--concurrency", type=int, default=1, help="number of worker threads for constructing records")
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing trajectory files")
    parser.add_argument("--split-episodes", action=argparse.BooleanOptionalAction, default=False, help="write one trajectory per API-planned coherent episode")
    parser.add_argument("--split-long-trajectories", action=argparse.BooleanOptionalAction, default=True, help="split selected actions into part files instead of truncating when max-selected-steps is exceeded")
    parser.add_argument("--trajectory-split-gap-seconds", type=float, default=120.0, help="split a selected trajectory when adjacent actions are separated by more than this many video seconds; <=0 disables gap splitting")
    parser.add_argument("--trajectory-split-min-steps", type=int, default=20, help="drop gap-split trajectory parts with fewer than this many steps unless every part is shorter")
    parser.add_argument("--require-transcript", action=argparse.BooleanOptionalAction, default=True, help="skip records without transcript files")
    parser.add_argument("--backend", choices=["passthrough", "none", "openai_chat", "gpt55", "gpt5.5"], default="passthrough", help="action labeling backend")
    parser.add_argument("--api-model", default=None, help="model for API backend; default ACTION_CONSTRUCTION_MODEL or GPT-5.5")
    parser.add_argument("--api-base-url", default=None, help="OpenAI-compatible API base URL; default from environment")
    parser.add_argument("--api-base-urls", nargs="+", default=None, help="OpenAI-compatible API base URLs for round-robin requests; overridden by --api-base-url")
    parser.add_argument("--api-endpoint-pool", default=None, help="YAML endpoint pool dynamically reloaded before requests")
    parser.add_argument("--api-key", default=None, help="API key for API backend; default ACTION_CONSTRUCTION_API_KEY/OPENAI_API_KEY/OPENROUTER_API_KEY")
    parser.add_argument("--api-key-env", default=None, help="environment variable containing the API key")
    parser.add_argument("--api-wire-api", choices=["chat_completions", "responses"], default="chat_completions")
    parser.add_argument("--api-store", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--api-trust-env", action=argparse.BooleanOptionalAction, default=True, help="allow HTTP clients to inherit proxy environment variables")
    parser.add_argument("--api-reasoning-effort", default="none", help="reasoning_effort for OpenAI-compatible chat backend")
    parser.add_argument("--api-plan-reasoning-effort", default=None, help="reasoning_effort for video planning; defaults to --api-reasoning-effort")
    parser.add_argument("--api-boundary-reasoning-effort", default=None, help="reasoning_effort for boundary refinement; defaults to --api-reasoning-effort")
    parser.add_argument("--api-selection-reasoning-effort", default=None, help="reasoning_effort for trajectory selection; defaults to --api-reasoning-effort")
    parser.add_argument("--api-step-reasoning-effort", default=None, help="reasoning_effort for per-step grounding; defaults to --api-reasoning-effort")
    parser.add_argument("--api-verify-reasoning-effort", default=None, help="reasoning_effort for whole-trajectory verification; defaults to --api-reasoning-effort")
    parser.add_argument("--api-max-tokens", type=int, default=2048, help="max completion tokens per API step refinement")
    parser.add_argument("--api-plan-max-tokens", type=int, default=4096, help="max completion tokens for video plan/boundary/selection/verification API calls")
    parser.add_argument("--api-max-retries", type=int, default=3, help="max retries per API request")
    parser.add_argument("--api-request-delay", type=float, default=0.0, help="sleep seconds between API step calls")
    parser.add_argument("--api-step-batch-size", type=int, default=1, help="ground this many contiguous steps per API call; 1 keeps legacy single-step grounding")
    parser.add_argument("--api-step-batch-fallback", action=argparse.BooleanOptionalAction, default=True, help="retry missing or suspicious batch-grounded steps with single-step grounding")
    parser.add_argument("--api-video-plan", action="store_true", help="use API to identify task, episodes, ignore ranges, and high-value CUA ranges")
    parser.add_argument("--api-overview-frames", type=int, default=12, help="overview frames for API video planning")
    parser.add_argument("--api-boundary-refine", action="store_true", help="use API to add/drop/split/merge action boundary candidates")
    parser.add_argument("--api-boundary-frames", type=int, default=24, help="candidate-centered frames for API boundary refinement")
    parser.add_argument("--api-trajectory-select", action="store_true", help="use API to select final action path from refined candidates")
    parser.add_argument("--api-ground-steps", action=argparse.BooleanOptionalAction, default=None, help="use API to ground per-step screenshots; defaults on for API-heavy modes and off for passthrough")
    parser.add_argument("--api-verify-trajectory", action="store_true", help="use API for whole-trajectory verification")
    parser.add_argument("--api-frame-width", type=int, default=512, help="width in px for temporary API planning/boundary frames")
    parser.add_argument("--enable-visual", action="store_true", help="add ffmpeg scene-change visual candidates")
    parser.add_argument("--scene-threshold", type=float, default=0.18, help="ffmpeg scene threshold for visual candidates")
    parser.add_argument("--max-visual-candidates", type=int, default=200, help="cap visual candidates per video")
    parser.add_argument("--semantic-pre-roll", type=float, default=1.0, help="seconds before semantic candidate window")
    parser.add_argument("--semantic-post-roll", type=float, default=1.5, help="seconds after semantic candidate window")
    parser.add_argument("--merge-tolerance", type=float, default=1.5, help="seconds for merging nearby candidates")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="minimum candidate confidence to keep in draft trajectory")
    parser.add_argument("--plan-ignore-overlap-ratio", type=float, default=0.8, help="drop candidates only when this fraction overlaps API ignore ranges")
    parser.add_argument("--plan-high-value-hard-filter", action="store_true", help="require candidates to overlap high-value API ranges; off by default because high-value ranges are advisory")
    parser.add_argument("--plan-primary-episode-filter", action=argparse.BooleanOptionalAction, default=False, help="keep candidates inside the API-selected primary long-horizon episode")
    parser.add_argument("--plan-primary-episode-tolerance", type=float, default=10.0, help="seconds of tolerance around the selected primary episode")
    parser.add_argument("--plan-empty-fallback-top-k", type=int, default=12, help="keep top-K raw candidates if plan filtering empties candidates without a no-CUA verdict")
    parser.add_argument("--max-boundary-candidates", type=int, default=240, help="cap candidates sent to API boundary refinement")
    parser.add_argument("--max-selected-steps", type=int, default=200, help="cap selected steps before screenshot grounding")
    parser.add_argument("--extend-short-trajectories", action=argparse.BooleanOptionalAction, default=False, help="add adjacent refined candidates back when API selection is high-quality but shorter than the long-horizon target")
    parser.add_argument("--extension-min-steps", type=int, default=40, help="target step count for short trajectory extension")
    parser.add_argument("--extension-max-gap-seconds", type=float, default=120.0, help="maximum adjacent time gap allowed while extending short selected trajectories")
    parser.add_argument("--extension-min-confidence", type=float, default=0.28, help="minimum candidate confidence for extension candidates")
    parser.add_argument("--long-min-steps", type=int, default=20, help="minimum step count for the long training tier")
    parser.add_argument("--long-target-steps", type=int, default=40, help="preferred target step count when extending selected trajectories")
    parser.add_argument("--include-long-candidates-in-train", action="store_true", help="include valid but non-exportable long candidates in trajectories.train.jsonl")
    parser.add_argument("--write-audit-jsonl", action="store_true", help="write verbose audit and rejected trajectory JSONL files during summarization")
    parser.add_argument("--extract-screenshots", action=argparse.BooleanOptionalAction, default=True, help="extract a state screenshot chain for trajectory steps")
    parser.add_argument("--screenshot-width", type=int, default=1280, help="width in px for extracted step screenshots")
    parser.add_argument("--observation-offset", type=float, default=0.0, help="seconds added to the first step start for the initial state screenshot")
    parser.add_argument("--post-observation-offset", type=float, default=0.0, help="seconds added to each step end for after-action state screenshots")
    parser.add_argument("--summarize", action="store_true", help="only rebuild trajectories.jsonl from existing outputs")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        args = _merge_config_args(parser, args, argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_started_at = time.monotonic()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize_started_at = time.monotonic()
        _summarize(
            output_dir,
            write_audit_jsonl=args.write_audit_jsonl,
            include_long_candidates_in_train=args.include_long_candidates_in_train,
            long_min_steps=args.long_min_steps,
        )
        logger.info(
            "summarize_analysis elapsed=%s output=%s",
            format_duration(time.monotonic() - summarize_started_at),
            output_dir,
        )
        return

    load_started_at = time.monotonic()
    records = load_jsonl(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    load_elapsed = time.monotonic() - load_started_at
    backend_started_at = time.monotonic()
    backend = make_backend(
        args.backend,
        output_dir=output_dir,
        model=args.api_model,
        base_url=args.api_base_url,
        base_urls=args.api_base_urls,
        endpoint_pool=args.api_endpoint_pool,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        wire_api=args.api_wire_api,
        store=args.api_store,
        trust_env=args.api_trust_env,
        reasoning_effort=args.api_reasoning_effort,
        plan_reasoning_effort=args.api_plan_reasoning_effort,
        boundary_reasoning_effort=args.api_boundary_reasoning_effort,
        selection_reasoning_effort=args.api_selection_reasoning_effort,
        step_reasoning_effort=args.api_step_reasoning_effort,
        verify_reasoning_effort=args.api_verify_reasoning_effort,
        max_tokens=args.api_max_tokens,
        plan_max_tokens=args.api_plan_max_tokens,
        max_retries=args.api_max_retries,
        request_delay=args.api_request_delay,
        step_batch_size=args.api_step_batch_size,
        step_batch_fallback=args.api_step_batch_fallback,
    )
    backend_elapsed = time.monotonic() - backend_started_at
    if args.api_ground_steps is None:
        args.api_ground_steps = args.backend not in {"passthrough", "none"}
    seen_video_ids = _prepare_video_ids_for_run(records, args)
    counts = {"ok": 0, "skipped": 0, "missing_video": 0, "missing_transcript": 0, "failed": 0}
    logger.info(
        "records=%d output=%s backend=%s visual=%s concurrency=%d",
        len(records),
        output_dir,
        args.backend,
        args.enable_visual,
        max(1, args.concurrency),
    )
    construct_started_at = time.monotonic()
    try:
        analysis = _construct_records(records, args, backend, counts, construct_started_at, seen_video_ids=seen_video_ids)
    except KeyboardInterrupt:
        _summarize(
            output_dir,
            write_audit_jsonl=args.write_audit_jsonl,
            include_long_candidates_in_train=args.include_long_candidates_in_train,
            long_min_steps=args.long_min_steps,
        )
        logger.warning("construction interrupted; wrote summaries from committed trajectories only")
        raise
    construct_elapsed = time.monotonic() - construct_started_at
    summarize_started_at = time.monotonic()
    _summarize(
        output_dir,
        write_audit_jsonl=args.write_audit_jsonl,
        include_long_candidates_in_train=args.include_long_candidates_in_train,
        long_min_steps=args.long_min_steps,
    )
    summarize_elapsed = time.monotonic() - summarize_started_at
    total_elapsed = time.monotonic() - run_started_at
    analysis["phase_seconds"] = {
        "load": round(load_elapsed, 3),
        "backend_init": round(backend_elapsed, 3),
        "construct": round(construct_elapsed, 3),
        "summarize": round(summarize_elapsed, 3),
        "total": round(total_elapsed, 3),
    }
    analysis["phase_durations"] = {
        key: format_duration(value)
        for key, value in {
            "load": load_elapsed,
            "backend_init": backend_elapsed,
            "construct": construct_elapsed,
            "summarize": summarize_elapsed,
            "total": total_elapsed,
        }.items()
    }
    _log_run_analysis(analysis)
    logger.info("done elapsed=%s %s", format_duration(total_elapsed), counts)


def _construct_records(
    records: list[dict],
    args: argparse.Namespace,
    backend,
    counts: dict[str, int],
    started_at: float,
    seen_video_ids: set[str] | None = None,
) -> dict[str, Any]:
    workers = max(1, int(getattr(args, "concurrency", 1) or 1))
    total = len(records)
    item_durations: list[float] = []
    if workers == 1:
        in_flight_peak = 0
        for index, record in enumerate(records, 1):
            if _claim_video_id_or_skip(record, seen_video_ids):
                _log_skipped_record(index, total, index, record, counts, started_at, item_durations)
                continue
            in_flight_peak = 1
            _construct_record_with_progress(index, total, record, args, backend, counts, started_at, item_durations)
        return _build_run_analysis(total, workers, counts, started_at, time.monotonic(), item_durations, in_flight_peak)

    max_in_flight = workers
    next_index = 0
    completed = 0
    in_flight_peak = 0
    futures = {}

    def submit(pool: ThreadPoolExecutor) -> None:
        nonlocal next_index, completed, in_flight_peak
        while next_index < total and len(futures) < max_in_flight:
            index = next_index + 1
            record = records[next_index]
            if _claim_video_id_or_skip(record, seen_video_ids):
                completed += 1
                _log_skipped_record(completed, total, index, record, counts, started_at, item_durations)
                next_index += 1
                continue
            _log_record_start(index, total, record)
            futures[pool.submit(_construct_record, record, args, backend)] = (index, record, time.monotonic())
            in_flight_peak = max(in_flight_peak, len(futures))
            next_index += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        submit(pool)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                index, record, item_started_at = futures.pop(future)
                video_id, status = future.result()
                completed += 1
                _log_record_result(completed, total, index, video_id, status, counts, started_at, item_started_at, item_durations)
            submit(pool)
    return _build_run_analysis(total, workers, counts, started_at, time.monotonic(), item_durations, in_flight_peak)


def _construct_record_with_progress(
    index: int,
    total: int,
    record: dict,
    args: argparse.Namespace,
    backend,
    counts: dict[str, int],
    started_at: float,
    item_durations: list[float] | None = None,
) -> None:
    item_started_at = time.monotonic()
    _log_record_start(index, total, record)
    video_id, status = _construct_record(record, args, backend)
    _log_record_result(index, total, index, video_id, status, counts, started_at, item_started_at, item_durations)


def _log_skipped_record(
    completed: int,
    total: int,
    record_index: int,
    record: dict,
    counts: dict[str, int],
    started_at: float,
    item_durations: list[float] | None = None,
) -> None:
    item_started_at = time.monotonic()
    video_id = str(record.get("video_id") or "")
    _log_record_result(completed, total, record_index, video_id, "skipped", counts, started_at, item_started_at, item_durations)


def _construct_record(record: dict, args: argparse.Namespace, backend) -> tuple[str, str]:
    try:
        return _construct_one(record, args, backend)
    except BaseException as exc:
        video_id = str(record.get("video_id") or "")
        try:
            task = resolve_video_task(record, Path(args.video_dir), Path(args.transcript_dir))
            out_path = _output_path(task.relative_path, Path(args.output_dir))
            _completion_manifest_path(Path(args.output_dir), video_id).unlink(missing_ok=True)
            _remove_base_trajectory_artifacts_for_video(out_path, video_id)
        except Exception as cleanup_exc:
            logger.warning("failed to clean interrupted trajectory %s: %s", video_id, cleanup_exc)
        if not isinstance(exc, Exception):
            raise
        logger.warning("failed %s: %s", video_id, exc)
        return video_id, "failed"


def _log_record_start(index: int, total: int, record: dict) -> None:
    logger.info(
        "[%d/%d] start %s app=%s category=%s/%s",
        index,
        total,
        str(record.get("video_id") or ""),
        record.get("application") or "",
        record.get("category") or "",
        record.get("subcategory") or "",
    )


def _log_record_result(
    completed: int,
    total: int,
    record_index: int,
    video_id: str,
    status: str,
    counts: dict[str, int],
    started_at: float,
    item_started_at: float,
    item_durations: list[float] | None = None,
) -> None:
    counts[status] = counts.get(status, 0) + 1
    elapsed = time.monotonic() - started_at
    item_elapsed = time.monotonic() - item_started_at
    if item_durations is not None:
        item_durations.append(item_elapsed)
    throughput = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / throughput if throughput > 0 else 0.0
    logger.info(
        "[%d/%d record=%d throughput=%.2f/min task=%s elapsed=%s eta=%s] %s %s counts=%s",
        completed,
        total,
        record_index,
        throughput * 60,
        format_duration(item_elapsed),
        format_duration(elapsed),
        format_duration(eta),
        status,
        video_id,
        counts,
    )


def _build_run_analysis(
    total: int,
    workers: int,
    counts: dict[str, int],
    started_at: float,
    ended_at: float,
    item_durations: list[float],
    in_flight_peak: int = 0,
) -> dict[str, Any]:
    elapsed = max(0.0, ended_at - started_at)
    completed = sum(counts.values())
    ok = counts.get("ok", 0)
    failed = counts.get("failed", 0)
    missing = counts.get("missing_video", 0) + counts.get("missing_transcript", 0)
    average = sum(item_durations) / len(item_durations) if item_durations else 0.0
    fastest = min(item_durations) if item_durations else 0.0
    slowest = max(item_durations) if item_durations else 0.0
    throughput = completed / elapsed if elapsed > 0 else 0.0
    return {
        "records_total": total,
        "records_completed": completed,
        "workers": workers,
        "in_flight_peak": in_flight_peak,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed": format_duration(elapsed),
        "throughput_per_minute": round(throughput * 60, 3),
        "success_rate": round(ok / completed, 4) if completed else 0.0,
        "failure_rate": round(failed / completed, 4) if completed else 0.0,
        "missing_rate": round(missing / completed, 4) if completed else 0.0,
        "average_record_seconds": round(average, 3),
        "average_record": format_duration(average),
        "fastest_record_seconds": round(fastest, 3),
        "fastest_record": format_duration(fastest),
        "slowest_record_seconds": round(slowest, 3),
        "slowest_record": format_duration(slowest),
        "counts": dict(counts),
    }


def _log_run_analysis(analysis: dict[str, Any]) -> None:
    logger.info(
        "run_analysis records=%d completed=%d workers=%d in_flight_peak=%d elapsed=%s throughput=%.2f/min "
        "success_rate=%.1f%% failure_rate=%.1f%% missing_rate=%.1f%% avg_record=%s "
        "fastest_record=%s slowest_record=%s phases=%s counts=%s",
        analysis["records_total"],
        analysis["records_completed"],
        analysis["workers"],
        analysis.get("in_flight_peak", 0),
        analysis["elapsed"],
        analysis["throughput_per_minute"],
        analysis["success_rate"] * 100,
        analysis["failure_rate"] * 100,
        analysis["missing_rate"] * 100,
        analysis["average_record"],
        analysis["fastest_record"],
        analysis["slowest_record"],
        analysis.get("phase_durations", {}),
        analysis["counts"],
    )


def _merge_config_args(parser: argparse.ArgumentParser, args: argparse.Namespace, argv: list[str] | None) -> argparse.Namespace:
    config_path = Path(args.config)
    raw = load_yaml_mapping(config_path)
    defaults = _flatten_config_defaults(raw)
    explicit = _explicit_cli_dests(parser, argv)
    pool_value = defaults.get("api_endpoint_pool")
    if pool_value and (defaults.get("api_base_url") or defaults.get("api_base_urls")):
        raise ValueError("api.endpoint_pool cannot be combined with api.base_url or api.base_urls")
    if pool_value:
        pool_path = resolve_endpoint_pool_path(pool_value, config_path.parent)
        pool_config = load_endpoint_pool_config(pool_path)
        defaults["api_endpoint_pool"] = str(pool_path)
        if pool_config.model:
            defaults.setdefault("api_model", pool_config.model)
        defaults.setdefault("api_base_urls", list(pool_config.base_urls))
        if pool_config.api_key_env:
            defaults.setdefault("api_key_env", pool_config.api_key_env)
        defaults.setdefault("api_wire_api", pool_config.wire_api)
        if pool_config.reasoning_effort:
            defaults.setdefault("api_reasoning_effort", pool_config.reasoning_effort)
        defaults.setdefault("api_store", pool_config.store)
        defaults.setdefault("api_trust_env", pool_config.trust_env)
    for key, value in defaults.items():
        if not hasattr(args, key):
            raise ValueError(f"unknown action construction config key: {key}")
        if key in explicit:
            continue
        setattr(args, key, value)
    if "api_base_url" in explicit or "api_base_urls" in explicit:
        args.api_endpoint_pool = None
    return args


def _flatten_config_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flat[f"{key}_{child_key}".replace("-", "_")] = child_value
        else:
            flat[key.replace("-", "_")] = value
    return flat


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str] | None) -> set[str]:
    tokens = list(argv) if argv is not None else os.sys.argv[1:]
    explicit: set[str] = set()
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    for token in tokens:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


if __name__ == "__main__":
    main()
