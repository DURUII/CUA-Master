"""Rewrite constructed GUI trajectories into mixed GUI, CLI, and tool data."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cuamaster.action_space import validate_action
from cuamaster.endpoints import (
    load_endpoint_pool_config,
    load_yaml_mapping,
    resolve_endpoint_pool_path,
)

from .backends import ActionRewriteBackend, OpenAIRewriteBackend
from .schema import RewriteSpan, RewriteTool, load_tools_config, tools_config_digest, validate_tool_call

logger = logging.getLogger("video2traj.rewrite_actions")

DEFAULT_INPUT_DIR = "data/action_trajectories"
DEFAULT_OUTPUT_DIR = "data/action_trajectories_rewritten"
_PLACEHOLDER_RE = re.compile(
    r"(?:<[^>]+>|\$\{[^}]+\}|\{\{[^}]+\}\}|\b(?:TODO|TBD|YOUR_[A-Z0-9_]+)\b|\.\.\.|\[(?:path|file|name|id|value)\])",
    re.IGNORECASE,
)


def rewrite_trajectory(
    trajectory: dict[str, Any],
    backend: ActionRewriteBackend,
    tools: list[RewriteTool],
    *,
    min_confidence: float = 0.8,
    allow_destructive_tools: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_steps = trajectory.get("steps")
    if not isinstance(source_steps, list) or not source_steps:
        raise ValueError("trajectory must contain a non-empty steps list")
    if any(not isinstance(step, dict) for step in source_steps):
        raise ValueError("every trajectory step must be an object")
    _validate_source_actions(source_steps)

    raw_spans = backend.propose_spans(trajectory, tools)
    spans, audit = _normalize_spans(
        raw_spans,
        source_steps,
        tools,
        min_confidence=min_confidence,
        allow_destructive_tools=allow_destructive_tools,
    )
    replacements: dict[int, dict[str, Any]] = {}
    covered_indices: set[int] = set()
    step_by_index = {int(step["index"]): step for step in source_steps}

    for span in spans:
        span_steps = [step_by_index[index] for index in range(span.start_index, span.end_index + 1)]
        image_paths = _span_image_paths(span_steps)
        try:
            payload = backend.rewrite_span(trajectory, span, span_steps, tools, image_paths)
            replacement = _replacement_step(
                payload,
                span,
                span_steps,
                tools,
                backend_model=getattr(backend, "model", "unknown"),
                min_confidence=min_confidence,
                allow_destructive_tools=allow_destructive_tools,
            )
        except Exception as exc:
            audit.append(
                {
                    **span.to_dict(),
                    "status": "rejected",
                    "stage": "rewrite",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue
        replacements[span.start_index] = replacement
        covered_indices.update(range(span.start_index, span.end_index + 1))
        audit.append(
            {
                **span.to_dict(),
                "status": "accepted",
                "stage": "rewrite",
                "action": replacement["action"],
                "rewrite_confidence": replacement["evidence"]["action_rewrite"]["rewrite_confidence"],
            }
        )

    rewritten_steps: list[dict[str, Any]] = []
    for step in source_steps:
        source_index = int(step["index"])
        if source_index in replacements:
            rewritten_steps.append(replacements[source_index])
        elif source_index in covered_indices:
            continue
        else:
            rewritten_steps.append(copy.deepcopy(step))
    for index, step in enumerate(rewritten_steps, start=1):
        step["index"] = index

    result = copy.deepcopy(trajectory)
    result["steps"] = rewritten_steps
    accepted = sum(item.get("status") == "accepted" for item in audit)
    result["rewrite"] = {
        "schema_version": 1,
        "status": "rewritten" if accepted else "no_change",
        "backend": backend.__class__.__name__,
        "model": getattr(backend, "model", "unknown"),
        "min_confidence": min_confidence,
        "allow_destructive_tools": allow_destructive_tools,
        "tools_digest": tools_config_digest(tools),
        "accepted_spans": accepted,
        "replaced_source_steps": len(covered_indices),
        "audit": audit,
        "semantics": (
            "counterfactual goal-equivalent execution selected for expected success or efficiency gains; "
            "source observations are retained and may show the original GUI interaction"
        ),
        "observation_policy": "preserve_source_observations_without_regeneration",
    }
    _refresh_quality(result)
    return result, audit


def _normalize_spans(
    raw_spans: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    tools: list[RewriteTool],
    *,
    min_confidence: float,
    allow_destructive_tools: bool,
) -> tuple[list[RewriteSpan], list[dict[str, Any]]]:
    step_by_index: dict[int, dict[str, Any]] = {}
    for step in steps:
        try:
            index = int(step.get("index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("trajectory step index must be an integer") from exc
        if index in step_by_index:
            raise ValueError(f"duplicate trajectory step index: {index}")
        step_by_index[index] = step

    tool_by_name = {tool.name: tool for tool in tools}
    parsed: list[RewriteSpan] = []
    audit: list[dict[str, Any]] = []
    for raw in raw_spans:
        base = {"proposal": copy.deepcopy(raw)}
        try:
            if not isinstance(raw, dict):
                raise TypeError("span proposal must be an object")
            start = int(raw.get("start_index"))
            end = int(raw.get("end_index"))
            target_type = str(raw.get("target_type") or "").strip().lower()
            confidence = float(raw.get("confidence"))
            reason = str(raw.get("reason") or "").strip()
            tool_name = str(raw.get("tool_name") or "").strip() or None
            if start > end:
                raise ValueError("start_index must be <= end_index")
            if target_type not in {"cli", "tool"}:
                raise ValueError("target_type must be cli or tool")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            if confidence < min_confidence:
                raise ValueError(f"confidence below threshold {min_confidence}")
            if not reason:
                raise ValueError("reason is required")
            improves_success_rate = _optional_bool(raw, "improves_success_rate", default=False)
            improves_efficiency = _optional_bool(
                raw,
                "improves_efficiency",
                default=end > start,
            )
            benefit_reason = str(raw.get("benefit_reason") or reason).strip()
            if not improves_success_rate and not improves_efficiency:
                raise ValueError("rewrite must improve expected success rate or efficiency")
            if not benefit_reason:
                raise ValueError("benefit_reason is required")
            indexes = list(range(start, end + 1))
            if any(index not in step_by_index for index in indexes):
                raise ValueError("span contains missing step indexes")
            if any(not _is_gui_action(step_by_index[index]) for index in indexes):
                raise ValueError("span must contain only computer.* actions")
            if all(_action_type(step_by_index[index]) == "computer.wait" for index in indexes):
                raise ValueError("passive wait spans are not meaningful CLI or tool rewrites")
            if target_type == "cli" and tool_name is not None:
                raise ValueError("CLI span must not set tool_name")
            if target_type == "tool":
                if tool_name not in tool_by_name:
                    raise ValueError(f"unregistered tool: {tool_name}")
                if tool_by_name[tool_name].destructive and not allow_destructive_tools:
                    raise ValueError(f"destructive tool is disabled: {tool_name}")
            parsed.append(
                RewriteSpan(
                    start_index=start,
                    end_index=end,
                    target_type=target_type,  # type: ignore[arg-type]
                    confidence=confidence,
                    reason=reason,
                    tool_name=tool_name,
                    improves_success_rate=improves_success_rate,
                    improves_efficiency=improves_efficiency,
                    benefit_reason=benefit_reason,
                )
            )
        except Exception as exc:
            audit.append(
                {
                    **base,
                    "status": "rejected",
                    "stage": "proposal",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    accepted: list[RewriteSpan] = []
    occupied: set[int] = set()
    for span in sorted(parsed, key=lambda item: (item.start_index, item.end_index, -item.confidence)):
        indexes = set(range(span.start_index, span.end_index + 1))
        if occupied & indexes:
            audit.append(
                {
                    **span.to_dict(),
                    "status": "rejected",
                    "stage": "proposal",
                    "error": "overlaps an earlier accepted proposal",
                }
            )
            continue
        accepted.append(span)
        occupied.update(indexes)
    return accepted, audit


def _replacement_step(
    payload: dict[str, Any],
    span: RewriteSpan,
    source_steps: list[dict[str, Any]],
    tools: list[RewriteTool],
    *,
    backend_model: str,
    min_confidence: float,
    allow_destructive_tools: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("rewrite response must be an object")
    if payload.get("equivalent") is not True:
        raise ValueError("model did not confirm semantic equivalence")
    confidence = float(payload.get("confidence"))
    if not 0.0 <= confidence <= 1.0 or confidence < min_confidence:
        raise ValueError(f"rewrite confidence below threshold {min_confidence}")
    action = payload.get("action")
    if not isinstance(action, dict):
        raise TypeError("rewrite action must be an object")
    expected_type = "shell.exec" if span.target_type == "cli" else "tool.call"
    if action.get("type") != expected_type:
        raise ValueError(f"expected {expected_type}, got {action.get('type')}")
    validate_action(action)
    if expected_type == "shell.exec":
        command = str((action.get("args") or {}).get("command") or "").strip()
        if not command:
            raise ValueError("shell command is empty")
        if "\n" in command or "\r" in command:
            raise ValueError("shell command must be a single submitted command")
        if _PLACEHOLDER_RE.search(command):
            raise ValueError("shell command contains a placeholder")
        if _is_delay_only_shell_command(command):
            raise ValueError("delay-only shell commands are not meaningful CLI rewrites")
    else:
        tool = validate_tool_call(action, tools, allow_destructive=allow_destructive_tools)
        if span.tool_name and tool.name != span.tool_name:
            raise ValueError(f"expected tool {span.tool_name}, got {tool.name}")

    intent = str(payload.get("intent") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    if not intent or not reason:
        raise ValueError("rewrite intent and reason are required")
    first = source_steps[0]
    last = source_steps[-1]
    source_confidences = [
        float(step.get("confidence"))
        for step in source_steps
        if isinstance(step.get("confidence"), (int, float))
    ]
    effective_confidence = min([confidence, span.confidence, *source_confidences]) if source_confidences else min(
        confidence, span.confidence
    )
    if effective_confidence < min_confidence:
        raise ValueError(
            f"effective confidence {effective_confidence:.3f} below threshold {min_confidence}"
        )
    return {
        "index": int(first["index"]),
        "start": first.get("start"),
        "end": last.get("end"),
        "intent": intent,
        "action": copy.deepcopy(action),
        "needs_grounding": False,
        "confidence": round(effective_confidence, 4),
        "evidence": {
            "action_rewrite": {
                "schema_version": 1,
                "model": backend_model,
                "target_type": span.target_type,
                "tool_name": span.tool_name,
                "proposal_confidence": span.confidence,
                "rewrite_confidence": confidence,
                "proposal_reason": span.reason,
                "rewrite_reason": reason,
                "improves_success_rate": span.improves_success_rate,
                "improves_efficiency": span.improves_efficiency,
                "benefit_reason": span.benefit_reason,
                "source_step_indices": [int(step["index"]) for step in source_steps],
                "source_actions": [copy.deepcopy(step.get("action")) for step in source_steps],
                "source_intents": [step.get("intent") for step in source_steps],
            }
        },
        "observation": copy.deepcopy(first.get("observation")),
        "post_observation": copy.deepcopy(last.get("post_observation")),
    }


def _is_gui_action(step: dict[str, Any]) -> bool:
    return _action_type(step).startswith("computer.")


def _action_type(step: dict[str, Any]) -> str:
    action = step.get("action")
    return str(action.get("type") or "") if isinstance(action, dict) else ""


def _optional_bool(raw: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in raw:
        return default
    value = raw[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _validate_source_actions(steps: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for step in steps:
        try:
            validate_action(step.get("action"))
        except (TypeError, ValueError) as exc:
            errors.append(f"step {step.get('index')}: {exc}")
    if errors:
        preview = "; ".join(errors[:10])
        if len(errors) > 10:
            preview += f"; ... {len(errors) - 10} more"
        raise ValueError(f"source trajectory contains invalid canonical actions: {preview}")


def _is_delay_only_shell_command(command: str) -> bool:
    normalized = " ".join(command.strip().lower().split())
    patterns = (
        r"sleep\s+\d+(?:\.\d+)?(?:ms|s|m|h|d)?",
        r"start-sleep\s+(?:-(?:seconds|milliseconds)\s+)?\d+(?:\.\d+)?",
        r"timeout\s+(?:/t\s+)?\d+(?:\s+/nobreak)?",
    )
    return any(re.fullmatch(pattern, normalized) for pattern in patterns)


def _span_image_paths(steps: list[dict[str, Any]]) -> list[Path]:
    values = [
        ((steps[0].get("observation") or {}).get("screenshot")),
        ((steps[-1].get("post_observation") or {}).get("screenshot")),
    ]
    if not all(values):
        return []
    paths = [Path(str(value)) for value in values]
    return paths if all(path.exists() for path in paths) else []


def _refresh_quality(trajectory: dict[str, Any]) -> None:
    steps = trajectory.get("steps") or []
    quality = trajectory.get("quality")
    if not isinstance(quality, dict):
        quality = {}
        trajectory["quality"] = quality
    action_types = Counter(
        str((step.get("action") or {}).get("type") or "unknown")
        for step in steps
        if isinstance(step, dict)
    )
    action_categories = Counter(_action_category(action_type) for action_type in action_types.elements())
    quality["num_steps"] = len(steps)
    quality["needs_grounding_steps"] = sum(bool(step.get("needs_grounding")) for step in steps)
    quality["has_screenshots"] = bool(steps) and all(
        ((step.get("observation") or {}).get("screenshot"))
        and ((step.get("post_observation") or {}).get("screenshot"))
        for step in steps
    )
    quality["num_screenshots"] = len(steps) + 1 if quality["has_screenshots"] else 0
    quality["action_types"] = dict(sorted(action_types.items()))
    quality["action_categories"] = dict(sorted(action_categories.items()))
    quality["rewrite_accepted_spans"] = int((trajectory.get("rewrite") or {}).get("accepted_spans") or 0)


def _action_category(action_type: str) -> str:
    if action_type.startswith("computer."):
        return "gui"
    if action_type == "shell.exec":
        return "cli"
    if action_type == "tool.call":
        return "tool"
    return "control"


def _materialize_screenshots(
    trajectory: dict[str, Any],
    destination_path: Path,
    *,
    frame_mode: str,
) -> tuple[Path | None, Path | None]:
    steps = trajectory.get("steps") or []
    if not steps or not any((step.get("observation") or {}).get("screenshot") for step in steps):
        return None, None
    final_dir = destination_path.with_suffix(".frames")
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.", dir=destination_path.parent))
    try:
        first_observation = copy.deepcopy(steps[0].get("observation") or {})
        initial_source = Path(str(first_observation.get("screenshot") or ""))
        if not initial_source.is_file():
            raise FileNotFoundError(f"missing initial screenshot: {initial_source}")
        initial_dest = temp_dir / "step_0000_state.jpg"
        _copy_or_link(initial_source, initial_dest, frame_mode=frame_mode)
        first_observation["screenshot"] = str(final_dir / initial_dest.name)
        previous_observation = first_observation
        for step in steps:
            step["observation"] = copy.deepcopy(previous_observation)
            post_observation = copy.deepcopy(step.get("post_observation") or {})
            post_source = Path(str(post_observation.get("screenshot") or ""))
            if not post_source.is_file():
                raise FileNotFoundError(f"missing post-action screenshot: {post_source}")
            post_dest = temp_dir / f"step_{int(step['index']):04d}_state.jpg"
            _copy_or_link(post_source, post_dest, frame_mode=frame_mode)
            post_observation["screenshot"] = str(final_dir / post_dest.name)
            step["post_observation"] = post_observation
            previous_observation = post_observation
        return temp_dir, final_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _copy_or_link(source: Path, destination: Path, *, frame_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if frame_mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_trajectory_atomic(
    trajectory: dict[str, Any],
    destination_path: Path,
    *,
    frame_mode: str,
    overwrite: bool,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(destination_path)
    temp_frame_dir, final_frame_dir = _materialize_screenshots(
        trajectory,
        destination_path,
        frame_mode=frame_mode,
    )
    temp_json = destination_path.with_name(f".{destination_path.name}.tmp")
    try:
        temp_json.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if final_frame_dir is not None and temp_frame_dir is not None:
            if final_frame_dir.exists():
                shutil.rmtree(final_frame_dir)
            os.replace(temp_frame_dir, final_frame_dir)
        os.replace(temp_json, destination_path)
    except Exception:
        temp_json.unlink(missing_ok=True)
        if temp_frame_dir is not None and temp_frame_dir.exists():
            shutil.rmtree(temp_frame_dir, ignore_errors=True)
        raise


def _process_path(
    source_path: Path,
    destination_path: Path,
    backend: ActionRewriteBackend,
    tools: list[RewriteTool],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if destination_path.exists() and not args.overwrite and _output_complete(destination_path):
        return {"status": "skipped", "source": str(source_path), "output": str(destination_path)}
    with open(source_path, encoding="utf-8") as handle:
        trajectory = json.load(handle)
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory JSON must be an object")
    rewritten, audit = rewrite_trajectory(
        trajectory,
        backend,
        tools,
        min_confidence=args.min_rewrite_confidence,
        allow_destructive_tools=args.allow_destructive_tools,
    )
    rewritten["rewrite"]["source_trajectory"] = str(source_path)
    if not args.dry_run:
        _write_trajectory_atomic(
            rewritten,
            destination_path,
            frame_mode=args.frame_mode,
            overwrite=args.overwrite,
        )
    return {
        "status": "rewritten" if rewritten["rewrite"]["accepted_spans"] else "no_change",
        "source": str(source_path),
        "output": str(destination_path),
        "accepted_spans": rewritten["rewrite"]["accepted_spans"],
        "audit_count": len(audit),
    }


def _output_complete(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as handle:
            trajectory = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    if not isinstance(steps, list) or not isinstance(trajectory.get("rewrite"), dict):
        return False
    for step in steps:
        if not isinstance(step, dict):
            return False
        try:
            validate_action(step.get("action"))
        except (TypeError, ValueError):
            return False
        for key in ("observation", "post_observation"):
            screenshot = ((step.get(key) or {}).get("screenshot"))
            if screenshot and not Path(str(screenshot)).is_file():
                return False
    return True


def _discover_inputs(input_dir: Path, output_dir: Path) -> list[Path]:
    output_resolved = output_dir.resolve()
    paths = []
    for path in input_dir.rglob("*.trajectory.json"):
        try:
            path.resolve().relative_to(output_resolved)
        except ValueError:
            paths.append(path)
    return sorted(paths)


def _summarize_output(output_dir: Path) -> None:
    trajectories: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    action_types: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    for path in sorted(output_dir.rglob("*.trajectory.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                trajectory = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("rewrite"), dict):
            continue
        row = copy.deepcopy(trajectory)
        row["trajectory_path"] = str(path)
        trajectories.append(row)
        for item in trajectory["rewrite"].get("audit") or []:
            audit_rows.append(
                {
                    "video_id": trajectory.get("video_id"),
                    "trajectory_path": str(path),
                    **copy.deepcopy(item),
                }
            )
        for step in trajectory.get("steps") or []:
            action = step.get("action") or {}
            action_type = str(action.get("type") or "unknown")
            action_types[action_type] += 1
            if action_type == "tool.call":
                tool_names[str((action.get("args") or {}).get("name") or "unknown")] += 1
    _write_jsonl_atomic(output_dir / "trajectories.rewritten.jsonl", trajectories)
    _write_jsonl_atomic(output_dir / "rewrite.audit.jsonl", audit_rows)
    stats = {
        "trajectories": len(trajectories),
        "rewritten_trajectories": sum((row.get("rewrite") or {}).get("status") == "rewritten" for row in trajectories),
        "accepted_spans": sum(int((row.get("rewrite") or {}).get("accepted_spans") or 0) for row in trajectories),
        "actions": sum(action_types.values()),
        "action_types": dict(sorted(action_types.items())),
        "tool_names": dict(sorted(tool_names.items())),
    }
    _write_json_atomic(output_dir / "rewrite.stats.json", stats)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def _write_failures(output_dir: Path, failures: list[dict[str, Any]]) -> None:
    _write_jsonl_atomic(output_dir / "rewrite.failures.jsonl", failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML defaults; explicit CLI flags override config")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tools-config", default=None, help="YAML/JSON registered tool definitions")
    parser.add_argument("--backend", choices=["openai_chat"], default="openai_chat")
    parser.add_argument("--api-model", default="GPT-5.5")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-base-urls", nargs="*", default=None)
    parser.add_argument("--api-endpoint-pool", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-wire-api", choices=["chat_completions", "responses"], default="chat_completions")
    parser.add_argument("--api-store", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--api-trust-env", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api-reasoning-effort", default="none")
    parser.add_argument("--api-max-tokens", type=int, default=4096)
    parser.add_argument("--api-max-retries", type=int, default=3)
    parser.add_argument("--api-request-delay", type=float, default=0.0)
    parser.add_argument("--min-rewrite-confidence", type=float, default=0.8)
    parser.add_argument("--allow-destructive-tools", action="store_true")
    parser.add_argument("--frame-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="only rebuild rewritten JSONL and stats")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        args = _merge_config_args(parser, args, argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not 0.0 <= args.min_rewrite_confidence <= 1.0:
        parser.error("--min-rewrite-confidence must be between 0 and 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        _summarize_output(output_dir)
        return

    tools = load_tools_config(Path(args.tools_config) if args.tools_config else None)
    backend = OpenAIRewriteBackend(
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
        max_tokens=args.api_max_tokens,
        max_retries=args.api_max_retries,
        request_delay=args.api_request_delay,
    )
    source_paths = _discover_inputs(input_dir, output_dir)
    if args.limit is not None:
        source_paths = source_paths[: args.limit]
    logger.info("trajectories=%d tools=%d output=%s", len(source_paths), len(tools), output_dir)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_path = {
            executor.submit(
                _process_path,
                source_path,
                output_dir / source_path.relative_to(input_dir),
                backend,
                tools,
                args,
            ): source_path
            for source_path in source_paths
        }
        for future in as_completed(future_to_path):
            source_path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                logger.info("%s %s", result["status"], source_path)
            except Exception as exc:
                failure = {
                    "source": str(source_path),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                failures.append(failure)
                logger.error("failed %s: %s", source_path, failure["error"])
    if not args.dry_run:
        _summarize_output(output_dir)
        _write_failures(output_dir, failures)
    counts = Counter(item["status"] for item in results)
    logger.info("done statuses=%s failures=%d", dict(sorted(counts.items())), len(failures))


def _merge_config_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> argparse.Namespace:
    config_path = Path(args.config)
    raw = load_yaml_mapping(config_path)
    defaults = _flatten_config_defaults(raw)
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
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - known)
    if unknown:
        raise ValueError(f"unknown action rewrite config keys: {', '.join(unknown)}")
    explicit = _explicit_destinations(parser, argv)
    for key, value in defaults.items():
        if key not in explicit and key != "config":
            setattr(args, key, value)
    if "api_base_url" in explicit or "api_base_urls" in explicit:
        args.api_endpoint_pool = None
    return args


def _flatten_config_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        normalized_key = key.replace("-", "_")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                normalized_child = child_key.replace("-", "_")
                flat[f"{normalized_key}_{normalized_child}"] = child_value
        else:
            flat[normalized_key] = value
    return flat


def _explicit_destinations(parser: argparse.ArgumentParser, argv: list[str] | None) -> set[str]:
    tokens = list(argv) if argv is not None else os.sys.argv[1:]
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    return {
        option_to_dest[token.split("=", 1)[0]]
        for token in tokens
        if token.split("=", 1)[0] in option_to_dest
    }


if __name__ == "__main__":
    main()
