#!/usr/bin/env python3
"""Convert mined CUA trajectories to ms-swift multimodal SFT JSONL."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from cuamaster.action_space import ACTION_SPECS, CUAAction


DEFAULT_OUTPUT_DIR = Path("data/cua_swift")
DEFAULT_OUTPUT_NAME = "cua_sft.jsonl"
DEFAULT_SYSTEM = (
    "You are a computer-use agent. Given the screenshot and task, predict the next canonical "
    "GUI, CLI, structured-tool, or termination action."
)


@dataclass
class BuildStats:
    rank_rows_seen: int = 0
    unique_target_videos: int = 0
    duplicate_target_videos: int = 0
    trajectories_seen: int = 0
    trajectories_kept: int = 0
    videos_kept: int = 0
    skipped_non_target_trajectory: int = 0
    skipped_empty_steps: int = 0
    skipped_bad_step: int = 0
    skipped_missing_image: int = 0
    skipped_bad_action: int = 0
    termination_examples_written: int = 0
    examples_written: int = 0
    action_types_written: dict[str, int] = field(default_factory=dict)
    action_categories_written: dict[str, int] = field(default_factory=dict)
    tool_names_written: dict[str, int] = field(default_factory=dict)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield line_no, obj


def load_target_video_ids(paths: list[Path], stats: BuildStats) -> set[str]:
    video_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for _, row in read_jsonl(path):
            stats.rank_rows_seen += 1
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            if video_id in video_ids:
                stats.duplicate_target_videos += 1
            video_ids.add(video_id)
    stats.unique_target_videos = len(video_ids)
    return video_ids


def format_history(history: list[str]) -> str:
    if not history:
        return "None"
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(history))


def py_literal(value: Any) -> str:
    return repr(value)


def action_to_code(action: dict[str, Any]) -> str | None:
    try:
        canonical = CUAAction.from_dict(action)
    except (TypeError, ValueError):
        return None
    action_type = canonical.type
    args = canonical.args

    if action_type == "computer.click":
        if "x" not in args or "y" not in args:
            return None
        button = args.get("button", "left")
        return f"pyautogui.click(x={args['x']}, y={args['y']}, button={py_literal(button)})"
    if action_type == "computer.double_click":
        if "x" not in args or "y" not in args:
            return None
        button = args.get("button", "left")
        return f"pyautogui.doubleClick(x={args['x']}, y={args['y']}, button={py_literal(button)})"
    if action_type == "computer.move":
        if "x" not in args or "y" not in args:
            return None
        return f"pyautogui.moveTo(x={args['x']}, y={args['y']})"
    if action_type == "computer.drag":
        call_args = [
            f"x1={args['x1']}",
            f"y1={args['y1']}",
            f"x2={args['x2']}",
            f"y2={args['y2']}",
            f"button={py_literal(args.get('button', 'left'))}",
        ]
        if "duration_ms" in args:
            call_args.append(f"duration_ms={args['duration_ms']}")
        return f"pyautogui.dragTo({', '.join(call_args)})"
    if action_type == "computer.scroll":
        call_args = [f"dx={args['dx']}", f"dy={args['dy']}"]
        for key in ("x", "y"):
            if key in args:
                call_args.append(f"{key}={args[key]}")
        return f"pyautogui.scroll({', '.join(call_args)})"
    if action_type == "computer.type":
        text = args.get("text")
        if text is None:
            return None
        return f"pyautogui.write(message={py_literal(str(text))})"
    if action_type == "computer.press":
        keys = args.get("keys")
        if isinstance(keys, list) and keys:
            if len(keys) == 1:
                return f"pyautogui.press({py_literal(str(keys[0]))})"
            key_args = ", ".join(py_literal(str(key)) for key in keys)
            return f"pyautogui.hotkey({key_args})"
        if isinstance(keys, str) and keys:
            return f"pyautogui.press({py_literal(keys)})"
        return None
    if action_type == "computer.wait":
        seconds = args.get("seconds")
        if seconds is None:
            return None
        return f"time.sleep({seconds})"
    if action_type == "shell.exec":
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        call_args = [f"command={py_literal(command)}"]
        cwd = args.get("cwd")
        if cwd is not None:
            if not isinstance(cwd, str):
                return None
            call_args.append(f"cwd={py_literal(cwd)}")
        timeout = args.get("timeout_sec")
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
                return None
            call_args.append(f"timeout_sec={timeout}")
        return f"shell.exec({', '.join(call_args)})"
    if action_type == "tool.call":
        name = args.get("name")
        tool_args = args.get("args")
        if not isinstance(name, str) or not name.strip() or not isinstance(tool_args, dict):
            return None
        return f"tool.call(name={py_literal(name)}, args={py_literal(tool_args)})"
    if action_type == "finish":
        status = args.get("status")
        if status not in {"success", "failure"}:
            return None
        return f"computer.terminate(status={py_literal(status)})"
    return None


def text_or_empty(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def evidence_reason(step: dict[str, Any]) -> str:
    evidence = step.get("evidence")
    if not isinstance(evidence, dict):
        return ""
    boundary = evidence.get("api_boundary")
    if isinstance(boundary, dict):
        boundary_evidence = text_or_empty(boundary.get("evidence"))
        if boundary_evidence:
            return boundary_evidence
    source_candidates = evidence.get("source_candidates")
    if isinstance(source_candidates, list):
        snippets: list[str] = []
        for candidate in source_candidates[:2]:
            if not isinstance(candidate, dict):
                continue
            text = text_or_empty(candidate.get("text"))
            nested_evidence = candidate.get("evidence")
            transcript = ""
            if isinstance(nested_evidence, dict):
                transcript = text_or_empty(nested_evidence.get("transcript"))
            snippet = transcript or text
            if snippet:
                snippets.append(snippet)
        if snippets:
            return " ".join(snippets)
    return ""


def format_assistant(step: dict[str, Any], code: str) -> str:
    action = step.get("action")
    if isinstance(action, dict) and action.get("type") == "finish":
        status = (action.get("args") or {}).get("status", "success")
        if status == "success":
            return "\n".join(
                [
                    "Observation: The requested task is complete in the current state.",
                    "Thought: The task has been completed successfully, so no further CUA action is needed.",
                    "Action: Terminate the task with success status.",
                    f"Code: {code}",
                ]
            )
        return "\n".join(
            [
                "Observation: The task cannot be completed from the current state.",
                "Thought: Further CUA actions will not complete the task, so terminate with failure status.",
                "Action: Terminate the task with failure status.",
                f"Code: {code}",
            ]
        )

    refinement = step.get("api_refinement")
    if not isinstance(refinement, dict):
        refinement = {}
    intent = text_or_empty(refinement.get("intent")) or text_or_empty(step.get("intent"))
    target = text_or_empty(refinement.get("target"))
    reason = text_or_empty(refinement.get("reason")) or evidence_reason(step) or intent
    changed_reason = text_or_empty(refinement.get("changed_action_type_reason"))
    evidence = evidence_reason(step)

    observation = reason
    if evidence and evidence != reason:
        observation = f"{observation} Supporting evidence: {evidence}"

    thought_parts: list[str] = []
    if target:
        thought_parts.append(f"Target: {target}.")
    if intent:
        thought_parts.append(f"Next, {intent}.")
    if changed_reason:
        thought_parts.append(f"Action type adjustment: {changed_reason}")
    thought = " ".join(thought_parts) or "Choose the next CUA action that advances the task."
    action = intent or "Perform the next CUA action."

    return "\n".join(
        [
            f"Observation: {observation}",
            f"Thought: {thought}",
            f"Action: {action}",
            f"Code: {code}",
        ]
    )


def make_task_id(trajectory: dict[str, Any]) -> str:
    video_id = str(trajectory.get("video_id") or "unknown_video")
    trajectory_path = Path(str(trajectory.get("trajectory_path") or "trajectory"))
    return f"{video_id}:{trajectory_path.stem}"


def make_example(
    trajectory: dict[str, Any],
    step: dict[str, Any],
    image_path: Path,
    code: str,
    history: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    task = text_or_empty(trajectory.get("task"))
    user_content = (
        "<image>\n"
        f"Task: {task}\n"
        f"Detailed task: {task}\n"
        f"Previous actions:\n{format_history(history)}\n"
        "Return the next canonical action as Observation, Thought, Action, and Code."
    )
    return {
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": format_assistant(step, code)},
        ],
        "images": [str(image_path)],
        "task_id": make_task_id(trajectory),
        "step_index": step.get("index"),
    }


def convert_dataset(args: argparse.Namespace, output_path: Path) -> BuildStats:
    stats = BuildStats()
    target_video_ids = load_target_video_ids(args.rank_paths, stats)
    source_path = args.cua_root / args.source_name
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    kept_videos: set[str] = set()
    remaining = args.limit_examples

    with tmp_path.open("w", encoding="utf-8") as out_f:
        for _, trajectory in read_jsonl(source_path):
            stats.trajectories_seen += 1
            video_id = str(trajectory.get("video_id") or "")
            if video_id not in target_video_ids:
                stats.skipped_non_target_trajectory += 1
                continue
            steps = trajectory.get("steps")
            if not isinstance(steps, list) or not steps:
                stats.skipped_empty_steps += 1
                continue

            trajectory_examples = 0
            history: list[str] = []
            for step in steps:
                if not isinstance(step, dict):
                    stats.skipped_bad_step += 1
                    continue
                observation = step.get("observation")
                image = observation.get("screenshot") if isinstance(observation, dict) else None
                image_path = Path(str(image or ""))
                if not image or not image_path.is_file():
                    stats.skipped_missing_image += 1
                    continue
                action = step.get("action")
                if not isinstance(action, dict):
                    stats.skipped_bad_action += 1
                    continue
                code = action_to_code(action)
                if not code:
                    stats.skipped_bad_action += 1
                    continue

                example = make_example(
                    trajectory,
                    step,
                    image_path,
                    code,
                    history[-args.max_history_actions :],
                    args,
                )
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats.examples_written += 1
                trajectory_examples += 1
                action_type = str(action.get("type") or "unknown")
                stats.action_types_written[action_type] = stats.action_types_written.get(action_type, 0) + 1
                category = ACTION_SPECS[action_type].category
                stats.action_categories_written[category] = (
                    stats.action_categories_written.get(category, 0) + 1
                )
                if action_type == "finish":
                    stats.termination_examples_written += 1
                if action_type == "tool.call":
                    tool_name = str((action.get("args") or {}).get("name") or "unknown")
                    stats.tool_names_written[tool_name] = stats.tool_names_written.get(tool_name, 0) + 1

                action_text = text_or_empty(step.get("intent")) or "Perform the next CUA action."
                history.append(f"{action_text} | {code}")
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break

            if trajectory_examples:
                stats.trajectories_kept += 1
                kept_videos.add(video_id)
            if remaining is not None and remaining <= 0:
                break

    stats.videos_kept = len(kept_videos)
    if not getattr(args, "no_validate", False):
        validation_rows, missing_images = validate_output(tmp_path)
        if validation_rows != stats.examples_written:
            raise ValueError(
                f"validation row mismatch: wrote {stats.examples_written}, read {validation_rows}"
            )
        if missing_images:
            raise ValueError(f"validation found {missing_images} missing images")
    minimums = {
        "shell.exec": int(getattr(args, "min_shell_actions", 0)),
        "tool.call": int(getattr(args, "min_tool_actions", 0)),
    }
    stats_path = output_path.with_name(f"{output_path.stem}.stats.json")
    failures = [
        f"{action_type}={stats.action_types_written.get(action_type, 0)} < {minimum}"
        for action_type, minimum in minimums.items()
        if stats.action_types_written.get(action_type, 0) < minimum
    ]
    minimum_tool_counts = parse_minimum_tool_counts(getattr(args, "min_tool_counts", []))
    failures.extend(
        f"tool:{name}={stats.tool_names_written.get(name, 0)} < {minimum}"
        for name, minimum in minimum_tool_counts.items()
        if stats.tool_names_written.get(name, 0) < minimum
    )
    stats_path.write_text(
        json.dumps(
            {
                "output": str(output_path),
                "passed": not failures,
                "minimum_action_counts": minimums,
                "minimum_tool_counts": minimum_tool_counts,
                "failures": failures,
                "stats": asdict(stats),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            "action count validation failed: " + "; ".join(failures) + f"; stats: {stats_path}"
        )
    tmp_path.replace(output_path)
    return stats


def validate_output(path: Path) -> tuple[int, int]:
    rows = 0
    missing_images = 0
    for line_no, row in read_jsonl(path):
        rows += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"invalid messages at {path}:{line_no}")
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"invalid images at {path}:{line_no}")
        if not Path(str(images[0])).is_file():
            missing_images += 1
        if "task_id" not in row or "step_index" not in row:
            raise ValueError(f"missing task_id/step_index at {path}:{line_no}")
    return rows, missing_images


def print_stats(path: Path, stats: BuildStats) -> None:
    print(f"output: {path}")
    for stat_field in stats.__dataclass_fields__:
        print(f"  {stat_field}: {getattr(stats, stat_field)}")


def parse_minimum_tool_counts(values: list[str] | None) -> dict[str, int]:
    minimums: dict[str, int] = {}
    for value in values or []:
        name, separator, raw_count = value.rpartition("=")
        if not separator or not name.strip():
            raise ValueError(f"invalid --min-tool-count value: {value!r}; expected name=count")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid --min-tool-count value: {value!r}; count must be an integer") from exc
        if count < 0:
            raise ValueError(f"invalid --min-tool-count value: {value!r}; count must be non-negative")
        minimums[name.strip()] = count
    return dict(sorted(minimums.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cua-root",
        type=Path,
        required=True,
        help="root directory containing the reconstructed trajectory dataset",
    )
    parser.add_argument("--source-name", default="trajectories.train.jsonl")
    parser.add_argument(
        "--rank-paths",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL files whose video_id column selects target videos",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--max-history-actions", type=int, default=3)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--min-shell-actions", type=int, default=0)
    parser.add_argument("--min-tool-actions", type=int, default=0)
    parser.add_argument(
        "--min-tool-count",
        dest="min_tool_counts",
        action="append",
        default=[],
        metavar="NAME=COUNT",
    )
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / args.output_name
    stats = convert_dataset(args, output_path)
    print_stats(output_path, stats)
    if not args.no_validate:
        rows, missing_images = validate_output(output_path)
        print(f"validation_rows: {rows}")
        print(f"validation_missing_images: {missing_images}")


if __name__ == "__main__":
    main()
