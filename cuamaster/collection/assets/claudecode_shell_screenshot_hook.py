#!/usr/bin/env python3
"""Record grounded execution results for Claude Code Bash tool calls."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path("/tmp_workspace")
TRAJECTORY_DIR = WORKSPACE / "_trajectory"
SCREENSHOT_DIR = WORKSPACE / "_shell_screenshots"
EVENT_LOG = TRAJECTORY_DIR / "shell_events.jsonl"
HOOK_NAME = "claudecode_shell_screenshot_hook.py"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def install_settings(settings_path: Path) -> None:
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        settings = {}
    if not isinstance(settings, dict):
        settings = {}
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        settings["hooks"] = hooks
    script = shlex.quote(str(Path(__file__).resolve()))
    for event_name in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
        entries = hooks.get(event_name)
        if not isinstance(entries, list):
            entries = []
        retained = []
        for entry in entries:
            commands = [
                hook.get("command", "")
                for hook in entry.get("hooks", [])
                if isinstance(hook, dict)
            ] if isinstance(entry, dict) else []
            if not any(HOOK_NAME in str(command) for command in commands):
                retained.append(entry)
        if event_name in {"PostToolUse", "PostToolUseFailure"}:
            retained.append({
                "matcher": "Bash",
                "hooks": [{
                    "type": "command",
                    "command": f"/usr/bin/python3 {script} capture",
                }],
            })
        hooks[event_name] = retained
    _atomic_json(settings_path, settings)


def _safe_identifier(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return cleaned[:160] or f"unknown_{os.getpid()}_{time.time_ns()}"


def _valid_png(path: Path) -> bool:
    try:
        return path.stat().st_size > 24 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _capture(path: Path) -> str | None:
    temporary = path.with_name(path.stem + ".tmp.png")
    temporary.unlink(missing_ok=True)
    errors = []
    try:
        completed = subprocess.run(
            ["gnome-screenshot", "-f", str(temporary)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12,
            check=False,
        )
        if completed.returncode != 0:
            errors.append((completed.stderr or "gnome-screenshot failed").strip())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
    if not _valid_png(temporary):
        try:
            import pyautogui

            pyautogui.screenshot(str(temporary))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pyautogui {type(exc).__name__}: {exc}")
    if _valid_png(temporary):
        os.replace(temporary, path)
        return None
    temporary.unlink(missing_ok=True)
    return " | ".join(error for error in errors if error) or "screenshot capture failed"


def _append_event(event: dict[str, Any]) -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _first_value(value: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(value, dict):
        return None
    for key in keys:
        if value.get(key) is not None:
            return value[key]
    return None


def _execution(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("tool_response")
    event_name = str(payload.get("hook_event_name") or "")
    returncode = _first_value(response, ("returncode", "return_code", "exit_code", "code"))
    try:
        returncode = int(returncode) if returncode is not None else None
    except (TypeError, ValueError):
        returncode = None

    explicit_ok = _first_value(response, ("ok", "success"))
    if isinstance(explicit_ok, bool):
        ok = explicit_ok
    elif returncode is not None:
        ok = returncode == 0
    else:
        ok = event_name == "PostToolUse"

    execution: dict[str, Any] = {
        "ok": ok,
        "status_source": "claude_code_hook",
        "hook_event_name": event_name or None,
    }
    if returncode is not None:
        execution["returncode"] = returncode
    stdout = _first_value(response, ("stdout", "output", "content"))
    stderr = _first_value(response, ("stderr", "error", "message"))
    if stdout is not None:
        execution["stdout"] = stdout
    if stderr is not None:
        execution["stderr"] = stderr
    if isinstance(response, dict):
        execution["tool_response"] = response
    elif response is not None:
        execution["output"] = response
    return execution


def capture_hook() -> None:
    try:
        payload = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError):
        return
    if str(payload.get("tool_name") or "").casefold() != "bash":
        return
    capture_screenshot = os.environ.get(
        "GUI_TRAJECTORY_CAPTURE_SHELL_SCREENSHOTS", ""
    ).strip().casefold() in {"1", "true", "yes", "on"}
    tool_call_id = str(payload.get("tool_use_id") or "")
    screenshot_path = None
    capture_error = None
    if capture_screenshot:
        try:
            delay = max(0.0, float(os.environ.get(
                "GUI_TRAJECTORY_SHELL_POST_SCREENSHOT_DELAY_S", "0.5"
            )))
        except ValueError:
            delay = 0.5
        time.sleep(delay)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        screenshot_path = SCREENSHOT_DIR / f"shell_{_safe_identifier(tool_call_id)}.png"
        capture_error = _capture(screenshot_path)
    _append_event({
        "schema_version": "gui_trajectory_collection.shell_capture.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "phase": "post",
        "hook_event_name": payload.get("hook_event_name"),
        "session_id": payload.get("session_id"),
        "tool_call_id": tool_call_id or None,
        "tool_name": payload.get("tool_name"),
        "tool_input": payload.get("tool_input"),
        "screenshot_path": (
            str(screenshot_path)
            if screenshot_path is not None and capture_error is None
            else None
        ),
        "error": capture_error,
        "execution": _execution(payload),
    })


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "install":
        install_settings(Path(sys.argv[2]))
        return 0
    if len(sys.argv) >= 2 and sys.argv[1] == "capture":
        capture_hook()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
