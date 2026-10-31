"""Host-side MCP bridge for an OSWorld-compatible desktop controller."""
from __future__ import annotations

import base64
import json
import os
import shlex
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .assets.weavebench_computer_mcp import server as core


CONTROLLER_URL = os.environ.get("GUI_TRAJECTORY_CONTROLLER_URL", "").rstrip("/")
CONTROLLER_PYTHON = os.environ.get("GUI_TRAJECTORY_CONTROLLER_PYTHON", "python")
CONTROLLER_PLATFORM = os.environ.get("GUI_TRAJECTORY_CONTROLLER_PLATFORM", "")
CONTROLLER_TIMEOUT_S = float(os.environ.get("GUI_TRAJECTORY_CONTROLLER_TIMEOUT_S", "90"))
ENABLE_GUEST_SHELL = os.environ.get(
    "GUI_TRAJECTORY_ENABLE_GUEST_SHELL", ""
).strip().casefold() in {"1", "true", "yes", "on"}
SHELL_EVENT_LOG = os.environ.get(
    "GUI_TRAJECTORY_SHELL_EVENT_LOG", str(Path.cwd() / "shell_events.jsonl")
)
_screenshot_lock = threading.Lock()
_screenshot_step = 0
_shell_lock = threading.Lock()
_shell_step = 0

SHELL_TOOL_NAME = "shell"
SHELL_TOOL = {
    "name": SHELL_TOOL_NAME,
    "description": (
        "Execute a command inside the connected desktop VM, not on the controller host. "
        "Commands use PowerShell on Windows and /bin/sh on macOS/Linux."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout_sec": {"type": "number", "minimum": 1, "maximum": 110},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}


def _request(
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> bytes:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{CONTROLLER_URL}{path}", data=data, headers=headers
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout or CONTROLLER_TIMEOUT_S
        ) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"controller {path} returned HTTP {exc.code}: {detail}") from exc


def _request_json(
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    raw = _request(path, payload=payload, timeout=timeout)
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"controller {path} returned a non-object response")
    return parsed


def _platform_action_to_python(action: dict[str, Any]) -> str:
    platform_name = CONTROLLER_PLATFORM.casefold()
    action_type = str(action.get("type") or "")
    if action_type in {"type", "keyboard"} and platform_name in {"windows", "darwin"}:
        text = str(action.get("text") or "")
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        if platform_name == "windows":
            powershell = (
                "$t=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"
                + encoded
                + "')); Set-Clipboard -Value $t"
            )
            return (
                "import subprocess; "
                f"subprocess.run(['powershell','-NoProfile','-Command',{json.dumps(powershell)}], "
                "check=True); pyautogui.hotkey('ctrl', 'v')"
            )
        return (
            "import base64, subprocess; "
            f"subprocess.run(['pbcopy'], input=base64.b64decode('{encoded}'), check=True); "
            "pyautogui.hotkey('command', 'v')"
        )
    if action_type in {"keypress", "key"} and platform_name in {"windows", "darwin"}:
        copied = dict(action)
        keys = copied.get("keys")
        if keys is None:
            keys = copied.get("text") or copied.get("key")
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(keys, list):
            replacement = "command" if platform_name == "darwin" else "win"
            copied["keys"] = [
                replacement
                if str(key).strip().upper() in {"CMD", "META", "WIN", "SUPER"}
                else key
                for key in keys
            ]
        action = copied
    return core.action_to_python(action)


def execute_action(action: dict[str, Any]) -> dict[str, Any]:
    snippet = _platform_action_to_python(action)
    if not snippet:
        action_type = action.get("type") if isinstance(action, dict) else None
        if action_type in {"screenshot", "cursor_position", None, ""}:
            return {"ok": True}
        return {"ok": False, "err": f"unsupported or malformed action: {action_type!r}"}
    wrapped = "import pyautogui, time\npyautogui.FAILSAFE = False\n" + snippet
    try:
        result = _request_json(
            "/execute",
            {"command": [CONTROLLER_PYTHON, "-c", wrapped], "shell": False},
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
    return_code = result.get("returncode")
    if result.get("status") == "success" and return_code in {None, 0}:
        return {"ok": True}
    error = result.get("error") or result.get("message") or result.get("output")
    return {"ok": False, "err": str(error or f"controller returncode={return_code}")}


def _capture_screenshot(action_label: str) -> tuple[str | None, str | None]:
    global _screenshot_step
    try:
        image = _request("/screenshot")
    except Exception as exc:  # noqa: BLE001
        core._log(f"controller screenshot failed: {type(exc).__name__}: {exc}")
        return None, None
    if not (
        image.startswith(b"\x89PNG\r\n\x1a\n")
        or image.startswith(b"\xff\xd8\xff")
    ):
        core._log("controller screenshot returned an invalid image payload")
        return None, None
    with _screenshot_lock:
        _screenshot_step += 1
        step = _screenshot_step
    screenshot_dir = Path(core.SCREENSHOT_DIR)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if image.startswith(b"\x89PNG") else ".jpg"
    path = screenshot_dir / f"screenshot_{step:04d}_{core._safe_label(action_label)}{suffix}"
    path.write_bytes(image)
    return core.base64.b64encode(image).decode("ascii"), str(path)


def _shell_argv(command: str, cwd: str | None = None) -> list[str]:
    platform_name = CONTROLLER_PLATFORM.casefold()
    if platform_name == "windows":
        script_parts = ["$ErrorActionPreference='Continue'"]
        if cwd:
            encoded_cwd = base64.b64encode(cwd.encode("utf-8")).decode("ascii")
            script_parts.append(
                "$cwd=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"
                + encoded_cwd
                + "')); Set-Location -LiteralPath $cwd"
            )
        script_parts.extend([
            "& {",
            command,
            "}",
            "$code = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } "
            "elseif ($?) { 0 } else { 1 }; exit $code",
        ])
        encoded = base64.b64encode("\n".join(script_parts).encode("utf-16le")).decode(
            "ascii"
        )
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ]
    shell = "/bin/sh"
    body = command if not cwd else f"cd -- {shlex.quote(cwd)} && {command}"
    return [shell, "-lc", body]


def _append_shell_event(event: dict[str, Any]) -> None:
    path = Path(SHELL_EVENT_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _shell_lock, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()


def handle_shell_call(arguments: dict[str, Any]) -> dict[str, Any]:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return {
            "content": [{"type": "text", "text": "command must be a non-empty string"}],
            "isError": True,
        }
    cwd = arguments.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        return {
            "content": [{"type": "text", "text": "cwd must be a string"}],
            "isError": True,
        }
    try:
        timeout_seconds = min(110.0, max(1.0, float(arguments.get("timeout_sec", 90))))
    except (TypeError, ValueError):
        return {
            "content": [{"type": "text", "text": "timeout_sec must be a number"}],
            "isError": True,
        }

    global _shell_step
    with _shell_lock:
        _shell_step += 1
        sequence = _shell_step

    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    _, pre_screenshot = _capture_screenshot("shell_pre")
    try:
        result = _request_json(
            "/execute",
            {"command": _shell_argv(command, cwd), "shell": False},
            timeout=max(CONTROLLER_TIMEOUT_S, timeout_seconds + 10),
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "error",
            "returncode": None,
            "output": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    post_image, post_screenshot = _capture_screenshot("shell_post")
    return_code = result.get("returncode")
    ok = result.get("status") == "success" and return_code in {None, 0}
    common = {
        "schema_version": "gui_trajectory_collection.shell_capture.v1",
        "sequence": sequence,
        "tool_name": SHELL_TOOL_NAME,
        "command": command,
        "cwd": cwd,
    }
    _append_shell_event({**common, "phase": "pre", "screenshot_path": pre_screenshot})
    _append_shell_event({
        **common,
        "phase": "post",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round((time.monotonic() - started_monotonic) * 1000, 3),
        "screenshot_path": post_screenshot,
        "execution": {
            "ok": ok,
            "status_source": "controller_execute",
            "returncode": return_code,
            "stdout": result.get("output") or result.get("stdout") or "",
            "stderr": result.get("error") or result.get("stderr") or result.get("message") or "",
        },
        "started_at": started_at,
    })
    payload = {
        "platform": CONTROLLER_PLATFORM,
        "returncode": return_code,
        "stdout": result.get("output") or result.get("stdout") or "",
        "stderr": result.get("error") or result.get("stderr") or result.get("message") or "",
    }
    content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    if post_image:
        mime_type = "image/jpeg" if str(post_screenshot).lower().endswith(".jpg") else "image/png"
        content.append({"type": "image", "data": post_image, "mimeType": mime_type})
    return {"content": content, "isError": not ok}


def configure_core() -> None:
    if not CONTROLLER_URL:
        raise RuntimeError("GUI_TRAJECTORY_CONTROLLER_URL is required")
    core.EVENT_LOG = os.environ.get(
        "GUI_TRAJECTORY_MCP_EVENT_LOG", str(Path.cwd() / "computer_events.jsonl")
    )
    core.SCREENSHOT_DIR = os.environ.get(
        "GUI_TRAJECTORY_MCP_SCREENSHOT_DIR", str(Path.cwd() / "screenshots")
    )
    core.SERVER_NAME = "gui-trajectory-host-computer"
    core.execute_action = execute_action
    core._capture_screenshot = _capture_screenshot


def dispatch(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    is_notification = request_id is None
    try:
        if method == "initialize":
            return core._make_response(request_id, result={
                "protocolVersion": core.PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": core.SERVER_NAME, "version": core.SERVER_VERSION},
            })
        if method in {"notifications/initialized", "initialized"}:
            return None
        if method == "ping":
            return core._make_response(request_id, result={})
        if method == "tools/list":
            tools = [{
                "name": core.TOOL_NAME,
                "description": core.TOOL_DESCRIPTION,
                "inputSchema": core.TOOL_INPUT_SCHEMA,
            }]
            if ENABLE_GUEST_SHELL:
                tools.append(SHELL_TOOL)
            return core._make_response(request_id, result={"tools": tools})
        if method == "tools/call":
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return core._make_response(
                    request_id,
                    error=core._make_error(-32602, "arguments must be an object"),
                )
            tool_name = params.get("name")
            if tool_name == core.TOOL_NAME:
                return core._make_response(
                    request_id, result=core.handle_tools_call(arguments)
                )
            if tool_name == SHELL_TOOL_NAME and ENABLE_GUEST_SHELL:
                return core._make_response(
                    request_id, result=handle_shell_call(arguments)
                )
            return core._make_response(
                request_id,
                error=core._make_error(-32602, f"unknown tool: {tool_name!r}"),
            )
        if is_notification:
            return None
        return core._make_response(
            request_id,
            error=core._make_error(-32601, f"method not found: {method!r}"),
        )
    except Exception as exc:  # noqa: BLE001
        core._log(f"dispatch crashed on {method!r}: {exc}")
        if is_notification:
            return None
        return core._make_response(
            request_id,
            error=core._make_error(-32603, "internal error", data=str(exc)),
        )


def main() -> int:
    configure_core()
    core._log(
        f"starting {core.SERVER_NAME} v{core.SERVER_VERSION} "
        f"(guest_shell={ENABLE_GUEST_SHELL})"
    )
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        raw = stdin.readline()
        if not raw:
            return 0
        try:
            request = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            response = core._make_response(
                None, error=core._make_error(-32700, "parse error", data=str(exc))
            )
        else:
            response = dispatch(request) if isinstance(request, dict) else None
        if response is not None:
            stdout.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
