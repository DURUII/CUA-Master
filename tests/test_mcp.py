from __future__ import annotations

import base64
import json
from pathlib import Path

from cuamaster.collection.assets.weavebench_computer_mcp import server
from cuamaster.collection import host_computer_mcp


def test_mcp_writes_actuator_event(monkeypatch, tmp_path):
    event_log = tmp_path / "events.jsonl"
    screenshot = tmp_path / "shot.png"
    screenshot.write_bytes(b"png")
    monkeypatch.setattr(server, "EVENT_LOG", str(event_log))
    monkeypatch.setattr(server, "execute_action", lambda action: {"ok": True})
    monkeypatch.setattr(server, "_capture_screenshot", lambda label: ("cG5n", str(screenshot)))
    result = server.handle_tools_call({
        "actions": [{"action": "click", "x": 1, "y": 2, "button": 1}],
    })
    event = json.loads(event_log.read_text(encoding="utf-8"))
    assert result["isError"] is False
    assert event["actions"][0]["type"] == "click"
    assert event["executions"][0]["ok"] is True
    assert event["pre_screenshot_path"] == str(screenshot)
    assert event["executions"][0]["screenshot_path"] == str(screenshot)


def test_host_mcp_executes_through_osworld_controller(monkeypatch, tmp_path):
    requests = []
    png = b"\x89PNG\r\n\x1a\n" + b"test-image"

    def fake_request(path, *, payload=None):
        assert path == "/screenshot"
        assert payload is None
        return png

    def fake_request_json(path, payload):
        assert path == "/execute"
        requests.append(payload)
        return {"status": "success", "output": "", "error": "", "returncode": 0}

    event_log = tmp_path / "computer_events.jsonl"
    screenshots = tmp_path / "screenshots"
    monkeypatch.setattr(host_computer_mcp, "CONTROLLER_URL", "http://controller")
    monkeypatch.setattr(host_computer_mcp, "_request", fake_request)
    monkeypatch.setattr(host_computer_mcp, "_request_json", fake_request_json)
    monkeypatch.setattr(server, "EVENT_LOG", str(event_log))
    monkeypatch.setattr(server, "SCREENSHOT_DIR", str(screenshots))
    monkeypatch.setattr(server, "execute_action", host_computer_mcp.execute_action)
    monkeypatch.setattr(server, "_capture_screenshot", host_computer_mcp._capture_screenshot)

    result = server.handle_tools_call({
        "actions": [{"action": "click", "x": 12, "y": 34, "button": 1}],
    })

    event = json.loads(event_log.read_text(encoding="utf-8"))
    assert result["isError"] is False
    assert requests[0]["command"][0] == "python"
    assert "pyautogui.click(12, 34" in requests[0]["command"][2]
    assert event["actions"][0]["type"] == "click"
    assert Path(event["pre_screenshot_path"]).is_file()
    assert Path(event["screenshot_path"]).is_file()


def test_host_mcp_uses_windows_clipboard_and_modifier(monkeypatch):
    monkeypatch.setattr(host_computer_mcp, "CONTROLLER_PLATFORM", "Windows")
    type_script = host_computer_mcp._platform_action_to_python({
        "type": "type",
        "text": "中文",
    })
    key_script = host_computer_mcp._platform_action_to_python({
        "type": "keypress",
        "keys": ["CMD", "L"],
    })
    assert "Set-Clipboard" in type_script
    assert "pyautogui.hotkey('ctrl', 'v')" in type_script
    assert "pyautogui.hotkey(\"win\", \"l\")" in key_script


def test_host_mcp_uses_macos_clipboard_and_command_key(monkeypatch):
    monkeypatch.setattr(host_computer_mcp, "CONTROLLER_PLATFORM", "Darwin")
    type_script = host_computer_mcp._platform_action_to_python({
        "type": "type",
        "text": "résumé",
    })
    key_script = host_computer_mcp._platform_action_to_python({
        "type": "keypress",
        "keys": ["META", "A"],
    })
    assert "pbcopy" in type_script
    assert "pyautogui.hotkey('command', 'v')" in type_script
    assert "pyautogui.hotkey(\"command\", \"a\")" in key_script


def test_host_mcp_lists_guest_shell_only_when_enabled(monkeypatch):
    monkeypatch.setattr(host_computer_mcp, "ENABLE_GUEST_SHELL", True)
    response = host_computer_mcp.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert [tool["name"] for tool in response["result"]["tools"]] == ["computer", "shell"]


def test_host_mcp_executes_windows_shell_in_guest(monkeypatch, tmp_path):
    requests = []
    screenshots = iter([
        ("cHJl", str(tmp_path / "screenshots" / "pre.png")),
        ("cG9zdA==", str(tmp_path / "screenshots" / "post.png")),
    ])
    (tmp_path / "screenshots").mkdir()
    (tmp_path / "screenshots" / "pre.png").write_bytes(b"pre")
    (tmp_path / "screenshots" / "post.png").write_bytes(b"post")

    def fake_request_json(path, payload, *, timeout=None):
        requests.append((path, payload, timeout))
        return {"status": "success", "output": "ok\n", "error": "", "returncode": 0}

    monkeypatch.setattr(host_computer_mcp, "CONTROLLER_PLATFORM", "Windows")
    monkeypatch.setattr(host_computer_mcp, "SHELL_EVENT_LOG", str(tmp_path / "shell.jsonl"))
    monkeypatch.setattr(host_computer_mcp, "_capture_screenshot", lambda label: next(screenshots))
    monkeypatch.setattr(host_computer_mcp, "_request_json", fake_request_json)
    monkeypatch.setattr(host_computer_mcp, "_shell_step", 0)

    result = host_computer_mcp.handle_shell_call({
        "command": "Write-Output 'ok'",
        "cwd": "C:\\Users\\oai",
        "timeout_sec": 30,
    })

    argv = requests[0][1]["command"]
    assert argv[0] == "powershell.exe"
    script = base64.b64decode(argv[-1]).decode("utf-16le")
    assert "Set-Location -LiteralPath $cwd" in script
    assert "Write-Output 'ok'" in script
    assert result["isError"] is False
    events = [json.loads(line) for line in (tmp_path / "shell.jsonl").read_text().splitlines()]
    assert [event["phase"] for event in events] == ["pre", "post"]
    assert events[1]["execution"]["returncode"] == 0


def test_host_mcp_executes_macos_shell_in_guest(monkeypatch):
    monkeypatch.setattr(host_computer_mcp, "CONTROLLER_PLATFORM", "Darwin")
    assert host_computer_mcp._shell_argv("pwd", "/Users/oai/My Files") == [
        "/bin/sh",
        "-lc",
        "cd -- '/Users/oai/My Files' && pwd",
    ]
