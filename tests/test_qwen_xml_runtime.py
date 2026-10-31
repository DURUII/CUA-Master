from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuamaster.action_space import ToolDefinition
from cuamaster.runtime import (
    CanonicalActionExecutor,
    CanonicalEpisodeRunner,
    DEFAULT_ACTION_PROTOCOL,
    ModelResponse,
    QwenXmlActionParser,
    ToolRegistry,
    action_to_qwen_xml_calls,
    format_qwen_xml_action,
)


def tool_call(function: str, **parameters: object) -> str:
    lines = ["<tool_call>", f"<function={function}>"]
    for name, value in parameters.items():
        rendered = (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else str(value)
        )
        lines.extend(
            [f"<parameter={name}>", rendered, "</parameter>"]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            tool_call("computer_use", action="left_click", coordinate=[999, 500]),
            {
                "type": "computer.click",
                "args": {"x": 1000, "y": 501, "button": "left"},
            },
        ),
        (
            tool_call("computer_use", action="key", keys=["CTRL", "C"]),
            {"type": "computer.press", "args": {"keys": ["ctrl", "c"]}},
        ),
        (
            tool_call("computer_use", action="key_down", keys=["SHIFT"]),
            {"type": "computer.key_down", "args": {"key": "shift"}},
        ),
        (
            tool_call("computer_use", action="triple_click", coordinate=[100, 200]),
            {
                "type": "computer.triple_click",
                "args": {"x": 100, "y": 200, "button": "left"},
            },
        ),
        (
            tool_call("computer_use", action="scroll", pixels=20),
            {"type": "computer.scroll", "args": {"dx": 0, "dy": -20.0}},
        ),
        (
            tool_call("computer_use", action="terminate", status="failed", text="blocked"),
            {"type": "finish", "args": {"status": "failure", "answer": "blocked"}},
        ),
    ],
)
def test_qwen_xml_parser_maps_computer_use(payload: str, expected: dict) -> None:
    assert QwenXmlActionParser().parse("Action: next\n\n" + payload).to_dict() == expected


def test_qwen_xml_parser_allows_angle_brackets_in_action_description() -> None:
    payload = tool_call("computer_use", action="left_click", coordinate=[100, 200])

    assert QwenXmlActionParser().parse(
        'Action: Click the "<number>" field.\n\n' + payload
    ).type == "computer.click"


def test_qwen_xml_parser_maps_drag_pair_to_one_canonical_action() -> None:
    payload = "\n".join(
        [
            tool_call("computer_use", action="mouse_move", coordinate=[100, 200]),
            tool_call("computer_use", action="left_click_drag", coordinate=[800, 900]),
        ]
    )

    assert QwenXmlActionParser().parse(payload).to_dict() == {
        "type": "computer.drag",
        "args": {
            "x1": 100,
            "y1": 200,
            "x2": 801,
            "y2": 901,
            "button": "left",
        },
    }


def test_qwen_xml_parser_combines_two_axis_scroll() -> None:
    payload = "\n".join(
        [
            tool_call("computer_use", action="scroll", pixels=-10),
            tool_call("computer_use", action="hscroll", pixels=5),
        ]
    )

    assert QwenXmlActionParser().parse(payload).to_dict() == {
        "type": "computer.scroll",
        "args": {"dx": -5.0, "dy": 10.0},
    }


def test_qwen_xml_parser_preserves_multiline_shell_command() -> None:
    command = "printf 'a\\nb\\n' | sed -n '2p'"
    payload = tool_call(
        "shell_exec", command=command, cwd="/tmp_workspace", timeout_sec=15
    )

    assert QwenXmlActionParser().parse(payload).to_dict() == {
        "type": "shell.exec",
        "args": {
            "command": command,
            "cwd": "/tmp_workspace",
            "timeout_sec": 15,
        },
    }


def test_qwen_xml_parser_maps_generic_and_direct_registered_tools() -> None:
    parser = QwenXmlActionParser(["filesystem.read_text"])
    generic = tool_call(
        "tool_call", name="filesystem.read_text", args={"path": "/tmp/a.txt"}
    )
    direct = tool_call(
        "filesystem_read_text", path="/tmp/a.txt", max_chars=100
    )

    assert parser.parse(generic).to_dict() == {
        "type": "tool.call",
        "args": {"name": "filesystem.read_text", "args": {"path": "/tmp/a.txt"}},
    }
    assert parser.parse(direct).to_dict() == {
        "type": "tool.call",
        "args": {
            "name": "filesystem.read_text",
            "args": {"path": "/tmp/a.txt", "max_chars": 100},
        },
    }


@pytest.mark.parametrize(
    "action",
    [
        {
            "type": "computer.triple_click",
            "args": {"x": 100, "y": 200, "button": "left"},
        },
        {"type": "computer.key_down", "args": {"key": "shift"}},
        {"type": "computer.mouse_up", "args": {"button": "left"}},
        {"type": "shell.exec", "args": {"command": "pwd", "cwd": "/tmp"}},
        {
            "type": "tool.call",
            "args": {"name": "filesystem.read_text", "args": {"path": "a.txt"}},
        },
        {"type": "finish", "args": {"status": "success", "answer": "done"}},
    ],
)
def test_qwen_xml_serializer_round_trips_canonical_actions(action: dict) -> None:
    parser = QwenXmlActionParser(["filesystem.read_text"])

    parsed = parser.parse("\n".join(action_to_qwen_xml_calls(action)))

    assert parsed.to_dict() == action


def test_qwen_xml_formatter_matches_training_shape() -> None:
    output = format_qwen_xml_action(
        {"type": "computer.click", "args": {"x": 500, "y": 500, "button": "left"}}
    )

    assert output.startswith("<think>\n\n</think>\n\nAction:")
    assert "<function=computer_use>" in output


@pytest.mark.parametrize(
    "payload",
    [
        tool_call("computer_use", action="call_user", text="Continue?"),
        tool_call("computer_use", action="left_click_drag", coordinate=[1, 2]),
        "\n".join(
            [
                tool_call("computer_use", action="left_click", coordinate=[1, 2]),
                tool_call("shell_exec", command="pwd"),
            ]
        ),
    ],
)
def test_qwen_xml_parser_rejects_unsupported_or_multiple_actions(payload: str) -> None:
    with pytest.raises(ValueError):
        QwenXmlActionParser().parse(payload)


def test_qwen_xml_parse_many_preserves_sequential_actions() -> None:
    parser = QwenXmlActionParser()
    payload = "\n".join(
        [
            tool_call("computer_use", action="type", text="echo ready"),
            tool_call("computer_use", action="key", keys=["return"]),
        ]
    )

    assert [action.to_dict() for action in parser.parse_many(payload)] == [
        {"type": "computer.type", "args": {"text": "echo ready"}},
        {"type": "computer.press", "args": {"keys": ["return"]}},
    ]


class XmlModel:
    model = "xml-model"

    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def complete(
        self, system_prompt: str, user_text: str, screenshot: bytes | None
    ) -> ModelResponse:
        self.system_prompt = system_prompt
        self.user_prompt = user_text
        return ModelResponse(
            tool_call("computer_use", action="terminate", status="success"),
            {},
        )


class XmlSequenceModel:
    model = "xml-sequence-model"

    def __init__(self) -> None:
        self.outputs = iter(
            [
                "\n".join(
                    [
                        tool_call("computer_use", action="type", text="echo ready"),
                        tool_call("computer_use", action="key", keys=["return"]),
                    ]
                ),
                tool_call("computer_use", action="terminate", status="success"),
            ]
        )

    def complete(
        self, system_prompt: str, user_text: str, screenshot: bytes | None
    ) -> ModelResponse:
        return ModelResponse(next(self.outputs), {})


class XmlEnvironment:
    def get_screenshot(self) -> bytes:
        return b"png"

    def get_screen_size(self) -> tuple[int, int]:
        return 1920, 1080

    def execute_gui(self, command: str, *, pause: float) -> dict:
        return {}

    def execute_shell(self, command: str, *, cwd: str, timeout_sec: int) -> dict:
        return {"returncode": 0}

    def finish(self, status: str) -> dict:
        return {"done": True}

    def get_platform(self) -> str:
        return "Linux"


def test_episode_runner_uses_qwen_xml_protocol(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "filesystem.read_text",
            "read text",
            {"type": "object"},
            {"type": "object"},
        ),
        lambda args, env: {},
    )
    model = XmlModel()
    result = CanonicalEpisodeRunner(
        model=model,
        tool_registry=registry,
        action_protocol="qwen_xml",
        max_steps=1,
    ).run("Complete task", XmlEnvironment(), tmp_path)

    assert result["final_status"] == "success"
    assert "<function=computer_use>" in model.system_prompt
    assert "filesystem_read_text" in model.system_prompt
    assert model.user_prompt.startswith("<image>\n")
    row = json.loads(
        (tmp_path / "canonical_trajectory.jsonl").read_text().splitlines()[0]
    )
    assert row["action"] == {"type": "finish", "args": {"status": "success"}}


def test_episode_runner_reads_qwen_xml_protocol_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUA_ACTION_PROTOCOL", "qwen_xml")
    model = XmlModel()

    result = CanonicalEpisodeRunner(model=model, max_steps=1).run(
        "Complete task", XmlEnvironment(), tmp_path
    )

    assert result["final_status"] == "success"
    assert "<function=computer_use>" in model.system_prompt


def test_episode_runner_defaults_to_qwen_xml_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUA_ACTION_PROTOCOL", raising=False)
    model = XmlModel()
    runner = CanonicalEpisodeRunner(model=model, max_steps=1)

    result = runner.run("Complete task", XmlEnvironment(), tmp_path)

    assert DEFAULT_ACTION_PROTOCOL == "qwen_xml"
    assert runner.action_protocol == DEFAULT_ACTION_PROTOCOL
    assert result["final_status"] == "success"


def test_episode_runner_executes_multiple_xml_calls_as_atomic_steps(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class Environment(XmlEnvironment):
        def execute_gui(self, command: str, *, pause: float) -> dict:
            calls.append(command)
            return {}

    result = CanonicalEpisodeRunner(
        model=XmlSequenceModel(), action_protocol="qwen_xml", max_steps=4
    ).run("Complete task", Environment(), tmp_path)

    assert result["final_status"] == "success"
    assert calls == [
        "pyautogui.write(message='echo ready')",
        "pyautogui.press('return')",
    ]
    rows = [
        json.loads(line)
        for line in (tmp_path / "canonical_trajectory.jsonl").read_text().splitlines()
    ]
    assert [row["action"]["type"] for row in rows] == [
        "computer.type",
        "computer.press",
        "finish",
    ]


def test_executor_routes_qwen_input_primitives() -> None:
    calls: list[str] = []

    class Environment(XmlEnvironment):
        def execute_gui(self, command: str, *, pause: float) -> dict:
            calls.append(command)
            return {}

    executor = CanonicalActionExecutor()
    parser = QwenXmlActionParser()
    environment = Environment()
    for payload in (
        tool_call("computer_use", action="triple_click", coordinate=[100, 200]),
        tool_call("computer_use", action="key_down", keys=["shift"]),
        tool_call("computer_use", action="left_mouse_up"),
    ):
        assert executor.execute(parser.parse(payload), environment).ok is True

    assert calls == [
        "pyautogui.tripleClick(x=192, y=216, button='left')",
        "pyautogui.keyDown('shift')",
        "pyautogui.mouseUp(button='left')",
    ]
