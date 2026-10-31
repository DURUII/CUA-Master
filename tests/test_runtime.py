from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuamaster.action_space import CUAAction, ToolDefinition
from cuamaster.runtime import (
    CallbackEnvironment,
    CanonicalActionExecutor,
    CanonicalActionParser,
    CanonicalEpisodeRunner,
    ModelResponse,
    OpenAICompatibleModel,
    ToolRegistry,
)


class FakeEnvironment:
    def __init__(self) -> None:
        self.gui_calls: list[tuple[str, float]] = []
        self.shell_calls: list[tuple[str, str, int]] = []
        self.finish_calls: list[str] = []
        self.shell_result = {"stdout": "ok\n", "stderr": "", "returncode": 0}

    def get_screenshot(self) -> bytes:
        return b"png"

    def get_screen_size(self) -> tuple[int, int]:
        return 1920, 1080

    def execute_gui(self, command: str, *, pause: float):
        self.gui_calls.append((command, pause))
        return {"done": False}

    def execute_shell(self, command: str, *, cwd: str, timeout_sec: int):
        self.shell_calls.append((command, cwd, timeout_sec))
        return self.shell_result

    def finish(self, status: str):
        self.finish_calls.append(status)
        return {"done": True}

    def get_platform(self) -> str:
        return "Linux"


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        (
            "<think>target</think><action>pyautogui.click(x=500, y=250)</action>",
            "computer.click",
        ),
        ("Code: shell.exec(command='pwd')", "shell.exec"),
        ("tool.call(name='lookup', args={'q': 'x'})", "tool.call"),
        ('{"type":"finish","args":{"status":"success"}}', "finish"),
    ],
)
def test_parser_supports_training_and_json_formats(
    text: str, expected_type: str
) -> None:
    assert CanonicalActionParser().parse(text).type == expected_type


@pytest.mark.parametrize(
    "text",
    [
        "pyautogui.click(x=1, y=2); shell.exec(command='id')",
        "Code: pyautogui.click(x=1, y=2)\nCode: shell.exec(command='id')",
        "__import__('os').system('id')",
    ],
)
def test_parser_rejects_multiple_actions_and_injection(text: str) -> None:
    with pytest.raises(ValueError):
        CanonicalActionParser().parse(text)


def test_parser_rejects_excessive_scroll() -> None:
    with pytest.raises(ValueError, match="dy must be between -50 and 50"):
        CanonicalActionParser().parse("pyautogui.scroll(5000000000000000000000)")


def test_executor_denormalizes_gui_and_routes_shell() -> None:
    env = FakeEnvironment()
    executor = CanonicalActionExecutor(max_shell_timeout=5)

    click = executor.execute(CUAAction("computer.click", {"x": 1000, "y": 500}), env)
    shell = executor.execute(
        CUAAction(
            "shell.exec", {"command": "pwd", "cwd": "/tmp/demo", "timeout_sec": 20}
        ),
        env,
    )

    assert click.ok is True
    assert env.gui_calls == [("pyautogui.click(x=1919, y=540, button='left')", 0.2)]
    assert shell.ok is True
    assert env.shell_calls == [("pwd", "/tmp/demo", 5)]
    assert shell.observation["stdout"] == "ok\n"


def test_callback_environment_is_the_only_benchmark_glue_required() -> None:
    calls: list[tuple[str, object]] = []
    env = CallbackEnvironment(
        screenshot=lambda: b"png",
        screen_size=lambda: (100, 50),
        gui=lambda command, pause: calls.append((command, pause)) or {},
        shell=lambda command, cwd, timeout: {},
        terminate=lambda status: calls.append(("finish", status)) or {},
    )

    CanonicalActionExecutor().execute(CUAAction("finish", {"status": "success"}), env)

    assert calls == [("finish", "success")]


def test_tool_registry_is_strict_and_structured() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "lookup",
            "look up a value",
            {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            },
            {"type": "object"},
        ),
        lambda args, env: {"value": args["query"].upper()},
    )
    executor = CanonicalActionExecutor(tool_registry=registry)

    success = executor.execute(
        CUAAction("tool.call", {"name": "lookup", "args": {"query": "cua"}}),
        FakeEnvironment(),
    )
    invalid = executor.execute(
        CUAAction("tool.call", {"name": "lookup", "args": {"query": 3}}),
        FakeEnvironment(),
    )
    unknown = executor.execute(
        CUAAction("tool.call", {"name": "missing", "args": {}}),
        FakeEnvironment(),
    )

    assert success.observation["result"] == {"value": "CUA"}
    assert invalid.observation["error"]["code"] == "invalid_tool_args"
    assert unknown.observation["error"]["code"] == "unknown_tool"


class FakeModel:
    model = "fake-model"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    def complete(
        self, system_prompt: str, user_text: str, screenshot: bytes | None
    ) -> ModelResponse:
        self.system_prompts.append(system_prompt)
        self.prompts.append(user_text)
        return ModelResponse(
            next(self.outputs), {"input": 1, "output": 1, "totalTokens": 2}
        )


class FlakyModel:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, system_prompt: str, user_text: str, screenshot: bytes | None
    ) -> ModelResponse:
        self.calls += 1
        if self.calls <= 2:
            raise ValueError("model response content is empty")
        return ModelResponse(
            '{"type":"finish","args":{"status":"success"}}',
            {"input": 1, "output": 1, "totalTokens": 2},
        )


class FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return self.response


class FakeOpenAIClient:
    def __init__(self, response: object) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


def test_episode_runner_handles_gui_cli_tool_and_finish(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition("lookup", "lookup", {"type": "object"}, {"type": "object"}),
        lambda args, env: {"found": True},
    )
    model = FakeModel(
        [
            "Code: pyautogui.click(x=100, y=200)",
            "Code: shell.exec(command='pwd')",
            "Code: tool.call(name='lookup', args={})",
            "Code: computer.terminate(status='success')",
        ]
    )
    env = FakeEnvironment()
    runner = CanonicalEpisodeRunner(
        model=model,
        tool_registry=registry,
        max_steps=8,
        action_protocol="canonical",
    )

    result = runner.run("Complete task", env, tmp_path)

    assert result["final_status"] == "success"
    assert env.shell_calls == [("pwd", "/tmp_workspace", 30)]
    assert env.finish_calls == ["success"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "canonical_trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["action"]["type"] for row in rows] == [
        "computer.click",
        "shell.exec",
        "tool.call",
        "finish",
    ]
    assert rows[0]["pre_screenshot_path"].endswith("screenshots/step_0000.png")
    assert rows[0]["post_screenshot_path"].endswith("screenshots/step_0001.png")
    assert rows[0]["screenshot_path"] == rows[0]["post_screenshot_path"]
    assert rows[1]["pre_screenshot_path"] == rows[0]["post_screenshot_path"]
    assert '"stdout": "ok\\n"' in model.prompts[2]


def test_openai_model_disables_qwen_thinking_and_passes_sampling_options() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"type":"finish","args":{"status":"success"}}'
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    client = FakeOpenAIClient(response)
    model = OpenAICompatibleModel(
        model="qwen36-27b",
        api_key="empty",
        base_url="http://localhost/v1",
        temperature=0.2,
        top_p=0.9,
        enable_thinking=False,
        client=client,
    )

    result = model.complete("system", "user", b"png")

    assert result.text.startswith("{")
    assert client.chat.completions.requests == [
        {
            "model": "qwen36-27b",
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "user"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,cG5n"},
                        },
                    ],
                },
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    ]


def test_episode_runner_retries_model_errors(tmp_path: Path) -> None:
    model = FlakyModel()
    env = FakeEnvironment()
    runner = CanonicalEpisodeRunner(
        model=model,
        max_steps=1,
        action_protocol="canonical",
        model_retries=2,
        model_retry_delay=0,
    )

    result = runner.run("Complete task", env, tmp_path)

    assert result["final_status"] == "success"
    assert model.calls == 3
    assert "RETRIES_USED=2" in (tmp_path / "agent.log").read_text(encoding="utf-8")
    chat_rows = [
        json.loads(line)
        for line in (tmp_path / "chat.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["type"] for row in chat_rows] == [
        "message",
        "model_error",
        "model_error",
        "message",
    ]


def test_user_prompt_truncates_only_at_history_boundaries() -> None:
    history = [
        {"step": step, "observation": {"stdout": "x" * 4000}} for step in range(12)
    ]

    prompt = CanonicalEpisodeRunner._user_prompt("Complete task", history)
    history_text = prompt.split("Previous canonical actions and observations: ", 1)[
        1
    ].split("\nThe attached image", 1)[0]

    parsed = json.loads(history_text)
    assert len(history_text) <= 16000
    assert parsed
    assert parsed[-1]["step"] == 11


def test_episode_runner_reports_nonstandard_shell_cwd(tmp_path: Path) -> None:
    class DesktopEnvironment(FakeEnvironment):
        def get_default_shell_cwd(self) -> str:
            return "/home/user/Desktop"

    model = FakeModel(['{"type":"finish","args":{"status":"success"}}'])
    runner = CanonicalEpisodeRunner(
        model=model, max_steps=1, action_protocol="canonical"
    )

    runner.run("Complete task", DesktopEnvironment(), tmp_path)

    assert (
        "default shell working directory is '/home/user/Desktop'"
        in model.system_prompts[0]
    )
    assert "instead of /tmp_workspace" in model.system_prompts[0]
