"""Shared model loop and trajectory logging for canonical CUA episodes."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from cuamaster.action_space import CUAResponse

from .core import (
    CanonicalActionExecutor,
    CanonicalActionParser,
    RuntimeEnvironment,
    ToolRegistry,
)
from .qwen_xml import (
    ACTION_PROTOCOL_CANONICAL,
    ACTION_PROTOCOL_QWEN_XML,
    DEFAULT_ACTION_PROTOCOL,
    SUPPORTED_ACTION_PROTOCOLS,
    CompatibleActionParser,
    QWEN_XML_SYSTEM_PROMPT,
    QwenXmlActionParser,
    qwen_xml_registered_tools_prompt,
)


DEFAULT_SYSTEM_PROMPT = """You are a computer-use agent operating one active environment.
Choose exactly one atomic action per turn. Coordinates use normalized 0-1000 space.

Allowed calls:
- pyautogui.click(x=..., y=..., button='left')
- pyautogui.doubleClick(x=..., y=..., button='left')
- pyautogui.tripleClick(x=..., y=..., button='left')
- pyautogui.moveTo(x=..., y=...)
- pyautogui.dragTo(x1=..., y1=..., x2=..., y2=..., button='left', duration_ms=500)
- pyautogui.scroll(dx=..., dy=..., x=..., y=...), where dx and dy are each in [-50, 50]
- pyautogui.write(message='...')
- pyautogui.press('enter') or pyautogui.hotkey('ctrl', 'c')
- pyautogui.keyDown('shift') or pyautogui.keyUp('shift')
- pyautogui.mouseDown(x=..., y=..., button='left') or pyautogui.mouseUp(...)
- time.sleep(seconds)
- shell.exec(command='...', cwd='/tmp_workspace', timeout_sec=30)
- tool.call(name='registered.name', args={...})
- computer.terminate(status='success'|'failure', answer='optional')

Return one canonical JSON object or one Code line containing one function call.
Never return multiple actions, imports, assignments, or executable statements.
"""


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: dict[str, Any]


class ModelClient(Protocol):
    model: str

    def complete(
        self, system_prompt: str, user_text: str, screenshot: bytes | None
    ) -> ModelResponse: ...


class OpenAICompatibleModel:
    """Small optional adapter for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: int = 7200,
        max_output_tokens: int = 1000,
        temperature: float | None = 0.0,
        top_p: float | None = 1.0,
        enable_thinking: bool | None = False,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAICompatibleModel requires `pip install cuamaster[annotator]`"
                ) from exc
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking

    def complete(
        self, system_prompt: str, user_text: str, screenshot: bytes | None
    ) -> ModelResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if screenshot:
            encoded = base64.b64encode(screenshot).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if self.enable_thinking is not None and "qwen" in self.model.lower():
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        response = self.client.chat.completions.create(**request)
        return ModelResponse(_response_text(response), _response_usage(response))


class CanonicalEpisodeRunner:
    def __init__(
        self,
        *,
        model: ModelClient,
        max_steps: int = 200,
        timeout: int = 7200,
        max_shell_timeout: int = 120,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        action_protocol: str | None = None,
        model_retries: int = 2,
        model_retry_delay: float = 1.0,
    ) -> None:
        if model_retries < 0:
            raise ValueError("model_retries must be non-negative")
        if model_retry_delay < 0:
            raise ValueError("model_retry_delay must be non-negative")
        resolved_action_protocol = action_protocol or os.environ.get(
            "CUA_ACTION_PROTOCOL", DEFAULT_ACTION_PROTOCOL
        )
        if resolved_action_protocol not in SUPPORTED_ACTION_PROTOCOLS:
            raise ValueError("action_protocol must be canonical, qwen_xml, or auto")
        self.model = model
        self.max_steps = max_steps
        self.timeout = timeout
        self.model_retries = model_retries
        self.model_retry_delay = model_retry_delay
        self.executor = CanonicalActionExecutor(
            max_shell_timeout=max_shell_timeout,
            tool_registry=tool_registry,
        )
        self.tool_registry = self.executor.tool_registry
        self.action_protocol = resolved_action_protocol
        canonical_parser = CanonicalActionParser()
        qwen_parser = QwenXmlActionParser(
            definition.name for definition in self.tool_registry.definitions()
        )
        self.parser = {
            ACTION_PROTOCOL_CANONICAL: canonical_parser,
            ACTION_PROTOCOL_QWEN_XML: qwen_parser,
            "auto": CompatibleActionParser(canonical_parser, qwen_parser),
        }[resolved_action_protocol]
        self.system_prompt = system_prompt or (
            DEFAULT_SYSTEM_PROMPT
            if resolved_action_protocol == ACTION_PROTOCOL_CANONICAL
            else QWEN_XML_SYSTEM_PROMPT
        )

    def run(
        self,
        instruction: str,
        env: RuntimeEnvironment,
        output_dir: str | Path,
        *,
        system_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshots_dir = output_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_dir / "canonical_trajectory.jsonl"
        chat_path = output_dir / "chat.jsonl"
        history: list[dict[str, Any]] = []
        started = time.monotonic()
        end_reason = "max_steps"
        final_status = "failure"
        retries_used = 0
        screenshot = env.get_screenshot()
        screenshot_path = self._save_screenshot(
            screenshots_dir / "step_0000.png", screenshot
        )
        system_prompt = self.system_prompt
        if system_prompt_override:
            system_prompt += "\nEnvironment policy:\n" + system_prompt_override
        shell_cwd_getter = getattr(env, "get_default_shell_cwd", None)
        shell_cwd = (
            shell_cwd_getter() if callable(shell_cwd_getter) else "/tmp_workspace"
        )
        if shell_cwd != "/tmp_workspace":
            system_prompt += (
                "\nRuntime environment:\n"
                f"- The default shell working directory is {shell_cwd!r}. "
                "Use it instead of /tmp_workspace unless that path has been confirmed to exist."
            )
        tool_prompt = self._tool_prompt()
        if tool_prompt:
            if self.action_protocol == ACTION_PROTOCOL_CANONICAL:
                system_prompt += "\nRegistered tools:\n" + tool_prompt
            else:
                system_prompt += "\n" + tool_prompt

        with (
            trajectory_path.open("w", encoding="utf-8") as trajectory_file,
            chat_path.open("w", encoding="utf-8") as chat_file,
        ):
            while len(history) < self.max_steps:
                if time.monotonic() - started >= self.timeout:
                    end_reason = "agent_timeout"
                    break
                user_text = self._model_user_prompt(instruction, history)
                self._write_chat(chat_file, "user", user_text)
                model_response = None
                model_error = None
                for attempt in range(self.model_retries + 1):
                    try:
                        model_response = self.model.complete(
                            system_prompt, user_text, screenshot
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        model_error = exc
                        if attempt >= self.model_retries:
                            break
                        retries_used += 1
                        self._write_model_error(chat_file, exc, attempt + 1)
                        if self.model_retry_delay:
                            time.sleep(self.model_retry_delay)
                if model_response is None:
                    assert model_error is not None
                    exc = model_error
                    result = _runtime_error("model_error", exc)
                    step_index = len(history) + 1
                    self._write_step(
                        trajectory_file,
                        step_index,
                        raw_text=None,
                        action=None,
                        result=result,
                        pre_screenshot_path=screenshot_path,
                        post_screenshot_path=None,
                        end_reason="model_error",
                    )
                    end_reason = "model_error"
                    break
                raw_text = model_response.text
                self._write_chat(
                    chat_file, "assistant", raw_text, usage=model_response.usage
                )

                try:
                    parse_many = getattr(self.parser, "parse_many", None)
                    actions = (
                        parse_many(raw_text)
                        if callable(parse_many)
                        else [self.parser.parse(raw_text)]
                    )
                except Exception as exc:  # noqa: BLE001
                    result = _runtime_error("parse_error", exc)
                    actions = []
                    step_index = len(history) + 1
                    post_screenshot = env.get_screenshot()
                    post_screenshot_path = self._save_screenshot(
                        screenshots_dir / f"step_{step_index:04d}.png", post_screenshot
                    )
                    self._write_step(
                        trajectory_file,
                        step_index,
                        raw_text=raw_text,
                        action=None,
                        result=result,
                        pre_screenshot_path=screenshot_path,
                        post_screenshot_path=post_screenshot_path,
                        end_reason=None,
                    )
                    screenshot = post_screenshot
                    screenshot_path = post_screenshot_path
                    history.append(
                        {
                            "step": step_index,
                            "action": None,
                            "observation": result.to_dict(),
                        }
                    )
                    continue

                terminal_reason = None
                for action in actions:
                    if len(history) >= self.max_steps:
                        break
                    step_index = len(history) + 1
                    result = self.executor.execute(action, env)
                    post_screenshot = env.get_screenshot()
                    post_screenshot_path = self._save_screenshot(
                        screenshots_dir / f"step_{step_index:04d}.png", post_screenshot
                    )
                    if action.type == "finish" and result.ok:
                        final_status = action.args["status"]
                        terminal_reason = f"finish_{final_status}"
                        end_reason = terminal_reason
                    self._write_step(
                        trajectory_file,
                        step_index,
                        raw_text=raw_text,
                        action=action.to_dict(),
                        result=result,
                        pre_screenshot_path=screenshot_path,
                        post_screenshot_path=post_screenshot_path,
                        end_reason=terminal_reason,
                    )
                    screenshot = post_screenshot
                    screenshot_path = post_screenshot_path
                    history.append(
                        {
                            "step": step_index,
                            "action": action.to_dict(),
                            "observation": result.to_dict(),
                        }
                    )
                    if terminal_reason or not result.ok:
                        break
                if terminal_reason:
                    break

        if not end_reason.startswith("finish_"):
            try:
                env.finish("failure")
            except Exception:
                pass
        elapsed = time.monotonic() - started
        (output_dir / "agent.log").write_text(
            "\n".join(
                [
                    "AGENT_EXIT=0",
                    f"RETRIES_USED={retries_used}",
                    f"END_REASON={end_reason}",
                    f"FINAL_STATUS={final_status}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "agent_done": True,
            "elapsed_seconds": elapsed,
            "end_reason": end_reason,
            "final_status": final_status,
            "steps": len(history),
        }

    def _tool_prompt(self) -> str:
        if self.action_protocol != ACTION_PROTOCOL_CANONICAL:
            return qwen_xml_registered_tools_prompt(self.tool_registry.definitions())
        return "\n".join(
            f"- {definition.name} "
            f"[{'read-only' if definition.read_only else 'writes state'}"
            f"{'/destructive' if definition.destructive else ''}]: "
            f"{definition.description}; "
            f"args schema={json.dumps(definition.input_schema, ensure_ascii=False, sort_keys=True)}"
            for definition in self.tool_registry.definitions()
        )

    @staticmethod
    def _user_prompt(instruction: str, history: list[dict[str, Any]]) -> str:
        recent = history[-12:]
        history_text = json.dumps(recent, ensure_ascii=False, default=str)
        while recent and len(history_text) > 16000:
            recent = recent[1:]
            history_text = json.dumps(recent, ensure_ascii=False, default=str)
        return (
            f"Task: {instruction}\n"
            f"Previous canonical actions and observations: {history_text if recent else 'None'}\n"
            "The attached image is the latest screenshot. Return the next single action."
        )

    def _model_user_prompt(
        self, instruction: str, history: list[dict[str, Any]]
    ) -> str:
        if self.action_protocol == ACTION_PROTOCOL_CANONICAL:
            return self._user_prompt(instruction, history)
        recent = history[-4:]
        history_text = json.dumps(recent, ensure_ascii=False, default=str)
        while recent and len(history_text) > 8000:
            recent = recent[1:]
            history_text = json.dumps(recent, ensure_ascii=False, default=str)
        return (
            "<image>\n"
            "Please generate the next move according to the UI screenshot, instruction, "
            "and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            "Previous canonical actions and observations: "
            f"{history_text if recent else 'None'}\n\n"
            "Return the next single action using the Qwen XML tool-call format."
        )

    def _write_model_error(self, stream: Any, exc: Exception, attempt: int) -> None:
        stream.write(
            json.dumps(
                {
                    "type": "model_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "model": self.model.model,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stream.flush()

    def _write_chat(
        self,
        stream: Any,
        role: str,
        content: str,
        *,
        usage: dict[str, Any] | None = None,
    ) -> None:
        message: dict[str, Any] = {"role": role, "content": content}
        if usage is not None:
            message["usage"] = usage
            message["model"] = self.model.model
        stream.write(
            json.dumps(
                {
                    "type": "message",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": message,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stream.flush()

    @staticmethod
    def _write_step(
        stream: Any,
        step_index: int,
        *,
        raw_text: str | None,
        action: dict[str, Any] | None,
        result: CUAResponse,
        pre_screenshot_path: str | None,
        post_screenshot_path: str | None,
        end_reason: str | None,
    ) -> None:
        stream.write(
            json.dumps(
                {
                    "step": step_index,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw_model_text": raw_text,
                    "action": action,
                    "execution_result": result.to_dict(),
                    "pre_screenshot_path": pre_screenshot_path,
                    "post_screenshot_path": post_screenshot_path,
                    "screenshot_path": post_screenshot_path,
                    "end_reason": end_reason,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stream.flush()

    @staticmethod
    def _save_screenshot(path: Path, screenshot: bytes | None) -> str | None:
        if not screenshot:
            return None
        path.write_bytes(screenshot)
        return str(path)


def _runtime_error(code: str, exc: Exception) -> CUAResponse:
    return CUAResponse(
        False,
        {"error": {"code": code, "message": f"{type(exc).__name__}: {exc}"}},
    )


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("model response has no choices")
    choice = choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                texts.append(item.text)
        if texts:
            return "\n".join(texts)
    reasoning = getattr(message, "reasoning_content", None)
    reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
    finish_reason = getattr(choice, "finish_reason", None)
    raise ValueError(
        "model response content is empty "
        f"(finish_reason={finish_reason!r}, reasoning_chars={reasoning_chars})"
    )


def _response_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", prompt + completion) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    return {
        "input": prompt,
        "output": completion,
        "cacheRead": 0,
        "cacheWrite": 0,
        "reasoningTokens": reasoning,
        "totalTokens": total,
    }
