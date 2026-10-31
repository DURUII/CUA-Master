"""Benchmark-independent canonical action parsing and execution."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from cuamaster.action_space import CUAAction, CUAResponse, ToolDefinition, validate_action


_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_LINE_RE = re.compile(r"^\s*Code:\s*(.+?)\s*$", re.MULTILINE)
_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


@runtime_checkable
class RuntimeEnvironment(Protocol):
    """Minimal benchmark adapter required by the shared executor."""

    def get_screenshot(self) -> bytes | None: ...

    def get_screen_size(self) -> tuple[int, int]: ...

    def execute_gui(self, command: str, *, pause: float) -> dict[str, Any]: ...

    def execute_shell(
        self,
        command: str,
        *,
        cwd: str,
        timeout_sec: int,
    ) -> dict[str, Any]: ...

    def finish(self, status: str) -> dict[str, Any]: ...

    def get_platform(self) -> str: ...


@dataclass(frozen=True)
class ToolPathRoot:
    """One guest path that registered tools may access."""

    path: str
    read: bool = True
    write: bool = False


@runtime_checkable
class ToolCapabilityEnvironment(RuntimeEnvironment, Protocol):
    """Optional benchmark capabilities used by the core CUA tool pack."""

    def get_accessibility_tree(self) -> str | None: ...

    def get_tool_roots(self) -> list[ToolPathRoot]: ...


@dataclass
class CallbackEnvironment:
    """Create an environment adapter from benchmark-specific callbacks."""

    screenshot: Callable[[], bytes | None]
    screen_size: Callable[[], tuple[int, int]]
    gui: Callable[[str, float], dict[str, Any]]
    shell: Callable[[str, str, int], dict[str, Any]]
    terminate: Callable[[str], dict[str, Any]]
    platform: Callable[[], str] = lambda: "unknown"
    accessibility_tree: Callable[[], str | None] | None = None
    tool_roots: Callable[[], list[ToolPathRoot]] | None = None
    default_shell_cwd: Callable[[], str] = lambda: "/tmp_workspace"

    def get_screenshot(self) -> bytes | None:
        return self.screenshot()

    def get_screen_size(self) -> tuple[int, int]:
        return self.screen_size()

    def execute_gui(self, command: str, *, pause: float) -> dict[str, Any]:
        return self.gui(command, pause)

    def execute_shell(self, command: str, *, cwd: str, timeout_sec: int) -> dict[str, Any]:
        return self.shell(command, cwd, timeout_sec)

    def finish(self, status: str) -> dict[str, Any]:
        return self.terminate(status)

    def get_platform(self) -> str:
        return self.platform()

    def get_accessibility_tree(self) -> str | None:
        if self.accessibility_tree is None:
            return None
        return self.accessibility_tree()

    def get_tool_roots(self) -> list[ToolPathRoot]:
        if self.tool_roots is None:
            return []
        return self.tool_roots()

    def get_default_shell_cwd(self) -> str:
        return self.default_shell_cwd()


ToolHandler = Callable[[dict[str, Any], RuntimeEnvironment], Any]


class ToolCallError(RuntimeError):
    """Expected structured failure raised by a registered tool handler."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        _validate_tool_definition(definition)
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        if definition.name in self._definitions:
            raise ValueError(f"tool already registered: {definition.name}")
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def definitions(self) -> list[ToolDefinition]:
        return [self._definitions[name] for name in sorted(self._definitions)]

    def call(self, name: str, args: dict[str, Any], env: RuntimeEnvironment) -> CUAResponse:
        definition = self._definitions.get(name)
        if definition is None:
            return _error_response("unknown_tool", f"tool is not registered: {name}")
        try:
            _validate_json_schema(args, definition.input_schema, path="args")
        except (TypeError, ValueError) as exc:
            return _error_response("invalid_tool_args", str(exc))
        try:
            result = self._handlers[name](args, env)
            json.dumps(result, ensure_ascii=False)
            if definition.output_schema:
                _validate_json_schema(result, definition.output_schema, path="result")
        except ToolCallError as exc:
            return _error_response(exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001
            return _error_response("tool_handler_error", f"{type(exc).__name__}: {exc}")
        return CUAResponse(True, {"category": "tool", "name": name, "result": result})


class CanonicalActionParser:
    """Parse one action using JSON or a restricted literal-only AST."""

    def parse(self, text: str) -> CUAAction:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("model output is empty")
        payload = self._extract_payload(text)
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            action = self._parse_call(payload)
        else:
            if not isinstance(value, dict):
                raise ValueError("canonical JSON action must be an object")
            action = CUAAction.from_dict(value)
        validate_action(action)
        return action

    def _extract_payload(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        tagged = _ACTION_TAG_RE.findall(stripped)
        if tagged:
            if len(tagged) != 1:
                raise ValueError("model output must contain exactly one <action> block")
            outside = _ACTION_TAG_RE.sub("", _THINK_TAG_RE.sub("", stripped)).strip()
            if outside:
                raise ValueError("unexpected text outside <think>/<action> blocks")
            return tagged[0].strip()
        code_lines = _CODE_LINE_RE.findall(stripped)
        if code_lines:
            if len(code_lines) != 1:
                raise ValueError("model output must contain exactly one Code line")
            return code_lines[0].strip()
        return stripped

    def _parse_call(self, payload: str) -> CUAAction:
        try:
            expression = ast.parse(payload, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"invalid action syntax: {exc.msg}") from exc
        call = expression.body
        if not isinstance(call, ast.Call):
            raise ValueError("action payload must be one function call")
        if any(keyword.arg is None for keyword in call.keywords):
            raise ValueError("expanded keyword arguments are not allowed")
        positional = [_literal(node) for node in call.args]
        keywords = {keyword.arg: _literal(keyword.value) for keyword in call.keywords}
        return _call_to_action(_call_name(call.func), positional, keywords)


class CanonicalActionExecutor:
    def __init__(
        self,
        *,
        max_shell_timeout: int = 120,
        default_shell_timeout: int = 30,
        tool_registry: ToolRegistry | None = None,
        gui_pause: float = 0.2,
    ) -> None:
        if max_shell_timeout <= 0:
            raise ValueError("max_shell_timeout must be positive")
        self.max_shell_timeout = int(max_shell_timeout)
        self.default_shell_timeout = min(int(default_shell_timeout), self.max_shell_timeout)
        self.tool_registry = tool_registry or ToolRegistry()
        self.gui_pause = gui_pause

    def execute(self, action: CUAAction, env: RuntimeEnvironment) -> CUAResponse:
        try:
            validate_action(action)
            if action.type.startswith("computer."):
                return self._execute_gui(action, env)
            if action.type == "shell.exec":
                return self._execute_shell(action, env)
            if action.type == "tool.call":
                return self.tool_registry.call(action.args["name"], action.args["args"], env)
            result = env.finish(action.args["status"])
            return CUAResponse(
                True,
                {
                    "category": "control",
                    "status": action.args["status"],
                    "answer": action.args.get("answer"),
                    **_dict_result(result),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return _error_response("executor_error", f"{type(exc).__name__}: {exc}")

    def _execute_gui(self, action: CUAAction, env: RuntimeEnvironment) -> CUAResponse:
        width, height = _screen_size(env.get_screen_size())
        command = _gui_command(action, width, height)
        result = env.execute_gui(command, pause=0 if action.type == "computer.wait" else self.gui_pause)
        return CUAResponse(
            True,
            {
                "category": "gui",
                "command": command,
                "screen_size": {"width": width, "height": height},
                **_dict_result(result),
            },
        )

    def _execute_shell(self, action: CUAAction, env: RuntimeEnvironment) -> CUAResponse:
        requested = action.args.get("timeout_sec", self.default_shell_timeout)
        timeout = min(max(1, int(requested)), self.max_shell_timeout)
        cwd = action.args.get("cwd") or _default_shell_cwd(env)
        result = env.execute_shell(action.args["command"], cwd=cwd, timeout_sec=timeout) or {}
        stderr = str(result.get("stderr") or result.get("error") or "")
        returncode = result.get("returncode")
        timed_out = bool(result.get("timed_out")) or "timed out" in stderr.lower()
        return CUAResponse(
            returncode == 0 and not timed_out,
            {
                "category": "cli",
                "stdout": str(result.get("stdout") or result.get("output") or ""),
                "stderr": stderr,
                "returncode": returncode,
                "timed_out": timed_out,
                "cwd": cwd,
                "timeout_sec": timeout,
            },
        )


def load_tool_registry(path: str | Path | None) -> ToolRegistry:
    registry = ToolRegistry()
    if path is None:
        return registry
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    limits = config.get("limits") if isinstance(config, dict) else None
    profile = config.get("profile") if isinstance(config, dict) else None
    entries = config.get("tools") if isinstance(config, dict) else config
    disabled = config.get("disabled_tools", []) if isinstance(config, dict) else []
    from .tools import builtin_tool_catalog, profile_tool_names

    selected: list[Any] = []
    if profile is not None:
        selected.extend(profile_tool_names(profile))
    if entries is not None:
        if not isinstance(entries, list):
            raise ValueError("tool registry config tools must be a list")
        selected.extend(entries)
    if profile is None and entries is None:
        raise ValueError("tool registry config must contain a profile or tools list")
    if not isinstance(disabled, list) or not all(isinstance(name, str) for name in disabled):
        raise ValueError("disabled_tools must be a list of tool names")
    disabled_names = set(disabled)
    entries = selected
    if not isinstance(entries, list):
        raise ValueError("tool registry config must contain a tools list")
    builtins = builtin_tool_catalog(limits)
    seen: set[str] = set()
    for entry in entries:
        name = entry if isinstance(entry, str) else entry.get("name") if isinstance(entry, dict) else None
        if name in disabled_names or name in seen:
            continue
        if name not in builtins:
            raise ValueError(f"unknown built-in tool in registry config: {name!r}")
        definition, handler = builtins[name]
        registry.register(definition, handler)
        seen.add(name)
    return registry


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise ValueError("action function must be a simple dotted name")
    parts.append(node.id)
    return ".".join(reversed(parts))


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError) as exc:
        raise ValueError("action arguments must be literal JSON/Python values") from exc


def _call_to_action(name: str, positional: list[Any], kwargs: dict[str, Any]) -> CUAAction:
    if name in {
        "pyautogui.click",
        "pyautogui.doubleClick",
        "pyautogui.tripleClick",
        "pyautogui.moveTo",
    }:
        x = kwargs.pop("x", positional.pop(0) if positional else None)
        y = kwargs.pop("y", positional.pop(0) if positional else None)
        args: dict[str, Any] = {"x": x, "y": y}
        if name != "pyautogui.moveTo":
            args["button"] = kwargs.pop("button", "left")
        _no_extra(positional, kwargs)
        action_type = {
            "pyautogui.click": "computer.click",
            "pyautogui.doubleClick": "computer.double_click",
            "pyautogui.tripleClick": "computer.triple_click",
            "pyautogui.moveTo": "computer.move",
        }[name]
        return CUAAction(action_type, args)
    if name == "pyautogui.dragTo":
        args = {
            "x1": kwargs.pop("x1", kwargs.pop("start_x", None)),
            "y1": kwargs.pop("y1", kwargs.pop("start_y", None)),
            "x2": kwargs.pop("x2", kwargs.pop("x", positional.pop(0) if positional else None)),
            "y2": kwargs.pop("y2", kwargs.pop("y", positional.pop(0) if positional else None)),
            "button": kwargs.pop("button", "left"),
        }
        duration = kwargs.pop("duration_ms", None)
        if duration is None and "duration" in kwargs:
            duration = float(kwargs.pop("duration")) * 1000
        if duration is not None:
            args["duration_ms"] = duration
        _no_extra(positional, kwargs)
        return CUAAction("computer.drag", args)
    if name == "pyautogui.scroll":
        args = {
            "dx": kwargs.pop("dx", 0),
            "dy": kwargs.pop("dy", positional.pop(0) if positional else None),
        }
        for key in ("x", "y"):
            if key in kwargs:
                args[key] = kwargs.pop(key)
        _no_extra(positional, kwargs)
        return CUAAction("computer.scroll", args)
    if name == "pyautogui.write":
        text = kwargs.pop("message", kwargs.pop("text", positional.pop(0) if positional else None))
        _no_extra(positional, kwargs)
        return CUAAction("computer.type", {"text": text})
    if name == "pyautogui.press":
        key = positional.pop(0) if positional else kwargs.pop("key", None)
        _no_extra(positional, kwargs)
        return CUAAction("computer.press", {"keys": [key]})
    if name == "pyautogui.hotkey":
        keys = list(positional) if positional else kwargs.pop("keys", None)
        positional.clear()
        _no_extra(positional, kwargs)
        return CUAAction("computer.press", {"keys": keys})
    if name in {"pyautogui.keyDown", "pyautogui.keyUp"}:
        key = positional.pop(0) if positional else kwargs.pop("key", None)
        _no_extra(positional, kwargs)
        action_type = (
            "computer.key_down" if name == "pyautogui.keyDown" else "computer.key_up"
        )
        return CUAAction(action_type, {"key": key})
    if name in {"pyautogui.mouseDown", "pyautogui.mouseUp"}:
        args = {"button": kwargs.pop("button", "left")}
        x = kwargs.pop("x", positional.pop(0) if positional else None)
        y = kwargs.pop("y", positional.pop(0) if positional else None)
        if x is not None or y is not None:
            args.update({"x": x, "y": y})
        _no_extra(positional, kwargs)
        action_type = (
            "computer.mouse_down"
            if name == "pyautogui.mouseDown"
            else "computer.mouse_up"
        )
        return CUAAction(action_type, args)
    if name == "time.sleep":
        seconds = positional.pop(0) if positional else kwargs.pop("seconds", None)
        _no_extra(positional, kwargs)
        return CUAAction("computer.wait", {"seconds": seconds})
    if name == "shell.exec":
        args = {"command": kwargs.pop("command", positional.pop(0) if positional else None)}
        for key in ("cwd", "timeout_sec"):
            if key in kwargs:
                args[key] = kwargs.pop(key)
        _no_extra(positional, kwargs)
        return CUAAction("shell.exec", args)
    if name == "tool.call":
        tool_name = kwargs.pop("name", positional.pop(0) if positional else None)
        tool_args = kwargs.pop("args", positional.pop(0) if positional else None)
        _no_extra(positional, kwargs)
        return CUAAction("tool.call", {"name": tool_name, "args": tool_args})
    if name in {"computer.terminate", "finish"}:
        status = kwargs.pop("status", positional.pop(0) if positional else None)
        args = {"status": "failure" if status in {"failed", "fail"} else status}
        answer = kwargs.pop("answer", positional.pop(0) if positional else None)
        if answer is not None:
            args["answer"] = answer
        _no_extra(positional, kwargs)
        return CUAAction("finish", args)
    raise ValueError(f"unsupported action call: {name}")


def _no_extra(positional: list[Any], kwargs: dict[str, Any]) -> None:
    if positional or kwargs:
        raise ValueError("unexpected action arguments")


def _screen_size(value: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise ValueError(f"invalid screen size: {value!r}")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid screen size: {value!r}")
    return width, height


def _default_shell_cwd(env: RuntimeEnvironment) -> str:
    getter = getattr(env, "get_default_shell_cwd", None)
    value = getter() if callable(getter) else "/tmp_workspace"
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"invalid default shell cwd: {value!r}")
    return value


def _pixel(value: float, extent: int) -> int:
    return max(0, min(extent - 1, round(value * extent / 1000)))


def _gui_command(action: CUAAction, width: int, height: int) -> str:
    args = action.args
    if action.type in {
        "computer.click",
        "computer.double_click",
        "computer.triple_click",
        "computer.move",
    }:
        x, y = _pixel(args["x"], width), _pixel(args["y"], height)
        if action.type == "computer.move":
            return f"pyautogui.moveTo(x={x}, y={y})"
        function = {
            "computer.click": "click",
            "computer.double_click": "doubleClick",
            "computer.triple_click": "tripleClick",
        }[action.type]
        return f"pyautogui.{function}(x={x}, y={y}, button={args.get('button', 'left')!r})"
    if action.type == "computer.drag":
        x1, y1 = _pixel(args["x1"], width), _pixel(args["y1"], height)
        x2, y2 = _pixel(args["x2"], width), _pixel(args["y2"], height)
        duration = args.get("duration_ms", 500) / 1000
        return (
            f"pyautogui.moveTo(x={x1}, y={y1}); "
            f"pyautogui.dragTo(x={x2}, y={y2}, duration={duration}, "
            f"button={args.get('button', 'left')!r})"
        )
    if action.type == "computer.scroll":
        commands: list[str] = []
        if "x" in args and "y" in args:
            commands.append(
                f"pyautogui.moveTo(x={_pixel(args['x'], width)}, y={_pixel(args['y'], height)})"
            )
        if args["dx"]:
            commands.append(f"pyautogui.hscroll({args['dx']!r})")
        if args["dy"]:
            commands.append(f"pyautogui.scroll({args['dy']!r})")
        return "; ".join(commands) or "pyautogui.scroll(0)"
    if action.type == "computer.type":
        return f"pyautogui.write(message={args['text']!r})"
    if action.type == "computer.wait":
        return f"time.sleep({args['seconds']!r})"
    if action.type in {"computer.key_down", "computer.key_up"}:
        function = "keyDown" if action.type == "computer.key_down" else "keyUp"
        return f"pyautogui.{function}({args['key']!r})"
    if action.type in {"computer.mouse_down", "computer.mouse_up"}:
        function = "mouseDown" if action.type == "computer.mouse_down" else "mouseUp"
        parameters = []
        if "x" in args and "y" in args:
            parameters.extend(
                (f"x={_pixel(args['x'], width)}", f"y={_pixel(args['y'], height)}")
            )
        parameters.append(f"button={args.get('button', 'left')!r}")
        return f"pyautogui.{function}({', '.join(parameters)})"
    keys = args["keys"]
    if len(keys) == 1:
        return f"pyautogui.press({keys[0]!r})"
    return f"pyautogui.hotkey({', '.join(repr(key) for key in keys)})"


def _dict_result(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"result": value}


def _error_response(code: str, message: str) -> CUAResponse:
    return CUAResponse(False, {"error": {"code": code, "message": message}})


def _validate_tool_definition(definition: ToolDefinition) -> None:
    if not isinstance(definition, ToolDefinition):
        raise TypeError("definition must be ToolDefinition")
    if not _TOOL_NAME_RE.fullmatch(definition.name):
        raise ValueError(f"invalid tool name: {definition.name!r}")
    if not isinstance(definition.input_schema, dict) or not isinstance(definition.output_schema, dict):
        raise TypeError("tool schemas must be objects")


def _validate_json_schema(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not schema:
        return
    expected = schema.get("type")
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected in checks and not checks[expected](value):
        raise TypeError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ValueError(f"{path} is shorter than minLength={schema['minLength']}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ValueError(f"{path} is longer than maxLength={schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} must be <= {schema['maximum']}")
    if expected == "object":
        for key in schema.get("required") or []:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValueError(f"{path} has unknown properties: {', '.join(unknown)}")
        for key, item in value.items():
            if key in properties:
                _validate_json_schema(item, properties[key], path=f"{path}.{key}")
    if expected == "array" and "items" in schema:
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ValueError(f"{path} has fewer than minItems={schema['minItems']}")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path} has more than maxItems={schema['maxItems']}")
        for index, item in enumerate(value):
            _validate_json_schema(item, schema["items"], path=f"{path}[{index}]")
