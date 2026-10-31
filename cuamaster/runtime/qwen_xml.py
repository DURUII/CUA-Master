"""Qwen-style XML tool-call protocol for canonical CUA actions."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from cuamaster.action_space import (
    MAX_SCROLL_DELTA,
    CUAAction,
    ToolDefinition,
    validate_action,
)


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z][A-Za-z0-9_.-]{0,127})>"
    r"(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z][A-Za-z0-9_.-]{0,127})>"
    r"(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FUNCTION_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")

ACTION_PROTOCOL_CANONICAL = "canonical"
ACTION_PROTOCOL_QWEN_XML = "qwen_xml"
ACTION_PROTOCOL_AUTO = "auto"
SUPPORTED_ACTION_PROTOCOLS = frozenset(
    {ACTION_PROTOCOL_CANONICAL, ACTION_PROTOCOL_QWEN_XML, ACTION_PROTOCOL_AUTO}
)
DEFAULT_ACTION_PROTOCOL = ACTION_PROTOCOL_QWEN_XML


QWEN_XML_SYSTEM_PROMPT = """You are a computer-use agent operating one active environment.
Choose exactly one atomic action per turn. Coordinates use normalized 0-999 space.

Available XML functions:
- computer_use(action=..., ...): GUI input, waiting, screenshots, and termination.
- shell_exec(command=..., cwd=..., timeout_sec=...): execute one guest-shell command.
- tool_call(name=..., args=...): call one registered structured tool.

computer_use actions:
- key(keys=[...])
- key_down(keys=[...]) and key_up(keys=[...])
- type(text=...)
- mouse_move(coordinate=[x,y])
- left_click/right_click/middle_click(coordinate=[x,y])
- double_click/triple_click(coordinate=[x,y])
- left_mouse_down/left_mouse_up(coordinate=[x,y] optional)
- left_click_drag(coordinate=[x,y]); emit it immediately after mouse_move in the same response
- scroll(pixels=...) and hscroll(pixels=...), each limited to [-50, 50]
- screenshot
- wait(time=...)
- terminate(status=success|failure, text=... optional)

If you call a function, respond with an optional short Action line followed by the call and no suffix:

<tool_call>
<function=computer_use>
<parameter=action>
left_click
</parameter>
<parameter=coordinate>
[500, 500]
</parameter>
</function>
</tool_call>

For a drag, exactly two calls are allowed: mouse_move followed by left_click_drag.
For simultaneous two-axis scrolling, scroll and hscroll may be emitted together.
Do not emit multiple unrelated actions, executable Python, or canonical JSON.
"""


@dataclass(frozen=True)
class _XmlToolCall:
    function: str
    parameters: dict[str, str]


def qwen_coordinate(value: Any) -> int:
    """Convert canonical 0-1000 coordinates to Qwen's 0-999 grid."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("canonical coordinates must be numbers")
    coordinate = float(value)
    if not math.isfinite(coordinate) or not 0 <= coordinate <= 1000:
        raise ValueError("canonical coordinates must be between 0 and 1000")
    return max(0, min(999, round(coordinate * 999 / 1000)))


def _xml_parameter(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _xml_tool_call(function: str, parameters: dict[str, Any]) -> str:
    lines = ["<tool_call>", f"<function={function}>"]
    for name, value in parameters.items():
        lines.extend(
            [f"<parameter={name}>", _xml_parameter(value), "</parameter>"]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def _computer_call(action: str, **parameters: Any) -> str:
    return _xml_tool_call("computer_use", {"action": action, **parameters})


def _bounded_scroll(value: Any, max_scroll_delta: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("scroll deltas must be numbers")
    delta = float(value or 0)
    if not math.isfinite(delta):
        raise ValueError("scroll deltas must be finite")
    return max(-max_scroll_delta, min(max_scroll_delta, delta))


def _compact_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def action_to_qwen_xml_calls(
    action: CUAAction | dict[str, Any],
    *,
    max_scroll_delta: float = MAX_SCROLL_DELTA,
) -> list[str]:
    """Serialize one canonical action using the shared Qwen XML protocol."""

    if (
        isinstance(max_scroll_delta, bool)
        or not isinstance(max_scroll_delta, (int, float))
        or not math.isfinite(max_scroll_delta)
        or not 0 < max_scroll_delta <= MAX_SCROLL_DELTA
    ):
        raise ValueError(
            f"max_scroll_delta must be between 0 and {MAX_SCROLL_DELTA:g}"
        )
    raw_action = action.to_dict() if isinstance(action, CUAAction) else action
    if not isinstance(raw_action, dict):
        raise TypeError("action must be a CUAAction or dict")
    normalized_action = dict(raw_action)
    normalized_args = dict(normalized_action.get("args") or {})
    if normalized_action.get("type") == "computer.scroll":
        normalized_args["dx"] = _bounded_scroll(
            normalized_args.get("dx"), max_scroll_delta
        )
        normalized_args["dy"] = _bounded_scroll(
            normalized_args.get("dy"), max_scroll_delta
        )
        normalized_action["args"] = normalized_args
    canonical = CUAAction.from_dict(normalized_action)
    validate_action(canonical)
    action_type = canonical.type
    args = canonical.args
    if action_type in {
        "computer.click",
        "computer.double_click",
        "computer.triple_click",
        "computer.move",
    }:
        button = str(args.get("button") or "left")
        if action_type in {"computer.double_click", "computer.triple_click"} and button != "left":
            raise ValueError(f"{action_type} only supports the left button in Qwen XML")
        qwen_action = {
            "computer.click": {
                "left": "left_click",
                "right": "right_click",
                "middle": "middle_click",
            }[button],
            "computer.double_click": "double_click",
            "computer.triple_click": "triple_click",
            "computer.move": "mouse_move",
        }[action_type]
        return [
            _computer_call(
                qwen_action,
                coordinate=[qwen_coordinate(args["x"]), qwen_coordinate(args["y"])],
            )
        ]
    if action_type == "computer.drag":
        if str(args.get("button") or "left") != "left":
            raise ValueError("Qwen XML drag only supports the left button")
        return [
            _computer_call(
                "mouse_move",
                coordinate=[qwen_coordinate(args["x1"]), qwen_coordinate(args["y1"])],
            ),
            _computer_call(
                "left_click_drag",
                coordinate=[qwen_coordinate(args["x2"]), qwen_coordinate(args["y2"])],
            ),
        ]
    if action_type == "computer.type":
        if not args["text"]:
            raise ValueError("Qwen XML type text must not be empty")
        return [_computer_call("type", text=args["text"])]
    if action_type == "computer.press":
        return [_computer_call("key", keys=[str(key).lower() for key in args["keys"]])]
    if action_type in {"computer.key_down", "computer.key_up"}:
        qwen_action = "key_down" if action_type.endswith("key_down") else "key_up"
        return [_computer_call(qwen_action, keys=[str(args["key"]).lower()])]
    if action_type in {"computer.mouse_down", "computer.mouse_up"}:
        if str(args.get("button") or "left") != "left":
            raise ValueError("Qwen XML mouse state actions only support the left button")
        parameters: dict[str, Any] = {}
        if "x" in args and "y" in args:
            parameters["coordinate"] = [
                qwen_coordinate(args["x"]),
                qwen_coordinate(args["y"]),
            ]
        qwen_action = (
            "left_mouse_down" if action_type.endswith("mouse_down") else "left_mouse_up"
        )
        return [_computer_call(qwen_action, **parameters)]
    if action_type == "computer.scroll":
        dx = _bounded_scroll(args.get("dx"), max_scroll_delta)
        dy = _bounded_scroll(args.get("dy"), max_scroll_delta)
        calls = []
        if dy:
            calls.append(_computer_call("scroll", pixels=_compact_number(-dy)))
        if dx:
            calls.append(_computer_call("hscroll", pixels=_compact_number(-dx)))
        if not calls:
            raise ValueError("scroll action has zero dx and dy")
        return calls
    if action_type == "computer.wait":
        return [_computer_call("wait", time=args["seconds"])]
    if action_type == "shell.exec":
        parameters = {"command": args["command"]}
        for name in ("cwd", "timeout_sec"):
            if name in args:
                parameters[name] = args[name]
        return [_xml_tool_call("shell_exec", parameters)]
    if action_type == "tool.call":
        return [
            _xml_tool_call(
                "tool_call", {"name": args["name"], "args": args["args"]}
            )
        ]
    if action_type == "finish":
        parameters = {"status": args["status"]}
        if "answer" in args:
            parameters["text"] = args["answer"]
        return [_computer_call("terminate", **parameters)]
    raise ValueError(f"unsupported canonical action for Qwen XML: {action_type}")


def describe_qwen_xml_action(action: CUAAction | dict[str, Any]) -> str:
    """Return the short Action line used by Qwen-style trajectories."""

    if isinstance(action, CUAAction):
        canonical = action
    elif isinstance(action, dict):
        canonical = CUAAction(
            str(action.get("type") or ""), dict(action.get("args") or {})
        )
    else:
        raise TypeError("action must be a CUAAction or dict")
    action_type = canonical.type
    args = canonical.args
    if action_type == "computer.click":
        return f"{str(args.get('button') or 'left').capitalize()}-click the target."
    if action_type == "computer.double_click":
        return "Double-click the target."
    if action_type == "computer.triple_click":
        return "Triple-click the target."
    if action_type == "computer.move":
        return "Move the cursor to the target."
    if action_type == "computer.drag":
        return "Drag from the current target to the destination."
    if action_type == "computer.type":
        return f"Type {args['text']!r}."
    if action_type == "computer.press":
        return "Press " + "+".join(str(key) for key in args["keys"]) + "."
    if action_type in {"computer.key_down", "computer.key_up"}:
        verb = "Hold" if action_type.endswith("key_down") else "Release"
        return f"{verb} the {args['key']} key."
    if action_type in {"computer.mouse_down", "computer.mouse_up"}:
        verb = "Press and hold" if action_type.endswith("mouse_down") else "Release"
        return f"{verb} the mouse button."
    if action_type == "computer.scroll":
        return "Scroll the view."
    if action_type == "computer.wait":
        return f"Wait {float(args['seconds']):g} seconds."
    if action_type == "shell.exec":
        return "Run the command in the guest shell."
    if action_type == "tool.call":
        return f"Call the {args['name']} tool."
    if action_type == "finish":
        return f"Terminate with {args['status']} status."
    raise ValueError(f"unsupported canonical action for Qwen XML: {action_type}")


def format_qwen_xml_action(
    action: CUAAction | dict[str, Any],
    *,
    description: str | None = None,
    include_empty_think: bool = True,
    max_scroll_delta: float = MAX_SCROLL_DELTA,
) -> str:
    """Format the complete assistant target used by Qwen XML SFT and runtime."""

    prefix = "<think>\n\n</think>\n\n" if include_empty_think else ""
    action_line = description or describe_qwen_xml_action(action)
    calls = action_to_qwen_xml_calls(action, max_scroll_delta=max_scroll_delta)
    return prefix + f"Action: {action_line}\n\n" + "\n".join(calls)


def qwen_xml_tool_name(name: str) -> str:
    """Return a stable XML function alias for a registered canonical tool."""

    alias = _FUNCTION_NAME_RE.sub("_", name).strip("_")
    if not alias or not alias[0].isalpha():
        alias = f"tool_{alias}"
    return alias


def qwen_xml_registered_tools_prompt(definitions: Iterable[ToolDefinition]) -> str:
    """Describe registered tools using Qwen-compatible XML function names."""

    lines = []
    for definition in definitions:
        alias = qwen_xml_tool_name(definition.name)
        schema = json.dumps(definition.input_schema, ensure_ascii=False, sort_keys=True)
        lines.append(
            f"- {alias}: {definition.description}; args schema={schema}; "
            f"canonical tool name={definition.name}"
        )
    if not lines:
        return ""
    return "Registered structured-tool XML functions:\n" + "\n".join(lines)


class QwenXmlActionParser:
    """Parse one Qwen XML tool call into the canonical action representation."""

    def __init__(self, tool_names: Iterable[str] = ()) -> None:
        aliases: dict[str, str] = {}
        for name in tool_names:
            for alias in (name, qwen_xml_tool_name(name)):
                existing = aliases.get(alias)
                if existing is not None and existing != name:
                    raise ValueError(
                        f"registered tools {existing!r} and {name!r} share XML alias {alias!r}"
                    )
                aliases[alias] = name
        self.tool_aliases = aliases

    def parse(self, text: str) -> CUAAction:
        actions = self.parse_many(text)
        if len(actions) != 1:
            raise ValueError("model output contains multiple canonical actions")
        return actions[0]

    def parse_many(self, text: str) -> list[CUAAction]:
        calls = self._extract_calls(text)
        if len(calls) > 16:
            raise ValueError("model output contains too many Qwen XML calls")
        actions: list[CUAAction] = []
        index = 0
        while index < len(calls):
            pair = calls[index : index + 2]
            if len(pair) == 2 and self._is_compound_pair(pair):
                action = self._compound_action(pair)
                index += 2
            else:
                action = self._single_action(calls[index])
                index += 1
            validate_action(action)
            actions.append(action)
        return actions

    def _is_compound_pair(self, calls: list[_XmlToolCall]) -> bool:
        if not all(call.function == "computer_use" for call in calls):
            return False
        action_names = [call.parameters.get("action", "").strip().lower() for call in calls]
        return action_names == ["mouse_move", "left_click_drag"] or set(
            action_names
        ) == {"scroll", "hscroll"}

    def _extract_calls(self, text: str) -> list[_XmlToolCall]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("model output is empty")
        stripped = _THINK_RE.sub("", text).strip()
        matches = list(_TOOL_CALL_RE.finditer(stripped))
        if not matches:
            raise ValueError("model output has no Qwen XML tool call")
        outside = []
        cursor = 0
        for match in matches:
            outside.append(stripped[cursor : match.start()])
            cursor = match.end()
        outside.append(stripped[cursor:])
        remainder = "".join(outside).strip()
        if "<tool_call" in remainder.lower() or "</tool_call" in remainder.lower():
            raise ValueError("malformed Qwen XML tool call")
        return [self._decode_call(match.group(1), match.group(2)) for match in matches]

    def _decode_call(self, function: str, body: str) -> _XmlToolCall:
        matches = list(_PARAMETER_RE.finditer(body))
        residue = _PARAMETER_RE.sub("", body).strip()
        if residue:
            raise ValueError(f"unexpected content in function {function!r}")
        parameters: dict[str, str] = {}
        for match in matches:
            name = match.group(1)
            if name in parameters:
                raise ValueError(f"duplicate XML parameter: {name}")
            parameters[name] = self._parameter_text(match.group(2))
        return _XmlToolCall(function=function, parameters=parameters)

    @staticmethod
    def _parameter_text(value: str) -> str:
        if value.startswith("\r\n"):
            value = value[2:]
        elif value.startswith("\n"):
            value = value[1:]
        if value.endswith("\r\n"):
            value = value[:-2]
        elif value.endswith("\n"):
            value = value[:-1]
        return value

    def _compound_action(self, calls: list[_XmlToolCall]) -> CUAAction:
        if len(calls) != 2:
            raise ValueError("model output must contain one atomic action")
        if all(call.function == "computer_use" for call in calls):
            action_names = [self._required(call.parameters, "action") for call in calls]
            if action_names == ["mouse_move", "left_click_drag"]:
                start = self._coordinate(calls[0].parameters)
                end = self._coordinate(calls[1].parameters)
                self._only(calls[0].parameters, {"action", "coordinate"})
                self._only(calls[1].parameters, {"action", "coordinate"})
                return CUAAction(
                    "computer.drag",
                    {
                        "x1": start[0],
                        "y1": start[1],
                        "x2": end[0],
                        "y2": end[1],
                        "button": "left",
                    },
                )
            if set(action_names) == {"scroll", "hscroll"}:
                dx = 0.0
                dy = 0.0
                for call, action_name in zip(calls, action_names):
                    self._only(call.parameters, {"action", "pixels"})
                    pixels = self._number(call.parameters, "pixels")
                    if action_name == "scroll":
                        dy = -pixels
                    else:
                        dx = -pixels
                return CUAAction("computer.scroll", {"dx": dx, "dy": dy})
        raise ValueError("multiple Qwen XML calls must form one drag or two-axis scroll")

    def _single_action(self, call: _XmlToolCall) -> CUAAction:
        if call.function == "computer_use":
            return self._computer_action(call.parameters)
        if call.function in {"shell_exec", "shell.exec"}:
            return self._shell_action(call.parameters)
        if call.function in {"tool_call", "tool.call"}:
            return self._generic_tool_action(call.parameters)
        tool_name = self.tool_aliases.get(call.function)
        if tool_name is not None:
            args = {
                name: self._json_or_text(value)
                for name, value in call.parameters.items()
            }
            return CUAAction("tool.call", {"name": tool_name, "args": args})
        raise ValueError(f"unsupported Qwen XML function: {call.function}")

    def _computer_action(self, parameters: dict[str, str]) -> CUAAction:
        name = self._required(parameters, "action").strip().lower()
        point_actions = {
            "left_click": ("computer.click", "left"),
            "right_click": ("computer.click", "right"),
            "middle_click": ("computer.click", "middle"),
            "double_click": ("computer.double_click", "left"),
            "triple_click": ("computer.triple_click", "left"),
        }
        if name in point_actions:
            self._only(parameters, {"action", "coordinate"})
            x, y = self._coordinate(parameters)
            action_type, button = point_actions[name]
            return CUAAction(action_type, {"x": x, "y": y, "button": button})
        if name == "mouse_move":
            self._only(parameters, {"action", "coordinate"})
            x, y = self._coordinate(parameters)
            return CUAAction("computer.move", {"x": x, "y": y})
        if name == "left_click_drag":
            raise ValueError("left_click_drag requires a preceding mouse_move call")
        if name == "type":
            self._only(parameters, {"action", "text"})
            return CUAAction("computer.type", {"text": self._required(parameters, "text")})
        if name == "key":
            self._only(parameters, {"action", "keys"})
            return CUAAction("computer.press", {"keys": self._keys(parameters)})
        if name in {"key_down", "key_up"}:
            self._only(parameters, {"action", "key", "keys"})
            keys = self._keys(parameters)
            if len(keys) != 1:
                raise ValueError(f"{name} requires exactly one key")
            action_type = "computer.key_down" if name == "key_down" else "computer.key_up"
            return CUAAction(action_type, {"key": keys[0]})
        if name in {"left_mouse_down", "left_mouse_up"}:
            self._only(parameters, {"action", "coordinate"})
            args: dict[str, Any] = {"button": "left"}
            if "coordinate" in parameters:
                x, y = self._coordinate(parameters)
                args.update({"x": x, "y": y})
            action_type = (
                "computer.mouse_down"
                if name == "left_mouse_down"
                else "computer.mouse_up"
            )
            return CUAAction(action_type, args)
        if name in {"scroll", "hscroll"}:
            self._only(parameters, {"action", "pixels"})
            pixels = self._number(parameters, "pixels")
            return CUAAction(
                "computer.scroll",
                {"dx": -pixels if name == "hscroll" else 0, "dy": -pixels if name == "scroll" else 0},
            )
        if name == "wait":
            self._only(parameters, {"action", "time"})
            return CUAAction("computer.wait", {"seconds": self._number(parameters, "time")})
        if name == "screenshot":
            self._only(parameters, {"action"})
            return CUAAction("computer.wait", {"seconds": 0.1})
        if name == "terminate":
            self._only(parameters, {"action", "status", "text", "answer"})
            status = self._required(parameters, "status").strip().lower()
            if status in {"failed", "fail"}:
                status = "failure"
            args = {"status": status}
            answer = parameters.get("answer", parameters.get("text"))
            if answer is not None:
                args["answer"] = answer
            return CUAAction("finish", args)
        if name == "call_user":
            raise ValueError("call_user is unsupported by the non-interactive runtime")
        raise ValueError(f"unsupported computer_use action: {name}")

    def _shell_action(self, parameters: dict[str, str]) -> CUAAction:
        self._only(parameters, {"command", "cwd", "timeout_sec"})
        args: dict[str, Any] = {"command": self._required(parameters, "command")}
        if "cwd" in parameters:
            args["cwd"] = parameters["cwd"]
        if "timeout_sec" in parameters:
            timeout = self._number(parameters, "timeout_sec")
            if not float(timeout).is_integer():
                raise ValueError("timeout_sec must be an integer")
            args["timeout_sec"] = int(timeout)
        return CUAAction("shell.exec", args)

    def _generic_tool_action(self, parameters: dict[str, str]) -> CUAAction:
        self._only(parameters, {"name", "args"})
        name = self._required(parameters, "name").strip()
        args = self._json_parameter(parameters, "args")
        if not isinstance(args, dict):
            raise TypeError("tool_call args must be a JSON object")
        return CUAAction("tool.call", {"name": name, "args": args})

    @staticmethod
    def _required(parameters: dict[str, str], name: str) -> str:
        value = parameters.get(name)
        if value is None or not value.strip():
            raise ValueError(f"missing XML parameter: {name}")
        return value

    @staticmethod
    def _only(parameters: dict[str, str], allowed: set[str]) -> None:
        unexpected = sorted(set(parameters) - allowed)
        if unexpected:
            raise ValueError(f"unexpected XML parameters: {unexpected}")

    def _coordinate(self, parameters: dict[str, str]) -> tuple[int, int]:
        coordinate = self._json_parameter(parameters, "coordinate")
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise TypeError("coordinate must be a two-item JSON array")
        return self._canonical_coordinate(coordinate[0]), self._canonical_coordinate(coordinate[1])

    @staticmethod
    def _canonical_coordinate(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("coordinate values must be numbers")
        if not 0 <= float(value) <= 999:
            raise ValueError("Qwen coordinates must be between 0 and 999")
        return max(0, min(1000, round(float(value) * 1000 / 999)))

    def _keys(self, parameters: dict[str, str]) -> list[str]:
        if "keys" in parameters:
            value = self._json_parameter(parameters, "keys")
        else:
            value = self._required(parameters, "key")
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not value:
            raise TypeError("keys must be a non-empty JSON array")
        if not all(isinstance(key, str) and key for key in value):
            raise TypeError("all keys must be non-empty strings")
        return [key.lower() for key in value]

    def _number(self, parameters: dict[str, str], name: str) -> float:
        value = self._json_parameter(parameters, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        return float(value)

    def _json_parameter(self, parameters: dict[str, str], name: str) -> Any:
        raw = self._required(parameters, name)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"XML parameter {name} must contain valid JSON") from exc

    @staticmethod
    def _json_or_text(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


class CompatibleActionParser:
    """Accept Qwen XML and canonical JSON/Code during protocol migration."""

    def __init__(self, canonical_parser: Any, qwen_parser: QwenXmlActionParser) -> None:
        self.canonical_parser = canonical_parser
        self.qwen_parser = qwen_parser

    def parse(self, text: str) -> CUAAction:
        if "<tool_call" in text.lower():
            return self.qwen_parser.parse(text)
        return self.canonical_parser.parse(text)

    def parse_many(self, text: str) -> list[CUAAction]:
        if "<tool_call" in text.lower():
            return self.qwen_parser.parse_many(text)
        return [self.canonical_parser.parse(text)]
