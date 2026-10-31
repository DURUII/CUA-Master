"""Schemas and validation helpers for action trajectory rewriting."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


RewriteTarget = Literal["cli", "tool"]


@dataclass(frozen=True)
class RewriteTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = True
    destructive: bool = False
    open_world: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "read_only": self.read_only,
            "destructive": self.destructive,
            "open_world": self.open_world,
        }


@dataclass(frozen=True)
class RewriteSpan:
    start_index: int
    end_index: int
    target_type: RewriteTarget
    confidence: float
    reason: str
    tool_name: str | None = None
    improves_success_rate: bool = False
    improves_efficiency: bool = False
    benefit_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "target_type": self.target_type,
            "tool_name": self.tool_name,
            "confidence": self.confidence,
            "reason": self.reason,
            "improves_success_rate": self.improves_success_rate,
            "improves_efficiency": self.improves_efficiency,
            "benefit_reason": self.benefit_reason,
        }


def load_tools_config(path: Path | None) -> list[RewriteTool]:
    if path is None:
        return []
    source = str(path)
    if source.startswith("builtin:"):
        return _load_builtin_tools(source.removeprefix("builtin:"))
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if isinstance(raw, dict):
        raw = raw.get("tools")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"tools config must be a list or contain a tools list: {path}")

    tools: list[RewriteTool] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"tools[{index}] must be an object")
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        input_schema = item.get("input_schema")
        if not name:
            raise ValueError(f"tools[{index}].name is required")
        if name in names:
            raise ValueError(f"duplicate tool name: {name}")
        if not description:
            raise ValueError(f"tools[{index}].description is required")
        if not isinstance(input_schema, dict):
            raise ValueError(f"tools[{index}].input_schema must be an object")
        if input_schema.get("type", "object") != "object":
            raise ValueError(f"tools[{index}].input_schema must describe an object")
        names.add(name)
        tools.append(
            RewriteTool(
                name=name,
                description=description,
                input_schema=input_schema,
                read_only=bool(item.get("read_only", True)),
                destructive=bool(item.get("destructive", False)),
                open_world=bool(item.get("open_world", False)),
            )
        )
    return tools


def _load_builtin_tools(profile: str) -> list[RewriteTool]:
    from cuamaster.runtime.tools import builtin_tool_catalog, profile_tool_names

    catalog = builtin_tool_catalog()
    names = tuple(catalog) if profile == "all" else profile_tool_names(profile)
    tools = []
    for name in names:
        definition, _ = catalog[name]
        tools.append(
            RewriteTool(
                name=definition.name,
                description=definition.description,
                input_schema=definition.input_schema,
                read_only=definition.read_only,
                destructive=definition.destructive,
                open_world=definition.open_world,
            )
        )
    return tools


def tools_config_digest(tools: list[RewriteTool]) -> str:
    payload = json.dumps([tool.to_dict() for tool in tools], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_tool_call(
    action: dict[str, Any],
    tools: list[RewriteTool],
    *,
    allow_destructive: bool,
) -> RewriteTool:
    args = action.get("args") if isinstance(action, dict) else None
    name = args.get("name") if isinstance(args, dict) else None
    tool_args = args.get("args") if isinstance(args, dict) else None
    tool_by_name = {tool.name: tool for tool in tools}
    if name not in tool_by_name:
        raise ValueError(f"unregistered tool: {name}")
    tool = tool_by_name[str(name)]
    if tool.destructive and not allow_destructive:
        raise ValueError(f"destructive tool is disabled: {tool.name}")
    validate_json_schema(tool_args, tool.input_schema, path=f"tool.{tool.name}.args")
    return tool


def validate_json_schema(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate the small JSON Schema subset used by tool definitions."""

    if not isinstance(schema, dict):
        raise TypeError(f"{path} schema must be an object")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']!r}")

    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_matches_type(value, item) for item in expected):
            raise TypeError(f"{path} must match one of {expected}")
    elif isinstance(expected, str) and not _matches_type(value, expected):
        raise TypeError(f"{path} must be {expected}")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise TypeError(f"{path} object schema has invalid properties or required")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} missing required fields: {', '.join(map(str, missing))}")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                validate_json_schema(item, properties[key], path=child_path)
            elif additional is False:
                raise ValueError(f"{path} has unknown field: {key}")
            elif isinstance(additional, dict):
                validate_json_schema(item, additional, path=child_path)
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ValueError(f"{path} must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path} must contain at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                validate_json_schema(item, item_schema, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ValueError(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ValueError(f"{path} is longer than maxLength")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above maximum")


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported JSON schema type: {expected}")
