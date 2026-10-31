"""Platform normalization and conservative static compatibility checks."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .schemas import AbilityPrimitive, EnvironmentContext, SourceTask


def platform_family(value: str | None) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in {"linux", "ubuntu"}:
        return "linux"
    if normalized in {"darwin", "mac", "macos", "osx"}:
        return "darwin"
    if normalized in {"windows", "win32"}:
        return "windows"
    return normalized or "unknown"


def environment_platform(environment: EnvironmentContext) -> str:
    return platform_family(environment.os.get("type"))


_PLATFORM_PATTERNS = {
    "linux": (
        (r"(?<![A-Za-z0-9_])/home/", "Linux /home path"),
        (r"\b(?:GNOME|Ubuntu|apt-get|apt |dpkg|systemctl|xdg-open)\b", "Linux desktop/tooling"),
        (r"\bActivities overview\b", "GNOME Activities UI"),
        (r"\bCtrl\+Alt\+T\b", "Linux terminal shortcut"),
        (r"\.local/share/evolution|evolution-calendar", "Evolution calendar store"),
    ),
    "darwin": (
        (r"(?<![A-Za-z0-9_])/Users/", "macOS /Users path"),
        (r"\b(?:Finder|AppleScript|osascript)\b", "macOS desktop/tooling"),
        (r"\b(?:Command|Cmd)[-+ ](?:Shift[-+ ])?[A-Z0-9]", "macOS keyboard shortcut"),
    ),
    "windows": (
        (r"\b[A-Za-z]:\\", "Windows drive path"),
        (r"\b(?:PowerShell|File Explorer|Registry Editor)\b", "Windows desktop/tooling"),
        (r"\bWin\+[A-Z0-9]", "Windows keyboard shortcut"),
    ),
}


def platform_text_errors(text: str, target_platform: str) -> list[str]:
    target = platform_family(target_platform)
    errors: list[str] = []
    for family, patterns in _PLATFORM_PATTERNS.items():
        if family == target:
            continue
        for pattern, label in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                errors.append(f"{label} is incompatible with target platform {target}")
    if target == "windows" and re.search(r"(?<![A-Za-z0-9_])~/(?:Desktop|Downloads|Documents|Pictures)\b", text):
        errors.append("POSIX home-relative path is incompatible with target platform windows")
    return sorted(set(errors))


def detected_platform_families(text: str) -> set[str]:
    families: set[str] = set()
    for family, patterns in _PLATFORM_PATTERNS.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern, _ in patterns):
            families.add(family)
    return families


def primitive_platform_errors(
    capability: AbilityPrimitive,
    environment: EnvironmentContext,
) -> list[str]:
    payload = "\n".join((
        capability.intent,
        *capability.preconditions,
        *capability.effects,
        *capability.side_effects,
        json.dumps(capability.inputs, sort_keys=True),
        json.dumps(capability.config_dependencies, sort_keys=True),
    ))
    return platform_text_errors(payload, environment_platform(environment))


def _task_class_reference(source: SourceTask) -> dict[str, Any] | None:
    for item in source.task.config:
        if isinstance(item, dict) and item.get("type") == "osworld_v2_task_class":
            return item
    return None


def _runtime_string_literals(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return []
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                values.append(child.value)
    return values


def source_platform_errors(
    source: SourceTask,
    environment: EnvironmentContext,
) -> list[str]:
    reference = _task_class_reference(source)
    if reference is None:
        return []
    path = Path(str(reference.get("path") or ""))
    if not path.is_file():
        return [f"source task class is unavailable: {path}"]
    runtime_text = "\n".join(_runtime_string_literals(path))
    target = environment_platform(environment)
    source_platform = platform_family(source.task.metadata.get("platform"))
    if target == source_platform:
        return []
    detected = detected_platform_families(runtime_text)
    if not detected or target in detected:
        return []
    labels = ", ".join(sorted(detected))
    return [
        f"source task class has runtime assumptions for {labels}, not target platform {target}"
    ]
