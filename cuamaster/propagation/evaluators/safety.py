"""Static safety checks for newly generated evaluator plugins."""
from __future__ import annotations

import ast
from typing import Any


_BANNED_IMPORTS = {
    "requests", "httpx", "socket", "paramiko", "fabric", "ftplib", "telnetlib",
}
_BANNED_CALLS = {
    "eval", "exec", "compile", "__import__", "os.system", "os.popen",
    "subprocess.Popen", "subprocess.call", "subprocess.run",
}


def _call_name(node: ast.Call) -> str:
    value = node.func
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        return f"{value.value.id}.{value.attr}"
    return ""


def inspect_evaluator_code(code: str, *, source_trust: str = "generated") -> dict[str, Any]:
    if source_trust == "source":
        return {"safe": True, "errors": [], "trusted_source": True}
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"safe": False, "errors": [f"syntax error: {exc}"], "trusted_source": False}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BANNED_IMPORTS:
                    errors.append(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _BANNED_IMPORTS:
                errors.append(f"banned import: {node.module}")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _BANNED_CALLS:
                errors.append(f"banned call: {name}")
    return {"safe": not errors, "errors": sorted(set(errors)), "trusted_source": False}
