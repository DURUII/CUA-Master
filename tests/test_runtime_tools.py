from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from cuamaster.runtime import (
    CORE_TOOL_NAMES,
    CUAAction,
    CanonicalActionExecutor,
    ToolPathRoot,
    build_tool_registry,
    load_tool_registry,
)


class LocalGuestEnvironment:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.shell_calls: list[tuple[str, str, int]] = []

    def get_screenshot(self) -> bytes:
        return b"png"

    def get_screen_size(self) -> tuple[int, int]:
        return 1920, 1080

    def execute_gui(self, command: str, *, pause: float):
        return {}

    def execute_shell(self, command: str, *, cwd: str, timeout_sec: int):
        self.shell_calls.append((command, cwd, timeout_sec))
        process = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "stdout": process.stdout,
            "stderr": process.stderr,
            "returncode": process.returncode,
        }

    def finish(self, status: str):
        return {}

    def get_platform(self) -> str:
        return "Linux"

    def get_accessibility_tree(self) -> str:
        return """
        <desktop>
          <button name="Save" states="enabled focused" screencoord="(10, 20)" size="(80, 30)" />
          <entry name="Filename" value="report.txt" states="enabled" />
        </desktop>
        """

    def get_tool_roots(self) -> list[ToolPathRoot]:
        return [ToolPathRoot(str(self.root), read=True, write=True)]


def call_tool(env: LocalGuestEnvironment, name: str, args: dict):
    executor = CanonicalActionExecutor(tool_registry=build_tool_registry())
    return executor.execute(CUAAction("tool.call", {"name": name, "args": args}), env)


def test_core_profile_registers_six_practical_tools() -> None:
    definitions = build_tool_registry().definitions()

    assert tuple(definition.name for definition in definitions) == tuple(sorted(CORE_TOOL_NAMES))
    assert {definition.name for definition in definitions if not definition.read_only} == {
        "archive.extract",
        "filesystem.write_text",
    }


def test_ui_inspect_returns_filtered_structured_nodes(tmp_path: Path) -> None:
    response = call_tool(LocalGuestEnvironment(tmp_path), "ui.inspect", {"query": "save"})

    assert response.ok is True
    assert response.observation["result"]["nodes"] == [
        {
            "role": "button",
            "name": "Save",
            "value": "",
            "state": "enabled focused",
            "bounds": {"x": 10, "y": 20, "width": 80, "height": 30},
        }
    ]


def test_file_tools_list_read_and_write_inside_guest_root(tmp_path: Path) -> None:
    env = LocalGuestEnvironment(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("alpha\nbeta\n", encoding="utf-8")

    listing = call_tool(env, "filesystem.list", {"path": str(tmp_path), "glob": "*.txt"})
    reading = call_tool(env, "filesystem.read_text", {"path": str(source), "max_chars": 5})
    writing = call_tool(
        env,
        "filesystem.write_text",
        {"path": str(tmp_path / "out.txt"), "content": "result", "mode": "create"},
    )

    assert listing.ok is True
    assert listing.observation["result"]["entries"][0]["path"] == str(source)
    assert reading.observation["result"]["text"] == "alpha"
    assert reading.observation["result"]["truncated"] is True
    assert writing.ok is True
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "result"
    assert all(cwd == "/" for _, cwd, _ in env.shell_calls)


def test_file_tools_reject_path_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    env = LocalGuestEnvironment(root)

    direct = call_tool(env, "filesystem.read_text", {"path": str(outside / "secret.txt")})
    symlink = call_tool(env, "filesystem.read_text", {"path": str(root / "escape/secret.txt")})

    assert direct.observation["error"]["code"] == "path_not_allowed"
    assert symlink.observation["error"]["code"] == "path_not_allowed"


def test_document_extracts_docx_text(tmp_path: Path) -> None:
    document = tmp_path / "sample.docx"
    with zipfile.ZipFile(document, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>Hello CUA</w:t></w:r></w:p></w:body></w:document>',
        )

    response = call_tool(
        LocalGuestEnvironment(tmp_path),
        "document.extract_text",
        {"path": str(document)},
    )

    assert response.ok is True
    assert response.observation["result"]["text"] == "Hello CUA"
    assert response.observation["result"]["format"] == "docx"


def test_archive_extracts_safe_zip_and_rejects_traversal(tmp_path: Path) -> None:
    safe = tmp_path / "safe.zip"
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(safe, "w") as archive:
        archive.writestr("folder/file.txt", "ok")
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    env = LocalGuestEnvironment(tmp_path)

    extracted = call_tool(
        env,
        "archive.extract",
        {"path": str(safe), "destination": str(tmp_path / "safe-out")},
    )
    rejected = call_tool(
        env,
        "archive.extract",
        {"path": str(unsafe), "destination": str(tmp_path / "unsafe-out")},
    )

    assert extracted.ok is True
    assert (tmp_path / "safe-out/folder/file.txt").read_text(encoding="utf-8") == "ok"
    assert rejected.observation["error"]["code"] == "unsafe_archive"
    assert not (tmp_path / "escape.txt").exists()


def test_registry_profile_config_and_limits(tmp_path: Path) -> None:
    config = tmp_path / "tools.json"
    config.write_text(
        json.dumps(
            {
                "profile": "cua-core-v1",
                "disabled_tools": ["archive.extract"],
                "limits": {"max_write_bytes": 4},
            }
        ),
        encoding="utf-8",
    )
    registry = load_tool_registry(config)
    env = LocalGuestEnvironment(tmp_path)

    response = CanonicalActionExecutor(tool_registry=registry).execute(
        CUAAction(
            "tool.call",
            {
                "name": "filesystem.write_text",
                "args": {"path": str(tmp_path / "large.txt"), "content": "12345", "mode": "create"},
            },
        ),
        env,
    )

    assert "archive.extract" not in {definition.name for definition in registry.definitions()}
    assert response.observation["error"]["code"] == "content_too_large"
