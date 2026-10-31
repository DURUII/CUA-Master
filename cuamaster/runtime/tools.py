"""Built-in structured tools for practical computer-use tasks."""
from __future__ import annotations

import base64
import json
import re
import shlex
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any

from cuamaster.action_space import ToolDefinition

from .core import RuntimeEnvironment, ToolCallError, ToolHandler, ToolPathRoot, ToolRegistry


CORE_TOOL_NAMES = (
    "ui.inspect",
    "filesystem.list",
    "filesystem.read_text",
    "filesystem.write_text",
    "document.extract_text",
    "archive.extract",
)


@dataclass(frozen=True)
class ToolLimits:
    max_list_entries: int = 200
    max_read_chars: int = 65_536
    max_write_bytes: int = 262_144
    max_document_chars: int = 100_000
    max_document_bytes: int = 26_214_400
    max_archive_entries: int = 1_000
    max_archive_bytes: int = 268_435_456


def profile_tool_names(profile: str) -> tuple[str, ...]:
    if profile == "cua-core-v1":
        return CORE_TOOL_NAMES
    if profile == "runtime-meta":
        return ("runtime.get_screen_size", "runtime.get_platform")
    raise ValueError(f"unknown tool registry profile: {profile!r}")


def build_tool_registry(
    profile: str = "cua-core-v1",
    *,
    limits: dict[str, Any] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    catalog = builtin_tool_catalog(limits)
    for name in profile_tool_names(profile):
        definition, handler = catalog[name]
        registry.register(definition, handler)
    return registry


def builtin_tool_catalog(
    configured_limits: dict[str, Any] | None = None,
) -> dict[str, tuple[ToolDefinition, ToolHandler]]:
    limits = _tool_limits(configured_limits)
    empty_args = {"type": "object", "properties": {}, "additionalProperties": False}
    object_output = {"type": "object"}
    return {
        "runtime.get_screen_size": (
            ToolDefinition(
                "runtime.get_screen_size",
                "Return the active environment screen width and height.",
                empty_args,
                object_output,
            ),
            lambda args, env: dict(zip(("width", "height"), env.get_screen_size())),
        ),
        "runtime.get_platform": (
            ToolDefinition(
                "runtime.get_platform",
                "Return the guest operating-system platform name.",
                empty_args,
                {"type": "string"},
            ),
            lambda args, env: env.get_platform(),
        ),
        "ui.inspect": (
            ToolDefinition(
                "ui.inspect",
                "Inspect the current accessibility tree and return matching visible UI nodes.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 256},
                        "roles": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 64},
                            "maxItems": 20,
                        },
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "additionalProperties": False,
                },
                object_output,
            ),
            _inspect_ui,
        ),
        "filesystem.list": (
            ToolDefinition(
                "filesystem.list",
                "List or recursively find files inside task-approved guest directories.",
                {
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "glob": {"type": "string", "maxLength": 256},
                        "recursive": {"type": "boolean"},
                        "max_depth": {"type": "integer", "minimum": 0, "maximum": 12},
                        "max_entries": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": limits.max_list_entries,
                        },
                    },
                    "additionalProperties": False,
                },
                object_output,
            ),
            lambda args, env: _guest_call("filesystem.list", args, env, limits),
        ),
        "filesystem.read_text": (
            ToolDefinition(
                "filesystem.read_text",
                "Read a bounded text slice from a task-approved guest file.",
                {
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "offset": {"type": "integer", "minimum": 0},
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": limits.max_read_chars,
                        },
                    },
                    "additionalProperties": False,
                },
                object_output,
            ),
            lambda args, env: _guest_call("filesystem.read_text", args, env, limits),
        ),
        "filesystem.write_text": (
            ToolDefinition(
                "filesystem.write_text",
                "Create, overwrite, or append bounded UTF-8 text in a task-approved guest directory.",
                {
                    "type": "object",
                    "required": ["path", "content", "mode"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "content": {"type": "string"},
                        "mode": {"type": "string", "enum": ["create", "overwrite", "append"]},
                        "create_parents": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
                object_output,
                read_only=False,
                destructive=True,
            ),
            lambda args, env: _guest_call("filesystem.write_text", args, env, limits),
        ),
        "document.extract_text": (
            ToolDefinition(
                "document.extract_text",
                "Extract bounded text and sections from PDF, Office, CSV, HTML, or text files.",
                {
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": limits.max_document_chars,
                        },
                        "include_metadata": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
                object_output,
            ),
            lambda args, env: _guest_call("document.extract_text", args, env, limits, timeout=60),
        ),
        "archive.extract": (
            ToolDefinition(
                "archive.extract",
                "Safely extract ZIP or TAR archives into a task-approved guest directory.",
                {
                    "type": "object",
                    "required": ["path", "destination"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "destination": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "overwrite": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
                object_output,
                read_only=False,
                destructive=True,
            ),
            lambda args, env: _guest_call("archive.extract", args, env, limits, timeout=120),
        ),
    }


def _tool_limits(config: dict[str, Any] | None) -> ToolLimits:
    if config is None:
        return ToolLimits()
    if not isinstance(config, dict):
        raise ValueError("tool registry limits must be an object")
    defaults = asdict(ToolLimits())
    unknown = sorted(set(config) - set(defaults))
    if unknown:
        raise ValueError(f"unknown tool limits: {', '.join(unknown)}")
    values = {**defaults, **config}
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
        raise ValueError("tool limits must be positive integers")
    return ToolLimits(**values)


def _inspect_ui(args: dict[str, Any], env: RuntimeEnvironment) -> dict[str, Any]:
    getter = getattr(env, "get_accessibility_tree", None)
    if not callable(getter):
        raise ToolCallError("capability_unavailable", "environment has no accessibility-tree capability")
    tree = getter()
    if not tree:
        raise ToolCallError("capability_unavailable", "accessibility tree is unavailable")
    try:
        root = ET.fromstring(tree)
    except ET.ParseError as exc:
        raise ToolCallError("invalid_accessibility_tree", str(exc)) from exc
    query = str(args.get("query") or "").casefold()
    roles = {str(role).casefold() for role in args.get("roles") or []}
    maximum = int(args.get("max_results", 50))
    nodes: list[dict[str, Any]] = []
    matched = 0
    for element in root.iter():
        node = _accessibility_node(element)
        if not node["name"] and not node["value"]:
            continue
        searchable = " ".join(str(value) for value in node.values() if value is not None).casefold()
        if query and query not in searchable:
            continue
        if roles and node["role"].casefold() not in roles:
            continue
        matched += 1
        if len(nodes) < maximum:
            nodes.append(node)
    return {"nodes": nodes, "matched": matched, "truncated": matched > len(nodes)}


def _accessibility_node(element: ET.Element) -> dict[str, Any]:
    attributes = {_local_name(key): value for key, value in element.attrib.items()}
    role = attributes.get("role") or _local_name(element.tag)
    name = attributes.get("name") or attributes.get("title") or ""
    value = attributes.get("value") or attributes.get("text") or (element.text or "").strip()
    state = attributes.get("states") or attributes.get("state") or ""
    bounds = _bounds(attributes)
    return {
        "role": role,
        "name": name,
        "value": value,
        "state": state,
        "bounds": bounds,
    }


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _bounds(attributes: dict[str, str]) -> dict[str, int] | None:
    combined = attributes.get("bounds")
    if combined:
        values = [int(value) for value in re.findall(r"-?\d+", combined)]
        if len(values) >= 4:
            return dict(zip(("x", "y", "width", "height"), values[:4]))
    coordinates = [int(value) for value in re.findall(r"-?\d+", attributes.get("screencoord", ""))]
    size = [int(value) for value in re.findall(r"-?\d+", attributes.get("size", ""))]
    if len(coordinates) >= 2 and len(size) >= 2:
        return {"x": coordinates[0], "y": coordinates[1], "width": size[0], "height": size[1]}
    return None


def _guest_call(
    operation: str,
    args: dict[str, Any],
    env: RuntimeEnvironment,
    limits: ToolLimits,
    *,
    timeout: int = 30,
) -> dict[str, Any]:
    roots = _tool_roots(env)
    payload = base64.b64encode(
        json.dumps(
            {
                "operation": operation,
                "args": args,
                "roots": [asdict(root) for root in roots],
                "limits": asdict(limits),
            },
            ensure_ascii=False,
        ).encode("utf-8")
    ).decode("ascii")
    command = f"python3 -c {shlex.quote(_GUEST_HELPER)} {shlex.quote(payload)}"
    result = env.execute_shell(command, cwd="/", timeout_sec=timeout) or {}
    stdout = str(result.get("stdout") or result.get("output") or "")
    stderr = str(result.get("stderr") or result.get("error") or "")
    returncode = result.get("returncode")
    if returncode not in {0, None}:
        raise ToolCallError("guest_tool_failed", stderr.strip() or f"guest helper exited {returncode}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ToolCallError("guest_tool_failed", stderr.strip() or "guest helper returned no result")
    try:
        response = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ToolCallError("guest_tool_failed", "guest helper returned invalid JSON") from exc
    if not response.get("ok"):
        error = response.get("error") or {}
        raise ToolCallError(str(error.get("code") or "guest_tool_failed"), str(error.get("message") or "tool failed"))
    return response.get("result") or {}


def _tool_roots(env: RuntimeEnvironment) -> list[ToolPathRoot]:
    getter = getattr(env, "get_tool_roots", None)
    if not callable(getter):
        raise ToolCallError("capability_unavailable", "environment has no task path whitelist")
    values = getter()
    roots: list[ToolPathRoot] = []
    for value in values or []:
        if isinstance(value, ToolPathRoot):
            root = value
        elif isinstance(value, dict):
            root = ToolPathRoot(**value)
        else:
            raise ToolCallError("invalid_tool_roots", f"invalid tool root: {value!r}")
        if not root.path or not root.path.startswith("/"):
            raise ToolCallError("invalid_tool_roots", f"tool root must be absolute: {root.path!r}")
        roots.append(root)
    if not roots:
        raise ToolCallError("capability_unavailable", "environment exposed no task path roots")
    return roots


_GUEST_HELPER = r'''
import base64
import csv
import fnmatch
import html.parser
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


class Failure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
operation = payload["operation"]
args = payload["args"]
roots = payload["roots"]
limits = payload["limits"]


def fail(code, message):
    raise Failure(code, message)


def allowed_roots(mode):
    selected = []
    for item in roots:
        if item.get(mode):
            selected.append(Path(item["path"]).expanduser().resolve(strict=False))
    if not selected:
        fail("permission_denied", f"no {mode} roots are available")
    return selected


def safe_path(raw, mode):
    candidates = allowed_roots(mode)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = candidates[0] / path
    resolved = path.resolve(strict=False)
    for root in candidates:
        if resolved == root or root in resolved.parents:
            return resolved
    fail("path_not_allowed", f"path is outside approved {mode} roots: {raw}")


def result(value):
    print(json.dumps({"ok": True, "result": value}, ensure_ascii=False))


def decode_text(data):
    if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        fail("binary_file", "file appears to be binary")
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    fail("binary_file", "file is not supported text encoding")


def list_files():
    root = safe_path(args["path"], "read")
    if not root.exists():
        fail("path_not_found", str(root))
    pattern = args.get("glob") or "*"
    recursive = bool(args.get("recursive", False))
    max_depth = int(args.get("max_depth", 4))
    maximum = min(int(args.get("max_entries", limits["max_list_entries"])), limits["max_list_entries"])
    entries = []
    matched = 0
    iterator = root.rglob("*") if recursive and root.is_dir() else root.iterdir() if root.is_dir() else [root]
    for path in iterator:
        try:
            relative = path.relative_to(root) if path != root else Path(path.name)
        except ValueError:
            continue
        if len(relative.parts) > max_depth + 1:
            continue
        if path.is_symlink():
            kind = "symlink"
        elif path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = "file"
        else:
            kind = "other"
        if not fnmatch.fnmatch(path.name, pattern) and not fnmatch.fnmatch(str(relative), pattern):
            continue
        matched += 1
        if len(entries) >= maximum:
            continue
        stat = path.lstat()
        entries.append({
            "path": str(path),
            "relative_path": str(relative),
            "type": kind,
            "size": stat.st_size if kind == "file" else None,
            "mtime": stat.st_mtime,
        })
    entries.sort(key=lambda item: item["path"])
    result({"root": str(root), "entries": entries, "matched": matched, "truncated": matched > len(entries)})


def read_text():
    path = safe_path(args["path"], "read")
    if not path.is_file():
        fail("path_not_found", str(path))
    if path.stat().st_size > 8 * 1024 * 1024:
        fail("file_too_large", "text file exceeds 8 MiB read limit")
    text, encoding = decode_text(path.read_bytes())
    offset = int(args.get("offset", 0))
    maximum = min(int(args.get("max_chars", limits["max_read_chars"])), limits["max_read_chars"])
    fragment = text[offset:offset + maximum]
    result({
        "path": str(path),
        "text": fragment,
        "encoding": encoding,
        "size_bytes": path.stat().st_size,
        "offset": offset,
        "truncated": offset + len(fragment) < len(text),
    })


def write_text():
    path = safe_path(args["path"], "write")
    content = args["content"].encode("utf-8")
    if len(content) > limits["max_write_bytes"]:
        fail("content_too_large", f"content exceeds {limits['max_write_bytes']} bytes")
    mode = args["mode"]
    existed = path.exists()
    if args.get("create_parents", False):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir():
        fail("parent_not_found", str(path.parent))
    if path.is_dir():
        fail("invalid_path", "target is a directory")
    if mode == "create" and existed:
        fail("path_exists", str(path))
    if mode == "append":
        with path.open("ab") as stream:
            stream.write(content)
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=".cua-write-", dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    result({"path": str(path), "bytes_written": len(content), "created": not existed, "mode": mode})


class TextHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)


def xml_text(data):
    root = ET.fromstring(data)
    return " ".join(text.strip() for text in root.itertext() if text and text.strip())


def office_sections(path, suffix):
    sections = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if suffix == ".docx":
            selected = [name for name in names if name == "word/document.xml"]
        elif suffix == ".pptx":
            selected = sorted(name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
        else:
            shared = []
            if "xl/sharedStrings.xml" in names:
                shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = ["".join(node.itertext()) for node in shared_root]
            selected = sorted(name for name in names if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
            for name in selected:
                root = ET.fromstring(archive.read(name))
                rows = []
                for row in root.iter():
                    if row.tag.rsplit("}", 1)[-1] != "row":
                        continue
                    values = []
                    for cell in row:
                        if cell.tag.rsplit("}", 1)[-1] != "c":
                            continue
                        kind = cell.attrib.get("t")
                        value = next((child.text for child in cell if child.tag.rsplit("}", 1)[-1] == "v"), "")
                        if kind == "s" and value:
                            try:
                                value = shared[int(value)]
                            except (ValueError, IndexError):
                                pass
                        values.append(value or "")
                    if values:
                        rows.append("\t".join(values))
                sections.append({"name": name, "text": "\n".join(rows)})
            return sections
        for name in selected:
            sections.append({"name": name, "text": xml_text(archive.read(name))})
    return sections


def extract_document():
    path = safe_path(args["path"], "read")
    if not path.is_file():
        fail("path_not_found", str(path))
    size = path.stat().st_size
    if size > limits["max_document_bytes"]:
        fail("file_too_large", f"document exceeds {limits['max_document_bytes']} bytes")
    suffix = path.suffix.lower()
    sections = []
    if suffix == ".pdf":
        executable = shutil.which("pdftotext")
        if not executable:
            fail("dependency_unavailable", "pdftotext is not installed in the guest")
        process = subprocess.run([executable, "-layout", str(path), "-"], capture_output=True, timeout=30)
        if process.returncode != 0:
            fail("document_parse_error", process.stderr.decode("utf-8", "replace"))
        text = process.stdout.decode("utf-8", "replace")
        sections = [{"name": "document", "text": text}]
    elif suffix in {".docx", ".pptx", ".xlsx"}:
        try:
            sections = office_sections(path, suffix)
        except (zipfile.BadZipFile, ET.ParseError) as exc:
            fail("document_parse_error", str(exc))
    elif suffix in {".html", ".htm"}:
        text, _ = decode_text(path.read_bytes())
        parser = TextHTMLParser()
        parser.feed(text)
        sections = [{"name": "document", "text": "\n".join(parser.parts)}]
    elif suffix in {".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml"}:
        text, _ = decode_text(path.read_bytes())
        sections = [{"name": "document", "text": text}]
    else:
        fail("unsupported_document", f"unsupported document type: {suffix or '<none>'}")
    full_text = "\n\n".join(section["text"] for section in sections)
    maximum = min(int(args.get("max_chars", limits["max_document_chars"])), limits["max_document_chars"])
    truncated = len(full_text) > maximum
    text = full_text[:maximum]
    remaining = maximum
    bounded_sections = []
    for section in sections:
        if remaining <= 0:
            break
        section_text = section["text"][:remaining]
        bounded_sections.append({"name": section["name"], "text": section_text})
        remaining -= len(section_text)
    output = {"path": str(path), "format": suffix.lstrip("."), "text": text, "sections": bounded_sections, "truncated": truncated}
    if args.get("include_metadata", True):
        output["metadata"] = {"size_bytes": size, "section_count": len(sections)}
    result(output)


def archive_members(path):
    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        values = []
        for item in archive.infolist():
            mode = item.external_attr >> 16
            if (mode & 0o170000) == 0o120000:
                fail("unsafe_archive", f"archive contains symlink: {item.filename}")
            values.append((item.filename, item.file_size, item.is_dir(), item))
        return archive, values
    if path.suffix.lower() == ".tar" or suffixes in {".tar.gz", ".tar.bz2", ".tar.xz"} or path.suffix.lower() == ".tgz":
        archive = tarfile.open(path, "r:*")
        values = []
        for item in archive.getmembers():
            if item.issym() or item.islnk() or item.isdev():
                fail("unsafe_archive", f"archive contains unsafe entry: {item.name}")
            values.append((item.name, item.size, item.isdir(), item))
        return archive, values
    fail("unsupported_archive", f"unsupported archive type: {path.name}")


def extract_archive():
    path = safe_path(args["path"], "read")
    destination = safe_path(args["destination"], "write")
    if not path.is_file():
        fail("path_not_found", str(path))
    overwrite = bool(args.get("overwrite", False))
    archive, members = archive_members(path)
    try:
        if len(members) > limits["max_archive_entries"]:
            fail("archive_too_large", "archive contains too many entries")
        total = sum(size for _, size, _, _ in members)
        if total > limits["max_archive_bytes"]:
            fail("archive_too_large", "archive expands beyond the configured byte limit")
        destination.mkdir(parents=True, exist_ok=True)
        extracted = []
        for name, _, is_directory, member in members:
            target = (destination / name).resolve(strict=False)
            if target != destination and destination not in target.parents:
                fail("unsafe_archive", f"archive path escapes destination: {name}")
            if target.exists() and not overwrite and not is_directory:
                fail("path_exists", str(target))
            if is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(archive, zipfile.ZipFile):
                source = archive.open(member)
            else:
                source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(str(target))
        result({
            "archive": str(path),
            "destination": str(destination),
            "entries_extracted": len(extracted),
            "bytes_extracted": total,
            "paths": extracted[:200],
            "truncated": len(extracted) > 200,
        })
    finally:
        archive.close()


try:
    if operation == "filesystem.list":
        list_files()
    elif operation == "filesystem.read_text":
        read_text()
    elif operation == "filesystem.write_text":
        write_text()
    elif operation == "document.extract_text":
        extract_document()
    elif operation == "archive.extract":
        extract_archive()
    else:
        fail("unknown_operation", operation)
except Failure as exc:
    print(json.dumps({"ok": False, "error": {"code": exc.code, "message": exc.message}}, ensure_ascii=False))
except Exception as exc:
    print(json.dumps({"ok": False, "error": {"code": "guest_tool_error", "message": f"{type(exc).__name__}: {exc}"}}, ensure_ascii=False))
'''
