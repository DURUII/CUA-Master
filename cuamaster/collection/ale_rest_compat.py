"""Translate the legacy OSWorld REST API to ALE's native ``/cmd`` API."""
from __future__ import annotations

import base64
import json
import logging
import shlex
import threading
import urllib.parse
import urllib.request
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from typing import Any

logger = logging.getLogger("cuamaster.collection.ale_rest_compat")

_FILE_CHUNK_BYTES = 8 * 1024 * 1024
_MULTIPART_READ_BYTES = 256 * 1024


class _MultipartReader:
    def __init__(self, stream: Any, content_length: int):
        self.stream = stream
        self.remaining = content_length
        self.buffer = bytearray()

    def _fill(self) -> bool:
        if self.remaining <= 0:
            return False
        chunk = self.stream.read(min(_MULTIPART_READ_BYTES, self.remaining))
        if not chunk:
            self.remaining = 0
            return False
        self.remaining -= len(chunk)
        self.buffer.extend(chunk)
        return True

    def read_exact(self, size: int) -> bytes:
        while len(self.buffer) < size and self._fill():
            pass
        if len(self.buffer) < size:
            raise OSError("multipart request ended unexpectedly")
        value = bytes(self.buffer[:size])
        del self.buffer[:size]
        return value

    def readline(self) -> bytes:
        while True:
            index = self.buffer.find(b"\n")
            if index >= 0:
                value = bytes(self.buffer[: index + 1])
                del self.buffer[: index + 1]
                return value
            if not self._fill():
                value = bytes(self.buffer)
                self.buffer.clear()
                return value

    def read_part(self, boundary: bytes, sink: Any) -> bool:
        marker = b"\r\n--" + boundary
        while True:
            index = self.buffer.find(marker)
            if index >= 0:
                if index:
                    sink(bytes(self.buffer[:index]))
                del self.buffer[: index + len(marker)]
                suffix = self.read_exact(2)
                if suffix == b"--":
                    if self.buffer.startswith(b"\r\n"):
                        del self.buffer[:2]
                    return True
                if suffix != b"\r\n":
                    raise OSError("invalid multipart boundary suffix")
                return False
            safe_length = len(self.buffer) - len(marker) + 1
            if safe_length > 0:
                sink(bytes(self.buffer[:safe_length]))
                del self.buffer[:safe_length]
            if not self._fill():
                raise OSError("multipart boundary was not found")


def _parse_sse_json(payload: bytes) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    for raw_line in payload.splitlines():
        if not raw_line.startswith(b"data:"):
            continue
        value = raw_line[len(b"data:") :].strip()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            result = parsed
    if result is not None:
        return result
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OSError("ALE controller returned no valid JSON event") from exc
    if not isinstance(parsed, dict):
        raise OSError("ALE controller returned a non-object response")
    return parsed


class AleCommandClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def status(self, timeout: float = 10) -> dict[str, Any]:
        with self._opener.open(f"{self.base_url}/status", timeout=timeout) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise OSError("ALE status response is not an object")
        return payload

    def command(
        self,
        command: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 120,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/cmd",
            data=json.dumps({"command": command, "params": params or {}}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self._opener.open(request, timeout=timeout) as response:
            result = _parse_sse_json(response.read())
        if result.get("success") is False:
            raise OSError(
                f"ALE command {command!r} failed: {result.get('error') or result}"
            )
        return result


class AleRestCompatibilityServer:
    def __init__(self, native_base_url: str, platform: str):
        self.client = AleCommandClient(native_base_url)
        self.platform = "Linux" if platform.casefold() == "linux" else platform
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("ALE compatibility server is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self._server is not None:
            return
        adapter = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("ALE REST compatibility: " + fmt, *args)

            def _send(
                self,
                status: int,
                payload: bytes,
                content_type: str = "application/json",
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _json(self, payload: Any, status: int = 200) -> None:
                self._send(
                    status,
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                )

            def _request_body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _fields(self) -> tuple[dict[str, str], dict[str, bytes]]:
                body = self._request_body()
                content_type = self.headers.get("Content-Type", "")
                if content_type.startswith("application/json"):
                    value = json.loads(body or b"{}")
                    return (
                        {str(key): value for key, value in value.items()},
                        {},
                    )
                values = urllib.parse.parse_qs(
                    body.decode("utf-8", errors="replace"), keep_blank_values=True
                )
                return {key: items[-1] for key, items in values.items()}, {}

            def _multipart_boundary(self) -> bytes:
                content_type = self.headers.get("Content-Type", "")
                message = Message()
                message["content-type"] = content_type
                boundary = message.get_param("boundary", header="content-type")
                if not boundary:
                    raise ValueError("multipart boundary is required")
                return str(boundary).encode("utf-8")

            def _stream_upload(self) -> str:
                boundary = self._multipart_boundary()
                reader = _MultipartReader(
                    self.rfile, int(self.headers.get("Content-Length") or 0)
                )
                if reader.readline().rstrip(b"\r\n") != b"--" + boundary:
                    raise OSError("invalid initial multipart boundary")
                fields: dict[str, str] = {}
                uploaded_path = ""
                finished = False
                while not finished:
                    headers: dict[str, str] = {}
                    while True:
                        line = reader.readline()
                        if line in {b"\r\n", b"\n", b""}:
                            break
                        key, separator, value = line.decode(
                            "utf-8", errors="replace"
                        ).partition(":")
                        if separator:
                            headers[key.strip().casefold()] = value.strip()
                    disposition = Message()
                    disposition["content-disposition"] = headers.get(
                        "content-disposition", ""
                    )
                    name = disposition.get_param(
                        "name", header="content-disposition"
                    )
                    if name == "file_data":
                        path = fields.get("file_path", "")
                        if not path:
                            raise ValueError(
                                "multipart file_path must precede file_data"
                            )
                        uploaded_path = path
                        parent = str(PurePosixPath(path).parent)
                        if parent not in {"", ".", "/"}:
                            self._command(
                                "run_command",
                                {
                                    "command": f"mkdir -p {shlex.quote(parent)}",
                                    "timeout": 30,
                                },
                                timeout=60,
                            )
                        pending = bytearray()
                        append = False

                        def flush(*, final: bool = False) -> None:
                            nonlocal append
                            while len(pending) >= _FILE_CHUNK_BYTES or (
                                final and pending
                            ):
                                size = min(len(pending), _FILE_CHUNK_BYTES)
                                chunk = bytes(pending[:size])
                                del pending[:size]
                                self._command(
                                    "write_bytes",
                                    {
                                        "path": path,
                                        "content_b64": base64.b64encode(chunk).decode(
                                            "ascii"
                                        ),
                                        "append": append,
                                    },
                                    timeout=180,
                                )
                                append = True

                        def write_chunk(chunk: bytes) -> None:
                            pending.extend(chunk)
                            flush()

                        finished = reader.read_part(boundary, write_chunk)
                        flush(final=True)
                        if not append:
                            self._command(
                                "write_bytes",
                                {"path": path, "content_b64": "", "append": False},
                                timeout=60,
                            )
                    else:
                        value = bytearray()
                        finished = reader.read_part(boundary, value.extend)
                        if name:
                            fields[str(name)] = value.decode(
                                "utf-8", errors="replace"
                            )
                if not uploaded_path:
                    raise ValueError("multipart file_data is required")
                return uploaded_path

            def _command(
                self,
                command: str,
                params: dict[str, Any] | None = None,
                *,
                timeout: float = 120,
            ) -> dict[str, Any]:
                return adapter.client.command(command, params, timeout=timeout)

            @staticmethod
            def _command_text(command: Any) -> str:
                if isinstance(command, (list, tuple)):
                    return shlex.join(str(item) for item in command)
                return str(command or "")

            @staticmethod
            def _legacy_result(result: dict[str, Any]) -> dict[str, Any]:
                returncode = int(result.get("return_code", result.get("returncode", 0)))
                output = str(result.get("stdout") or result.get("output") or "")
                error = str(result.get("stderr") or result.get("error") or "")
                return {
                    "status": "success" if returncode == 0 else "error",
                    "returncode": returncode,
                    "output": output,
                    "error": error,
                }

            def _execute(self, fields: dict[str, Any]) -> dict[str, Any]:
                command = self._command_text(fields.get("command"))
                timeout = float(fields.get("timeout") or 120)
                result = self._command(
                    "run_command",
                    {"command": command, "timeout": timeout},
                    timeout=max(30, timeout + 30),
                )
                return self._legacy_result(result)

            def _read_file(self, path: str) -> bytes:
                chunks: list[bytes] = []
                offset = 0
                while True:
                    result = self._command(
                        "read_bytes",
                        {
                            "path": path,
                            "offset": offset,
                            "length": _FILE_CHUNK_BYTES,
                        },
                        timeout=180,
                    )
                    chunk = base64.b64decode(result.get("content_b64") or "")
                    chunks.append(chunk)
                    offset += len(chunk)
                    if len(chunk) < _FILE_CHUNK_BYTES:
                        return b"".join(chunks)

            def _write_file(self, path: str, payload: bytes) -> None:
                parent = str(PurePosixPath(path).parent)
                if parent not in {"", ".", "/"}:
                    self._command(
                        "run_command",
                        {"command": f"mkdir -p {shlex.quote(parent)}", "timeout": 30},
                        timeout=60,
                    )
                offsets = range(0, len(payload), _FILE_CHUNK_BYTES)
                for index, offset in enumerate(offsets):
                    chunk = payload[offset : offset + _FILE_CHUNK_BYTES]
                    self._command(
                        "write_bytes",
                        {
                            "path": path,
                            "content_b64": base64.b64encode(chunk).decode("ascii"),
                            "append": index > 0,
                        },
                        timeout=180,
                    )
                if not payload:
                    self._command(
                        "write_bytes",
                        {"path": path, "content_b64": "", "append": False},
                        timeout=60,
                    )

            def do_GET(self) -> None:  # noqa: N802
                try:
                    if self.path == "/platform":
                        self._send(200, adapter.platform.encode("utf-8"), "text/plain")
                    elif self.path == "/screenshot":
                        result = self._command("screenshot", {"format": "png"})
                        self._send(
                            200,
                            base64.b64decode(result.get("image_data") or ""),
                            "image/png",
                        )
                    elif self.path == "/accessibility":
                        result = self._command("get_accessibility_tree")
                        tree = result.get("tree") or result.get("accessibility_tree") or ""
                        self._json({"AT": tree})
                    elif self.path == "/terminal":
                        self._json({"output": ""})
                    else:
                        self._json({"error": "not found"}, 404)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("ALE compatibility GET %s failed", self.path)
                    self._json({"error": f"{type(exc).__name__}: {exc}"}, 502)

            def do_POST(self) -> None:  # noqa: N802
                try:
                    if self.path == "/setup/upload":
                        path = self._stream_upload()
                        self._json({"status": "success", "path": path})
                        return
                    fields, files = self._fields()
                    if self.path in {
                        "/execute",
                        "/setup/execute",
                        "/setup/execute_with_verification",
                    }:
                        self._json(self._execute(fields))
                    elif self.path == "/run_python":
                        command = f"/usr/bin/python3 -c {shlex.quote(str(fields.get('code') or ''))}"
                        self._json(self._execute({"command": command, "timeout": 90}))
                    elif self.path == "/run_bash_script":
                        script = str(fields.get("script") or "")
                        working_dir = str(fields.get("working_dir") or "").strip()
                        command = f"/bin/bash -lc {shlex.quote(script)}"
                        if working_dir:
                            command = f"cd {shlex.quote(working_dir)} && {command}"
                        self._json(
                            self._execute(
                                {"command": command, "timeout": fields.get("timeout", 30)}
                            )
                        )
                    elif self.path == "/screen_size":
                        result = self._command("get_screen_size")
                        size = result.get("size") if isinstance(result.get("size"), dict) else result
                        self._json(
                            {
                                "width": int(size.get("width", 0)),
                                "height": int(size.get("height", 0)),
                            }
                        )
                    elif self.path == "/file":
                        path = str(fields.get("file_path") or "")
                        if not path:
                            self._json({"error": "file_path is required"}, 400)
                        else:
                            self._send(200, self._read_file(path), "application/octet-stream")
                    elif self.path == "/setup/open_file":
                        result = self._command("open", {"target": str(fields.get("path") or "")})
                        self._json(result)
                    elif self.path == "/setup/change_wallpaper":
                        result = self._command(
                            "set_wallpaper", {"path": str(fields.get("path") or "")}
                        )
                        self._json(result)
                    elif self.path == "/setup/launch":
                        command = fields.get("command")
                        if isinstance(command, (list, tuple)) and command:
                            app = str(command[0])
                            args = [str(item) for item in command[1:]]
                        else:
                            app = "bash"
                            args = ["-lc", str(command or "")]
                        launched = self._command(
                            "launch", {"app": app, "args": args}, timeout=30
                        )
                        self._json(
                            {
                                "status": "success",
                                "returncode": 0,
                                "output": str(launched.get("pid") or ""),
                                "error": "",
                                "pid": launched.get("pid"),
                            }
                        )
                    elif self.path == "/desktop_path":
                        result = self._execute(
                            {
                                "command": "xdg-user-dir DESKTOP 2>/dev/null || printf '%s/Desktop\\n' \"$HOME\"",
                                "timeout": 30,
                            }
                        )
                        self._json({"desktop_path": result["output"].strip()})
                    elif self.path == "/list_directory":
                        result = self._command("list_dir", {"path": str(fields.get("path") or "")})
                        tree = result.get("directory_tree") or result.get("entries") or result
                        self._json({"directory_tree": tree})
                    elif self.path == "/setup/activate_window":
                        title = shlex.quote(str(fields.get("window_name") or ""))
                        self._json(self._execute({"command": f"wmctrl -a {title}", "timeout": 30}))
                    elif self.path == "/setup/close_window":
                        title = shlex.quote(str(fields.get("window_name") or ""))
                        self._json(self._execute({"command": f"wmctrl -c {title}", "timeout": 30}))
                    elif self.path in {"/start_recording", "/end_recording"}:
                        self._json({"status": "success"})
                    else:
                        self._json({"error": "not found"}, 404)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("ALE compatibility POST %s failed", self.path)
                    self._json({"error": f"{type(exc).__name__}: {exc}"}, 502)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ale-rest-compat",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "ALE REST compatibility server listening at %s for %s",
            self.base_url,
            getattr(self.client, "base_url", "<custom-client>"),
        )

    def close(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5)
