"""Claude Code adapter that runs on the host and controls tools inside the guest."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cuamaster.benchmarks import claude_code_tool_policy

from .collection_policy import append_collection_policy


def _controller_url(env: Any) -> str:
    value = getattr(env, "vm_base_url", None)
    if value:
        return str(value).rstrip("/")
    controller = getattr(env, "controller", None)
    value = getattr(controller, "http_server", None)
    if value:
        return str(value).rstrip("/")
    raise RuntimeError("DesktopEnv does not expose an OSWorld controller URL")


def _controller_platform(controller_url: str) -> str:
    try:
        with urllib.request.urlopen(f"{controller_url}/platform", timeout=15) as response:
            raw = response.read().decode("utf-8", errors="replace").strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            return str(parsed).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({type(exc).__name__}: {exc})"


def _expected_platform(os_type: str) -> str:
    normalized = os_type.strip().casefold()
    if normalized in {"ubuntu", "linux"}:
        return "Linux"
    if normalized in {"windows", "win32"}:
        return "Windows"
    if normalized in {"darwin", "macos", "mac"}:
        return "Darwin"
    return os_type.strip()


def validate_controller_platform(env: Any) -> str:
    controller_url = _controller_url(env)
    actual = _controller_platform(controller_url)
    declared = os.environ.get("WEAVEBENCH_VM_PROFILE_JSON")
    if not declared:
        return actual
    try:
        profile = json.loads(declared)
    except json.JSONDecodeError as exc:
        raise RuntimeError("WEAVEBENCH_VM_PROFILE_JSON is invalid JSON") from exc
    os_spec = profile.get("os") if isinstance(profile, dict) else None
    declared_os = os_spec.get("type") if isinstance(os_spec, dict) else None
    if not declared_os:
        return actual
    expected = _expected_platform(str(declared_os))
    if actual.casefold() != expected.casefold():
        raise RuntimeError(
            "controller platform mismatch: "
            f"profile declares {declared_os!r} but guest reports {actual!r} "
            f"at {controller_url}"
        )
    return actual


def _controller_execute(
    controller_url: str, command: list[str], *, timeout: int = 30
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{controller_url}/execute",
        data=json.dumps({"command": command, "shell": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result if isinstance(result, dict) else {"error": "non-object response"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _platform_probes(controller_url: str, platform_name: str) -> dict[str, Any]:
    probes = {
        "python_platform": [
            "python",
            "-c",
            "import json,platform,sys; print(json.dumps({"
            "'platform':platform.platform(),'python':sys.version,'machine':platform.machine()}))",
        ],
    }
    if platform_name.casefold() == "windows":
        probes.update({
            "os_version": ["cmd", "/c", "ver"],
            "powershell": [
                "powershell",
                "-NoProfile",
                "-Command",
                "$PSVersionTable.PSVersion.ToString()",
            ],
            "applications": [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Command chrome,msedge,soffice,winword -ErrorAction SilentlyContinue | "
                "Select-Object -ExpandProperty Source",
            ],
        })
    elif platform_name.casefold() == "darwin":
        probes.update({
            "os_version": ["sw_vers"],
            "kernel": ["uname", "-a"],
            "applications": [
                "/bin/sh",
                "-c",
                "for app in Safari Pages 'Google Chrome' LibreOffice; do "
                "test -d \"/Applications/$app.app\" && echo \"$app\"; done",
            ],
        })
    return {
        name: _controller_execute(controller_url, command)
        for name, command in probes.items()
    }


def _host_mcp_config(
    controller_url: str,
    output_dir: Path,
    controller_platform: str,
    *,
    enable_guest_shell: bool = False,
) -> dict[str, Any]:
    env = {
        "GUI_TRAJECTORY_CONTROLLER_URL": controller_url,
        "GUI_TRAJECTORY_CONTROLLER_PLATFORM": controller_platform,
        "GUI_TRAJECTORY_MCP_EVENT_LOG": str(output_dir / "computer_events.jsonl"),
        "GUI_TRAJECTORY_MCP_SCREENSHOT_DIR": str(output_dir / "screenshots"),
        "GUI_TRAJECTORY_SHELL_EVENT_LOG": str(output_dir / "shell_events.jsonl"),
        "GUI_TRAJECTORY_ENABLE_GUEST_SHELL": str(enable_guest_shell).lower(),
        "GUI_TRAJECTORY_CONTROLLER_TIMEOUT_S": os.environ.get(
            "GUI_TRAJECTORY_CONTROLLER_TIMEOUT_S", "90"
        ),
    }
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return {
        "mcpServers": {
            "weavebench_computer": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "cuamaster.collection.host_computer_mcp"],
                "env": env,
            }
        }
    }


def _claude_tool_permissions(interaction_harness: str) -> list[str]:
    policy = claude_code_tool_policy(
        interaction_harness,
        runtime_location="host",
    )
    return [
        "--allowedTools",
        policy.allowed_argument,
        "--disallowedTools",
        policy.disallowed_argument,
    ]


def build_host_claude_agent(upstream: Any) -> type:
    base_agent = upstream.ClaudeCodeAgent

    class HostClaudeCodeAgent(base_agent):
        runtime_location = "host"

        def bootstrap(self, env: Any) -> None:
            validate_controller_platform(env)

        def configure(self, env: Any) -> None:
            validate_controller_platform(env)

        def _materialize_host_chat(self, output_dir: Path) -> None:
            raw_path = output_dir / "claude_stream.jsonl"
            chat_path = output_dir / "chat.jsonl"
            events = upstream.parse_claudecode_stream_file(raw_path)
            records = upstream.claudecode_stream_to_openclaw_v3(
                events,
                model=self.model,
                cwd="controller-host",
                thinking_level="medium",
            )
            upstream.write_openclaw_v3_jsonl(records, chat_path)

        def _capture_environment_fingerprint(
            self, env: Any, output_dir: Path
        ) -> None:
            controller_url = _controller_url(env)
            declared = os.environ.get("WEAVEBENCH_VM_PROFILE_JSON")
            try:
                declared_profile: Any = json.loads(declared) if declared else None
            except json.JSONDecodeError:
                declared_profile = declared
            fingerprint = {
                "schema_version": "gui_trajectory_collection.environment.v1",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "profile_id": os.environ.get("WEAVEBENCH_VM_PROFILE_ID"),
                "declared_profile": declared_profile,
                "runtime_location": "host",
                "controller_url": controller_url,
                "controller_platform": _controller_platform(controller_url),
                "probes": {},
            }
            fingerprint["probes"] = _platform_probes(
                controller_url, fingerprint["controller_platform"]
            )
            (output_dir / "environment_fingerprint.json").write_text(
                json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        def run(
            self,
            env: Any,
            instruction: str,
            output_dir: Path,
            system_prompt_override: str | None = None,
            already_configured: bool = False,
            **_ignored: Any,
        ) -> dict[str, Any]:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                screenshot = env.controller.get_screenshot()
                if screenshot:
                    (output_dir / "init_screenshot.png").write_bytes(screenshot)
            except Exception:  # noqa: BLE001
                pass
            controller_url = _controller_url(env)
            mcp_config = output_dir / "host_mcp.json"
            controller_platform = validate_controller_platform(env)
            requested_mode = os.environ.get(
                "CUA_INTERACTION_HARNESS",
                os.environ.get("GUI_TRAJECTORY_REQUESTED_MODE", "gui"),
            ).strip().lower()
            enable_guest_shell = requested_mode in {"cli", "hybrid"}
            mcp_config.write_text(
                json.dumps(
                    _host_mcp_config(
                        controller_url,
                        output_dir,
                        controller_platform,
                        enable_guest_shell=enable_guest_shell,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            workspace = output_dir / "host_workspace"
            workspace.mkdir(exist_ok=True)
            raw_path = output_dir / "claude_stream.jsonl"
            log_path = output_dir / "agent.log"
            prompt = append_collection_policy(
                self._build_prompt(instruction, system_prompt_override)
            )
            if enable_guest_shell:
                prompt += (
                    "\n\n=== HYBRID GUEST SHELL ===\n"
                    "The built-in Bash tool is disabled because it would execute on the "
                    "controller host. Use `mcp__weavebench_computer__shell` for every "
                    "command-line operation; it executes inside the connected desktop VM. "
                    "Use PowerShell syntax when the guest platform is Windows and POSIX shell "
                    "syntax when it is macOS or Linux. Continue to use "
                    "`mcp__weavebench_computer__computer` for visual interaction.\n"
                    "=== END HYBRID GUEST SHELL ===\n"
                )
            claude_bin = os.environ.get("GUI_TRAJECTORY_CLAUDE_BIN", "claude")
            command = [
                claude_bin,
                "--print",
                "--output-format",
                "stream-json",
                "--verbose",
                "--bare",
                "--strict-mcp-config",
                "--mcp-config",
                str(mcp_config),
                "--effort",
                "high",
                "--model",
                self.model,
                *_claude_tool_permissions(requested_mode),
            ]
            child_env = os.environ.copy()
            base_url = self.litellm_base_url.rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            child_env.update({
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_API_KEY": self.litellm_api_key,
                "ANTHROPIC_AUTH_TOKEN": self.litellm_api_key,
                "ANTHROPIC_MODEL": self.model,
                "CLAUDE_CONFIG_DIR": str(output_dir / ".claude"),
                "DISABLE_PROMPT_CACHING": "1",
                "DISABLE_INTERLEAVED_THINKING": "1",
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            })
            started = time.monotonic()
            timed_out = False
            exit_code = -1
            try:
                with raw_path.open("wb") as stdout, log_path.open("wb") as stderr:
                    completed = subprocess.run(
                        command,
                        input=prompt.encode("utf-8"),
                        stdout=stdout,
                        stderr=stderr,
                        cwd=workspace,
                        env=child_env,
                        timeout=self.timeout,
                        check=False,
                    )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                with log_path.open("ab") as stderr:
                    stderr.write(f"\n[host_claude_agent] timeout={self.timeout}\n".encode())
            elapsed = time.monotonic() - started
            if not raw_path.exists():
                raw_path.write_text("", encoding="utf-8")
            self._materialize_host_chat(output_dir)
            try:
                screenshot = env.controller.get_screenshot()
                if screenshot:
                    (output_dir / "final_screenshot.png").write_bytes(screenshot)
            except Exception:  # noqa: BLE001
                pass
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[host_claude_agent] AGENT_EXIT={exit_code}\n")
                log.write("[host_claude_agent] RETRIES_USED=0\n")
            self._capture_environment_fingerprint(env, output_dir)
            status = {
                "agent_done": not timed_out and exit_code == 0,
                "elapsed_seconds": round(elapsed, 2),
                "timed_out": timed_out,
                "exit_code": exit_code,
                "model": self.model,
                "gui": self.gui,
                "runtime_location": "host",
                "shell_runtime_location": "guest" if enable_guest_shell else None,
                "mode": requested_mode,
                "controller_platform": controller_platform,
            }
            (output_dir / "execution_status.json").write_text(
                json.dumps(status, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return status

    HostClaudeCodeAgent.__name__ = "HostClaudeCodeAgent"
    return HostClaudeCodeAgent
