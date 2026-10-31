"""Generic backend that executes an arbitrary configured command."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from cuamaster.runtime import DEFAULT_ACTION_PROTOCOL, SUPPORTED_ACTION_PROTOCOLS

from .base import CollectionBackend, RunSpec
from ..profiles import MatrixConfig, VMProfile


class _Fields(dict):
    def __missing__(self, key):
        raise ValueError(f"unknown command backend placeholder: {{{key}}}")


class CommandBackend(CollectionBackend):
    """Connect any runner without importing its Python package.

    The configured process receives the image/profile through placeholders and
    should write task directories containing ``chat.jsonl`` and/or
    ``computer_events.jsonl`` below ``output_dir``. Tool calls from chat are
    merged with actuator-grounded GUI events into one canonical CUA timeline.
    """

    def prepare(
        self,
        *,
        config: MatrixConfig,
        profile: VMProfile,
        output_dir: Path,
        cli: Any,
        base_env: Mapping[str, str],
    ) -> RunSpec:
        defaults = config.defaults.get("backend_options") or {}
        options = {
            **defaults,
            **profile.backend_options,
            **getattr(cli, "backend_options", {}),
        }
        action_protocol = str(
            options.get("action_protocol") or DEFAULT_ACTION_PROTOCOL
        ).strip()
        if action_protocol not in SUPPORTED_ACTION_PROTOCOLS - {"auto"}:
            raise ValueError("command backend action_protocol must be canonical or qwen_xml")
        raw_command = options.get("command")
        if not isinstance(raw_command, list) or not raw_command:
            raise ValueError(
                f"profile {profile.id}: command backend requires backend_options.command array"
            )
        fields = _Fields(
            profile_id=profile.id,
            output_dir=str(output_dir),
            image_path=profile.image_path or "",
            image_id=profile.image_id or "",
            os_type=profile.os_type,
            os_version=profile.os_version,
            architecture=profile.architecture,
            provider=profile.provider,
            num_envs=str(profile.num_envs),
            config_dir=str(config.path.parent),
            project_root=str(Path(__file__).resolve().parents[2]),
        )
        command = [str(value).format_map(fields) for value in raw_command]
        raw_cwd = str(options.get("cwd") or config.path.parent)
        cwd = Path(raw_cwd.format_map(fields)).expanduser()
        if not cwd.is_absolute():
            cwd = config.path.parent / cwd
        cwd = cwd.resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(f"command backend cwd not found: {cwd}")
        env = dict(base_env)
        env.update(profile.env)
        configured_env = options.get("env") or {}
        if not isinstance(configured_env, dict):
            raise ValueError("command backend env must be an object")
        env.update({str(key): str(value).format_map(fields) for key, value in configured_env.items()})
        env.update({
            "GUI_TRAJECTORY_PROFILE_ID": profile.id,
            "GUI_TRAJECTORY_IMAGE_PATH": profile.image_path or "",
            "GUI_TRAJECTORY_OUTPUT_DIR": str(output_dir),
            "GUI_TRAJECTORY_ACTION_PROTOCOL": action_protocol,
            "CUA_ACTION_PROTOCOL": action_protocol,
        })
        if options.get("require_api_key") and not (
            env.get("OPENROUTER_API_KEY") or env.get("ANTHROPIC_API_KEY")
        ):
            raise ValueError("command backend requires an LLM API key")
        executable = command[0]
        if os.path.sep in executable and not Path(executable).expanduser().exists():
            raise FileNotFoundError(f"command executable not found: {executable}")
        return RunSpec(
            command=command,
            env=env,
            cwd=cwd,
            metadata={
                "backend": "command",
                "action_protocol": action_protocol,
                "options": options,
            },
        )
