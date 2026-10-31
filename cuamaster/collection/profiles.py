"""Strict, dependency-free loader for the GUI VM profile matrix."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ALL_PLATFORMS = ("linux", "windows", "macos")
_PLATFORM_MODES = {"configured", "all"}
_WORKER_SELECTION_MODES = {"balanced", "random_balanced", "random"}


@dataclass(frozen=True)
class VMProfile:
    id: str
    backend: str
    os_type: str
    os_version: str
    architecture: str
    provider: str
    image_path: str | None
    image_id: str | None
    snapshot: str
    num_envs: int
    software: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    backend_options: dict[str, Any] = field(default_factory=dict)
    image_metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def as_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend": self.backend,
            "os": {
                "type": self.os_type,
                "version": self.os_version,
                "architecture": self.architecture,
            },
            "provider": self.provider,
            "image": {
                "path": self.image_path,
                "id": self.image_id,
                "snapshot": self.snapshot,
                "metadata": self.image_metadata,
            },
            "num_envs": self.num_envs,
            "software": self.software,
        }


@dataclass(frozen=True)
class MatrixConfig:
    path: Path
    defaults: dict[str, Any]
    profiles: tuple[VMProfile, ...]

    def enabled_profiles(self, selected: set[str] | None = None) -> list[VMProfile]:
        profiles = [profile for profile in self.profiles if profile.enabled]
        if selected is not None:
            unknown = selected - {profile.id for profile in self.profiles}
            if unknown:
                raise ValueError(f"unknown profile id(s): {', '.join(sorted(unknown))}")
            profiles = [profile for profile in profiles if profile.id in selected]
        return profiles


def expand_platform_profiles(
    config: MatrixConfig,
    profiles: list[VMProfile],
    platform_mode: str | None = None,
) -> list[VMProfile]:
    mode = str(platform_mode or config.defaults.get("platform_mode") or "configured")
    mode = mode.strip().lower()
    if mode not in _PLATFORM_MODES:
        raise ValueError(
            f"unsupported platform_mode {mode!r}; expected one of "
            f"{', '.join(sorted(_PLATFORM_MODES))}"
        )
    if mode == "configured":
        return list(profiles)

    platform_profiles = config.defaults.get("platform_profiles")
    if not isinstance(platform_profiles, dict):
        raise ValueError(
            "platform_mode=all requires defaults.platform_profiles with "
            "linux, windows, and macos entries"
        )
    missing = [name for name in _ALL_PLATFORMS if name not in platform_profiles]
    if missing:
        raise ValueError(
            "platform_mode=all is missing defaults.platform_profiles entries: "
            + ", ".join(missing)
        )

    worker_selection = str(
        config.defaults.get("platform_worker_selection") or "random_balanced"
    ).strip().lower()
    if worker_selection not in _WORKER_SELECTION_MODES:
        raise ValueError(
            f"unsupported platform_worker_selection {worker_selection!r}; expected one of "
            f"{', '.join(sorted(_WORKER_SELECTION_MODES))}"
        )
    worker_seed = config.defaults.get("platform_worker_seed")
    if worker_seed in (None, "") and worker_selection in {"random", "random_balanced"}:
        worker_seed = "{resource_id}"

    expanded: list[VMProfile] = []
    seen: set[str] = set()
    for profile in profiles:
        for platform in _ALL_PLATFORMS:
            row = platform_profiles[platform]
            if not isinstance(row, dict):
                raise ValueError(
                    f"defaults.platform_profiles.{platform} must be an object"
                )
            os_spec = row.get("os") or {}
            image = row.get("image") or {}
            if not isinstance(os_spec, dict) or not isinstance(image, dict):
                raise ValueError(
                    f"defaults.platform_profiles.{platform}.os and image must be objects"
                )
            os_type = str(os_spec.get("type") or "").strip()
            if not os_type:
                raise ValueError(
                    f"defaults.platform_profiles.{platform}.os.type is required"
                )
            image_path = _expand_path(image.get("path"), config.path.parent)
            image_id = str(image.get("id") or "").strip() or None
            if not image_id and not image_path:
                raise ValueError(
                    f"defaults.platform_profiles.{platform}.image requires id or path"
                )
            expanded_id = f"{profile.id}-{platform}"
            if not _PROFILE_ID.fullmatch(expanded_id):
                raise ValueError(
                    f"expanded profile id is not a safe slug: {expanded_id!r}"
                )
            if expanded_id in seen:
                raise ValueError(f"duplicate expanded profile id: {expanded_id}")
            seen.add(expanded_id)

            row_options = row.get("backend_options") or {}
            if not isinstance(row_options, dict):
                raise ValueError(
                    f"defaults.platform_profiles.{platform}.backend_options must be an object"
                )
            backend_options = {
                **profile.backend_options,
                **row_options,
                "force_all_tasks_on_platform": True,
                "cross_platform_parent_id": profile.id,
                "cross_platform_platform": platform,
            }
            backend_options.setdefault(
                "sandbox_worker_selection", worker_selection
            )
            if worker_seed not in (None, ""):
                backend_options.setdefault(
                    "sandbox_worker_selection_seed", str(worker_seed)
                )

            num_envs = int(row.get("num_envs", profile.num_envs))
            if num_envs < 1:
                raise ValueError(
                    f"defaults.platform_profiles.{platform}.num_envs must be >= 1"
                )
            expanded.append(replace(
                profile,
                id=expanded_id,
                backend=str(row.get("backend") or profile.backend),
                provider=str(row.get("provider") or profile.provider),
                os_type=os_type,
                os_version=str(os_spec.get("version") or "unknown"),
                architecture=str(os_spec.get("architecture") or "x86_64"),
                image_path=image_path,
                image_id=image_id,
                snapshot=str(image.get("snapshot") or profile.snapshot),
                num_envs=num_envs,
                software={
                    **profile.software,
                    **_string_map(
                        row.get("software"),
                        f"defaults.platform_profiles.{platform}.software",
                    ),
                },
                env={
                    **profile.env,
                    **_string_map(
                        row.get("env"),
                        f"defaults.platform_profiles.{platform}.env",
                    ),
                },
                backend_options=backend_options,
                image_metadata={},
            ))
    return expanded


def _string_map(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): str(item) for key, item in value.items()}


def _expand_path(value: Any, base_dir: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_matrix_config(path: str | Path) -> MatrixConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("matrix config must be a JSON object")
    if raw.get("schema_version") != 1:
        raise ValueError("matrix config schema_version must be 1")
    defaults = raw.get("defaults") or {}
    rows = raw.get("profiles")
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    if not isinstance(rows, list) or not rows:
        raise ValueError("profiles must be a non-empty array")

    profiles: list[VMProfile] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"profiles[{index}] must be an object")
        profile_id = str(row.get("id") or "")
        if not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError(f"profiles[{index}].id is not a safe slug: {profile_id!r}")
        if profile_id in seen:
            raise ValueError(f"duplicate profile id: {profile_id}")
        seen.add(profile_id)
        os_spec = row.get("os") or {}
        image = row.get("image") or {}
        if not isinstance(os_spec, dict) or not isinstance(image, dict):
            raise ValueError(f"profile {profile_id}: os and image must be objects")
        os_type = str(os_spec.get("type") or defaults.get("os_type") or "Ubuntu")
        backend = str(row.get("backend") or defaults.get("backend") or "command")
        provider = str(
            row.get("provider")
            or defaults.get("provider")
            or (
                "sandbox"
                if backend.strip().lower() == "weavebench"
                else "docker"
            )
        )
        if not os_type.strip():
            raise ValueError(f"profile {profile_id}: os.type cannot be empty")
        if not backend.strip():
            raise ValueError(f"profile {profile_id}: backend cannot be empty")
        if not provider.strip():
            raise ValueError(f"profile {profile_id}: provider cannot be empty")
        num_envs = int(row.get("num_envs", defaults.get("num_envs", 1)))
        if num_envs < 1:
            raise ValueError(f"profile {profile_id}: num_envs must be >= 1")
        profiles.append(VMProfile(
            id=profile_id,
            backend=backend,
            os_type=os_type,
            os_version=str(os_spec.get("version") or "unknown"),
            architecture=str(os_spec.get("architecture") or "x86_64"),
            provider=provider,
            image_path=_expand_path(image.get("path"), config_path.parent),
            image_id=str(image["id"]) if image.get("id") else None,
            snapshot=str(image.get("snapshot") or "clean"),
            num_envs=num_envs,
            software=_string_map(row.get("software"), f"profile {profile_id}.software"),
            env=_string_map(row.get("env"), f"profile {profile_id}.env"),
            backend_options=(
                dict(row.get("backend_options") or {})
                if isinstance(row.get("backend_options") or {}, dict)
                else {}
            ),
            enabled=bool(row.get("enabled", True)),
        ))
    return MatrixConfig(config_path, dict(defaults), tuple(profiles))
