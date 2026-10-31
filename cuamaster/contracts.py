"""Trajectory data contracts shared across the CUAMaster pipeline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


TASK_SCHEMA = "gui_trajectory.task.v2"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EnvironmentRequirement:
    capability: str
    version: str | None = None
    optional: bool = False

    @classmethod
    def from_value(cls, value: Any) -> "EnvironmentRequirement":
        if isinstance(value, str):
            return cls(capability=value)
        if not isinstance(value, Mapping) or not value.get("capability"):
            raise ValueError("environment requirement must be a string or capability object")
        return cls(
            capability=str(value["capability"]),
            version=str(value["version"]) if value.get("version") else None,
            optional=bool(value.get("optional", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "version": self.version,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class AssetReference:
    path: str
    sha256: str
    source: str = "generated"
    media_type: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AssetReference":
        path = str(raw.get("path") or "")
        digest = str(raw.get("sha256") or "")
        if not path or len(digest) != 64:
            raise ValueError("asset path and SHA-256 are required")
        return cls(
            path=path,
            sha256=digest,
            source=str(raw.get("source") or "generated"),
            media_type=str(raw["media_type"]) if raw.get("media_type") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "source": self.source,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class TargetEnvironment:
    os_type: str
    os_versions: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TargetEnvironment":
        os_type = str(raw.get("os_type") or "").strip()
        if not os_type:
            raise ValueError("target environment os_type cannot be empty")
        versions = raw.get("os_versions") or []
        architectures = raw.get("architectures") or []
        if not isinstance(versions, list) or not isinstance(architectures, list):
            raise ValueError(
                "target environment os_versions and architectures must be arrays"
            )
        return cls(
            os_type=os_type,
            os_versions=tuple(sorted({str(item) for item in versions if str(item)})),
            architectures=tuple(
                sorted({str(item) for item in architectures if str(item)})
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_type": self.os_type,
            "os_versions": list(self.os_versions),
            "architectures": list(self.architectures),
        }


@dataclass(frozen=True)
class TaskSpec:
    id: str
    instruction: str
    snapshot: str = "clean"
    config: tuple[dict[str, Any], ...] = ()
    evaluator: dict[str, Any] = field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()
    target_environments: tuple[TargetEnvironment, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskSpec":
        task_id = str(raw.get("id") or "").strip()
        instruction = str(raw.get("instruction") or "").strip()
        if not task_id:
            raise ValueError("task.id cannot be empty")
        if not instruction:
            raise ValueError(f"task {task_id}: instruction cannot be empty")
        config = raw.get("config") or []
        evaluator = raw.get("evaluator") or {}
        metadata = raw.get("metadata") or {}
        capabilities = raw.get("required_capabilities") or []
        target_environments = raw.get("target_environments") or []
        if not isinstance(config, list) or not all(isinstance(item, dict) for item in config):
            raise ValueError(f"task {task_id}: config must be an array of objects")
        if not isinstance(evaluator, dict):
            raise ValueError(f"task {task_id}: evaluator must be an object")
        if not isinstance(metadata, dict):
            raise ValueError(f"task {task_id}: metadata must be an object")
        if not isinstance(capabilities, list):
            raise ValueError(f"task {task_id}: required_capabilities must be an array")
        if not isinstance(target_environments, list) or not all(
            isinstance(item, Mapping) for item in target_environments
        ):
            raise ValueError(f"task {task_id}: target_environments must be an array")
        return cls(
            id=task_id,
            instruction=instruction,
            snapshot=str(raw.get("snapshot") or "clean"),
            config=tuple(dict(item) for item in config),
            evaluator=dict(evaluator),
            required_capabilities=tuple(sorted({str(item) for item in capabilities if str(item)})),
            target_environments=tuple(
                TargetEnvironment.from_dict(item) for item in target_environments
            ),
            metadata=dict(metadata),
            version=str(raw["version"]) if raw.get("version") else None,
        )

    def version_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "instruction": self.instruction,
            "snapshot": self.snapshot,
            "config": list(self.config),
            "evaluator": self.evaluator,
            "required_capabilities": list(self.required_capabilities),
            "metadata": self.metadata,
        }
        if self.target_environments:
            payload["target_environments"] = [
                environment.to_dict() for environment in self.target_environments
            ]
        return payload

    @property
    def resolved_version(self) -> str:
        return self.version or content_digest(self.version_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_SCHEMA,
            **self.version_payload(),
            "version": self.resolved_version,
        }


@dataclass(frozen=True)
class TaskValidation:
    task_id: str
    task_version: str
    valid: bool
    checks: dict[str, bool]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "gui_trajectory.task_validation.v1",
            "task_id": self.task_id,
            "task_version": self.task_version,
            "valid": self.valid,
            "checks": self.checks,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }
