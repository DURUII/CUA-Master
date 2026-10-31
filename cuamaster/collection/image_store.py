"""Versioned VM image catalog shared by all collection backends."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLATFORM_DIRECTORIES = {
    "darwin": "macos",
    "linux": "linux",
    "macos": "macos",
    "ubuntu": "linux",
    "windows": "windows",
}


@dataclass(frozen=True)
class ImageRecord:
    id: str
    path: Path
    descriptor_path: Path | None
    format: str
    os_type: str
    os_version: str
    architecture: str
    snapshot: str
    software: dict[str, str]
    source: str | None
    size_bytes: int | None
    derived_from: str | None
    build: dict[str, Any]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "descriptor": str(self.descriptor_path) if self.descriptor_path else None,
            "format": self.format,
            "os": {
                "type": self.os_type,
                "version": self.os_version,
                "architecture": self.architecture,
            },
            "snapshot": self.snapshot,
            "software": self.software,
            "source": self.source,
            "size_bytes": self.size_bytes,
            "derived_from": self.derived_from,
            "build": self.build,
        }


class ImageStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.catalog_path = self.root / "catalog.json"

    def _raw(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"image catalog not found: {self.catalog_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid image catalog {self.catalog_path}: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError(f"unsupported image catalog schema: {self.catalog_path}")
        if not isinstance(raw.get("images"), dict):
            raise ValueError("image catalog 'images' must be an object")
        return raw

    def list(self) -> list[ImageRecord]:
        return [self._record(image_id, row) for image_id, row in self._raw()["images"].items()]

    def resolve(self, image_id: str, *, verify: bool = True) -> ImageRecord:
        images = self._raw()["images"]
        if image_id not in images:
            raise KeyError(f"image id not found in {self.catalog_path}: {image_id}")
        record = self._record(image_id, images[image_id])
        if verify:
            self.verify_record(record)
        return record

    def _record(self, image_id: str, row: Any) -> ImageRecord:
        if not isinstance(row, dict):
            raise ValueError(f"image {image_id}: entry must be an object")
        raw_path = Path(str(row.get("path") or ""))
        path = raw_path if raw_path.is_absolute() else self.root / raw_path
        raw_descriptor = row.get("descriptor")
        descriptor_path = None
        if raw_descriptor:
            descriptor = Path(str(raw_descriptor))
            descriptor_path = (
                descriptor if descriptor.is_absolute() else self.root / descriptor
            )
        os_spec = row.get("os") if isinstance(row.get("os"), dict) else {}
        software = row.get("software") if isinstance(row.get("software"), dict) else {}
        size = row.get("size_bytes")
        return ImageRecord(
            id=image_id,
            # Keep the managed store path in manifests even when it is a
            # symlink; Path.resolve() would leak the original external path.
            path=Path(os.path.abspath(path)),
            descriptor_path=(
                Path(os.path.abspath(descriptor_path)) if descriptor_path else None
            ),
            format=str(row.get("format") or path.suffix.lstrip(".") or "unknown"),
            os_type=str(os_spec.get("type") or "unknown"),
            os_version=str(os_spec.get("version") or "unknown"),
            architecture=str(os_spec.get("architecture") or "unknown"),
            snapshot=str(row.get("snapshot") or "clean"),
            software={str(key): str(value) for key, value in software.items()},
            source=str(row["source"]) if row.get("source") else None,
            size_bytes=int(size) if size is not None else None,
            derived_from=(
                str(row["derived_from"]) if row.get("derived_from") else None
            ),
            build=dict(row.get("build") or {}) if isinstance(row.get("build"), dict) else {},
        )

    @staticmethod
    def verify_record(record: ImageRecord) -> None:
        if not record.path.is_file():
            raise FileNotFoundError(f"image {record.id} is missing: {record.path}")
        if not os.access(record.path, os.R_OK):
            raise PermissionError(f"image {record.id} is not readable: {record.path}")
        actual_size = record.path.stat().st_size
        if record.size_bytes is not None and actual_size != record.size_bytes:
            raise ValueError(
                f"image {record.id} size mismatch: catalog={record.size_bytes}, actual={actual_size}"
            )
        if record.descriptor_path is None:
            raise ValueError(f"image {record.id} has no descriptor")
        try:
            descriptor = json.loads(record.descriptor_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"image {record.id} descriptor is missing: {record.descriptor_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"image {record.id} descriptor is invalid: {record.descriptor_path}: {exc}"
            ) from exc
        if not isinstance(descriptor, dict) or descriptor.get("schema_version") != 1:
            raise ValueError(f"image {record.id} descriptor has unsupported schema")
        if descriptor.get("image_id") != record.id:
            raise ValueError(
                f"image {record.id} descriptor image_id mismatch: "
                f"{descriptor.get('image_id')!r}"
            )
        descriptor_os = descriptor.get("os")
        if not isinstance(descriptor_os, dict):
            raise ValueError(f"image {record.id} descriptor has no OS metadata")
        expected_os = {
            "type": record.os_type,
            "version": record.os_version,
            "architecture": record.architecture,
        }
        for field, expected in expected_os.items():
            actual = descriptor_os.get(field)
            if field == "type":
                matches = (
                    isinstance(actual, str)
                    and actual.casefold() == expected.casefold()
                )
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"image {record.id} descriptor OS {field} mismatch: "
                    f"catalog={expected!r}, descriptor={actual!r}"
                )
        apps = descriptor.get("apps")
        if not isinstance(apps, list) or not all(
            isinstance(app, dict) and isinstance(app.get("name"), str) for app in apps
        ):
            raise ValueError(f"image {record.id} descriptor apps must be a list of apps")
        inventory_status = descriptor.get("inventory_status", "complete")
        if inventory_status not in {"complete", "pending"}:
            raise ValueError(
                f"image {record.id} descriptor inventory_status must be complete or pending"
            )
        if record.build:
            legacy_verified = record.build.get("verified") is True
            artifact_verified = record.build.get("artifact_verified") is True
            if not legacy_verified and not artifact_verified:
                raise ValueError(f"image {record.id} artifact is not marked verified")
            if (
                inventory_status == "complete"
                and record.build.get("boot_verified") is False
            ):
                raise ValueError(
                    f"image {record.id} inventory is complete but boot is not verified"
                )

    def _write_catalog(self, raw: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=".catalog.json.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(raw, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, self.catalog_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def register(
        self,
        *,
        image_id: str,
        source: str | Path,
        os_type: str,
        os_version: str,
        architecture: str,
        snapshot: str,
        software: dict[str, str],
        mode: str = "link",
        derived_from: str | None = None,
        build: dict[str, Any] | None = None,
    ) -> ImageRecord:
        if mode not in {"link", "hardlink", "copy", "reference", "adopt"}:
            raise ValueError("mode must be link, hardlink, copy, reference, or adopt")
        if not _SAFE_ID.fullmatch(image_id):
            raise ValueError(f"unsafe image id: {image_id!r}")
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        raw = self._raw() if self.catalog_path.exists() else {"schema_version": 1, "images": {}}
        if image_id in raw["images"]:
            raise ValueError(f"image id already exists: {image_id}")
        normalized_os_type = os_type.strip().lower()
        platform_dir = _PLATFORM_DIRECTORIES.get(
            normalized_os_type,
            re.sub(r"[^a-z0-9._-]+", "-", normalized_os_type).strip("-."),
        )
        if not platform_dir:
            raise ValueError(f"unsafe os type: {os_type!r}")
        suffix = source_path.suffix or ".img"
        relative = Path(platform_dir) / image_id / f"disk{suffix}"
        destination = self.root / relative
        if mode == "reference":
            stored_path = source_path
        elif mode == "adopt":
            destination_path = Path(os.path.abspath(destination))
            if source_path != destination_path:
                raise ValueError(
                    f"adopt source must already be at its managed path: {destination_path}"
                )
            if Path(source).expanduser().is_symlink():
                raise ValueError("adopt source must be an independent regular file")
            stored_path = destination
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            if mode == "link":
                destination.symlink_to(source_path)
            elif mode == "hardlink":
                os.link(source_path, destination)
            else:
                shutil.copy2(source_path, destination)
            stored_path = destination
        catalog_path = str(stored_path if stored_path.is_absolute() and mode == "reference" else relative)
        raw["images"][image_id] = {
            "path": catalog_path,
            "descriptor": str(relative.parent / "descriptor.json"),
            "format": suffix.lstrip("."),
            "os": {
                "type": os_type,
                "version": os_version,
                "architecture": architecture,
            },
            "snapshot": snapshot,
            "software": software,
            "source": str(source_path),
            "size_bytes": source_path.stat().st_size,
        }
        if derived_from:
            raw["images"][image_id]["derived_from"] = derived_from
        if build:
            raw["images"][image_id]["build"] = dict(build)
        self._write_catalog(raw)
        # Inventory is collected from a booted guest after registration. Until
        # descriptor.json exists, list works but verify intentionally fails.
        return self.resolve(image_id, verify=False)
