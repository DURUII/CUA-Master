"""Build environment capability indexes from the managed VM store."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from cuamaster.contracts import content_digest

from ..schemas import EnvironmentContext


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "code": "visual_studio_code",
        "command_prompt": "terminal",
        "libreoffice": "libreoffice",
        "libreoffice_calc": "libreoffice_calc",
        "libreoffice_writer": "libreoffice_writer",
        "libreoffice_impress": "libreoffice_impress",
        "google_chrome": "google_chrome",
        "visual_studio_code": "visual_studio_code",
        "gnome_terminal": "terminal",
        "powershell": "terminal",
        "terminal": "terminal",
        "vlc_media_player": "vlc",
        "windows_powershell": "terminal",
        "windows_terminal": "terminal",
    }
    if normalized.startswith("gimp_"):
        return "gimp"
    if normalized.startswith("libreoffice_") and normalized[12:13].isdigit():
        return "libreoffice"
    if normalized.startswith("mozilla_thunderbird_"):
        return "thunderbird"
    return aliases.get(normalized, normalized)


def load_environment_context(
    image_store: str | Path,
    image_id: str,
    *,
    allowed_download_domains: tuple[str, ...] = (),
    primitive_overrides: tuple[str, ...] = (),
) -> EnvironmentContext:
    root = Path(image_store).expanduser().resolve()
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    row = (catalog.get("images") or {}).get(image_id)
    if not isinstance(row, dict):
        raise KeyError(f"image not found in catalog: {image_id}")
    descriptor_path = root / str(row.get("descriptor") or "")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    apps: dict[str, dict] = {}
    capabilities = {"computer", "filesystem", "shell", "desktop_os"}
    for app in descriptor.get("apps") or []:
        if not isinstance(app, dict) or not app.get("name"):
            continue
        slug = _slug(str(app["name"]))
        apps[slug] = dict(app)
        capabilities.add(slug)
        executable = app.get("executable")
        if executable:
            capabilities.add(f"executable:{Path(str(executable)).name}")
    for name, version in (row.get("software") or {}).items():
        slug = _slug(str(name))
        capabilities.add(slug)
        apps.setdefault(slug, {"name": name, "version": version})
    if "libreoffice" in capabilities:
        capabilities.update({"libreoffice_calc", "libreoffice_writer", "libreoffice_impress"})
    if capabilities & {"visual_studio_code", "vscode"}:
        capabilities.add("vs_code")
    if capabilities & {"files", "nautilus"}:
        capabilities.add("file_manager")
    if "document_viewer" in capabilities or "executable:evince" in capabilities:
        capabilities.update({"evince", "pdfviewer"})
    if capabilities & {"wps_2019", "wps_office", "wps_spreadsheets"}:
        capabilities.update({"wps", "excel"})
    if "musescore_studio" in capabilities:
        capabilities.add("musescore")
    for capability in primitive_overrides:
        slug = _slug(capability)
        capabilities.add(str(capability))
        capabilities.add(slug)
        apps.setdefault(slug, {
            "name": capability,
            "inventory_source": "configured_primitive_override",
        })
    inventory_status = str(descriptor.get("inventory_status") or "complete")
    image_digest = str(row.get("sha256") or row.get("digest") or content_digest({
        "image_id": image_id,
        "size_bytes": row.get("size_bytes"),
        "descriptor": descriptor,
    }))
    return EnvironmentContext(
        image_id=image_id,
        image_digest=image_digest,
        os={str(key): str(value) for key, value in (descriptor.get("os") or row.get("os") or {}).items()},
        applications=apps,
        capabilities=tuple(sorted(capabilities)),
        evaluator_types=("weavebench_grade", "composite", "file_exists", "task_evaluator"),
        policy={
            "allow_local_fixtures": True,
            "allowed_download_domains": list(allowed_download_domains),
            "forbid_external_accounts": True,
            "forbid_host_access": True,
            "inventory_status": inventory_status,
            "inventory_verified": inventory_status not in {"pending", "failed", "unavailable"},
            "primitive_overrides": list(primitive_overrides),
        },
    )


def select_environment_contexts(
    image_store: str | Path,
    image_ids: tuple[str, ...],
    *,
    selection: str,
    sample_size: int,
    random_seed: int,
    allowed_download_domains: tuple[str, ...] = (),
    include_pending: bool = False,
    primitive_overrides: dict[str, tuple[str, ...]] | None = None,
) -> list[EnvironmentContext]:
    root = Path(image_store).expanduser().resolve()
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    catalog_images = catalog.get("images") or {}
    candidates: list[str] = []
    for image_id in image_ids:
        if image_id == "*":
            candidates.extend(
                candidate
                for candidate, row in sorted(catalog_images.items())
                if _catalog_image_ready(root, row, include_pending=include_pending)
            )
        else:
            candidates.append(image_id)
    candidates = list(dict.fromkeys(candidates))
    missing = [image_id for image_id in candidates if image_id not in catalog_images]
    if missing:
        raise KeyError(f"images not found in catalog: {', '.join(missing)}")
    if not candidates:
        raise ValueError("environment image selection has no eligible candidates")
    if selection == "fixed":
        selected = candidates[:1]
    elif selection == "random":
        count = min(sample_size, len(candidates))
        selected = random.Random(random_seed).sample(candidates, count)
    elif selection == "all":
        selected = candidates
    else:
        raise ValueError(f"unknown environment selection: {selection}")
    return [
        load_environment_context(
            root,
            image_id,
            allowed_download_domains=allowed_download_domains,
            primitive_overrides=(primitive_overrides or {}).get(image_id, ()),
        )
        for image_id in selected
    ]


def _catalog_image_ready(
    root: Path,
    row: object,
    *,
    include_pending: bool,
) -> bool:
    if not isinstance(row, dict):
        return False
    descriptor_path = root / str(row.get("descriptor") or "")
    if not descriptor_path.is_file():
        return False
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("inventory_status") in {"pending", "failed", "unavailable"}:
        return include_pending
    return bool(descriptor.get("apps") or row.get("software"))
