#!/usr/bin/env python3
"""Collect a VM descriptor through an OSWorld-compatible session gateway."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import requests

from cuamaster.collection import assets as _collection_assets


DEFAULT_GUEST_SCRIPT = Path(_collection_assets.__file__).parent / "guest_app_inventory.py"


def platform_name(os_type: str) -> str:
    normalized = os_type.strip().casefold()
    if normalized in {"darwin", "mac", "macos", "osx"}:
        return "macos"
    if normalized.startswith("win"):
        return "windows"
    return "linux"


def normalize_architecture(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if normalized in {"arm64", "aarch64"}:
        return "aarch64"
    return value


def request_with_retry(
    operation: Callable[[], requests.Response],
    *,
    attempts: int,
    delay: float,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = operation()
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boot a temporary VM and write its real application inventory descriptor."
    )
    parser.add_argument("--gateway", required=True, help="Session gateway base URL.")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--os-type", default="Ubuntu")
    parser.add_argument(
        "--platform",
        choices=("linux", "windows", "macos"),
        default=None,
    )
    parser.add_argument("--vm-backend", default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--guest-script", type=Path, default=DEFAULT_GUEST_SCRIPT)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args()


def build_descriptor(
    *,
    image_id: str,
    guest_inventory: dict[str, Any],
) -> dict[str, Any]:
    inventory = guest_inventory["inventory"]
    guest = guest_inventory["guest"]
    os_release = guest["os_release"]
    os_type = os_release.get("name") or os_release.get("id")
    os_version = os_release.get("version_id") or os_release.get("version")
    architecture = guest.get("architecture")
    if not all(
        isinstance(value, str) and value
        for value in (os_type, os_version, architecture)
    ):
        raise ValueError("guest inventory is missing OS type, version, or architecture")
    applications = inventory.get("applications")
    if not isinstance(applications, list):
        applications = inventory.get("desktop_applications") or []
    apps = []
    seen = set()
    for index, entry in enumerate(applications):
        if not isinstance(entry, dict) or entry.get("visible") is False:
            continue
        name = str(entry.get("name") or "").strip()
        app_id = str(
            entry.get("app_id")
            or entry.get("desktop_id")
            or entry.get("package")
            or f"application-{index}"
        ).strip()
        if not name:
            continue
        identity = (name.casefold(), app_id.casefold())
        if identity in seen:
            continue
        seen.add(identity)
        app = {
            "name": name,
            "desktop_id": app_id,
            "executable": entry.get("executable"),
            "categories": entry.get("categories") or [],
            "package_manager": entry.get("package_manager"),
            "package": entry.get("package"),
            "version": entry.get("version"),
            "version_output": entry.get("version_output"),
            "architecture": entry.get("architecture"),
            "publisher": entry.get("publisher"),
            "inventory_source": entry.get("source"),
        }
        apps.append({key: value for key, value in app.items() if value is not None})
    apps.sort(key=lambda item: (item["name"].casefold(), item["desktop_id"].casefold()))
    return {
        "schema_version": 1,
        "image_id": image_id,
        "os": {
            "type": os_type,
            "version": os_version,
            "architecture": normalize_architecture(architecture),
        },
        "inventory_status": "complete",
        "inventory_collected_at": guest_inventory.get("collected_at"),
        "inventory_collection_method": guest_inventory.get("collection_method"),
        "apps": apps,
    }


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        raise SystemExit("--attempts must be >= 1")
    gateway = args.gateway.rstrip("/")
    platform = args.platform or platform_name(args.os_type)
    vm_backend = args.vm_backend or (
        "macos_kvm" if platform == "macos" else "desktop_env"
    )
    windows = platform == "windows"
    guest_script_path = (
        r"C:\Windows\Temp\guest_app_inventory.py"
        if windows
        else "/tmp/guest_app_inventory.py"
    )
    python_executable = "python" if windows else "python3"
    job_id = args.job_id or f"vm-inventory-{args.image_id}"
    guest_script = args.guest_script.resolve()
    if not guest_script.is_file():
        raise FileNotFoundError(guest_script)

    http = requests.Session()
    http.trust_env = False
    session_id: str | None = None
    try:
        response = request_with_retry(
            lambda: http.post(
                f"{gateway}/sessions",
                json={
                    "job_id": job_id,
                    "headless": True,
                    "image_id": args.image_id,
                    "os_type": args.os_type,
                    "platform": platform,
                    "vm_backend": vm_backend,
                    "lease_ttl_seconds": 3600,
                    "entry": {"task_id": "inventory-installed-apps"},
                },
                headers={"x-rollout-id": job_id},
                timeout=(10, 1800),
            ),
            attempts=args.attempts,
            delay=args.retry_delay,
        )
        session_id = response.json()["session_id"]
        print(f"created session {session_id}", file=sys.stderr, flush=True)

        guest_script_bytes = guest_script.read_bytes()
        request_with_retry(
            lambda: http.post(
                f"{gateway}/sessions/{session_id}/vm/setup/upload",
                data={"file_path": guest_script_path},
                files={
                    "file_data": (
                        guest_script.name,
                        guest_script_bytes,
                        "text/x-python",
                    )
                },
                timeout=(10, 120),
            ),
            attempts=args.attempts,
            delay=args.retry_delay,
        )

        response = request_with_retry(
            lambda: http.post(
                f"{gateway}/sessions/{session_id}/vm/setup/execute",
                json={
                    "command": [python_executable, guest_script_path],
                    "shell": False,
                },
                timeout=(10, 300),
            ),
            attempts=args.attempts,
            delay=args.retry_delay,
        )
        execution = response.json()
        if execution.get("returncode") != 0:
            raise RuntimeError(
                f"guest inventory failed: {execution.get('error') or execution.get('output')}"
            )
        guest_inventory = json.loads(execution["output"])
        descriptor = build_descriptor(
            image_id=args.image_id,
            guest_inventory=guest_inventory,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(descriptor, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
        print(f"wrote {args.output}", file=sys.stderr, flush=True)
        return 0
    finally:
        if session_id:
            try:
                request_with_retry(
                    lambda: http.delete(
                        f"{gateway}/sessions/{session_id}",
                        timeout=(10, 180),
                    ),
                    attempts=args.attempts,
                    delay=args.retry_delay,
                )
                print(f"released session {session_id}", file=sys.stderr, flush=True)
            except requests.RequestException as exc:
                print(
                    f"WARNING: could not release session {session_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    raise SystemExit(main())
