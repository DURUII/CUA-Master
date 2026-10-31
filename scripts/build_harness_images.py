#!/usr/bin/env python3
"""Build, verify, and register independent harness-baked VM images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cuamaster.collection.image_store import ImageRecord, ImageStore


HARNESSES = ("openclaw", "codex", "claudecode", "hermes")
ASSET_FILES = {
    "openclaw": ("openclaw.tar.gz",),
    "codex": ("codex.tar.gz",),
    "claudecode": ("claudecode.tar.gz",),
    "hermes": ("hermes.tar.gz", "hermes_mcp_wheels.tar.gz"),
}
VERSION_MEMBERS = {
    "openclaw": ("openclaw.tar.gz", "openclaw/package.json", "json"),
    "codex": (
        "codex.tar.gz",
        "./usr/lib/node_modules/@openai/codex/package.json",
        "json",
    ),
    "claudecode": (
        "claudecode.tar.gz",
        "./usr/lib/node_modules/@anthropic-ai/claude-code/package.json",
        "json",
    ),
    "hermes": ("hermes.tar.gz", "./opt/hermes/pyproject.toml", "toml"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_version(harness: str, cache_dir: Path) -> str:
    archive_name, member_name, kind = VERSION_MEMBERS[harness]
    with tarfile.open(cache_dir / archive_name, "r:gz") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise ValueError(f"missing {member_name} in {archive_name}")
        content = member.read().decode("utf-8")
    if kind == "json":
        version = json.loads(content).get("version")
    else:
        version = None
        in_project = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_project = stripped == "[project]"
            elif in_project and stripped.startswith("version") and "=" in stripped:
                version = stripped.split("=", 1)[1].strip().strip("'\"")
                break
    if not version:
        raise ValueError(f"could not detect {harness} version from {archive_name}")
    return str(version)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def image_id(harness: str, build_date: str) -> str:
    return f"osworld-ubuntu-22.04-{harness}-{build_date}"


def preflight(
    *,
    harnesses: list[str],
    store: ImageStore,
    base: ImageRecord,
    weavebench_root: Path,
    cache_dir: Path,
    execute: bool,
) -> None:
    bake_script = weavebench_root / "scripts" / "bake_harness_into_qcow2.sh"
    if not bake_script.is_file():
        raise FileNotFoundError(f"bake script not found: {bake_script}")
    store.verify_record(base)
    for harness in harnesses:
        for asset in ASSET_FILES[harness]:
            path = cache_dir / asset
            if not path.is_file():
                raise FileNotFoundError(f"missing {harness} asset: {path}")
        detect_version(harness, cache_dir)
    free = shutil.disk_usage(store.root).free
    required = base.path.stat().st_size * (len(harnesses) + 1)
    if free < required:
        raise OSError(
            f"insufficient free space in {store.root}: "
            f"need {required} bytes, have {free} bytes"
        )
    if not execute:
        return
    if not Path("/dev/kvm").exists():
        raise RuntimeError("/dev/kvm is unavailable; KVM is required for VM baking")
    if not Path("/dev/net/tun").exists():
        raise RuntimeError("/dev/net/tun is unavailable; VM port forwarding requires TUN")
    if shutil.which("docker") is None:
        raise RuntimeError("docker is not installed")
    subprocess.run(
        ["docker", "info"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["docker", "image", "inspect", "happysixd/osworld-docker:latest"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def materialize_bake_script(upstream: Path, destination: Path) -> None:
    """Create a Podman-compatible bake script without modifying WeaveBench."""
    content = upstream.read_text(encoding="utf-8")
    replacements = (
        (
            'REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"',
            'REPO_ROOT="${WEAVEBENCH_ROOT:?WEAVEBENCH_ROOT must be set}"',
            1,
        ),
        (
            'r = sess.get(base + "/screenshot", timeout=10)',
            'r = sess.get(base + "/platform", timeout=10)',
            1,
        ),
        (
            'docker run -d --rm --name "${NAME}" \\',
            'docker run -d --rm --privileged --name "${NAME}" \\',
            1,
        ),
        (
            'docker run -d --rm --name "${NAME}_verify" \\',
            'docker run -d --rm --privileged --name "${NAME}_verify" \\',
            1,
        ),
        (
            '--cap-add NET_ADMIN --device /dev/kvm \\',
            '--cap-add NET_ADMIN --device /dev/kvm --device /dev/net/tun \\',
            2,
        ),
    )
    for original, replacement, expected_count in replacements:
        actual_count = content.count(original)
        if actual_count != expected_count:
            raise ValueError(
                f"unsupported WeaveBench bake script: expected {expected_count} "
                f"occurrence(s) of {original!r}, found {actual_count}"
            )
        content = content.replace(original, replacement)
    destination.write_text(content, encoding="utf-8")
    destination.chmod(0o755)


def _run_bake(
    *,
    harness: str,
    base_path: Path,
    partial_path: Path,
    cache_dir: Path,
    weavebench_root: Path,
    log_path: Path,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "SRC_QCOW": str(base_path),
            "DST_QCOW": str(partial_path),
            "CACHE_DIR": str(cache_dir),
            "PROMOTE": "0",
            "WEAVEBENCH_ROOT": str(weavebench_root),
        }
    )
    with tempfile.TemporaryDirectory(prefix="gui-vm-bake-") as temporary_dir:
        bake_script = Path(temporary_dir) / "bake_harness_into_qcow2.sh"
        materialize_bake_script(
            weavebench_root / "scripts" / "bake_harness_into_qcow2.sh",
            bake_script,
        )
        command = ["bash", str(bake_script), harness]
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=weavebench_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _derived_descriptor(base: ImageRecord, target_id: str, harness: str, version: str) -> dict[str, Any]:
    assert base.descriptor_path is not None
    descriptor = json.loads(base.descriptor_path.read_text(encoding="utf-8"))
    descriptor["image_id"] = target_id
    descriptor["os"] = {
        "type": base.os_type,
        "version": base.os_version,
        "architecture": base.architecture,
    }
    descriptor["derived_from"] = base.id
    descriptor["tools"] = [{"name": harness, "version": version, "kind": "cli-harness"}]
    return descriptor


def _read_verified_build(path: Path, target_id: str, harness: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    build = json.loads(path.read_text(encoding="utf-8"))
    if (
        build.get("image_id") != target_id
        or build.get("harness") != harness
        or build.get("verified") is not True
    ):
        raise ValueError(f"incompatible build metadata: {path}")
    return build


def build_one(
    *,
    harness: str,
    build_date: str,
    store: ImageStore,
    base: ImageRecord,
    weavebench_root: Path,
    cache_dir: Path,
    keep_failed: bool,
) -> ImageRecord:
    target_id = image_id(harness, build_date)
    existing_ids = {record.id for record in store.list()}
    if target_id in existing_ids:
        record = store.resolve(target_id)
        print(f"[skip] verified image already registered: {target_id}")
        return record

    target_dir = store.root / "linux" / target_id
    disk = target_dir / "disk.qcow2"
    partial = target_dir / "disk.qcow2.partial"
    descriptor_path = target_dir / "descriptor.json"
    build_path = target_dir / "build.json"
    log_path = target_dir / "build.log"
    version = detect_version(harness, cache_dir)
    resumed = _read_verified_build(build_path, target_id, harness)
    if disk.exists() and resumed is None:
        raise FileExistsError(
            f"unregistered disk has no verified build metadata: {disk}"
        )
    if partial.exists():
        raise FileExistsError(
            f"stale partial image exists: {partial}; remove it or inspect build.log"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(target_dir, 0o755)
    if resumed is None:
        try:
            _run_bake(
                harness=harness,
                base_path=base.path,
                partial_path=partial,
                cache_dir=cache_dir,
                weavebench_root=weavebench_root,
                log_path=log_path,
            )
            if not partial.is_file():
                raise FileNotFoundError(f"bake did not create output: {partial}")
            os.replace(partial, disk)
            os.chmod(disk, 0o644)
            assets = []
            for name in ASSET_FILES[harness]:
                asset = cache_dir / name
                assets.append(
                    {
                        "name": name,
                        "size_bytes": asset.stat().st_size,
                        "sha256": _sha256(asset),
                    }
                )
            resumed = {
                "schema_version": 1,
                "image_id": target_id,
                "base_image_id": base.id,
                "harness": harness,
                "harness_version": version,
                "assets": assets,
                "builder": "WeaveBench/scripts/bake_harness_into_qcow2.sh",
                "built_at": datetime.now(timezone.utc).isoformat(),
                "verified": True,
            }
            _write_json_atomic(build_path, resumed)
        except Exception:
            if partial.exists() and not keep_failed:
                partial.unlink()
                print(f"[cleanup] removed failed partial image: {partial}", file=sys.stderr)
            raise

    _write_json_atomic(
        descriptor_path,
        _derived_descriptor(base, target_id, harness, version),
    )
    readme = (
        f"# {target_id}\n\n"
        f"`disk.qcow2` is a complete image copied independently from `{base.id}` "
        f"with {harness} {version} pre-installed. The build process ran a "
        "read-only reboot verification; the image does not depend on baseline "
        "files, symlinks, or incremental layers.\n"
    )
    (target_dir / "README.md").write_text(readme, encoding="utf-8")
    os.chmod(target_dir / "README.md", 0o644)
    software = {**base.software, harness: version}
    record = store.register(
        image_id=target_id,
        source=disk,
        os_type=base.os_type,
        os_version=base.os_version,
        architecture=base.architecture,
        snapshot="clean-v1",
        software=software,
        mode="adopt",
        derived_from=base.id,
        build=resumed,
    )
    store.verify_record(record)
    print(f"[done] registered and verified: {target_id}")
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build independent harness-baked Ubuntu VM images."
    )
    parser.add_argument(
        "--harness",
        action="append",
        choices=HARNESSES,
        required=True,
        help="Harness to bake; repeat for multiple images.",
    )
    parser.add_argument("--build-date", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--store", type=Path, default=Path("vm_images"))
    parser.add_argument(
        "--weavebench-root",
        type=Path,
        default=Path(os.environ.get("WEAVEBENCH_ROOT", "WeaveBench")),
        help="WeaveBench checkout providing scripts/bake_harness_into_qcow2.sh.",
    )
    parser.add_argument(
        "--base-image",
        default="osworld-v2-ubuntu-x86-example",
        help="Registered base image ID in the image store.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.harness) != len(set(args.harness)):
        raise SystemExit("--harness values must be unique")
    if len(args.build_date) != 8 or not args.build_date.isdigit():
        raise SystemExit("--build-date must be YYYYMMDD")
    store = ImageStore(args.store)
    base = store.resolve(args.base_image)
    weavebench_root = args.weavebench_root.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir
        else weavebench_root / "cache" / "runtime_assets"
    )
    preflight(
        harnesses=args.harness,
        store=store,
        base=base,
        weavebench_root=weavebench_root,
        cache_dir=cache_dir,
        execute=not args.dry_run,
    )
    for harness in args.harness:
        target_id = image_id(harness, args.build_date)
        print(
            f"[plan] {harness} -> {store.root / 'linux' / target_id / 'disk.qcow2'}"
        )
    if args.dry_run:
        return 0
    for harness in args.harness:
        build_one(
            harness=harness,
            build_date=args.build_date,
            store=store,
            base=base,
            weavebench_root=weavebench_root,
            cache_dir=cache_dir,
            keep_failed=args.keep_failed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
