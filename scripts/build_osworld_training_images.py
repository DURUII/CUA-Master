#!/usr/bin/env python3
"""Build OSWorld images with common document-processing dependencies."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cuamaster.collection.image_store import ImageRecord, ImageStore


DEFAULT_BASE_IMAGES = ("osworld-v2-ubuntu-x86-example",)
APT_PACKAGES = (
    "ghostscript",
    "imagemagick",
    "libreoffice",
    "poppler-utils",
    "python3-docx",
    "python3-fitz",
    "python3-odf",
    "python3-openpyxl",
    "python3-pandas",
    "python3-pip",
    "python3-pil",
    "python3-pypdf2",
    "python3-uno",
    "python3-xlrd",
    "qpdf",
    "tesseract-ocr",
    "tesseract-ocr-eng",
)
PIP_PACKAGES = (
    "pytesseract",
    "python-pptx",
)
VERIFY_COMMAND = """set -eu
python3 - <<'PY'
import docx
import fitz
import odf
import openpyxl
import pandas
import PIL
import pptx
import PyPDF2
import pytesseract
import xlrd
PY
command -v libreoffice
command -v pdftotext
command -v pdfinfo
command -v qpdf
command -v tesseract
"""


def training_image_id(base_image_id: str, suffix: str) -> str:
    return f"{base_image_id}-{suffix}"


def _run(command: list[str], *, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
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


def _virt_customize_command(image: Path) -> list[str]:
    apt_options = (
        "-o Dir::Etc::sourcelist=/etc/apt/sources.list "
        "-o Dir::Etc::sourceparts=-"
    )
    apt_install = (
        f"DEBIAN_FRONTEND=noninteractive apt-get {apt_options} update && "
        f"DEBIAN_FRONTEND=noninteractive apt-get {apt_options} install -y "
        f"--no-install-recommends {' '.join(APT_PACKAGES)}"
    )
    pip_install = (
        "python3 -m pip install --no-cache-dir --disable-pip-version-check "
        + " ".join(PIP_PACKAGES)
    )
    return [
        "virt-customize",
        "--format",
        "qcow2",
        "-a",
        str(image),
        "--network",
        "--no-logfile",
        "--run-command",
        apt_install,
        "--run-command",
        pip_install,
        "--run-command",
        VERIFY_COMMAND,
    ]


def _descriptor(base: ImageRecord, target_id: str) -> dict[str, Any]:
    assert base.descriptor_path is not None
    descriptor = json.loads(base.descriptor_path.read_text(encoding="utf-8"))
    descriptor["image_id"] = target_id
    return descriptor


def build_one(
    *,
    store: ImageStore,
    base: ImageRecord,
    suffix: str,
    keep_failed: bool,
) -> ImageRecord:
    target_id = training_image_id(base.id, suffix)
    existing_ids = {record.id for record in store.list()}
    if target_id in existing_ids:
        record = store.resolve(target_id)
        print(f"[skip] verified image already registered: {target_id}")
        return record

    target_dir = store.root / "linux" / target_id
    disk = target_dir / "disk.qcow2"
    partial = target_dir / "disk.qcow2.partial"
    descriptor_path = target_dir / "descriptor.json"
    log_path = target_dir / "build.log"
    if disk.exists():
        raise FileExistsError(f"unregistered image already exists: {disk}")
    if partial.exists():
        raise FileExistsError(f"stale partial image exists: {partial}")

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run(
            [
                "qemu-img",
                "convert",
                "-p",
                "-O",
                "qcow2",
                str(base.path),
                str(partial),
            ],
            log_path=log_path,
        )
        _run(_virt_customize_command(partial), log_path=log_path)
        os.replace(partial, disk)
        os.chmod(disk, 0o644)
    except Exception:
        if partial.exists() and not keep_failed:
            partial.unlink()
        raise

    descriptor_path.write_text(
        json.dumps(_descriptor(base, target_id), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(descriptor_path, 0o644)
    built_at = datetime.now(timezone.utc).isoformat()
    build = {
        "schema_version": 1,
        "artifact_verified": True,
        "dependency_verified": True,
        "base_image_id": base.id,
        "builder": "scripts/build_osworld_training_images.py",
        "built_at": built_at,
        "apt_packages": list(APT_PACKAGES),
        "pip_packages": list(PIP_PACKAGES),
    }
    software = {
        **base.software,
        "osworld-training-dependencies": suffix,
    }
    record = store.register(
        image_id=target_id,
        source=disk,
        os_type=base.os_type,
        os_version=base.os_version,
        architecture=base.architecture,
        snapshot=base.snapshot,
        software=software,
        mode="adopt",
        derived_from=base.id,
        build=build,
    )
    store.verify_record(record)
    print(f"[done] registered dependency image: {target_id}")
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake common OCR, PDF, spreadsheet, and Office dependencies into OSWorld images."
    )
    parser.add_argument(
        "--base-image",
        action="append",
        dest="base_images",
        help="Base image ID; repeat for multiple images. Defaults to the catalog's Linux image.",
    )
    parser.add_argument(
        "--suffix",
        default="training-deps-v1",
        help="Suffix appended to each base image ID.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("vm_images"),
    )
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not shutil.which("qemu-img"):
        raise SystemExit("qemu-img is required")
    if not shutil.which("virt-customize"):
        raise SystemExit("virt-customize is required")
    store = ImageStore(args.store)
    base_ids = args.base_images or list(DEFAULT_BASE_IMAGES)
    if len(base_ids) != len(set(base_ids)):
        raise SystemExit("--base-image values must be unique")
    bases = [store.resolve(image_id) for image_id in base_ids]
    for base in bases:
        target_id = training_image_id(base.id, args.suffix)
        print(f"[plan] {base.id} -> {target_id}")
    if args.dry_run:
        return 0
    for base in bases:
        build_one(
            store=store,
            base=base,
            suffix=args.suffix,
            keep_failed=args.keep_failed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
