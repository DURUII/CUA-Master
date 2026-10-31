from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import build_harness_images as MODULE
from scripts.build_harness_images import detect_version, image_id


def test_detects_codex_version_from_cached_archive(tmp_path):
    archive_path = tmp_path / "codex.tar.gz"
    payload = json.dumps({"name": "@openai/codex", "version": "1.2.3"}).encode()
    info = tarfile.TarInfo("./usr/lib/node_modules/@openai/codex/package.json")
    info.size = len(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    assert detect_version("codex", tmp_path) == "1.2.3"


def test_stable_harness_image_id():
    assert image_id("hermes", "20260813") == "osworld-ubuntu-22.04-hermes-20260813"


def test_materializes_host_compatible_bake_script(tmp_path):
    upstream = tmp_path / "upstream.sh"
    destination = tmp_path / "generated.sh"
    upstream.write_text(
        '\n'.join((
            'REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"',
            'r = sess.get(base + "/screenshot", timeout=10)',
            'docker run -d --rm --name "${NAME}" \\',
            '  --cap-add NET_ADMIN --device /dev/kvm \\',
            'docker run -d --rm --name "${NAME}_verify" \\',
            '  --cap-add NET_ADMIN --device /dev/kvm \\',
        )),
        encoding="utf-8",
    )

    MODULE.materialize_bake_script(upstream, destination)

    generated = destination.read_text(encoding="utf-8")
    assert "WEAVEBENCH_ROOT must be set" in generated
    assert 'base + "/platform"' in generated
    assert generated.count("--privileged") == 2
    assert generated.count("--device /dev/net/tun") == 2


def test_failed_build_cleans_partial_without_catalog_entry(tmp_path, monkeypatch):
    store_root = tmp_path / "vm_images"
    base_dir = store_root / "linux" / "base"
    base_dir.mkdir(parents=True)
    (base_dir / "disk.qcow2").write_bytes(b"base")
    (base_dir / "descriptor.json").write_text(json.dumps({
        "schema_version": 1,
        "image_id": "base",
        "os": {"type": "Ubuntu", "version": "22.04", "architecture": "x86_64"},
        "apps": [{"name": "Test App"}],
    }), encoding="utf-8")
    (store_root / "catalog.json").write_text(json.dumps({
        "schema_version": 1,
        "images": {
            "base": {
                "path": "linux/base/disk.qcow2",
                "descriptor": "linux/base/descriptor.json",
                "format": "qcow2",
                "os": {"type": "Ubuntu", "version": "22.04", "architecture": "x86_64"},
                "size_bytes": 4,
            }
        },
    }), encoding="utf-8")
    store = MODULE.ImageStore(store_root)

    monkeypatch.setattr(MODULE, "detect_version", lambda *_: "1.2.3")

    def fail_bake(*, partial_path, **_kwargs):
        partial_path.write_bytes(b"partial")
        raise RuntimeError("bake failed")

    monkeypatch.setattr(MODULE, "_run_bake", fail_bake)
    with pytest.raises(RuntimeError, match="bake failed"):
        MODULE.build_one(
            harness="codex",
            build_date="20260813",
            store=store,
            base=store.resolve("base"),
            weavebench_root=tmp_path,
            cache_dir=tmp_path,
            keep_failed=False,
        )

    target = store_root / "linux" / "osworld-ubuntu-22.04-codex-20260813"
    assert not (target / "disk.qcow2.partial").exists()
    assert [record.id for record in store.list()] == ["base"]
