from __future__ import annotations

import os
from pathlib import Path

import pytest

from cuamaster.endpoints import (
    DynamicEndpointPool,
    EndpointLoad,
    load_yaml_mapping,
    normalize_base_urls,
)


def _write_pool(path: Path, urls: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "model: kimi-k3",
                "base_urls:",
                *[f"  - {url}" for url in urls],
                "routing:",
                "  reload_on_request: true",
                "  failure_threshold: 1",
                "  cooldown_seconds: 60",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_endpoint_pool_reloads_and_skips_failed_url(tmp_path: Path) -> None:
    path = tmp_path / "pool.yaml"
    _write_pool(path, ["model-a:30000", "http://model-b:30000/v1"])
    pool = DynamicEndpointPool(config_path=path)

    first = pool.acquire()
    pool.mark_failure(first, TimeoutError("timed out"))
    assert pool.acquire() == "http://model-b:30000/v1"

    previous_mtime = path.stat().st_mtime_ns
    _write_pool(path, ["http://model-c:30000/v1"])
    os.utime(path, ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000))

    assert pool.acquire() == "http://model-c:30000/v1"
    assert pool.base_urls == ("http://model-c:30000/v1",)


def test_endpoint_pool_reports_when_all_endpoints_are_cooling_down() -> None:
    pool = DynamicEndpointPool(
        base_urls=["model-a:30000", "model-b:30000"],
        cooldown_seconds=60,
    )
    for base_url in pool.base_urls:
        pool.mark_failure(base_url, TimeoutError("timed out"))

    retry_after = pool.retry_after()

    assert 0 < retry_after <= 60


def test_endpoint_pool_rejects_malformed_numeric_host() -> None:
    with pytest.raises(ValueError, match="invalid endpoint IP address"):
        normalize_base_urls(["http://10.0.0.1.2:30000/v1"])


def test_invalid_endpoint_pool_update_keeps_last_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "pool.yaml"
    _write_pool(path, ["http://model-a:30000/v1"])
    pool = DynamicEndpointPool(config_path=path)
    previous_mtime = path.stat().st_mtime_ns

    path.write_text("base_urls: []\n", encoding="utf-8")
    os.utime(path, ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000))

    assert pool.acquire() == "http://model-a:30000/v1"


def test_load_aware_pool_prefers_small_queue_and_skips_unknown(monkeypatch) -> None:
    pool = DynamicEndpointPool(
        base_urls=["model-a:30000", "model-b:30000", "model-c:30000"],
        load_aware=True,
        load_probe_interval_seconds=60,
        max_waiting_requests=2,
    )
    loads = {
        "http://model-a:30000/v1": EndpointLoad(8, 12, 1200, 0.9),
        "http://model-b:30000/v1": EndpointLoad(3, 0, 0, 0.2),
    }

    def probe(base_url: str) -> EndpointLoad:
        if base_url not in loads:
            raise TimeoutError("load probe timed out")
        return loads[base_url]

    monkeypatch.setattr(pool, "_probe_load", probe)

    assert pool.acquire() == "http://model-b:30000/v1"
    assert pool.acquire() == "http://model-b:30000/v1"


def test_load_aware_pool_uses_least_busy_endpoint_when_all_queued(monkeypatch) -> None:
    pool = DynamicEndpointPool(
        base_urls=["model-a:30000", "model-b:30000"],
        load_aware=True,
        load_probe_interval_seconds=60,
        max_waiting_requests=2,
    )
    loads = {
        "http://model-a:30000/v1": EndpointLoad(20, 9, 900, 0.8),
        "http://model-b:30000/v1": EndpointLoad(30, 4, 400, 0.9),
    }
    monkeypatch.setattr(pool, "_probe_load", loads.__getitem__)

    assert pool.acquire() == "http://model-b:30000/v1"


def test_yaml_extends_merges_nested_mappings(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "backend: openai_chat\napi:\n  model: kimi-k3\n  max_tokens: 100\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: parent.yaml\napi:\n  max_tokens: 200\n",
        encoding="utf-8",
    )

    assert load_yaml_mapping(child) == {
        "backend": "openai_chat",
        "api": {"model": "kimi-k3", "max_tokens": 200},
    }
