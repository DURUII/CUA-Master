from __future__ import annotations

import json
import threading
import time
from argparse import Namespace
from types import SimpleNamespace

import pytest

from cuamaster.collection import collector
from cuamaster.collection.ledger import EpisodeLedger
from cuamaster.collection.profiles import MatrixConfig, VMProfile
from cuamaster.collection.schemas import content_digest


def _profile(profile_id: str, num_envs: int) -> VMProfile:
    return VMProfile(
        id=profile_id,
        backend="weavebench",
        os_type="Ubuntu",
        os_version="22.04",
        architecture="x86_64",
        provider="sandbox",
        image_path=None,
        image_id=None,
        snapshot="clean",
        num_envs=num_envs,
    )


def test_default_vm_concurrency_matches_largest_profile(tmp_path):
    profiles = [_profile("large", 4), _profile("small", 2)]
    config = MatrixConfig(tmp_path / "matrix.json", {}, tuple(profiles))

    assert collector._resolve_vm_concurrency(config, profiles, None) == 4


def test_stable_config_digest_ignores_endpoint_pool_runtime_url():
    payload = {
        "defaults": {
            "backend_options": {
                "model_endpoint_pool": "/config/models.yaml",
                "litellm_base_url": "http://model-a.example/v1",
            }
        },
        "profiles": [{
            "id": "ubuntu",
            "backend_options": {
                "litellm_base_url": "http://model-b.example/v1",
            },
        }],
    }

    stable = collector._stable_config_payload(payload)

    assert "litellm_base_url" not in stable["defaults"]["backend_options"]
    assert "litellm_base_url" not in stable["profiles"][0]["backend_options"]
    assert payload["defaults"]["backend_options"]["litellm_base_url"] == (
        "http://model-a.example/v1"
    )


def test_legacy_endpoint_digest_matches_ledger_runtime_url(tmp_path):
    root = tmp_path / "collection"
    run_id = "run-id"
    ledger = EpisodeLedger(root / ".collection_state" / "ledger.sqlite3")
    ledger.create_run(run_id, config_path="matrix.json")
    ledger.register_episode(
        "profile-run",
        run_id=run_id,
        profile_id="ubuntu",
        task_id="__profile__",
        metadata={
            "backend": {
                "litellm_base_url": "http://old-model.example/v1",
            }
        },
    )
    current = {
        "defaults": {
            "backend_options": {
                "model_endpoint_pool": "/config/models.yaml",
                "litellm_base_url": "http://new-model.example/v1",
            }
        },
        "profiles": [{"id": "ubuntu"}],
    }
    legacy = json.loads(json.dumps(current))
    legacy["defaults"]["backend_options"]["litellm_base_url"] = (
        "http://old-model.example/v1"
    )

    assert collector._matches_legacy_config_digest(
        current,
        content_digest(legacy),
        root,
        run_id,
    )


def test_collector_migrates_legacy_endpoint_digest_on_resume(tmp_path):
    output_root = tmp_path / "collection"
    state_root = output_root / ".collection_state"
    state_root.mkdir(parents=True)
    run_id = "legacy-run"
    current = {
        "schema_version": 1,
        "defaults": {
            "provider": "docker",
            "num_envs": 1,
            "backend_options": {
                "model_endpoint_pool": "/config/models.yaml",
                "litellm_base_url": "http://new-model.example/v1",
            },
        },
        "profiles": [{
            "id": "ubuntu",
            "os": {"type": "Ubuntu"},
            "image": {},
        }],
    }
    legacy = json.loads(json.dumps(current))
    legacy["defaults"]["backend_options"]["litellm_base_url"] = (
        "http://old-model.example/v1"
    )
    config_path = tmp_path / "matrix.json"
    config_path.write_text(json.dumps(current), encoding="utf-8")
    (state_root / "run.json").write_text(
        json.dumps({
            "run_id": run_id,
            "config": str(config_path),
            "config_digest": content_digest(legacy),
        }),
        encoding="utf-8",
    )
    ledger = EpisodeLedger(state_root / "ledger.sqlite3")
    ledger.create_run(run_id, config_path=str(config_path))
    ledger.register_episode(
        "profile-run",
        run_id=run_id,
        profile_id="ubuntu",
        task_id="__profile__",
        metadata={
            "backend": {
                "litellm_base_url": "http://old-model.example/v1",
            }
        },
    )

    assert collector.main([
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--export-only",
    ]) == 0

    marker = json.loads((state_root / "run.json").read_text(encoding="utf-8"))
    assert marker["config_digest"] == content_digest(
        collector._stable_config_payload(current)
    )
    assert marker["config_digest_scope"] == "stable_runtime_routing_v1"


def test_collector_allows_explicit_config_change_on_resume(tmp_path):
    output_root = tmp_path / "collection"
    state_root = output_root / ".collection_state"
    state_root.mkdir(parents=True)
    run_id = "existing-run"
    previous = {
        "schema_version": 1,
        "defaults": {"provider": "docker", "num_envs": 1},
        "profiles": [{
            "id": "ubuntu",
            "os": {"type": "Ubuntu"},
            "image": {},
            "env": {"OSWORLD_EVAL_MODEL_PROVIDER": "anthropic"},
        }],
    }
    current = json.loads(json.dumps(previous))
    current["profiles"][0]["env"] = {
        "OSWORLD_EVAL_MODEL_PROVIDER": "openai_compatible"
    }
    config_path = tmp_path / "matrix.json"
    config_path.write_text(json.dumps(current), encoding="utf-8")
    previous_digest = content_digest(collector._stable_config_payload(previous))
    (state_root / "run.json").write_text(
        json.dumps({
            "run_id": run_id,
            "config": str(config_path),
            "config_digest": previous_digest,
        }),
        encoding="utf-8",
    )
    EpisodeLedger(state_root / "ledger.sqlite3").create_run(
        run_id,
        config_path=str(config_path),
    )

    assert collector.main([
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--export-only",
        "--allow-config-change",
    ]) == 0

    marker = json.loads((state_root / "run.json").read_text(encoding="utf-8"))
    assert marker["config_digest"] == content_digest(
        collector._stable_config_payload(current)
    )
    assert marker["config_digest_history"] == [{
        "config_digest": previous_digest,
        "replaced_at": marker["config_digest_history"][0]["replaced_at"],
    }]


def test_matrix_default_can_set_global_vm_concurrency(tmp_path):
    profiles = [_profile("first", 4), _profile("second", 4)]
    config = MatrixConfig(
        tmp_path / "matrix.json",
        {"concurrency": 8},
        tuple(profiles),
    )

    assert collector._resolve_vm_concurrency(config, profiles, None) == 8


def test_profile_cannot_request_more_vms_than_global_concurrency(tmp_path):
    profiles = [_profile("too-large", 4)]
    config = MatrixConfig(tmp_path / "matrix.json", {}, tuple(profiles))

    with pytest.raises(ValueError, match="too-large=4"):
        collector._resolve_vm_concurrency(config, profiles, 3)


def test_retry_can_cap_profile_vm_count_to_global_concurrency():
    profiles = [_profile("large", 24), _profile("small", 4)]

    capped = collector._cap_profile_vm_concurrency(
        profiles,
        12,
        enabled=True,
    )

    assert [profile.num_envs for profile in capped] == [12, 4]
    assert [profile.num_envs for profile in profiles] == [24, 4]


def test_profile_scheduler_bounds_total_active_vm_slots(monkeypatch, tmp_path):
    profiles = [
        _profile("large", 2),
        _profile("small-a", 1),
        _profile("small-b", 1),
    ]
    config = MatrixConfig(tmp_path / "matrix.json", {}, tuple(profiles))
    lock = threading.Lock()
    active_vms = 0
    peak_vms = 0

    def fake_run_profile(config, profile, root, args, run_id, ledger, trace_store):
        del config, root, args, run_id, ledger, trace_store
        nonlocal active_vms, peak_vms
        with lock:
            active_vms += profile.num_envs
            peak_vms = max(peak_vms, active_vms)
        time.sleep(0.03)
        with lock:
            active_vms -= profile.num_envs
        return {"profile": profile.id, "status": "completed"}

    monkeypatch.setattr(collector, "_run_profile", fake_run_profile)
    monkeypatch.setattr(collector, "_progress", lambda message: None)

    results = collector._run_profiles(
        config,
        profiles,
        tmp_path,
        Namespace(concurrency=2, max_parallel_profiles=None),
        "run-id",
        object(),
        object(),
    )

    assert peak_vms == 2
    assert {result["profile"] for result in results} == {
        "large",
        "small-a",
        "small-b",
    }


def test_aborted_run_marks_running_profiles_cancelled():
    calls = []

    class FakeLedger:
        def rows(self, *, run_id):
            assert run_id == "run-id"
            return [
                {"episode_id": "active", "status": "running"},
                {"episode_id": "done", "status": "succeeded"},
            ]

        def finish_episode(self, episode_id, status, *, error=None):
            calls.append(("episode", episode_id, status.value, error))

        def finish_run(self, run_id, status):
            calls.append(("run", run_id, status))

    collector._finish_aborted_run(FakeLedger(), "run-id", KeyboardInterrupt())

    assert calls == [
        ("episode", "active", "cancelled", "KeyboardInterrupt"),
        ("run", "run-id", "cancelled"),
    ]


def test_run_profile_marks_backend_exception_as_infra_failed(
    monkeypatch, tmp_path
):
    profile = _profile("ubuntu", 1)
    config = MatrixConfig(tmp_path / "matrix.json", {}, (profile,))
    calls = []

    class FakeLedger:
        def episode(self, episode_id):
            return None

        def register_episode(self, *args, **kwargs):
            pass

        def mark_running(self, episode_id, *, worker_id):
            calls.append(("running", episode_id, worker_id))

        def finish_episode(self, episode_id, status, *, error=None):
            calls.append(("finished", episode_id, status.value, error))

    class FakeBackend:
        def prepare(self, **kwargs):
            return SimpleNamespace(
                command=["backend"],
                cwd=tmp_path,
                env={},
                metadata={"backend": "fake"},
            )

    monkeypatch.setattr(collector, "create_backend", lambda name: FakeBackend())
    monkeypatch.setattr(
        collector,
        "_run_streaming_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(collector, "_progress", lambda message: None)

    with pytest.raises(RuntimeError, match="boom"):
        collector._run_profile(
            config,
            profile,
            tmp_path,
            Namespace(dry_run=False, force_rerun=False),
            "run-id",
            FakeLedger(),
            object(),
        )

    assert calls[-1][0] == "finished"
    assert calls[-1][2:] == ("infra_failed", "RuntimeError: boom")
