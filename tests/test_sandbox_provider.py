"""Tests for the snapshot-rollback sandbox provider interface."""
from __future__ import annotations

import json

import pytest

from cuamaster.collection.providers.sandbox import (
    SANDBOX_PROVIDER_NAMES,
    CommandSandboxConfig,
    CommandSandboxProvider,
    MockSandboxProvider,
    SandboxProvider,
    SandboxVMManager,
    create_sandbox_provider,
    register_sandbox_provider,
)


def test_mock_provider_lifecycle_and_rollback():
    provider = MockSandboxProvider()
    provider.start_emulator("sandbox://image", headless=True, os_type="Ubuntu")
    assert provider.get_ip_address("sandbox://image").startswith("127.0.0.1:")

    provider.save_state("sandbox://image", "snapshot_1")
    provider.save_state("sandbox://image", "snapshot_2")
    provider.save_state("sandbox://image", "snapshot_3")
    assert provider.current_snapshot == "snapshot_3"

    provider.revert_to_snapshot("sandbox://image", "snapshot_2")
    assert provider.current_snapshot == "snapshot_2"
    assert [snapshot.name for snapshot in provider.snapshots] == [
        "snapshot_1",
        "snapshot_2",
    ]

    provider.stop_emulator("sandbox://image")
    assert not provider.running
    assert ("rollback", "snapshot_2") in provider.events


def test_mock_provider_snapshot_signature():
    provider = MockSandboxProvider()
    provider.start_emulator("sandbox://image", headless=True, os_type="Windows")
    provider.save_state("sandbox://image", "boundary")
    signature = provider.snapshot_signature("boundary")
    assert signature == {"snapshot": "boundary", "os_type": "Windows", "ordinal": 0}
    assert provider.snapshot_signature("missing") is None


def test_mock_provider_rejects_unknown_snapshot():
    provider = MockSandboxProvider()
    provider.start_emulator("sandbox://image", headless=True)
    with pytest.raises(KeyError):
        provider.revert_to_snapshot("sandbox://image", "never-saved")


def test_mock_provider_rejects_double_start():
    provider = MockSandboxProvider()
    provider.start_emulator("sandbox://image", headless=True)
    with pytest.raises(RuntimeError):
        provider.start_emulator("sandbox://image", headless=True)


def test_registry_resolves_builtin_names():
    for name in [*SANDBOX_PROVIDER_NAMES, "mock"]:
        if name == "mock":
            manager, provider = create_sandbox_provider(name)
            assert isinstance(provider, MockSandboxProvider)
            assert isinstance(manager, SandboxVMManager)


def test_registry_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown sandbox provider"):
        create_sandbox_provider("no-such-provider")


def test_registry_accepts_custom_factory():
    class CustomProvider(MockSandboxProvider):
        pass

    register_sandbox_provider(
        "custom-test", lambda region: (SandboxVMManager(), CustomProvider(region))
    )
    _, provider = create_sandbox_provider("custom-test", region="eu")
    assert isinstance(provider, CustomProvider)
    assert provider.region == "eu"


def test_sandbox_provider_is_abstract():
    with pytest.raises(TypeError):
        SandboxProvider()  # type: ignore[abstract]


def test_manager_resolves_image_reference(monkeypatch):
    manager = SandboxVMManager()
    monkeypatch.delenv("CUAMASTER_SANDBOX_IMAGE_PATH", raising=False)
    monkeypatch.setenv("CUAMASTER_SANDBOX_IMAGE_ID", "ubuntu-base")
    assert manager.get_vm_path() == "sandbox://ubuntu-base"
    monkeypatch.setenv("CUAMASTER_SANDBOX_IMAGE_PATH", "/images/disk.qcow2")
    assert manager.get_vm_path() == "/images/disk.qcow2"


def test_command_config_requires_commands(monkeypatch):
    monkeypatch.setenv("CUAMASTER_SANDBOX_IMAGE_ID", "ubuntu-base")
    monkeypatch.delenv("CUAMASTER_SANDBOX_START_COMMAND", raising=False)
    monkeypatch.delenv("CUAMASTER_SANDBOX_STOP_COMMAND", raising=False)
    with pytest.raises(ValueError, match="CUAMASTER_SANDBOX_START_COMMAND"):
        CommandSandboxConfig.from_env()


def test_command_config_requires_image(monkeypatch):
    monkeypatch.delenv("CUAMASTER_SANDBOX_IMAGE_ID", raising=False)
    monkeypatch.delenv("CUAMASTER_SANDBOX_IMAGE_PATH", raising=False)
    monkeypatch.setenv("CUAMASTER_SANDBOX_START_COMMAND", "orchestrator start")
    monkeypatch.setenv("CUAMASTER_SANDBOX_STOP_COMMAND", "orchestrator stop")
    with pytest.raises(ValueError, match="CUAMASTER_SANDBOX_IMAGE_ID"):
        CommandSandboxConfig.from_env()


def _configure_command_provider(monkeypatch, tmp_path):
    endpoints = {
        "resource_id": "lease-42",
        "server_host": "relay.example.com",
        "server_port": 5000,
        "chromium_host": "relay.example.com",
        "chromium_port": 9222,
        "vnc_host": "relay.example.com",
        "vnc_port": 8006,
        "vlc_host": "relay.example.com",
        "vlc_port": 8080,
    }
    start_script = tmp_path / "start.sh"
    start_script.write_text(
        "#!/bin/bash\n"
        f"echo '{json.dumps(endpoints)}'\n",
        encoding="utf-8",
    )
    start_script.chmod(0o755)
    stop_marker = tmp_path / "stopped.flag"
    monkeypatch.setenv("CUAMASTER_SANDBOX_IMAGE_ID", "ubuntu-base")
    monkeypatch.setenv(
        "CUAMASTER_SANDBOX_START_COMMAND",
        f"{start_script} {{image_id}} {{resource_id}} {{platform}}",
    )
    monkeypatch.setenv(
        "CUAMASTER_SANDBOX_STOP_COMMAND", f"touch {stop_marker} # {{resource_id}}"
    )
    monkeypatch.setenv("CUAMASTER_SANDBOX_START_TIMEOUT_S", "5")
    return stop_marker


def test_command_provider_parses_endpoints_and_releases(monkeypatch, tmp_path):
    stop_marker = _configure_command_provider(monkeypatch, tmp_path)
    provider = CommandSandboxProvider()
    monkeypatch.setattr(provider, "_wait_until_ready", lambda os_type: None)

    provider.start_emulator("sandbox://ubuntu-base", headless=True, os_type="Ubuntu")
    assert provider.resource_id == "lease-42"
    assert provider.get_ip_address("sandbox://ubuntu-base") == (
        "relay.example.com:5000:9222:8006:8080"
    )
    assert provider.vm_base_url == "http://relay.example.com:5000"

    with pytest.raises(NotImplementedError):
        provider.save_state("sandbox://ubuntu-base", "snapshot_1")

    provider.stop_emulator("sandbox://ubuntu-base")
    assert stop_marker.exists()
    assert provider.resource_id is None
    assert provider.vm_base_url is None


def test_command_provider_cleans_up_on_bad_endpoints(monkeypatch, tmp_path):
    stop_marker = _configure_command_provider(monkeypatch, tmp_path)
    bad_start = tmp_path / "bad_start.sh"
    bad_start.write_text(
        '#!/bin/bash\necho \'{"resource_id": "lease-43"}\'\n', encoding="utf-8"
    )
    bad_start.chmod(0o755)
    monkeypatch.setenv("CUAMASTER_SANDBOX_START_COMMAND", str(bad_start))
    provider = CommandSandboxProvider()
    with pytest.raises(ValueError, match="relay host"):
        provider.start_emulator(
            "sandbox://ubuntu-base", headless=True, os_type="Ubuntu"
        )
    assert stop_marker.exists()
