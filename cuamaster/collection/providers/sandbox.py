"""Snapshot-and-rollback sandbox provider interface (paper Sec. 2.4.2).

The resilient trajectory collection loop materializes each task in a VM
selected from the image pool, creates incremental snapshots at stable
boundaries, and rolls the environment back to the latest valid snapshot
preceding the estimated divergence step when verification fails.

This module defines the provider contract only. The orchestration service
that actually leases, boots, snapshots, and releases VMs is deployment
specific, so implementations must be supplied (and registered) by the user:

* :class:`SandboxProvider` — the abstract VM lifecycle + snapshot/rollback
  contract consumed by the collection runner.
* :class:`SandboxVMManager` — a pool bookkeeping duck type; the default
  implementation delegates pooling to the external orchestrator and only
  resolves the configured image reference.
* :class:`CommandSandboxProvider` — a generic implementation that shells out
  to user-configured start/stop commands. Point it at any CLI that can lease
  a VM and print its controller endpoints as JSON; no vendor protocol is
  assumed.
* :class:`MockSandboxProvider` — an in-memory provider for tests.

Register a custom provider with :func:`register_sandbox_provider`, or pass a
``"module.path:factory"`` provider name to resolve one dynamically.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import time
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("cuamaster.collection.providers.sandbox")

#: Profile ``provider`` values routed to the registered sandbox provider.
SANDBOX_PROVIDER_NAMES = {"sandbox", "sandbox_manager"}

#: Name of the environment variable that carries the active provider name
#: from the backend into the runner process.
SANDBOX_PROVIDER_ENV = "CUAMASTER_SANDBOX_PROVIDER"


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _expected_platform(os_type: str) -> str:
    normalized = os_type.strip().casefold()
    if normalized in {"ubuntu", "linux"}:
        return "Linux"
    if normalized in {"windows", "win32"}:
        return "Windows"
    if normalized in {"darwin", "macos", "mac"}:
        return "Darwin"
    return os_type.strip()


class SandboxProvider(ABC):
    """VM lifecycle, incremental snapshot, and rollback contract.

    The method signatures mirror the desktop-environment provider duck type
    used by the rollout runner, so an implementation can be injected without
    touching the runner itself. ``path_to_vm`` is an opaque image reference
    (a local disk path or a catalog URI); providers that lease remote VMs may
    ignore it in favor of their own configuration.
    """

    @abstractmethod
    def start_emulator(
        self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"
    ) -> None:
        """Materialize a VM for ``path_to_vm`` and block until its controller
        answers platform and screenshot probes."""

    @abstractmethod
    def get_ip_address(self, path_to_vm: str) -> str:
        """Return the controller address of the running VM.

        The collection runner expects ``host:server:chromium:vnc:vlc`` with
        one relay host followed by the four channel ports.
        """

    @abstractmethod
    def save_state(self, path_to_vm: str, snapshot_name: str) -> None:
        """Create an incremental snapshot at a stable boundary.

        Implementations should persist the VM state together with a
        lightweight state signature (active applications, window layout,
        relevant files, and controller metadata) so a later
        :meth:`revert_to_snapshot` can restore the exact prefix state and the
        detector can match the estimated divergence step to a snapshot.
        """

    @abstractmethod
    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> None:
        """Restore the environment to a previously saved snapshot."""

    @abstractmethod
    def stop_emulator(self, path_to_vm: str, region: str | None = None) -> None:
        """Release the VM and any provider-side lease or overlay state."""

    def snapshot_signature(self, snapshot_name: str) -> dict[str, Any] | None:
        """Return the state signature stored with ``snapshot_name``.

        Optional: providers without signature support return ``None`` and the
        retry loop falls back to rolling back to the initial snapshot.
        """
        del snapshot_name
        return None


class SandboxVMManager:
    """Pool bookkeeping stub for externally orchestrated VMs.

    The external sandbox manager owns scheduling and slot accounting, so
    registry operations are no-ops; only :meth:`get_vm_path` matters — it
    resolves the opaque image reference handed to the provider.
    """

    def initialize_registry(self, **kwargs: Any) -> None:
        pass

    def add_vm(self, vm_path: str, **kwargs: Any) -> None:
        pass

    def delete_vm(self, vm_path: str, **kwargs: Any) -> None:
        pass

    def occupy_vm(self, vm_path: str, pid: int, **kwargs: Any) -> None:
        pass

    def list_free_vms(self, **kwargs: Any) -> list[str]:
        return []

    def check_and_clean(self, **kwargs: Any) -> None:
        pass

    def get_vm_path(self, **kwargs: Any) -> str:
        image_path = os.environ.get("CUAMASTER_SANDBOX_IMAGE_PATH", "").strip()
        if image_path:
            return image_path
        image_id = os.environ.get("CUAMASTER_SANDBOX_IMAGE_ID", "vm").strip()
        return f"sandbox://{image_id or 'vm'}"


@dataclass(frozen=True)
class CommandSandboxConfig:
    """Configuration for :class:`CommandSandboxProvider`.

    ``start_command`` and ``stop_command`` are shell templates expanded with
    ``str.format_map``; available placeholders: ``image_id``, ``image_path``,
    ``image_selector``, ``resource_id``, ``os_type``, ``platform``,
    ``architecture``, ``vm_backend``. The start command must print one JSON
    object on its final stdout line::

        {"resource_id": "...", "server_host": "...", "server_port": 1,
         "chromium_host": "...", "chromium_port": 2,
         "vnc_host": "...", "vnc_port": 3, "vlc_host": "...", "vlc_port": 4}
    """

    image_path: str
    start_command: str
    stop_command: str
    start_timeout: float
    image_id: str = ""
    vm_backend: str = ""
    architecture: str = "x86_64"

    @classmethod
    def from_env(cls) -> "CommandSandboxConfig":
        image_path = os.environ.get("CUAMASTER_SANDBOX_IMAGE_PATH", "").strip()
        image_id = os.environ.get("CUAMASTER_SANDBOX_IMAGE_ID", "").strip()
        start_command = os.environ.get("CUAMASTER_SANDBOX_START_COMMAND", "").strip()
        stop_command = os.environ.get("CUAMASTER_SANDBOX_STOP_COMMAND", "").strip()
        if not image_id and not image_path:
            raise ValueError(
                "CUAMASTER_SANDBOX_IMAGE_ID or CUAMASTER_SANDBOX_IMAGE_PATH is required"
            )
        if not start_command or not stop_command:
            raise ValueError(
                "CUAMASTER_SANDBOX_START_COMMAND and CUAMASTER_SANDBOX_STOP_COMMAND "
                "are required; they must invoke your sandbox orchestrator CLI"
            )
        return cls(
            image_path=image_path,
            start_command=start_command,
            stop_command=stop_command,
            start_timeout=float(
                os.environ.get("CUAMASTER_SANDBOX_START_TIMEOUT_S", "2400")
            ),
            image_id=image_id,
            vm_backend=os.environ.get("CUAMASTER_SANDBOX_VM_BACKEND", "").strip(),
            architecture=os.environ.get(
                "CUAMASTER_SANDBOX_ARCHITECTURE", "x86_64"
            ).strip()
            or "x86_64",
        )


class CommandSandboxProvider(SandboxProvider):
    """Sandbox provider driven by configured orchestrator commands.

    This keeps the collection stack decoupled from any specific sandbox
    manager: the lease/boot/release protocol lives entirely in the two
    configured commands. Snapshots are delegated to the orchestrator when it
    exposes them; otherwise :meth:`save_state` raises and the retry loop
    treats every rollback as a full environment reset.
    """

    def __init__(self, region: str | None = None):
        self.region = region
        self.config = CommandSandboxConfig.from_env()
        self.resource_id: str | None = None
        self.endpoint_host: str | None = None
        self.ports: dict[str, int] = {}
        self.vm_base_url: str | None = None
        self.os_type = "Ubuntu"
        self.platform = "linux"
        self.vm_backend = self.config.vm_backend or "desktop_env"
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _run(self, command: str, *, timeout: float, check: bool = True):
        completed = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.stderr:
            logger.info("sandbox orchestrator: %s", completed.stderr.strip())
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-4000:]
            raise RuntimeError(
                "sandbox orchestrator command failed"
                + (f": {detail}" if detail else "")
            )
        return completed

    def _command_fields(self, resource_id: str, os_type: str) -> dict[str, str]:
        platform = _expected_platform(os_type).casefold()
        if platform == "darwin":
            platform = "macos"
        image_path = self.config.image_path or ""
        return {
            "image_id": self.config.image_id,
            "image_path": image_path,
            "image_selector": self.config.image_id or image_path,
            "resource_id": resource_id,
            "os_type": os_type,
            "platform": platform,
            "architecture": self.config.architecture,
            "vm_backend": self.vm_backend,
        }

    def start_emulator(
        self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"
    ) -> None:
        del headless
        if self.resource_id is not None:
            raise RuntimeError("sandbox VM is already running")
        requested_id = "cuamaster-" + uuid.uuid4().hex[:12]
        self.resource_id = requested_id
        platform = _expected_platform(os_type).casefold()
        if platform == "darwin":
            platform = "macos"
        self.os_type = os_type
        self.platform = platform
        self.vm_backend = self.config.vm_backend or (
            "macos_kvm" if platform == "macos" else "desktop_env"
        )
        if not self.config.image_path and not path_to_vm.startswith("sandbox://"):
            fields = self._command_fields(requested_id, os_type)
            fields["image_path"] = path_to_vm
            fields["image_selector"] = self.config.image_id or path_to_vm
        else:
            fields = self._command_fields(requested_id, os_type)
        try:
            completed = self._run(
                self.config.start_command.format_map(fields),
                timeout=self.config.start_timeout,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.resource_id = str(payload.get("resource_id") or requested_id)
            hosts = {
                str(payload.get(f"{name}_host") or "")
                for name in ("server", "chromium", "vnc", "vlc")
            }
            if "" in hosts or len(hosts) != 1:
                raise ValueError(
                    "sandbox endpoints must share one non-empty relay host"
                )
            self.endpoint_host = hosts.pop()
            self.ports = {
                name: int(payload[f"{name}_port"])
                for name in ("server", "chromium", "vnc", "vlc")
            }
            self.vm_base_url = f"http://{self.endpoint_host}:{self.ports['server']}"
            self._wait_until_ready(os_type)
        except Exception:
            self.stop_emulator(path_to_vm)
            raise

    def _wait_until_ready(self, os_type: str) -> None:
        deadline = time.monotonic() + self.config.start_timeout
        last_error = "controller did not respond"
        while time.monotonic() < deadline:
            try:
                with self._opener.open(
                    f"{self.vm_base_url}/platform", timeout=10
                ) as response:
                    actual = response.read().decode("utf-8", errors="replace").strip()
                try:
                    actual = str(json.loads(actual))
                except json.JSONDecodeError:
                    pass
                expected = _expected_platform(os_type)
                if actual.casefold() != expected.casefold():
                    raise RuntimeError(
                        f"remote guest platform mismatch: requested {os_type!r}, "
                        f"controller reports {actual!r}"
                    )
                with self._opener.open(
                    f"{self.vm_base_url}/screenshot", timeout=20
                ) as response:
                    screenshot = response.read()
                if screenshot.startswith((b"\x89PNG", b"\xff\xd8\xff")):
                    return
                last_error = "controller returned an invalid screenshot"
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5)
        raise TimeoutError(
            "sandbox VM controller was not ready after "
            f"{self.config.start_timeout}s: {last_error}"
        )

    def get_ip_address(self, path_to_vm: str) -> str:
        del path_to_vm
        if not self.endpoint_host or not self.ports:
            raise RuntimeError("sandbox VM has not been started")
        return ":".join(
            [
                self.endpoint_host,
                str(self.ports["server"]),
                str(self.ports["chromium"]),
                str(self.ports["vnc"]),
                str(self.ports["vlc"]),
            ]
        )

    def save_state(self, path_to_vm: str, snapshot_name: str) -> None:
        del path_to_vm, snapshot_name
        raise NotImplementedError(
            "the configured sandbox orchestrator does not expose snapshots; "
            "implement SandboxProvider.save_state/revert_to_snapshot against "
            "your orchestrator to enable prefix-preserving rollback retries"
        )

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> None:
        del snapshot_name
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region: str | None = None) -> None:
        del path_to_vm, region
        resource_id = self.resource_id
        cleanup_succeeded = True
        if resource_id:
            command = self.config.stop_command.format_map(
                self._command_fields(resource_id, self.os_type)
            )
            cleanup_succeeded = False
            last_detail = "orchestrator cleanup did not run"
            for attempt in range(3):
                try:
                    completed = self._run(command, timeout=120, check=False)
                    if completed.returncode == 0:
                        cleanup_succeeded = True
                        break
                    last_detail = (completed.stderr or completed.stdout).strip()[-2000:]
                except Exception as exc:  # noqa: BLE001
                    last_detail = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    time.sleep(1)
            if not cleanup_succeeded:
                logger.error(
                    "failed to release sandbox VM %s after 3 attempts: %s",
                    resource_id,
                    last_detail,
                )
        if cleanup_succeeded:
            self.resource_id = None
        self.endpoint_host = None
        self.ports = {}
        self.vm_base_url = None


@dataclass
class _MockSnapshot:
    name: str
    signature: dict[str, Any] = field(default_factory=dict)


class MockSandboxProvider(SandboxProvider):
    """In-memory provider used by the test suite and dry runs."""

    def __init__(self, region: str | None = None):
        self.region = region
        self.running = False
        self.os_type = "Ubuntu"
        self.snapshots: list[_MockSnapshot] = []
        self.current_snapshot: str | None = None
        self.events: list[tuple[str, str]] = []

    def start_emulator(
        self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"
    ) -> None:
        del headless
        if self.running:
            raise RuntimeError("mock sandbox VM is already running")
        self.running = True
        self.os_type = os_type
        self.events.append(("start", path_to_vm))

    def get_ip_address(self, path_to_vm: str) -> str:
        del path_to_vm
        if not self.running:
            raise RuntimeError("mock sandbox VM has not been started")
        return "127.0.0.1:5000:9222:8006:8080"

    def save_state(self, path_to_vm: str, snapshot_name: str) -> None:
        del path_to_vm
        if not self.running:
            raise RuntimeError("mock sandbox VM has not been started")
        self.snapshots.append(
            _MockSnapshot(
                name=snapshot_name,
                signature={
                    "snapshot": snapshot_name,
                    "os_type": self.os_type,
                    "ordinal": len(self.snapshots),
                },
            )
        )
        self.current_snapshot = snapshot_name
        self.events.append(("snapshot", snapshot_name))

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> None:
        del path_to_vm
        names = [snapshot.name for snapshot in self.snapshots]
        if snapshot_name not in names:
            raise KeyError(f"unknown snapshot: {snapshot_name}")
        index = names.index(snapshot_name)
        self.snapshots = self.snapshots[: index + 1]
        self.current_snapshot = snapshot_name
        self.events.append(("rollback", snapshot_name))

    def stop_emulator(self, path_to_vm: str, region: str | None = None) -> None:
        del path_to_vm, region
        self.running = False
        self.events.append(("stop", ""))

    def snapshot_signature(self, snapshot_name: str) -> dict[str, Any] | None:
        for snapshot in self.snapshots:
            if snapshot.name == snapshot_name:
                return dict(snapshot.signature)
        return None


ProviderFactory = Callable[
    [str | None], tuple[SandboxVMManager, SandboxProvider]
]

_PROVIDER_REGISTRY: dict[str, ProviderFactory] = {}


def register_sandbox_provider(name: str, factory: ProviderFactory) -> None:
    """Register a factory returning ``(manager, provider)`` for ``name``."""
    key = name.strip().lower()
    if not key:
        raise ValueError("provider name cannot be empty")
    _PROVIDER_REGISTRY[key] = factory


def create_sandbox_provider(
    name: str, region: str | None = None
) -> tuple[SandboxVMManager, SandboxProvider]:
    """Instantiate the sandbox provider registered under ``name``.

    ``name`` may also be ``"module.path:factory"`` to load a factory from an
    installed package without prior registration.
    """
    key = str(name or "").strip().lower()
    factory = _PROVIDER_REGISTRY.get(key)
    if factory is None and ":" in str(name or ""):
        module_name, attribute = str(name).split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute, None)
    if factory is None:
        raise ValueError(
            f"unknown sandbox provider {name!r}; register one with "
            "register_sandbox_provider() or use module.path:factory"
        )
    return factory(region)


def _command_factory(region: str | None):
    return SandboxVMManager(), CommandSandboxProvider(region)


def _mock_factory(region: str | None):
    return SandboxVMManager(), MockSandboxProvider(region)


for _name in SANDBOX_PROVIDER_NAMES:
    register_sandbox_provider(_name, _command_factory)
register_sandbox_provider("mock", _mock_factory)
