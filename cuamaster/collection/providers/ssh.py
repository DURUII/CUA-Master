"""OSWorld VM provider that boots a remote guest over SSH."""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("cuamaster.collection.remote_ssh_provider")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _expected_platform(os_type: str) -> str:
    normalized = os_type.strip().casefold()
    if normalized in {"ubuntu", "linux"}:
        return "Linux"
    if normalized in {"windows", "win32"}:
        return "Windows"
    if normalized in {"darwin", "macos", "mac"}:
        return "Darwin"
    return os_type.strip()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class RemoteSSHConfig:
    host: str
    port: int
    user: str
    password: str
    identity_file: str | None
    remote_image_path: str
    docker_image: str
    docker_image_tar: str | None
    stage_image: bool
    allow_pull: bool
    start_timeout: float
    connect_timeout: int
    strict_host_key_checking: str
    known_hosts_file: str | None
    remote_start_command: str | None
    remote_stop_command: str | None
    ram_size: str
    cpu_cores: int
    disk_size: str

    @classmethod
    def from_env(cls) -> "RemoteSSHConfig":
        host = os.environ.get("GUI_TRAJECTORY_SSH_HOST", "").strip()
        remote_image_path = os.environ.get(
            "GUI_TRAJECTORY_SSH_REMOTE_IMAGE_PATH", ""
        ).strip()
        if not host:
            raise ValueError("GUI_TRAJECTORY_SSH_HOST is required")
        if not remote_image_path:
            raise ValueError("GUI_TRAJECTORY_SSH_REMOTE_IMAGE_PATH is required")
        return cls(
            host=host,
            port=int(os.environ.get("GUI_TRAJECTORY_SSH_PORT", "22")),
            user=os.environ.get("GUI_TRAJECTORY_SSH_USER", "root").strip() or "root",
            password=os.environ.get("GUI_TRAJECTORY_SSH_PASSWORD", ""),
            identity_file=os.environ.get("GUI_TRAJECTORY_SSH_IDENTITY_FILE") or None,
            remote_image_path=remote_image_path,
            docker_image=os.environ.get(
                "GUI_TRAJECTORY_SSH_DOCKER_IMAGE", "happysixd/osworld-docker:latest"
            ),
            docker_image_tar=os.environ.get(
                "GUI_TRAJECTORY_SSH_DOCKER_IMAGE_TAR"
            ) or None,
            stage_image=_truthy(os.environ.get("GUI_TRAJECTORY_SSH_STAGE_IMAGE")),
            allow_pull=_truthy(os.environ.get("GUI_TRAJECTORY_SSH_ALLOW_PULL")),
            start_timeout=float(
                os.environ.get("GUI_TRAJECTORY_SSH_START_TIMEOUT_S", "1800")
            ),
            connect_timeout=int(
                os.environ.get("GUI_TRAJECTORY_SSH_CONNECT_TIMEOUT_S", "15")
            ),
            strict_host_key_checking=os.environ.get(
                "GUI_TRAJECTORY_SSH_STRICT_HOST_KEY_CHECKING", "accept-new"
            ),
            known_hosts_file=os.environ.get(
                "GUI_TRAJECTORY_SSH_KNOWN_HOSTS_FILE"
            ) or None,
            remote_start_command=os.environ.get(
                "GUI_TRAJECTORY_SSH_REMOTE_START_COMMAND"
            ) or None,
            remote_stop_command=os.environ.get(
                "GUI_TRAJECTORY_SSH_REMOTE_STOP_COMMAND"
            ) or None,
            ram_size=os.environ.get("GUI_TRAJECTORY_SSH_RAM_SIZE", "8G"),
            cpu_cores=int(os.environ.get("GUI_TRAJECTORY_SSH_CPU_CORES", "4")),
            disk_size=os.environ.get("GUI_TRAJECTORY_SSH_DISK_SIZE", "64G"),
        )


ForwardTarget = int | tuple[str, int]


class _SSHControlForward:
    def __init__(self, client: "SSHClient", forwards: dict[int, ForwardTarget]):
        self.client = client
        self.forwards = dict(forwards)
        self.closed = False

    def terminate(self) -> None:
        if not self.closed:
            self.client._change_forwards("cancel", self.forwards, check=False)
            self.closed = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        self.terminate()


class SSHClient:
    def __init__(self, config: RemoteSSHConfig):
        self.config = config
        self._temp_dir = Path(tempfile.mkdtemp(prefix="gui-trajectory-ssh-"))
        configured_control_path = os.environ.get(
            "GUI_TRAJECTORY_SSH_CONTROL_PATH", ""
        ).strip()
        self.control_path = Path(configured_control_path or self._temp_dir / "control")
        self._owns_control = not configured_control_path
        self.known_hosts_file = Path(
            config.known_hosts_file or self._temp_dir / "known_hosts"
        )
        self.askpass_path: Path | None = None
        if config.password:
            self.askpass_path = self._temp_dir / "askpass.sh"
            self.askpass_path.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$GUI_TRAJECTORY_SSH_PASSWORD\"\n",
                encoding="utf-8",
            )
            self.askpass_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    @property
    def target(self) -> str:
        return f"{self.config.user}@{self.config.host}"

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.password:
            env.update({
                "GUI_TRAJECTORY_SSH_PASSWORD": self.config.password,
                "SSH_ASKPASS": str(self.askpass_path),
                "SSH_ASKPASS_REQUIRE": "force",
                "DISPLAY": env.get("DISPLAY") or "gui-trajectory:0",
            })
        return env

    def _options(self) -> list[str]:
        options = [
            "-o", f"ConnectTimeout={self.config.connect_timeout}",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4",
            "-o", "NumberOfPasswordPrompts=1",
            "-o", f"StrictHostKeyChecking={self.config.strict_host_key_checking}",
            "-o", f"UserKnownHostsFile={self.known_hosts_file}",
            "-o", f"ControlPath={self.control_path}",
        ]
        if self.config.password and not self.config.identity_file:
            options.extend([
                "-o", "PreferredAuthentications=password",
                "-o", "PubkeyAuthentication=no",
                "-o", "GSSAPIAuthentication=no",
            ])
        if self.config.identity_file:
            options.extend(["-i", self.config.identity_file])
        return options

    def _ensure_master(self) -> None:
        check_command = [
            "ssh", "-p", str(self.config.port),
            "-o", f"ControlPath={self.control_path}",
            "-O", "check", self.target,
        ]
        checked = subprocess.run(
            check_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self._environment(),
            check=False,
        )
        if checked.returncode == 0:
            return
        if not self._owns_control:
            raise RuntimeError(
                f"configured SSH control socket is unavailable: {self.control_path}"
            )
        command = [
            "ssh", "-fN", "-M", "-p", str(self.config.port),
            *self._options(),
            "-o", "ControlPersist=10m",
            self.target,
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._environment(),
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(
                "failed to establish persistent SSH connection"
                + (f": {detail}" if detail else "")
            )

    def _change_forwards(
        self,
        operation: str,
        forwards: dict[int, ForwardTarget],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "ssh", "-p", str(self.config.port),
            "-o", f"ControlPath={self.control_path}",
            "-O", operation,
        ]
        for local_port, target in forwards.items():
            remote_host, remote_port = (
                target if isinstance(target, tuple) else ("127.0.0.1", target)
            )
            command.extend([
                "-L", f"127.0.0.1:{local_port}:{remote_host}:{remote_port}"
            ])
        command.append(self.target)
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._environment(),
            check=False,
        )
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(
                f"failed to {operation} SSH port forwards"
                + (f": {detail}" if detail else "")
            )
        return completed

    def run(
        self,
        script: str,
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self._ensure_master()
        command = [
            "ssh", "-p", str(self.config.port), *self._options(), self.target,
            "bash", "-lc", shlex.quote(script),
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._environment(),
            timeout=timeout or self.config.start_timeout,
            check=False,
        )
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(
                f"remote SSH command failed with exit {completed.returncode}: {detail}"
            )
        return completed

    def copy_to(self, source: Path, destination: str) -> None:
        self._ensure_master()
        command = [
            "scp", "-P", str(self.config.port), *self._options(),
            str(source), f"{self.target}:{destination}",
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._environment(),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "remote image upload failed: "
                + (completed.stderr or completed.stdout).strip()[-2000:]
            )

    def start_tunnel(
        self, forwards: dict[int, ForwardTarget]
    ) -> _SSHControlForward:
        self._ensure_master()
        self._change_forwards("forward", forwards)
        return _SSHControlForward(self, forwards)

    def close(self) -> None:
        if self._owns_control and self.control_path.exists():
            subprocess.run(
                [
                    "ssh", "-p", str(self.config.port),
                    "-o", f"ControlPath={self.control_path}",
                    "-O", "exit", self.target,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._environment(),
                check=False,
            )
        shutil.rmtree(self._temp_dir, ignore_errors=True)


class RemoteSSHProvider:
    def __init__(self, region: str | None = None):
        self.region = region
        self.config = RemoteSSHConfig.from_env()
        self.ssh = SSHClient(self.config)
        self.container_name: str | None = None
        self.remote_resource_id: str | None = None
        self.tunnel: _SSHControlForward | None = None
        self.local_ports: dict[str, int] = {}
        self.remote_ports: dict[str, int] = {}
        self.remote_hosts: dict[str, str] = {}
        self.vm_base_url: str | None = None

    def _ensure_remote_image(self, local_image_path: str) -> None:
        remote = shlex.quote(self.config.remote_image_path)
        present = self.ssh.run(f"test -r {remote}", check=False)
        if present.returncode == 0:
            return
        if present.returncode != 1:
            detail = (present.stderr or present.stdout).strip()[-2000:]
            raise RuntimeError(
                "failed to check the remote VM image over SSH"
                + (f": {detail}" if detail else "")
            )
        if not self.config.stage_image:
            raise FileNotFoundError(
                f"remote VM image is missing: {self.config.remote_image_path}; "
                "stage it first or set stage_image=true"
            )
        local = Path(local_image_path)
        if not local.is_file():
            raise FileNotFoundError(f"local VM image is missing: {local}")
        parent = str(Path(self.config.remote_image_path).parent)
        partial = self.config.remote_image_path + ".partial"
        self.ssh.run(f"mkdir -p {shlex.quote(parent)}")
        self.ssh.copy_to(local, partial)
        self.ssh.run(
            f"mv -f {shlex.quote(partial)} {shlex.quote(self.config.remote_image_path)}"
        )

    def _default_start(self, os_type: str) -> dict[str, Any]:
        expected = _expected_platform(os_type)
        if expected == "Darwin":
            raise ValueError(
                "default remote SSH Docker launcher supports Ubuntu/Linux and Windows only; "
                "macOS requires a compatible remote_start_command"
            )
        name = "gui-trajectory-" + uuid.uuid4().hex[:12]
        self.container_name = name
        docker_image = shlex.quote(self.config.docker_image)
        image_path = shlex.quote(self.config.remote_image_path)
        image_tar = (
            shlex.quote(self.config.docker_image_tar)
            if self.config.docker_image_tar else ""
        )
        ensure_image = (
            f"docker image inspect {docker_image} >/dev/null 2>&1"
            + (
                f" || docker load -i {image_tar} >/dev/null"
                if image_tar else ""
            )
            + (
                f" || docker pull {docker_image} >/dev/null"
                if self.config.allow_pull else ""
            )
        )
        run_parts = [
            "docker", "run", "-d", "--name", name, "--privileged",
            "--cap-add", "NET_ADMIN",
            "--device", "/dev/kvm",
            "--device", "/dev/net/tun",
            "-e", f"DISK_SIZE={self.config.disk_size}",
            "-e", f"RAM_SIZE={self.config.ram_size}",
            "-e", f"CPU_CORES={self.config.cpu_cores}",
            "-v", f"{self.config.remote_image_path}:/System.qcow2:ro",
            self.config.docker_image,
        ]
        run_command = " ".join(shlex.quote(value) for value in run_parts)
        script = "\n".join([
            "set -e",
            "command -v docker >/dev/null",
            "docker info >/dev/null",
            "test -c /dev/kvm",
            "test -c /dev/net/tun",
            f"test -r {image_path}",
            ensure_image,
            run_command + " >/dev/null",
            "container_ip=$(docker inspect -f "
            "'{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "
            + shlex.quote(name) + ")",
            "case \"$container_ip\" in ''|*[!0-9a-fA-F:.]*) exit 1 ;; esac",
            "printf '{\"resource_id\":\"%s\",\"server_host\":\"%s\","
            "\"server_port\":5000,\"chromium_host\":\"%s\","
            "\"chromium_port\":9222,\"vnc_host\":\"%s\","
            "\"vnc_port\":8006,\"vlc_host\":\"%s\",\"vlc_port\":8080}\\n' "
            + " ".join([
                shlex.quote(name), '"$container_ip"', '"$container_ip"',
                '"$container_ip"', '"$container_ip"',
            ]),
        ])
        completed = self.ssh.run(script)
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def _custom_start(self, os_type: str) -> dict[str, Any]:
        resource_id = "gui-trajectory-" + uuid.uuid4().hex[:12]
        self.remote_resource_id = resource_id
        fields = {
            "image_path": self.config.remote_image_path,
            "resource_id": resource_id,
            "os_type": os_type,
            "ram_size": self.config.ram_size,
            "cpu_cores": str(self.config.cpu_cores),
            "disk_size": self.config.disk_size,
        }
        command = str(self.config.remote_start_command).format_map(fields)
        completed = self.ssh.run(command)
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        payload.setdefault("resource_id", resource_id)
        return payload

    def start_emulator(
        self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"
    ) -> None:
        del headless
        if self.tunnel is not None or self.remote_resource_id is not None:
            raise RuntimeError("remote VM is already running")
        self._ensure_remote_image(path_to_vm)
        try:
            payload = (
                self._custom_start(os_type)
                if self.config.remote_start_command
                else self._default_start(os_type)
            )
            self.remote_resource_id = str(payload["resource_id"])
            self.remote_ports = {
                name: int(payload[f"{name}_port"])
                for name in ("server", "chromium", "vnc", "vlc")
            }
            self.remote_hosts = {
                name: str(payload.get(f"{name}_host") or "127.0.0.1")
                for name in self.remote_ports
            }
            self.local_ports = {name: _free_local_port() for name in self.remote_ports}
            self.tunnel = self.ssh.start_tunnel({
                self.local_ports[name]: (
                    self.remote_hosts[name], self.remote_ports[name]
                )
                for name in self.remote_ports
            })
            self.vm_base_url = f"http://127.0.0.1:{self.local_ports['server']}"
            self._wait_until_ready(os_type)
        except Exception:
            self.stop_emulator(path_to_vm)
            raise

    def _wait_until_ready(self, os_type: str) -> None:
        deadline = time.monotonic() + self.config.start_timeout
        last_error = "controller did not respond"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
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
                with urllib.request.urlopen(
                    f"{self.vm_base_url}/screenshot", timeout=20
                ) as response:
                    screenshot = response.read()
                if screenshot.startswith(b"\x89PNG") or screenshot.startswith(b"\xff\xd8\xff"):
                    return
                last_error = "controller returned an invalid screenshot"
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5)
        raise TimeoutError(
            f"remote VM controller was not ready after {self.config.start_timeout}s: {last_error}"
        )

    def get_ip_address(self, path_to_vm: str) -> str:
        del path_to_vm
        if not self.local_ports:
            raise RuntimeError("remote VM has not been started")
        return ":".join([
            "127.0.0.1",
            str(self.local_ports["server"]),
            str(self.local_ports["chromium"]),
            str(self.local_ports["vnc"]),
            str(self.local_ports["vlc"]),
        ])

    def save_state(self, path_to_vm: str, snapshot_name: str) -> None:
        del path_to_vm, snapshot_name
        raise NotImplementedError("remote SSH provider does not support snapshots")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> None:
        del snapshot_name
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region: str | None = None) -> None:
        del path_to_vm, region
        if self.tunnel is not None:
            self.tunnel.terminate()
            try:
                self.tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.tunnel.kill()
            self.tunnel = None
        resource_id = self.remote_resource_id or self.container_name
        cleanup_succeeded = True
        if resource_id:
            if self.config.remote_stop_command:
                command = self.config.remote_stop_command.format_map({
                    "resource_id": resource_id,
                    "image_path": self.config.remote_image_path,
                })
            else:
                quoted_resource = shlex.quote(resource_id)
                command = (
                    f"if docker container inspect {quoted_resource} >/dev/null 2>&1; then "
                    f"docker rm -f {quoted_resource} >/dev/null; fi"
                )
            cleanup_succeeded = False
            last_detail = "remote cleanup did not run"
            for attempt in range(3):
                try:
                    completed = self.ssh.run(command, timeout=120, check=False)
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
                    "failed to clean remote VM resource %s after 3 attempts: %s",
                    resource_id,
                    last_detail,
                )
        if cleanup_succeeded:
            self.remote_resource_id = None
            self.container_name = None
        self.local_ports = {}
        self.remote_ports = {}
        self.remote_hosts = {}
        self.vm_base_url = None


class RemoteSSHVMManager:
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
        return os.environ.get(
            "GUI_TRAJECTORY_SSH_REMOTE_IMAGE_PATH", "remote-ssh://vm"
        )


def create_remote_ssh_provider(region: str | None = None):
    return RemoteSSHVMManager(), RemoteSSHProvider(region)
