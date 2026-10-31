"""Experimental macOS QEMU/OpenCore launcher for a remote Linux KVM host."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


RESOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_DOCKER_IMAGE = "happysixd/osworld-docker:latest"
DEFAULT_RUNTIME_ROOT = "/tmp/gui-trajectory-macos-kvm"
DEFAULT_CPU_MODEL = (
    "Skylake-Client,-hle,-rtm,kvm=on,vendor=GenuineIntel,+invtsc,"
    "vmware-cpuid-freq=on,+ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,"
    "+xsaveopt,check"
)


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise RuntimeError(
            f"command {command[0]!r} failed with exit {completed.returncode}"
            + (f"\n{detail}" if detail else "")
        )
    return completed


def _validate_resource_id(resource_id: str) -> str:
    if not RESOURCE_ID_PATTERN.fullmatch(resource_id):
        raise ValueError(f"invalid resource id: {resource_id!r}")
    return resource_id


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _free_vnc_port() -> int:
    for port in range(5901, 7000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free VNC port in range 5901-6999")


def _configure_kvm() -> None:
    parameters = {
        Path("/sys/module/kvm/parameters/ignore_msrs"): "1\n",
        Path("/sys/module/kvm/parameters/report_ignored_msrs"): "0\n",
    }
    for path, value in parameters.items():
        try:
            path.write_text(value, encoding="utf-8")
        except OSError:
            pass


def select_boot(args: argparse.Namespace) -> dict[str, object]:
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    monitor_path = runtime_dir / "monitor.sock"
    deadline = time.monotonic() + args.monitor_timeout
    while not monitor_path.exists() and time.monotonic() < deadline:
        time.sleep(0.5)
    if not monitor_path.exists():
        raise TimeoutError(f"QEMU monitor socket was not created: {monitor_path}")
    time.sleep(args.delay)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as monitor:
        monitor.settimeout(10)
        monitor.connect(str(monitor_path))
        try:
            monitor.recv(4096)
        except TimeoutError:
            pass
        monitor.sendall(b"sendkey right\n")
        time.sleep(1)
        monitor.sendall(b"sendkey ret\n")
    return {"status": "boot-selected", "runtime_dir": str(runtime_dir)}


def build_qemu_args(
    *,
    ram_size: str,
    cpu_cores: int,
    cpu_model: str,
    osk: str,
    server_port: int,
    chromium_port: int,
    vnc_port: int,
    vlc_port: int,
) -> list[str]:
    host_forwards = [
        f"hostfwd=tcp:127.0.0.1:{server_port}-:5000",
        f"hostfwd=tcp:127.0.0.1:{chromium_port}-:9222",
        f"hostfwd=tcp:127.0.0.1:{vlc_port}-:8080",
    ]
    return [
        "-enable-kvm",
        "-m", ram_size,
        "-cpu", cpu_model,
        "-machine", "q35",
        "-smp", f"{cpu_cores},cores={cpu_cores},sockets=1",
        "-device", "qemu-xhci,id=xhci",
        "-device", "usb-kbd,bus=xhci.0",
        "-device", "usb-tablet,bus=xhci.0",
        "-global", "ICH9-LPC.acpi-pci-hotplug-with-bridge-support=off",
        "-device", f"isa-applesmc,osk={osk}",
        "-drive", "if=pflash,format=raw,readonly=on,file=/opencore/OVMF_CODE_4M.fd",
        "-drive", "if=pflash,format=raw,file=/runtime/OVMF_VARS.fd",
        "-smbios", "type=2",
        "-device", "ich9-intel-hda",
        "-device", "hda-duplex",
        "-device", "ich9-ahci,id=sata",
        "-drive", (
            "id=OpenCoreBoot,if=none,snapshot=on,format=qcow2,"
            "file=/opencore/OpenCore/OpenCore.qcow2"
        ),
        "-device", "ide-hd,bus=sata.2,drive=OpenCoreBoot",
        "-drive", "id=MacHDD,if=none,format=qcow2,file=/runtime/disk.qcow2",
        "-device", "ide-hd,bus=sata.4,drive=MacHDD",
        "-netdev", "user,id=net0," + ",".join(host_forwards),
        "-device", "vmxnet3,netdev=net0,id=net0,mac=52:54:00:c9:18:27",
        "-device", "vmware-svga",
        "-display", "none",
        "-vnc", f"127.0.0.1:{vnc_port - 5900}",
        "-monitor", "unix:/runtime/monitor.sock,server,nowait",
        "-serial", "stdio",
    ]


def start(args: argparse.Namespace) -> dict[str, object]:
    resource_id = _validate_resource_id(args.resource_id)
    image_path = Path(args.image_path).expanduser().resolve()
    seed_overlay = (
        Path(args.seed_overlay).expanduser().resolve()
        if args.seed_overlay else None
    )
    opencore_dir = Path(args.opencore_dir).expanduser().resolve()
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    runtime_dir = runtime_root / resource_id
    osk_file = Path(args.osk_file).expanduser().resolve()

    required = [
        image_path,
        opencore_dir / "OVMF_CODE_4M.fd",
        opencore_dir / "OVMF_VARS-1920x1080.fd",
        opencore_dir / "OpenCore" / "OpenCore.qcow2",
        osk_file,
    ]
    if seed_overlay is not None:
        required.append(seed_overlay)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing macOS KVM asset(s): " + ", ".join(missing))
    if runtime_dir.exists():
        raise FileExistsError(f"runtime directory already exists: {runtime_dir}")

    osk = osk_file.read_text(encoding="utf-8").strip()
    if not osk:
        raise ValueError(f"Apple SMC OSK file is empty: {osk_file}")

    server_port = _free_port()
    chromium_port = _free_port()
    vlc_port = _free_port()
    vnc_port = _free_vnc_port()
    runtime_dir.mkdir(parents=True)
    _configure_kvm()
    shutil.copyfile(
        opencore_dir / "OVMF_VARS-1920x1080.fd",
        runtime_dir / "OVMF_VARS.fd",
    )

    try:
        overlay_mounts = ["-v", f"{image_path}:/System.qcow2:ro"]
        backing_path = "/System.qcow2"
        if seed_overlay is not None:
            overlay_mounts.extend(["-v", f"{seed_overlay}:/Seed.qcow2:ro"])
            backing_path = "/Seed.qcow2"
        _run([
            "docker", "run", "--rm",
            *overlay_mounts,
            "-v", f"{runtime_dir}:/runtime",
            "--entrypoint", "qemu-img",
            args.docker_image,
            "create", "-f", "qcow2", "-F", "qcow2",
            "-b", backing_path, "/runtime/disk.qcow2",
        ])
        qemu_args = build_qemu_args(
            ram_size=args.ram_size,
            cpu_cores=args.cpu_cores,
            cpu_model=args.cpu_model,
            osk=osk,
            server_port=server_port,
            chromium_port=chromium_port,
            vnc_port=vnc_port,
            vlc_port=vlc_port,
        )
        qemu_mounts = ["-v", f"{image_path}:/System.qcow2:ro"]
        if seed_overlay is not None:
            qemu_mounts.extend(["-v", f"{seed_overlay}:/Seed.qcow2:ro"])
        _run([
            "docker", "run", "-d",
            "--name", resource_id,
            "--privileged",
            "--device", "/dev/kvm",
            "--network", "host",
            *qemu_mounts,
            "-v", f"{runtime_dir}:/runtime",
            "-v", f"{opencore_dir}:/opencore:ro",
            "--entrypoint", "qemu-system-x86_64",
            args.docker_image,
            *qemu_args,
        ])
        time.sleep(2)
        running = _run([
            "docker", "inspect", "-f", "{{.State.Running}}", resource_id
        ]).stdout.strip().casefold()
        if running != "true":
            logs = _run(["docker", "logs", resource_id], check=False)
            raise RuntimeError(
                f"macOS QEMU container exited during startup: "
                f"{(logs.stderr or logs.stdout).strip()[-4000:]}"
            )
        helper_log = runtime_dir / "boot-selector.log"
        with helper_log.open("ab") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "select-boot",
                    "--runtime-dir", str(runtime_dir),
                    "--delay", str(args.boot_selection_delay),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception:
        _run(["docker", "rm", "-f", resource_id], check=False)
        shutil.rmtree(runtime_dir, ignore_errors=True)
        raise

    return {
        "resource_id": resource_id,
        "server_host": "127.0.0.1",
        "server_port": server_port,
        "chromium_host": "127.0.0.1",
        "chromium_port": chromium_port,
        "vnc_host": "127.0.0.1",
        "vnc_port": vnc_port,
        "vlc_host": "127.0.0.1",
        "vlc_port": vlc_port,
        "runtime_dir": str(runtime_dir),
    }


def stop(args: argparse.Namespace) -> dict[str, object]:
    resource_id = _validate_resource_id(args.resource_id)
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    runtime_dir = runtime_root / resource_id
    _run(["docker", "rm", "-f", resource_id], check=False)
    shutil.rmtree(runtime_dir, ignore_errors=True)
    return {"resource_id": resource_id, "status": "stopped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--resource-id", required=True)
    start_parser.add_argument("--image-path", required=True)
    start_parser.add_argument("--opencore-dir", required=True)
    start_parser.add_argument("--osk-file", required=True)
    start_parser.add_argument("--seed-overlay")
    start_parser.add_argument("--runtime-root", default=DEFAULT_RUNTIME_ROOT)
    start_parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    start_parser.add_argument("--ram-size", default="8G")
    start_parser.add_argument("--cpu-cores", type=int, default=4)
    start_parser.add_argument("--cpu-model", default=DEFAULT_CPU_MODEL)
    start_parser.add_argument("--boot-selection-delay", type=float, default=12.0)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--resource-id", required=True)
    stop_parser.add_argument("--runtime-root", default=DEFAULT_RUNTIME_ROOT)

    select_parser = subparsers.add_parser("select-boot")
    select_parser.add_argument("--runtime-dir", required=True)
    select_parser.add_argument("--delay", type=float, default=12.0)
    select_parser.add_argument("--monitor-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "start":
        result = start(args)
    elif args.command == "stop":
        result = stop(args)
    else:
        result = select_boot(args)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
