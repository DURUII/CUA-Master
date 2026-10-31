#!/usr/bin/env python3
"""Inventory GUI applications from inside a booted Linux, Windows, or macOS VM."""
from __future__ import annotations

import configparser
import datetime as dt
import glob
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


LINUX_DESKTOP_ROOTS = (
    "/usr/share/applications",
    "/usr/local/share/applications",
    "/var/lib/snapd/desktop/applications",
    "/var/lib/flatpak/exports/share/applications",
    "/home/user/.local/share/applications",
    "/home/user/.local/share/flatpak/exports/share/applications",
)

LINUX_VERSION_COMMANDS = {
    "Google Chrome": ["google-chrome", "--version"],
    "Chromium": ["chromium", "--version"],
    "Firefox": ["firefox", "--version"],
    "LibreOffice": ["libreoffice", "--version"],
    "GIMP": ["gimp", "--version"],
    "VLC": ["vlc", "--version"],
    "Visual Studio Code": ["code", "--version"],
    "Thunderbird": ["thunderbird", "--version"],
    "GNOME Terminal": ["gnome-terminal", "--version"],
    "GNOME Files": ["nautilus", "--version"],
    "XTerm": ["xterm", "-version"],
    "Claude Code": ["claude", "--version"],
}


def run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )


def first_line(value: str) -> str:
    return next((line.strip() for line in value.splitlines() if line.strip()), "")


def executable_from_exec(raw: str) -> str | None:
    try:
        tokens = shlex.split(re.sub(r"%[fFuUdDnNickvm]", "", raw))
    except ValueError:
        return None
    if not tokens:
        return None
    if tokens[0] == "env":
        tokens = tokens[1:]
        while tokens and "=" in tokens[0] and not tokens[0].startswith("/"):
            tokens = tokens[1:]
    return tokens[0] if tokens else None


def package_for_path(path: Path) -> str | None:
    result = run(["dpkg-query", "-S", str(path)])
    if result.returncode != 0 or ": " not in result.stdout:
        return None
    owners = result.stdout.splitlines()[0].rsplit(": ", 1)[0]
    return owners.split(",", 1)[0].strip()


def package_metadata(package: str, cache: dict[str, dict[str, str]]) -> dict[str, str]:
    if package not in cache:
        result = run(
            [
                "dpkg-query",
                "-W",
                "-f=${binary:Package}\\t${Version}\\t${Architecture}\\n",
                package,
            ]
        )
        fields = result.stdout.strip().split("\t") if result.returncode == 0 else []
        cache[package] = (
            {"name": fields[0], "version": fields[1], "architecture": fields[2]}
            if len(fields) == 3
            else {"name": package}
        )
    return cache[package]


def linux_desktop_applications() -> list[dict[str, Any]]:
    applications = []
    package_cache: dict[str, dict[str, str]] = {}
    paths = sorted(
        {
            Path(path)
            for root in LINUX_DESKTOP_ROOTS
            for path in glob.glob(f"{root}/**/*.desktop", recursive=True)
        }
    )
    for path in paths:
        parser = configparser.ConfigParser(
            interpolation=None,
            strict=False,
            delimiters=("=",),
        )
        parser.optionxform = str
        try:
            parser.read(path, encoding="utf-8")
            entry = parser["Desktop Entry"]
        except (OSError, UnicodeError, KeyError, configparser.Error):
            continue
        if entry.get("Type", "Application") != "Application":
            continue
        raw_exec = entry.get("Exec", "")
        executable = executable_from_exec(raw_exec)
        package = package_for_path(path)
        item: dict[str, Any] = {
            "name": entry.get("Name", path.stem),
            "app_id": path.name,
            "desktop_id": path.name,
            "source": str(path),
            "visible": entry.get("Hidden", "false").lower() != "true"
            and entry.get("NoDisplay", "false").lower() != "true",
            "executable": shutil.which(executable) if executable else None,
            "categories": [value for value in entry.get("Categories", "").split(";") if value],
        }
        if package:
            metadata = package_metadata(package, package_cache)
            item.update(
                {
                    "package_manager": "dpkg",
                    "package": metadata["name"],
                    "version": metadata.get("version"),
                    "architecture": metadata.get("architecture"),
                }
            )
        applications.append(item)
    return applications


def linux_command_versions() -> list[dict[str, Any]]:
    products = []
    for name, command in LINUX_VERSION_COMMANDS.items():
        executable = shutil.which(command[0])
        if not executable:
            continue
        result = run([executable, *command[1:]])
        products.append(
            {
                "name": name,
                "executable": executable,
                "version_output": first_line(result.stdout) or first_line(result.stderr),
                "returncode": result.returncode,
            }
        )
    return products


def snap_packages() -> list[dict[str, str]]:
    if not shutil.which("snap"):
        return []
    result = run(["snap", "list"])
    rows = []
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5:
            rows.append(
                {
                    "name": fields[0],
                    "version": fields[1],
                    "revision": fields[2],
                    "tracking": fields[3],
                    "publisher": fields[4],
                }
            )
    return rows


def flatpak_applications() -> list[dict[str, str]]:
    if not shutil.which("flatpak"):
        return []
    result = run(
        ["flatpak", "list", "--app", "--columns=application,name,version,branch,origin"]
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 5:
            rows.append(dict(zip(("id", "name", "version", "branch", "origin"), fields)))
    return rows


def linux_inventory() -> dict[str, Any]:
    applications = linux_desktop_applications()
    products = linux_command_versions()
    detected_by_executable = {
        item["executable"]: item for item in products if item.get("executable")
    }
    for application in applications:
        detected = detected_by_executable.get(application.get("executable"))
        if detected and detected.get("version_output"):
            application["version_output"] = detected["version_output"]
    flatpaks = flatpak_applications()
    known_ids = {item["app_id"] for item in applications}
    for flatpak in flatpaks:
        app_id = f"{flatpak['id']}.desktop"
        if app_id not in known_ids:
            applications.append(
                {
                    "name": flatpak["name"],
                    "app_id": app_id,
                    "desktop_id": app_id,
                    "visible": True,
                    "executable": f"flatpak run {flatpak['id']}",
                    "categories": [],
                    "package_manager": "flatpak",
                    "package": flatpak["id"],
                    "version": flatpak["version"],
                }
            )
    dpkg_count = run(["dpkg-query", "-W", "-f=${binary:Package}\\n"])
    return {
        "applications": applications,
        "desktop_applications": applications,
        "detected_products": products,
        "snap_packages": snap_packages(),
        "flatpak_applications": flatpaks,
        "installed_dpkg_package_count": len(dpkg_count.stdout.splitlines()),
    }


def powershell_json(script: str) -> list[dict[str, Any]]:
    executable = shutil.which("powershell") or shutil.which("powershell.exe")
    if not executable:
        return []
    result = run([executable, "-NoProfile", "-NonInteractive", "-Command", script], timeout=90)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout.lstrip("\ufeff"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    return payload if isinstance(payload, list) else []


def windows_registry_applications() -> list[dict[str, Any]]:
    try:
        import winreg
    except ImportError:
        return []
    roots = (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER)
    key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    views = (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY)
    applications = []
    seen = set()
    for root in roots:
        for view in views:
            try:
                parent = winreg.OpenKey(root, key_path, 0, winreg.KEY_READ | view)
            except OSError:
                continue
            with parent:
                for index in range(winreg.QueryInfoKey(parent)[0]):
                    try:
                        subkey_name = winreg.EnumKey(parent, index)
                        subkey = winreg.OpenKey(parent, subkey_name)
                    except OSError:
                        continue
                    with subkey:
                        values = {}
                        for name in (
                            "DisplayName",
                            "DisplayVersion",
                            "DisplayIcon",
                            "InstallLocation",
                            "Publisher",
                            "SystemComponent",
                        ):
                            try:
                                values[name] = winreg.QueryValueEx(subkey, name)[0]
                            except OSError:
                                pass
                    display_name = str(values.get("DisplayName") or "").strip()
                    if not display_name or values.get("SystemComponent") == 1:
                        continue
                    identity = (display_name.casefold(), str(values.get("DisplayVersion") or ""))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    display_icon = os.path.expandvars(str(values.get("DisplayIcon") or ""))
                    executable = display_icon.strip('"').split(",", 1)[0] or None
                    applications.append(
                        {
                            "name": display_name,
                            "app_id": subkey_name,
                            "visible": True,
                            "executable": executable,
                            "version": values.get("DisplayVersion"),
                            "publisher": values.get("Publisher"),
                            "package_manager": "windows_registry",
                            "package": subkey_name,
                            "source": str(values.get("InstallLocation") or key_path),
                            "categories": [],
                        }
                    )
    return applications


def windows_inventory() -> dict[str, Any]:
    applications = windows_registry_applications()
    appx_rows = powershell_json(
        "Get-AppxPackage | Select-Object Name,PackageFullName,PackageFamilyName,Version,InstallLocation | ConvertTo-Json -Compress"
    )
    appx_by_family = {
        str(row.get("PackageFamilyName") or "").casefold(): row
        for row in appx_rows
        if row.get("PackageFamilyName")
    }
    start_apps = powershell_json(
        "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress"
    )
    known_names = {str(item["name"]).casefold() for item in applications}
    for row in start_apps:
        name = str(row.get("Name") or "").strip()
        app_id = str(row.get("AppID") or name).strip()
        if not name or name.casefold() in known_names:
            continue
        known_names.add(name.casefold())
        package_family = app_id.split("!", 1)[0].casefold()
        appx = appx_by_family.get(package_family, {})
        applications.append(
            {
                "name": name,
                "app_id": app_id,
                "visible": True,
                "version": appx.get("Version"),
                "package_manager": "appx" if appx else "start_menu",
                "package": appx.get("PackageFullName") or app_id,
                "source": appx.get("InstallLocation"),
                "categories": [],
            }
        )
    return {
        "applications": applications,
        "windows_registry_application_count": sum(
            item.get("package_manager") == "windows_registry" for item in applications
        ),
        "windows_launchable_appx_count": sum(
            item.get("package_manager") == "appx" for item in applications
        ),
        "windows_installed_appx_package_count": len(appx_rows),
        "windows_start_application_count": len(start_apps),
    }


def macos_applications() -> list[dict[str, Any]]:
    applications = []
    roots = [Path("/Applications"), Path("/System/Applications"), Path.home() / "Applications"]
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for directory, names, _files in os.walk(root):
            directory_path = Path(directory)
            if directory_path.suffix == ".app":
                names[:] = []
                continue
            app_names = [name for name in names if name.endswith(".app")]
            names[:] = [name for name in names if not name.endswith(".app")]
            for name in app_names:
                bundle = directory_path / name
                try:
                    with (bundle / "Contents" / "Info.plist").open("rb") as handle:
                        info = plistlib.load(handle)
                except (OSError, plistlib.InvalidFileException):
                    info = {}
                app_id = str(info.get("CFBundleIdentifier") or bundle)
                if app_id.casefold() in seen:
                    continue
                seen.add(app_id.casefold())
                executable_name = info.get("CFBundleExecutable")
                executable = bundle / "Contents" / "MacOS" / str(executable_name) if executable_name else None
                category = info.get("LSApplicationCategoryType")
                applications.append(
                    {
                        "name": str(
                            info.get("CFBundleDisplayName")
                            or info.get("CFBundleName")
                            or bundle.stem
                        ),
                        "app_id": app_id,
                        "visible": not bool(
                            info.get("LSBackgroundOnly") or info.get("LSUIElement")
                        ),
                        "executable": str(executable) if executable else None,
                        "version": info.get("CFBundleShortVersionString")
                        or info.get("CFBundleVersion"),
                        "package_manager": "app_bundle",
                        "package": app_id,
                        "source": str(bundle),
                        "categories": [str(category)] if category else [],
                    }
                )
    return applications


def macos_inventory() -> dict[str, Any]:
    applications = macos_applications()
    return {"applications": applications, "application_bundle_count": len(applications)}


def linux_os_release() -> dict[str, str]:
    values = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.lower()] = value.strip().strip('"')
    except OSError:
        pass
    return values


def main() -> int:
    system = platform.system()
    if system == "Windows":
        inventory = windows_inventory()
        os_release = {"name": "Windows", "version_id": platform.release()}
    elif system == "Darwin":
        inventory = macos_inventory()
        version = platform.mac_ver()[0] or platform.release()
        os_release = {"name": "Darwin", "version_id": version}
    else:
        inventory = linux_inventory()
        os_release = linux_os_release()
    result = {
        "schema_version": 1,
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "collection_method": "booted_guest_runtime",
        "guest": {
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "kernel": platform.release(),
            "platform": system,
            "os_release": os_release,
            "desktop_session": os.environ.get("XDG_CURRENT_DESKTOP"),
        },
        "inventory": inventory,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
