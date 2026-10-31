from __future__ import annotations

from scripts import collect_vm_inventory as MODULE


def test_default_guest_script_ships_with_the_package():
    assert MODULE.DEFAULT_GUEST_SCRIPT.is_file()


def test_descriptor_contains_guest_os_metadata():
    descriptor = MODULE.build_descriptor(
        image_id="ubuntu-test",
        guest_inventory={
            "guest": {
                "architecture": "x86_64",
                "os_release": {"name": "Ubuntu", "version_id": "24.04"},
            },
            "inventory": {
                "desktop_applications": [],
                "detected_products": [],
                "snap_packages": [],
                "flatpak_applications": [],
            },
        },
    )

    assert descriptor["os"] == {
        "type": "Ubuntu",
        "version": "24.04",
        "architecture": "x86_64",
    }
    assert descriptor["inventory_status"] == "complete"


def test_windows_descriptor_normalizes_applications_and_architecture():
    descriptor = MODULE.build_descriptor(
        image_id="windows-test",
        guest_inventory={
            "collected_at": "2026-08-25T00:00:00+00:00",
            "collection_method": "booted_guest_runtime",
            "guest": {
                "architecture": "AMD64",
                "os_release": {"name": "Windows", "version_id": "10"},
            },
            "inventory": {
                "applications": [
                    {
                        "name": "Visual Studio Code",
                        "app_id": "Microsoft Visual Studio Code",
                        "visible": True,
                        "executable": r"C:\Program Files\Microsoft VS Code\Code.exe",
                        "version": "1.104.0",
                        "publisher": "Microsoft Corporation",
                        "package_manager": "windows_registry",
                        "package": "Microsoft Visual Studio Code",
                        "categories": [],
                    },
                    {
                        "name": "Background Helper",
                        "app_id": "helper",
                        "visible": False,
                    },
                ]
            },
        },
    )

    assert descriptor["os"] == {
        "type": "Windows",
        "version": "10",
        "architecture": "x86_64",
    }
    assert descriptor["inventory_collected_at"] == "2026-08-25T00:00:00+00:00"
    assert descriptor["inventory_collection_method"] == "booted_guest_runtime"
    assert descriptor["apps"] == [
        {
            "name": "Visual Studio Code",
            "desktop_id": "Microsoft Visual Studio Code",
            "executable": r"C:\Program Files\Microsoft VS Code\Code.exe",
            "categories": [],
            "package_manager": "windows_registry",
            "package": "Microsoft Visual Studio Code",
            "version": "1.104.0",
            "publisher": "Microsoft Corporation",
        }
    ]


def test_macos_descriptor_preserves_bundle_metadata():
    descriptor = MODULE.build_descriptor(
        image_id="macos-test",
        guest_inventory={
            "guest": {
                "architecture": "x86_64",
                "os_release": {"name": "Darwin", "version_id": "15.6"},
            },
            "inventory": {
                "applications": [
                    {
                        "name": "Safari",
                        "app_id": "com.apple.Safari",
                        "visible": True,
                        "executable": "/Applications/Safari.app/Contents/MacOS/Safari",
                        "version": "18.6",
                        "package_manager": "app_bundle",
                        "package": "com.apple.Safari",
                        "source": "/Applications/Safari.app",
                        "categories": ["public.app-category.productivity"],
                    }
                ]
            },
        },
    )

    assert descriptor["os"] == {
        "type": "Darwin",
        "version": "15.6",
        "architecture": "x86_64",
    }
    assert descriptor["apps"][0]["desktop_id"] == "com.apple.Safari"
    assert descriptor["apps"][0]["inventory_source"] == "/Applications/Safari.app"


def test_platform_name_maps_supported_operating_systems():
    assert MODULE.platform_name("Ubuntu") == "linux"
    assert MODULE.platform_name("Windows") == "windows"
    assert MODULE.platform_name("Darwin") == "macos"
