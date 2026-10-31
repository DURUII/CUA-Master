from __future__ import annotations

import pytest

from cuamaster.collection.macos_kvm_launcher import (
    _validate_resource_id,
    build_qemu_args,
)


def test_macos_kvm_qemu_args_include_opencore_and_controller_forwards():
    args = build_qemu_args(
        ram_size="8G",
        cpu_cores=4,
        cpu_model="Skylake-Client,kvm=on",
        osk="test-osk",
        server_port=15000,
        chromium_port=19222,
        vnc_port=5907,
        vlc_port=18080,
    )
    joined = " ".join(args)

    assert "isa-applesmc,osk=test-osk" in joined
    assert "/opencore/OpenCore/OpenCore.qcow2" in joined
    assert "/runtime/disk.qcow2" in joined
    assert "hostfwd=tcp:127.0.0.1:15000-:5000" in joined
    assert "hostfwd=tcp:127.0.0.1:19222-:9222" in joined
    assert "hostfwd=tcp:127.0.0.1:18080-:8080" in joined
    assert "127.0.0.1:7" in args


@pytest.mark.parametrize("resource_id", ["../escape", "bad value", "bad/value"])
def test_macos_kvm_rejects_unsafe_resource_id(resource_id):
    with pytest.raises(ValueError, match="invalid resource id"):
        _validate_resource_id(resource_id)
