from __future__ import annotations

from scripts import build_osworld_training_images as MODULE


def test_training_image_id_preserves_base_identity():
    assert MODULE.training_image_id("osworld-v2-ubuntu", "training-deps-v1") == (
        "osworld-v2-ubuntu-training-deps-v1"
    )


def test_virt_customize_installs_and_verifies_document_dependencies(tmp_path):
    command = MODULE._virt_customize_command(tmp_path / "disk.qcow2")

    assert command[:5] == [
        "virt-customize",
        "--format",
        "qcow2",
        "-a",
        str(tmp_path / "disk.qcow2"),
    ]
    run_commands = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--run-command"
    ]
    assert any(
        "apt-get" in value
        and "python3-openpyxl" in value
        and "poppler-utils" in value
        and "tesseract-ocr" in value
        for value in run_commands
    )
    assert any("pytesseract" in value and "python-pptx" in value for value in run_commands)
    assert any("command -v tesseract" in value for value in run_commands)
