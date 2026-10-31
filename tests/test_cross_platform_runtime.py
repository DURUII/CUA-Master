from __future__ import annotations

import importlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import requests

from cuamaster.collection.host_claude_agent import (
    _host_mcp_config,
    validate_controller_platform,
)
from cuamaster.collection.providers.ssh import (
    RemoteSSHConfig,
    RemoteSSHProvider,
    SSHClient,
    _expected_platform,
)
from cuamaster.collection.weavebench_runner import (
    VMTransportUnavailable,
    _OSWorldSetupControllerAdapter,
    _capture_legacy_in_place_image,
    _cleanup_guest_runtime,
    _dedent_claudecode_retry_parser,
    _harden_openclaw_bootstrap,
    _harden_claudecode_tool_policy,
    _enable_legacy_osworld_proxy,
    _install_agent_adapter,
    _install_hybrid_mode,
    _install_osworld_task_adapter,
    _install_screen_size_fallback,
    _install_task_endpoint_failover,
    _install_task_platform_filter,
    _install_vm_transport_guard,
    _install_vm_provider,
    _install_worker_exit_monitor,
    _materialize_changed_legacy_image,
    _install_proxy_log_redaction,
    _prepare_guest_artifact_dirs,
    _prepare_osworld_setup_command,
    _record_has_endpoint_failure,
    _runner_module,
    _score_allows_resume_skip,
    _stage_legacy_osworld_expected_files,
    _shared_remote_ssh_control,
    _screenshot_size,
    _worker_aware_monitor,
    _write_task_metadata,
)
from cuamaster.collection.host_claude_agent import _claude_tool_permissions
from cuamaster.propagation.compilers.weavebench import compile_weavebench_taskset


def _remote_config(**overrides):
    values = {
        "host": "example.invalid",
        "port": 13022,
        "user": "root",
        "password": "",
        "identity_file": None,
        "remote_image_path": "/srv/vms/windows.qcow2",
        "docker_image": "happysixd/osworld-docker:latest",
        "docker_image_tar": None,
        "stage_image": False,
        "allow_pull": False,
        "start_timeout": 30.0,
        "connect_timeout": 5,
        "strict_host_key_checking": "accept-new",
        "known_hosts_file": None,
        "remote_start_command": None,
        "remote_stop_command": None,
        "ram_size": "8G",
        "cpu_cores": 4,
        "disk_size": "64G",
    }
    values.update(overrides)
    return RemoteSSHConfig(**values)


def test_screenshot_size_reads_png_header():
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (1280).to_bytes(4, "big") + (
        800
    ).to_bytes(4, "big")

    assert _screenshot_size(payload) == {"width": 1280, "height": 800}


def test_install_screen_size_fallback_uses_screenshot(monkeypatch):
    controllers = ModuleType("weavebench.desktop_env.controllers.python")

    class FakeController:
        def get_vm_screen_size(self):
            return None

        def get_screenshot(self):
            return (
                b"\x89PNG\r\n\x1a\n"
                + b"\x00" * 8
                + (1920).to_bytes(4, "big")
                + (1080).to_bytes(4, "big")
            )

    controllers.PythonController = FakeController
    monkeypatch.setitem(
        sys.modules, "weavebench.desktop_env.controllers.python", controllers
    )
    monkeypatch.setenv("CUAMASTER_SANDBOX_PROVIDER", "sandbox")

    _install_screen_size_fallback()

    assert FakeController().get_vm_screen_size() == {"width": 1920, "height": 1080}


def test_prepare_guest_artifact_dirs_uses_sudo_and_chown():
    calls = []
    upstream = SimpleNamespace(
        _vm_exec=lambda env, command: calls.append((env, command)) or {
            "returncode": 0
        }
    )
    env = object()

    _prepare_guest_artifact_dirs(upstream, env, "p'ass word")

    assert calls[0][0] is env
    shell_command = calls[0][1][2]
    assert "sudo -S -p '' sh -c" in shell_command
    assert "mkdir -p /tmp_workspace/_trajectory" in shell_command
    assert "chown -R user:user" in shell_command


def test_host_mcp_config_is_episode_local(tmp_path):
    config = _host_mcp_config(
        "http://controller", tmp_path, "Windows", enable_guest_shell=True
    )
    server = config["mcpServers"]["weavebench_computer"]
    assert server["args"] == ["-m", "cuamaster.collection.host_computer_mcp"]
    assert server["env"]["GUI_TRAJECTORY_CONTROLLER_PLATFORM"] == "Windows"
    assert server["env"]["GUI_TRAJECTORY_MCP_EVENT_LOG"] == str(
        tmp_path / "computer_events.jsonl"
    )
    assert server["env"]["GUI_TRAJECTORY_SHELL_EVENT_LOG"] == str(
        tmp_path / "shell_events.jsonl"
    )
    assert server["env"]["GUI_TRAJECTORY_ENABLE_GUEST_SHELL"] == "true"


def test_host_runtime_rejects_guest_platform_mismatch(monkeypatch):
    env = SimpleNamespace(vm_base_url="http://controller")
    monkeypatch.setenv(
        "WEAVEBENCH_VM_PROFILE_JSON",
        json.dumps({"id": "windows", "os": {"type": "Windows"}}),
    )
    monkeypatch.setattr(
        "cuamaster.collection.host_claude_agent._controller_platform",
        lambda url: "Linux",
    )
    with pytest.raises(RuntimeError, match="profile declares 'Windows'.*guest reports 'Linux'"):
        validate_controller_platform(env)


@pytest.mark.parametrize(
    ("os_type", "platform"),
    [("Ubuntu", "Linux"), ("Windows", "Windows"), ("macOS", "Darwin")],
)
def test_remote_ssh_expected_platform(os_type, platform):
    assert _expected_platform(os_type) == platform


def test_remote_ssh_password_auth_disables_other_auth_methods(tmp_path):
    client = SSHClient(_remote_config(password="secret", known_hosts_file=str(
        tmp_path / "known_hosts"
    )))
    try:
        options = client._options()
    finally:
        client.close()

    assert "PreferredAuthentications=password" in options
    assert "PubkeyAuthentication=no" in options
    assert "GSSAPIAuthentication=no" in options
    assert any(option.startswith("ControlPath=") for option in options)


def test_shared_remote_ssh_control_preflights_and_restores_env(monkeypatch, tmp_path):
    calls = []
    config = SimpleNamespace(
        user="root",
        host="remote.example",
        port=13022,
        connect_timeout=5,
    )

    class FakeClient:
        def __init__(self, received):
            assert received is config
            self.control_path = tmp_path / "control"

        def run(self, script, timeout=None):
            calls.append((script, timeout))

        def close(self):
            calls.append(("close", None))

    monkeypatch.setenv("GUI_TRAJECTORY_REMOTE_SSH_PROVIDER", "1")
    monkeypatch.setenv("GUI_TRAJECTORY_SSH_CONTROL_PATH", "/tmp/original-control")
    monkeypatch.setattr(
        "cuamaster.collection.providers.ssh.RemoteSSHConfig.from_env",
        lambda: config,
    )
    monkeypatch.setattr(
        "cuamaster.collection.providers.ssh.SSHClient",
        FakeClient,
    )

    with _shared_remote_ssh_control():
        assert os.environ["GUI_TRAJECTORY_SSH_CONTROL_PATH"] == str(
            tmp_path / "control"
        )

    assert calls == [("true", 10), ("close", None)]
    assert os.environ["GUI_TRAJECTORY_SSH_CONTROL_PATH"] == "/tmp/original-control"


def test_dedent_claudecode_retry_parser_produces_valid_python():
    marker = 'RETRY_REASON=$(echo "$LAST_RESULT" | python3 -c \'\n'
    terminator = "' 2>/dev/null || echo PARSE_ERR)"
    script = (
        marker
        + "\timport json\n"
        + "\ttry:\n"
        + "\t    print(json.loads('null'))\n"
        + "\texcept Exception:\n"
        + "\t    print('PARSE_ERR')\n"
        + terminator
    )

    patched = _dedent_claudecode_retry_parser(script)
    body = patched.split(marker, 1)[1].split(terminator, 1)[0]

    compile(body, "<retry-parser>", "exec")
    assert body.startswith("import json\n")


def test_host_claude_uses_explicit_mcp_allowlist_without_unsafe_bypass():
    args = _claude_tool_permissions("hybrid")
    assert "--dangerously-skip-permissions" not in args
    assert args[args.index("--allowedTools") + 1] == (
        "mcp__weavebench_computer__computer,mcp__weavebench_computer__shell"
    )
    assert "Bash" in args[args.index("--disallowedTools") + 1]
    assert "TodoWrite" in args[args.index("--disallowedTools") + 1]
    assert "TaskOutput" in args[args.index("--disallowedTools") + 1]


def test_guest_claude_disables_internal_tools_and_gui_shortcuts(monkeypatch):
    script = '''DISALLOWED_TOOLS="WebFetch,WebSearch"
claude --verbose --dangerously-skip-permissions --setting-sources user \\
      --disallowedTools "$DISALLOWED_TOOLS" \\
      < prompt.txt
'''
    monkeypatch.setenv("GUI_TRAJECTORY_REQUESTED_MODE", "gui")

    gui = _harden_claudecode_tool_policy(script)

    assert "TodoWrite" in gui
    assert "TaskOutput" in gui
    assert 'ALLOWED_TOOLS="mcp__weavebench_computer__computer"' in gui
    assert "Bash" in gui.split('DISALLOWED_TOOLS="', 1)[1]
    assert "Read" in gui.split('DISALLOWED_TOOLS="', 1)[1]
    assert "--dangerously-skip-permissions" not in gui
    assert '--allowedTools "$ALLOWED_TOOLS"' in gui
    monkeypatch.setenv("GUI_TRAJECTORY_REQUESTED_MODE", "hybrid")
    hybrid = _harden_claudecode_tool_policy(script)
    assert "TodoWrite" in hybrid
    assert (
        'ALLOWED_TOOLS="mcp__weavebench_computer__computer,Bash"'
        in hybrid
    )
    disallowed = hybrid.split('DISALLOWED_TOOLS="', 1)[1].split('"', 1)[0]
    assert "Bash" not in disallowed.split(",")
    assert {"Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit"} <= set(
        disallowed.split(",")
    )


def test_guest_claude_tool_policy_fails_closed_on_unknown_script(monkeypatch):
    monkeypatch.setenv("GUI_TRAJECTORY_REQUESTED_MODE", "hybrid")

    with pytest.raises(RuntimeError, match="permission mode"):
        _harden_claudecode_tool_policy(
            'DISALLOWED_TOOLS="WebFetch,WebSearch"\n'
            'claude --disallowedTools "$DISALLOWED_TOOLS" \\\n'
        )


def test_remote_ssh_default_launcher_builds_osworld_docker_command():
    scripts = []

    class FakeSSH:
        def run(self, script, **kwargs):
            scripts.append((script, kwargs))
            return subprocess.CompletedProcess(
                [],
                0,
                stdout=(
                    '{"resource_id":"remote-vm","server_port":15000,'
                    '"chromium_port":19222,"vnc_port":18006,"vlc_port":18080}\n'
                ),
                stderr="",
            )

    provider = object.__new__(RemoteSSHProvider)
    provider.config = _remote_config()
    provider.ssh = FakeSSH()
    provider.container_name = None

    payload = provider._default_start("Windows")

    assert payload["resource_id"] == "remote-vm"
    script = scripts[0][0]
    assert "docker info >/dev/null" in script
    assert "test -c /dev/kvm" in script
    assert "test -c /dev/net/tun" in script
    assert "--cap-add NET_ADMIN" in script
    assert "--device /dev/kvm" in script
    assert "--device /dev/net/tun" in script
    assert "/srv/vms/windows.qcow2:/System.qcow2:ro" in script
    assert "docker inspect -f" in script
    assert "server_host" in script
    assert "-p 127.0.0.1::5000" not in script


def test_remote_ssh_default_launcher_rejects_macos():
    provider = object.__new__(RemoteSSHProvider)
    provider.config = _remote_config()
    with pytest.raises(ValueError, match="compatible remote_start_command"):
        provider._default_start("Darwin")


def test_remote_ssh_custom_launcher_receives_platform_fields():
    scripts = []

    class FakeSSH:
        def run(self, script, **kwargs):
            scripts.append(script)
            return subprocess.CompletedProcess(
                [],
                0,
                stdout=(
                    '{"server_port":25000,"chromium_port":29222,'
                    '"vnc_port":28006,"vlc_port":28080}\n'
                ),
                stderr="",
            )

    provider = object.__new__(RemoteSSHProvider)
    provider.config = _remote_config(
        remote_start_command="launch {resource_id} {os_type} {image_path} {cpu_cores}"
    )
    provider.ssh = FakeSSH()

    payload = provider._custom_start("Darwin")

    assert payload["resource_id"].startswith("gui-trajectory-")
    assert provider.remote_resource_id == payload["resource_id"]
    assert " Darwin /srv/vms/windows.qcow2 4" in scripts[0]


def test_remote_ssh_custom_launcher_reserves_resource_before_failure():
    class FakeSSH:
        def run(self, script, **kwargs):
            raise RuntimeError("launcher failed")

    provider = object.__new__(RemoteSSHProvider)
    provider.config = _remote_config(remote_start_command="launch {resource_id}")
    provider.ssh = FakeSSH()
    provider.remote_resource_id = None

    with pytest.raises(RuntimeError, match="launcher failed"):
        provider._custom_start("Darwin")

    assert provider.remote_resource_id.startswith("gui-trajectory-")


def test_remote_ssh_image_probe_distinguishes_connection_failure():
    class FakeSSH:
        def run(self, script, **kwargs):
            return subprocess.CompletedProcess([], 255, stdout="", stderr="network down")

    provider = object.__new__(RemoteSSHProvider)
    provider.config = _remote_config()
    provider.ssh = FakeSSH()

    with pytest.raises(RuntimeError, match="failed to check.*network down"):
        provider._ensure_remote_image("/local/windows.qcow2")


def test_remote_ssh_cleanup_retries_transient_ssh_failure(monkeypatch):
    results = [
        subprocess.CompletedProcess([], 255, stdout="", stderr="connection reset"),
        subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    ]

    class FakeSSH:
        def run(self, script, **kwargs):
            return results.pop(0)

    provider = object.__new__(RemoteSSHProvider)
    provider.config = _remote_config()
    provider.ssh = FakeSSH()
    provider.tunnel = None
    provider.remote_resource_id = "remote-vm"
    provider.container_name = "remote-vm"
    provider.local_ports = {"server": 15000}
    provider.remote_ports = {"server": 25000}
    provider.vm_base_url = "http://127.0.0.1:15000"
    monkeypatch.setattr("cuamaster.collection.providers.ssh.time.sleep", lambda _: None)

    provider.stop_emulator("/srv/vms/windows.qcow2")

    assert results == []
    assert provider.remote_resource_id is None
    assert provider.container_name is None


def test_runner_replaces_manager_factory_with_remote_ssh(monkeypatch):
    weavebench = ModuleType("weavebench")
    desktop_package = ModuleType("weavebench.desktop_env")
    desktop_package.__path__ = []
    providers_module = ModuleType("weavebench.desktop_env.providers")
    desktop_module = ModuleType("weavebench.desktop_env.desktop_env")
    original_result = (object(), object())
    remote_result = (object(), object())

    def original_factory(provider_name, region, use_proxy=False):
        return original_result

    providers_module.create_vm_manager_and_provider = original_factory
    desktop_module.create_vm_manager_and_provider = original_factory
    desktop_package.providers = providers_module
    desktop_package.desktop_env = desktop_module
    weavebench.desktop_env = desktop_package
    monkeypatch.setitem(sys.modules, "weavebench", weavebench)
    monkeypatch.setitem(sys.modules, "weavebench.desktop_env", desktop_package)
    monkeypatch.setitem(
        sys.modules, "weavebench.desktop_env.providers", providers_module
    )
    monkeypatch.setitem(
        sys.modules, "weavebench.desktop_env.desktop_env", desktop_module
    )
    monkeypatch.setattr(
        "cuamaster.collection.providers.ssh.create_remote_ssh_provider",
        lambda region: remote_result,
    )
    monkeypatch.setenv("GUI_TRAJECTORY_REMOTE_SSH_PROVIDER", "1")

    _install_vm_provider()

    assert providers_module.create_vm_manager_and_provider("docker", "cn") == remote_result
    assert providers_module.create_vm_manager_and_provider("manager", "cn") == original_result
    assert desktop_module.create_vm_manager_and_provider("docker", "cn") == remote_result


def test_platform_filter_keeps_only_compatible_tasks(monkeypatch, tmp_path):
    tasks = [
        {"task_id": "windows-task"},
        {"task_id": "mac-task"},
        {"task_id": "unannotated-task"},
    ]
    manifest = {
        "tasks": [
            {
                "id": "windows-task",
                "required_capabilities": ["computer", "app:notepad"],
                "target_environments": [{"os_type": "Windows", "os_versions": ["11"]}],
            },
            {
                "id": "mac-task",
                "target_environments": [{"os_type": "Darwin"}],
            },
            {"id": "unannotated-task", "target_environments": []},
        ]
    }
    (tmp_path / "taskset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    ds = SimpleNamespace(discover_tasks=lambda root, categories: list(tasks))
    messages_gui = SimpleNamespace(ds=ds)
    monkeypatch.setenv("GUI_TRAJECTORY_REQUIRE_EXPLICIT_PLATFORM", "true")
    monkeypatch.setenv(
        "WEAVEBENCH_VM_PROFILE_JSON",
        json.dumps({
            "id": "windows-11",
            "os": {"type": "Windows", "version": "11", "architecture": "x86_64"},
            "software": {"notepad": "11"},
        }),
    )
    _install_task_platform_filter(messages_gui)
    assert messages_gui.ds.discover_tasks(Path(tmp_path), ["DOC"]) == [tasks[0]]


def test_platform_filter_treats_ubuntu_as_linux(monkeypatch, tmp_path):
    task = {"task_id": "linux-task"}
    (tmp_path / "taskset_manifest.json").write_text(
        json.dumps({
            "tasks": [{
                "id": "linux-task",
                "required_capabilities": ["computer", "filesystem"],
                "target_environments": [{"os_type": "Linux"}],
            }]
        }),
        encoding="utf-8",
    )
    ds = SimpleNamespace(discover_tasks=lambda root, categories: [task])
    runner = SimpleNamespace(ds=ds)
    monkeypatch.setenv("GUI_TRAJECTORY_REQUIRE_EXPLICIT_PLATFORM", "true")
    monkeypatch.setenv(
        "WEAVEBENCH_VM_PROFILE_JSON",
        json.dumps({
            "id": "ubuntu-profile",
            "os": {"type": "Ubuntu", "version": "22.04", "architecture": "x86_64"},
            "software": {},
        }),
    )

    _install_task_platform_filter(runner)

    assert runner.ds.discover_tasks(Path(tmp_path), ["os"]) == [task]


def test_platform_filter_can_be_disabled_for_all_platform_replay(monkeypatch, tmp_path):
    tasks = [{"task_id": "windows-task"}, {"task_id": "mac-task"}]

    def discover(root, categories):
        return list(tasks)

    ds = SimpleNamespace(discover_tasks=discover)
    messages_gui = SimpleNamespace(ds=ds)
    monkeypatch.setenv("GUI_TRAJECTORY_FORCE_ALL_TASKS_ON_PLATFORM", "true")
    monkeypatch.setenv(
        "WEAVEBENCH_VM_PROFILE_JSON",
        json.dumps({
            "id": "linux-replay",
            "os": {"type": "Ubuntu", "version": "22.04"},
        }),
    )

    _install_task_platform_filter(messages_gui)

    assert messages_gui.ds.discover_tasks is discover
    assert messages_gui.ds.discover_tasks(Path(tmp_path), ["DOC"]) == tasks


def test_platform_filter_routes_generated_tasks_to_exact_image(monkeypatch, tmp_path):
    tasks = [{"task_id": "controller-task"}, {"task_id": "v2-task"}]
    manifest = {
        "tasks": [
            {
                "id": "controller-task",
                "environment_id": "controller-image",
                "required_capabilities": ["unlisted-generated-capability"],
                "target_environments": [{"os_type": "Ubuntu", "os_versions": ["22.04"]}],
            },
            {
                "id": "v2-task",
                "environment_id": "v2-image",
                "required_capabilities": ["unlisted-generated-capability"],
                "target_environments": [{"os_type": "Ubuntu", "os_versions": ["22.04"]}],
            },
        ]
    }
    (tmp_path / "taskset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    ds = SimpleNamespace(discover_tasks=lambda root, categories: list(tasks))
    runner = SimpleNamespace(ds=ds)
    monkeypatch.setenv("GUI_TRAJECTORY_REQUIRE_EXPLICIT_PLATFORM", "true")
    monkeypatch.setenv(
        "WEAVEBENCH_VM_PROFILE_JSON",
        json.dumps({
            "id": "controller-profile",
            "os": {"type": "Ubuntu", "version": "22.04", "architecture": "x86_64"},
            "image": {"id": "controller-image"},
            "software": {},
        }),
    )
    _install_task_platform_filter(runner)
    assert runner.ds.discover_tasks(Path(tmp_path), ["OSWORLD_V2"]) == [tasks[0]]


def test_platform_filter_routes_generated_tasks_to_derived_image(monkeypatch, tmp_path):
    tasks = [{"task_id": "base-task"}, {"task_id": "other-task"}]
    manifest = {
        "tasks": [
            {
                "id": "base-task",
                "environment_id": "base-image",
                "target_environments": [{"os_type": "Ubuntu", "os_versions": ["22.04"]}],
            },
            {
                "id": "other-task",
                "environment_id": "other-image",
                "target_environments": [{"os_type": "Ubuntu", "os_versions": ["22.04"]}],
            },
        ]
    }
    (tmp_path / "taskset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    ds = SimpleNamespace(discover_tasks=lambda root, categories: list(tasks))
    runner = SimpleNamespace(ds=ds)
    monkeypatch.setenv("GUI_TRAJECTORY_REQUIRE_EXPLICIT_PLATFORM", "true")
    monkeypatch.setenv(
        "WEAVEBENCH_VM_PROFILE_JSON",
        json.dumps({
            "id": "derived-profile",
            "os": {"type": "Ubuntu", "version": "22.04", "architecture": "x86_64"},
            "image": {
                "id": "base-image-training-deps-v1",
                "metadata": {
                    "derived_from": "base-image",
                    "build": {"base_image_id": "base-image"},
                },
            },
            "software": {},
        }),
    )

    _install_task_platform_filter(runner)

    assert runner.ds.discover_tasks(Path(tmp_path), ["OSWORLD_V2"]) == [tasks[0]]


def test_osworld_task_adapter_runs_setup_and_authoritative_evaluator(
    monkeypatch, tmp_path
):
    task_id = "generated-task"
    (tmp_path / "taskset_manifest.json").write_text(json.dumps({
        "tasks": [{
            "id": task_id,
            "version": "task-version",
            "evaluator": {
                "type": "osworld_v2_task_class",
                "path": "/unused/task.py",
                "class_name": "Task001",
            },
        }],
    }), encoding="utf-8")
    calls = []

    class FakeTask:
        proxy = False

        def setup(self, setup_controller, use_proxy=False):
            calls.append(("setup", setup_controller, use_proxy))

        def evaluate(self, env):
            calls.append(("evaluate", env))
            return {"score": 1.0, "detail": "passed"}

    class FakeEnv:
        setup_controller = object()
        is_environment_used = False

        def reset(self, task_config):
            calls.append(("reset", task_config))

    def inner_run_one(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        return {"task_id": task_id, "score": 0.4, "scores": {"judge": 0.4}}

    runner = SimpleNamespace(ds=SimpleNamespace(run_one=inner_run_one))
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._load_osworld_task_class",
        lambda evaluator: FakeTask,
    )
    _install_osworld_task_adapter(runner)
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    record = runner.ds.run_one(
        FakeEnv(), object(), {"task_id": task_id}, "gui", output_dir, tmp_path
    )

    assert calls[0] == ("reset", None)
    assert calls[1][0] == "setup"
    assert calls[2][0] == "evaluate"
    assert record["score"] == 1.0
    assert record["success"] is True
    assert record["scores"]["agent_judge_score"] == 0.4
    assert json.loads((output_dir / "score.json").read_text())["evaluator"] == (
        "osworld_v2_task_class"
    )


def test_osworld_task_adapter_runs_legacy_runtime_config_and_evaluator(
    monkeypatch, tmp_path
):
    task_id = "legacy-task"
    runtime_config = [{"type": "open", "parameters": {"path": "/tmp/a.txt"}}]
    evaluator = {
        "func": "exact_match",
        "result": {"type": "vm_command_line", "command": "true"},
        "expected": {"type": "rule", "rules": {"expected": ""}},
    }
    (tmp_path / "taskset_manifest.json").write_text(json.dumps({
        "tasks": [{
            "id": task_id,
            "version": "task-version",
            "execution_adapter": "legacy_osworld",
            "proxy": False,
            "runtime_config": runtime_config,
            "evaluator": evaluator,
        }],
    }), encoding="utf-8")
    calls = []

    class FakeEnv:
        setup_controller = object()
        is_environment_used = False
        enable_proxy = False

        def reset(self, task_config):
            calls.append(("reset", task_config, self.setup_controller))

        def evaluate(self):
            calls.append(("evaluate",))
            return 0.75

    def inner_run_one(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        calls.append(("run",))
        return {"task_id": task_id, "score": 0.4, "scores": {"judge": 0.4}}

    runner = SimpleNamespace(ds=SimpleNamespace(
        run_one=inner_run_one,
        _original_run_one=inner_run_one,
    ))
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._cleanup_guest_runtime",
        lambda controller: None,
    )
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._install_legacy_osworld_evaluators",
        lambda env: None,
    )
    _install_osworld_task_adapter(runner)
    output_dir = tmp_path / "result"

    record = runner.ds.run_one(
        FakeEnv(),
        object(),
        {"task_id": task_id, "category": "os", "prompt": "Do it"},
        "gui",
        output_dir,
        tmp_path,
    )

    reset = calls[0]
    assert reset[0] == "reset"
    assert reset[1]["config"] == runtime_config
    assert reset[1]["evaluator"] == evaluator
    assert isinstance(reset[2], _OSWorldSetupControllerAdapter)
    assert calls[1] == ("run",)
    assert calls[2] == ("evaluate",)
    assert record["score"] == 0.75
    assert record["scores"]["agent_judge_score"] == 0.4
    assert record["evaluator"] == "legacy_osworld"


def test_osworld_task_adapter_skips_evaluator_after_agent_failure(
    monkeypatch, tmp_path
):
    task_id = "failed-legacy-task"
    (tmp_path / "taskset_manifest.json").write_text(json.dumps({
        "tasks": [{
            "id": task_id,
            "version": "task-version",
            "execution_adapter": "legacy_osworld",
            "proxy": False,
            "runtime_config": [],
            "evaluator": {
                "func": "check_structure_sim",
                "result": {"type": "vm_file", "path": "/tmp/missing.png"},
            },
        }],
    }), encoding="utf-8")
    calls = []

    class FakeEnv:
        setup_controller = object()
        is_environment_used = False
        enable_proxy = False

        def reset(self, task_config):
            calls.append(("reset", task_config))

        def evaluate(self):
            calls.append(("evaluate",))
            raise AssertionError("failed agents must not run file evaluators")

    def inner_run_one(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        calls.append(("run",))
        return {
            "task_id": task_id,
            "score": 0.0,
            "error": "agent_crash: selected model is unavailable",
            "agent_done": True,
            "agent_exit": 1,
        }

    runner = SimpleNamespace(ds=SimpleNamespace(
        run_one=inner_run_one,
        _original_run_one=inner_run_one,
    ))
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._cleanup_guest_runtime",
        lambda controller: None,
    )
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._install_legacy_osworld_evaluators",
        lambda env: None,
    )
    _install_osworld_task_adapter(runner)

    record = runner.ds.run_one(
        FakeEnv(),
        object(),
        {"task_id": task_id, "category": "gimp", "prompt": "Do it"},
        "gui",
        tmp_path / "result",
        tmp_path,
    )

    assert calls[-1] == ("run",)
    assert record["score"] == 0.0
    assert record["success"] is False
    assert record["evaluator_skipped"] == "agent_execution_failed"


def test_osworld_task_adapter_rejects_proxy_task_without_pool(monkeypatch, tmp_path):
    task_id = "proxy-task"
    (tmp_path / "taskset_manifest.json").write_text(json.dumps({
        "tasks": [{
            "id": task_id,
            "execution_adapter": "legacy_osworld",
            "proxy": True,
            "runtime_config": [],
            "evaluator": {"func": "exact_match"},
        }],
    }), encoding="utf-8")

    class FakeEnv:
        setup_controller = object()
        enable_proxy = False

    runner = SimpleNamespace(ds=SimpleNamespace(run_one=lambda *args, **kwargs: {}))
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._cleanup_guest_runtime",
        lambda controller: None,
    )
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._install_legacy_osworld_evaluators",
        lambda env: None,
    )
    _install_osworld_task_adapter(runner)

    with pytest.raises(RuntimeError, match="requires proxy support"):
        runner.ds.run_one(
            FakeEnv(),
            object(),
            {"task_id": task_id, "category": "chrome", "prompt": "Do it"},
            "gui",
            tmp_path / "result",
            tmp_path,
        )


def test_enable_legacy_osworld_proxy_uses_environment_without_exposing_url(
    monkeypatch,
):
    setup_module = SimpleNamespace(get_global_proxy_pool=None)
    real_import_module = importlib.import_module

    def import_module(name):
        if name == "weavebench.desktop_env.controllers.setup":
            return setup_module
        return real_import_module(name)

    forwarding_calls = []
    env = SimpleNamespace(
        enable_proxy=False,
        _setup_docker_proxy_forwarding=lambda: forwarding_calls.append(True),
    )
    monkeypatch.setenv("GUI_TRAJECTORY_LEGACY_OSWORLD_PROXY_ENABLED", "true")
    monkeypatch.setenv(
        "GUI_TRAJECTORY_LEGACY_OSWORLD_PROXY_URL",
        "http://proxy-user:proxy-pass@10.0.0.8:3128",
    )
    monkeypatch.setenv("CUAMASTER_SANDBOX_PROVIDER", "sandbox")
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner.importlib.import_module",
        import_module,
    )

    _enable_legacy_osworld_proxy(env)

    proxy = setup_module.get_global_proxy_pool().get_next_proxy()
    assert env.enable_proxy is True
    assert proxy.host == "10.0.0.8"
    assert proxy.port == 3128
    assert proxy.username == "proxy-user"
    assert proxy.password == "proxy-pass"
    env._setup_docker_proxy_forwarding()
    assert forwarding_calls == []


def test_osworld_setup_adapter_uses_direct_proxy_for_remote_provider(monkeypatch):
    calls = []
    controller = SimpleNamespace(
        _proxy_setup=lambda password, is_docker=False: calls.append(
            (password, is_docker)
        )
    )
    monkeypatch.setenv("CUAMASTER_SANDBOX_PROVIDER", "sandbox")

    _OSWorldSetupControllerAdapter(controller)._proxy_setup(
        "guest-password", is_docker=True
    )

    assert calls == [("guest-password", False)]


def test_proxy_log_filter_redacts_url_and_authority():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        _install_proxy_log_redaction(
            "http://proxy-user:proxy-pass@10.0.0.8:3128"
        )
        logging.getLogger("desktopenv.setup").warning(
            "proxy=%s upstream=%s",
            "http://proxy-user:proxy-pass@10.0.0.8:3128",
            "proxy-user:proxy-pass@10.0.0.8:3128",
        )
    finally:
        root_logger.removeHandler(handler)

    output = stream.getvalue()
    assert "proxy-pass" not in output
    assert output.count("<redacted-proxy>") == 2


def test_legacy_osworld_expected_files_are_staged_from_file_urls(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}', encoding="utf-8")
    cache_dir = tmp_path / "cache"
    task_config = {
        "evaluator": {
            "expected": {
                "type": "cloud_file",
                "path": source.as_uri(),
                "dest": "expected/result.json",
            }
        }
    }

    _stage_legacy_osworld_expected_files(
        task_config, SimpleNamespace(cache_dir=str(cache_dir))
    )

    assert (cache_dir / "expected/result.json").read_text() == '{"ok": true}'


def test_osworld_task_adapter_exposes_current_setup_controller_methods(
    monkeypatch, tmp_path
):
    task_id = "generated-task"
    (tmp_path / "taskset_manifest.json").write_text(json.dumps({
        "tasks": [{
            "id": task_id,
            "evaluator": {
                "type": "osworld_v2_task_class",
                "path": "/unused/task.py",
                "class_name": "Task001",
            },
        }],
    }), encoding="utf-8")
    calls = []

    class FakeSetupController:
        cache_dir = "/tmp/cache"

        def _download_setup(self, files):
            calls.append(("download", files))

        def _execute_setup(self, command, **kwargs):
            calls.append(("execute", command, kwargs))

        def _launch_setup(self, command, **kwargs):
            calls.append(("launch", command, kwargs))

    class FakeTask:
        proxy = False

        def setup(self, setup_controller, use_proxy=False):
            assert setup_controller.cache_dir == "/tmp/cache"
            setup_controller.download([{"url": "asset", "path": "/tmp/asset"}])
            setup_controller.execute(["echo", "ok"], quiet=True, timeout=5)
            setup_controller.launch(["app"], shell=True)

        def evaluate(self, env):
            env.setup_controller.execute(["echo", "eval"])
            return 1.0

    class FakeEnv:
        setup_controller = FakeSetupController()
        is_environment_used = False

        def reset(self, task_config):
            pass

    runner = SimpleNamespace(ds=SimpleNamespace(
        run_one=lambda *args, **kwargs: {"score": 0.0}
    ))
    original_import = __import__(
        "cuamaster.collection.weavebench_runner", fromlist=["importlib"]
    ).importlib.import_module

    def import_without_osworld_setup(name):
        if name == "desktop_env.controllers.setup":
            raise ImportError(name)
        return original_import(name)

    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner.importlib.import_module",
        import_without_osworld_setup,
    )
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._load_osworld_task_class",
        lambda evaluator: FakeTask,
    )
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._cleanup_guest_runtime",
        lambda controller: None,
    )
    _install_osworld_task_adapter(runner)
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    env = FakeEnv()
    runner.ds.run_one(
        env, object(), {"task_id": task_id}, "gui", output_dir, tmp_path
    )

    assert env.cache_dir == "/tmp/cache"
    assert calls == [
        ("download", [{"url": "asset", "path": "/tmp/asset"}]),
        (
            "execute",
            ["echo", "ok"],
            {"stdout": "", "stderr": "", "shell": False, "until": None},
        ),
        ("launch", ["app"], {"shell": True}),
        (
            "execute",
            ["echo", "eval"],
            {"stdout": "", "stderr": "", "shell": False, "until": None},
        ),
    ]


@pytest.mark.parametrize(
    "record",
    [
        {"error": "Setup execute failed with status 500: command timed out"},
        {"error": "500 Server Error for /setup/upload: No space left on device"},
        {"error": "osworld_evaluator_failed: setup step 5 failed"},
    ],
)
def test_endpoint_failover_ignores_non_model_failures(record):
    assert not _record_has_endpoint_failure(record)


def test_endpoint_failover_still_detects_model_transport_failures():
    assert _record_has_endpoint_failure({"error": "connection timeout"})
    assert _record_has_endpoint_failure({"agent_status": "API error: rate limit"})
    assert _record_has_endpoint_failure({
        "error": "openclaw runtime exited non-zero (AGENT_EXIT=1)",
    })


def test_osworld_setup_adapter_uploads_percent_encoded_local_file(tmp_path):
    source = tmp_path / "gt_Young’s_Modulus.docx"
    source.write_bytes(b"docx")
    calls = []

    class FakeSetupController:
        def _upload_file_setup(self, files):
            calls.append(files)

    adapter = _OSWorldSetupControllerAdapter(FakeSetupController())
    adapter.download([{
        "url": str(source).replace("’", "%E2%80%99"),
        "path": "/home/user/Desktop/ground_truth.docx",
    }])

    assert calls == [[{
        "local_path": str(source),
        "path": "/home/user/Desktop/ground_truth.docx",
    }]]


def test_osworld_setup_adapter_supports_check_for_legacy_controller():
    class FakeSetupController:
        def execute(
            self,
            command,
            stdout="",
            stderr="",
            shell=False,
            until=None,
            quiet=False,
            timeout=120,
        ):
            return {"returncode": 7, "error": "failed"}

    adapter = _OSWorldSetupControllerAdapter(FakeSetupController())

    with pytest.raises(RuntimeError, match="exit code 7"):
        adapter.execute(["false"], check=True)


def test_openclaw_bootstrap_hardening_is_idempotent():
    script = """#!/bin/bash
SUDO() { sudo "$@"; }
if [ -f /home/user/.openclaw_bootstrap.done ]; then
  exit 0
fi
NODE_MAJ=$(/opt/node22/bin/node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\\1/' || echo 0)
SUDO tar -xJf "$NODE_TAR" -C /opt/node22 --strip-components=1 || { echo "node extract failed"; exit 12; }
SUDO tar xzf /tmp/openclaw.tar.gz -C /usr/lib/node_modules || { echo "extract failed"; exit 17; }
"""

    hardened = _harden_openclaw_bootstrap(script)

    assert 'OPENCLAW_RUNTIME_TMP="/tmp/openclaw-$(id -u)"' in hardened
    assert 'SYSTEM_NODE=$(command -v node 2>/dev/null || true)' in hardened
    assert 'rm -f "$NODE_TAR"' in hardened
    assert "rm -f /tmp/openclaw.tar.gz" in hardened
    assert _harden_openclaw_bootstrap(hardened) == hardened


def test_osworld_setup_adapter_disables_gimp_color_profile_prompt():
    calls = []

    class FakeSetupController:
        def _execute_setup(self, command, **kwargs):
            calls.append(("execute", command, kwargs))

        def _launch_setup(self, command, **kwargs):
            calls.append(("launch", command, kwargs))

    _OSWorldSetupControllerAdapter(FakeSetupController()).launch(
        ["gimp", "/home/user/Desktop/image.jpg"]
    )

    assert calls[0][0] == "execute"
    assert "(color-profile-policy keep)" in calls[0][1][2]
    assert calls[1] == (
        "launch",
        ["gimp", "/home/user/Desktop/image.jpg"],
        {"shell": False},
    )


@pytest.mark.parametrize(
    "evaluator_func",
    ["check_image_mirror", "check_saturation_increase_and_structure_sim"],
)
def test_legacy_image_evaluator_uses_changed_in_place_input(evaluator_func):
    input_path = "/home/user/Desktop/source.jpg"
    result_path = "/home/user/Desktop/source_mirror.png"
    task_config = {
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [{"path": input_path, "url": "file:///source.jpg"}]
                },
            }
        ],
        "evaluator": {
            "func": evaluator_func,
            "expected": {"type": "cloud_file", "dest": "source.jpg"},
            "result": {"type": "vm_file", "path": result_path},
            "postconfig": [{"type": "activate_window"}],
        },
    }
    payloads = iter((b"original", b"mirrored"))
    commands = []
    env = SimpleNamespace(
        controller=SimpleNamespace(get_file=lambda path: next(payloads)),
        setup_controller=SimpleNamespace(
            execute=lambda command, **kwargs: commands.append((command, kwargs))
        ),
        evaluator=dict(task_config["evaluator"]),
    )

    snapshot = _capture_legacy_in_place_image(task_config, env)

    assert _materialize_changed_legacy_image(env, snapshot)
    assert input_path in commands[0][0][2]
    assert result_path in commands[0][0][2]
    assert env.evaluator["postconfig"] == []


def test_legacy_image_evaluator_keeps_postconfig_for_unchanged_input():
    task_config = {
        "config": [{
            "type": "download",
            "parameters": {
                "files": [{
                    "path": "/home/user/Desktop/source.jpg",
                    "url": "file:///source.jpg",
                }]
            },
        }],
        "evaluator": {
            "func": "check_image_mirror",
            "expected": {"type": "cloud_file", "dest": "source.jpg"},
            "result": {
                "type": "vm_file",
                "path": "/home/user/Desktop/source_mirror.png",
            },
            "postconfig": [{"type": "activate_window"}],
        },
    }
    env = SimpleNamespace(
        controller=SimpleNamespace(get_file=lambda path: b"unchanged"),
        setup_controller=SimpleNamespace(execute=lambda *args, **kwargs: None),
        evaluator=dict(task_config["evaluator"]),
    )

    snapshot = _capture_legacy_in_place_image(task_config, env)

    assert not _materialize_changed_legacy_image(env, snapshot)
    assert env.evaluator["postconfig"] == [{"type": "activate_window"}]


def test_osworld_setup_command_tunes_slow_installers_and_servers():
    command, timeout, launch, background_wait = _prepare_osworld_setup_command(
        ["pip3", "install", "-q", "fpdf2"], 120
    )
    assert command == ["pip3", "install", "--no-cache-dir", "-q", "fpdf2"]
    assert timeout == 600
    assert launch is False
    assert background_wait is True

    command, timeout, launch, background_wait = _prepare_osworld_setup_command(
        ["bash", "-c", "cd /tmp/app && npm start"], 120
    )
    assert command == ["bash", "-c", "cd /tmp/app && npm start"]
    assert timeout == 120
    assert launch is True
    assert background_wait is False


def test_cleanup_guest_runtime_removes_transient_caches():
    calls = []
    controller = SimpleNamespace(
        execute=lambda command, **kwargs: calls.append((command, kwargs))
    )

    _cleanup_guest_runtime(controller)

    command, kwargs = calls[0]
    assert "/tmp/weavebench_claudecode_install" in command[2]
    assert "/tmp/openclaw.tar.gz" in command[2]
    assert "/tmp/openclaw-$(id -u)" in command[2]
    assert "/home/user/.cache/pip" in command[2]
    assert kwargs == {"quiet": True, "timeout": 300}


def test_resume_skips_only_when_trajectory_evidence_exists(tmp_path):
    task = {"task_id": "task-1", "category": "OSWORLD_V2"}
    score_dir = tmp_path / "hybrid/model/OSWORLD_V2/task-1"
    score_dir.mkdir(parents=True)
    (score_dir / "score.json").write_text(
        json.dumps({"score": 0.0, "error": None, "agent_done": False}),
        encoding="utf-8",
    )

    assert not _score_allows_resume_skip(task, "hybrid", "model", tmp_path)

    (score_dir / "score.json").write_text(
        json.dumps({"score": 0.0, "error": None, "agent_done": True}),
        encoding="utf-8",
    )
    assert not _score_allows_resume_skip(task, "hybrid", "model", tmp_path)

    (score_dir / "computer_events.jsonl").write_text(
        json.dumps({"step": 1, "action": {"type": "click"}}) + "\n",
        encoding="utf-8",
    )
    assert _score_allows_resume_skip(task, "hybrid", "model", tmp_path)

    (score_dir / "score.json").write_text(
        json.dumps({
            "score": 0.0,
            "error": " | judge_failed: host judge unavailable",
            "agent_done": True,
            "agent_status": "ok",
            "agent_exit": 0,
        }),
        encoding="utf-8",
    )
    assert _score_allows_resume_skip(task, "hybrid", "model", tmp_path)

    (score_dir / "score.json").write_text(
        json.dumps({
            "score": 0.0,
            "error": "agent_empty_run: no model output | judge_failed: unavailable",
            "agent_done": True,
            "agent_status": "empty_run",
            "agent_exit": 0,
        }),
        encoding="utf-8",
    )
    assert _score_allows_resume_skip(task, "hybrid", "model", tmp_path)

    (score_dir / "score.json").write_text(
        json.dumps({"score": 0.0, "error": None}),
        encoding="utf-8",
    )
    assert _score_allows_resume_skip(task, "hybrid", "model", tmp_path)

    (score_dir / "score.json").write_text(
        json.dumps({"score": 1.0, "error": None, "success": True}),
        encoding="utf-8",
    )
    assert _score_allows_resume_skip(task, "hybrid", "model", tmp_path)

    (score_dir / "computer_events.jsonl").write_text("", encoding="utf-8")
    assert not _score_allows_resume_skip(task, "hybrid", "model", tmp_path)


def test_osworld_task_adapter_keeps_absolute_model_under_result_root(
    monkeypatch, tmp_path
):
    task_id = "generated-task"
    (tmp_path / "taskset_manifest.json").write_text(
        json.dumps({"tasks": [{"id": task_id}]}), encoding="utf-8"
    )
    seen = {}

    def inner_run_one(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        seen["output_dir"] = output_dir
        return {"score": 0.0}

    runner = SimpleNamespace(ds=SimpleNamespace(
        run_one=inner_run_one,
        already_done=lambda task, mode, model, result_root: (
            seen.update(model=model, result_root=result_root) or False
        ),
    ))
    monkeypatch.setenv("WEAVEBENCH_MODEL", "/models/moonshotai--Kimi-K3")
    monkeypatch.setenv("GUI_TRAJECTORY_RESULT_DIR", str(tmp_path / "results"))
    _install_osworld_task_adapter(runner)

    task = {"task_id": task_id, "category": "OSWORLD_V2"}
    runner.ds.run_one(
        object(), object(), task, "gui", Path("/models/escaped"), tmp_path
    )
    runner.ds.already_done(
        task, "gui", "/models/moonshotai--Kimi-K3", tmp_path / "results"
    )

    assert seen["output_dir"] == (
        tmp_path / "results/gui/moonshotai--Kimi-K3/OSWORLD_V2/generated-task"
    )
    assert seen["model"] == "moonshotai--Kimi-K3"


def test_osworld_task_adapter_uses_requested_mode_for_result_paths(
    monkeypatch, tmp_path
):
    task_id = "generated-task"
    (tmp_path / "taskset_manifest.json").write_text(
        json.dumps({"tasks": [{"id": task_id}]}), encoding="utf-8"
    )
    seen = {}

    def inner_run_one(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        seen["mode"] = mode
        seen["output_dir"] = output_dir
        return {"score": 0.0}

    runner = SimpleNamespace(ds=SimpleNamespace(
        run_one=inner_run_one,
        already_done=lambda task, mode, model, result_root: (
            seen.update(done_mode=mode) or False
        ),
    ))
    monkeypatch.setenv("WEAVEBENCH_MODEL", "/models/moonshotai--Kimi-K3")
    monkeypatch.setenv("GUI_TRAJECTORY_RESULT_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("GUI_TRAJECTORY_REQUESTED_MODE", "hybrid")
    _install_osworld_task_adapter(runner)

    task = {"task_id": task_id, "category": "OSWORLD_V2"}
    runner.ds.run_one(
        object(), object(), task, "gui", Path("/models/escaped"), tmp_path
    )
    runner.ds.already_done(
        task, "gui", "/models/moonshotai--Kimi-K3", tmp_path / "results"
    )

    assert seen["mode"] == "gui"
    assert seen["done_mode"] == "hybrid"
    assert seen["output_dir"] == (
        tmp_path / "results/hybrid/moonshotai--Kimi-K3/OSWORLD_V2/generated-task"
    )


def test_runner_module_uses_chat_runner_for_openclaw(monkeypatch):
    sentinel = object()
    imported = []

    def fake_import(name):
        imported.append(name)
        return sentinel

    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner.importlib.import_module",
        fake_import,
    )
    assert _runner_module("openclaw", "chat") is sentinel
    assert imported == ["weavebench.eval._runners.chat_gui"]


def test_hybrid_mode_shim_translates_argv_and_records_hybrid(
    monkeypatch, tmp_path
):
    calls = []

    def run_one(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        calls.append(mode)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "score.json").write_text(
            json.dumps({"mode": mode}), encoding="utf-8"
        )
        return {"mode": mode}

    messages_gui = SimpleNamespace(ds=SimpleNamespace(run_one=run_one))
    monkeypatch.setenv("GUI_TRAJECTORY_RUNTIME_LOCATION", "guest")
    argv = ["prog", "--mode", "hybrid", "--model", "m"]
    _install_hybrid_mode(messages_gui, argv)
    assert argv[argv.index("--mode") + 1] == "gui"
    output_dir = tmp_path / "result"
    record = messages_gui.ds.run_one(
        None,
        None,
        {"category": "CAT", "task_id": "task"},
        "gui",
        output_dir,
        tmp_path,
    )
    assert record == {"mode": "hybrid"}
    assert calls == ["gui"]
    assert json.loads((output_dir / "score.json").read_text())["mode"] == "hybrid"


def test_hybrid_mode_shim_ignores_gui_mode(monkeypatch):
    sentinel = object()
    messages_gui = SimpleNamespace(ds=SimpleNamespace(run_one=sentinel))
    argv = ["prog", "--mode", "gui"]
    _install_hybrid_mode(messages_gui, argv)
    assert argv == ["prog", "--mode", "gui"]
    assert messages_gui.ds.run_one is sentinel


def test_hybrid_mode_shim_supports_host_runtime_with_guest_shell(monkeypatch, tmp_path):
    calls = []
    messages_gui = SimpleNamespace(
        ds=SimpleNamespace(run_one=lambda *args, **kwargs: calls.append(args[3]) or {})
    )
    monkeypatch.setenv("GUI_TRAJECTORY_RUNTIME_LOCATION", "host")
    monkeypatch.setenv("GUI_TRAJECTORY_ENABLE_GUEST_SHELL", "true")
    argv = ["prog", "--mode", "hybrid"]
    _install_hybrid_mode(messages_gui, argv)
    assert argv == ["prog", "--mode", "gui"]
    messages_gui.ds.run_one(None, None, {}, "gui", tmp_path, tmp_path)
    assert calls == ["gui"]


def test_hybrid_mode_shim_rejects_host_runtime_without_guest_shell(monkeypatch):
    messages_gui = SimpleNamespace(ds=SimpleNamespace(run_one=lambda *a, **k: None))
    monkeypatch.setenv("GUI_TRAJECTORY_RUNTIME_LOCATION", "host")
    monkeypatch.delenv("GUI_TRAJECTORY_ENABLE_GUEST_SHELL", raising=False)
    with pytest.raises(ValueError, match="guest shell MCP bridge"):
        _install_hybrid_mode(messages_gui, ["prog", "--mode", "hybrid"])


def test_worker_aware_monitor_accepts_terminal_event():
    progress = [
        {"event": "start", "name": "env-1", "mode": "gui", "task": "task-1"},
        {
            "event": "done",
            "name": "env-1",
            "mode": "gui",
            "task": "task-1",
            "score": 1.0,
        },
    ]

    failure = _worker_aware_monitor(
        progress,
        1,
        1,
        poll_interval=0,
        active_children_fn=lambda: [],
    )

    assert failure is None


def test_worker_aware_monitor_detects_exited_active_worker(caplog):
    process = SimpleNamespace(name="env-1", is_alive=lambda: True, exitcode=0)
    children = iter(([process], []))
    progress = [
        {"event": "start", "name": "env-1", "mode": "gui", "task": "task-1"}
    ]

    with caplog.at_level(logging.ERROR):
        failure = _worker_aware_monitor(
            progress,
            1,
            1,
            poll_interval=0,
            active_children_fn=lambda: next(children),
        )

    assert failure == (
        "env-1:task-1:worker exited without a terminal progress event "
        "(exitcode=0)"
    )
    assert "☠ CRASH  [env-1][gui] task-1" in caplog.text


def test_install_worker_exit_monitor_records_failure(monkeypatch):
    runner = SimpleNamespace(_monitor=lambda *args: None)
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._worker_aware_monitor",
        lambda *args: "worker failure",
    )

    _install_worker_exit_monitor(runner)
    runner._monitor([], 1, 10)

    module = __import__(
        "cuamaster.collection.weavebench_runner",
        fromlist=["_worker_monitor_failure"],
    )
    assert module._worker_monitor_failure == "worker failure"


def test_task_endpoint_failover_retries_endpoint_errors(monkeypatch, tmp_path):
    pool_path = tmp_path / "kimi.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: kimi-k3\nbase_urls:\n"
        "  - http://model-a:30000/v1\n  - http://model-b:30000/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GUI_TRAJECTORY_ENDPOINT_POOL", str(pool_path))
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._endpoint_pool_instance",
        None,
    )
    calls = []

    def run_one(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        calls.append(agent.litellm_base_url)
        if len(calls) == 1:
            return {"error": "connection timeout"}
        return {"error": None, "score": 1.0}

    resets = []
    env = SimpleNamespace(reset=lambda **kwargs: resets.append(kwargs))
    agent = SimpleNamespace(litellm_base_url="")
    runner = SimpleNamespace(ds=SimpleNamespace(run_one=run_one))
    _install_task_endpoint_failover(runner)

    result = runner.ds.run_one(
        env,
        agent,
        {"task_id": "task-1"},
        "gui",
        tmp_path / "result",
        tmp_path,
    )

    assert result["error"] is None
    assert calls == ["http://model-a:30000/v1", "http://model-b:30000/v1"]
    assert resets == [{"task_config": {"task_id": "task-1"}}]
    audit = json.loads((tmp_path / "result/endpoint_failover.json").read_text())
    assert [item["endpoint_failure"] for item in audit["attempts"]] == [True, False]


def test_task_endpoint_failover_defers_osworld_reset_to_task_adapter(
    monkeypatch, tmp_path
):
    pool_path = tmp_path / "kimi.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: kimi-k3\nbase_urls:\n"
        "  - http://model-a:30000/v1\n  - http://model-b:30000/v1\n",
        encoding="utf-8",
    )
    (tmp_path / "taskset_manifest.json").write_text(json.dumps({
        "tasks": [{
            "id": "task-1",
            "evaluator": {"type": "osworld_v2_task_class"},
        }],
    }), encoding="utf-8")
    monkeypatch.setenv("GUI_TRAJECTORY_ENDPOINT_POOL", str(pool_path))
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._endpoint_pool_instance",
        None,
    )
    calls = []

    def run_one(env, agent, task, mode, output_dir, tasks_root, **kwargs):
        calls.append(agent.litellm_base_url)
        if len(calls) == 1:
            return {"error": "connection timeout"}
        return {"error": None, "score": 1.0}

    resets = []
    env = SimpleNamespace(reset=lambda **kwargs: resets.append(kwargs))
    runner = SimpleNamespace(ds=SimpleNamespace(run_one=run_one))
    _install_task_endpoint_failover(runner)

    result = runner.ds.run_one(
        env,
        SimpleNamespace(litellm_base_url=""),
        {"task_id": "task-1", "category": "OSWORLD_V2"},
        "gui",
        tmp_path / "result",
        tmp_path,
    )

    assert result["error"] is None
    assert calls == ["http://model-a:30000/v1", "http://model-b:30000/v1"]
    assert resets == []


def test_claudecode_adapter_overrides_upstream_task_timeout(monkeypatch):
    class FakeClaudeCodeAgent:
        def __init__(self, **kwargs):
            self.timeout = kwargs["timeout"]
            self.gui = kwargs.get("gui", False)

    upstream = ModuleType("weavebench.agents.claudecode_agent")
    upstream.ClaudeCodeAgent = FakeClaudeCodeAgent
    weavebench = ModuleType("weavebench")
    agents = ModuleType("weavebench.agents")
    weavebench.agents = agents
    agents.claudecode_agent = upstream
    monkeypatch.setitem(sys.modules, "weavebench", weavebench)
    monkeypatch.setitem(sys.modules, "weavebench.agents", agents)
    monkeypatch.setitem(sys.modules, "weavebench.agents.claudecode_agent", upstream)
    monkeypatch.setenv("GUI_TRAJECTORY_RUNTIME_LOCATION", "guest")
    monkeypatch.setenv("GUI_TRAJECTORY_TASK_TIMEOUT_SECONDS", "3600")

    adapter = _install_agent_adapter("claudecode")
    agent = adapter(timeout=7200, gui=False)

    assert agent.timeout == 3600


def test_vm_transport_guard_fails_wait_after_consecutive_failures(monkeypatch):
    calls = []

    def vm_exec(env, command, shell=False, timeout=120):
        calls.append((env, command, shell, timeout))
        raise requests.ConnectionError("relay closed connection")

    upstream = SimpleNamespace(
        _vm_exec=vm_exec,
        _vm_fetch=lambda env, remote, local: True,
    )
    monkeypatch.setenv("GUI_TRAJECTORY_VM_TRANSPORT_FAILURE_LIMIT", "2")
    monkeypatch.setenv("GUI_TRAJECTORY_VM_TRANSPORT_POLL_INTERVAL_SECONDS", "0.01")
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _install_vm_transport_guard(upstream)
    env = SimpleNamespace(
        vm_base_url="http://worker.example:21000",
        provider=SimpleNamespace(resource_id="lease-123", ports={"server": 21000}),
        _gui_trajectory_task_id="task-123",
    )

    with pytest.raises(VMTransportUnavailable, match="wait:/tmp/run.done"):
        upstream._wait_file(env, "/tmp/run.done", 3600)

    assert len(calls) == 2
    assert all(call[2] is False for call in calls)
    assert all(0 < call[3] <= 10 for call in calls)


def test_vm_transport_guard_retries_artifact_fetch(monkeypatch, tmp_path):
    calls = []

    def vm_fetch(env, remote, local):
        calls.append((env, remote, local))
        if len(calls) < 3:
            raise requests.ConnectionError("temporary relay failure")
        return True

    upstream = SimpleNamespace(
        _vm_exec=lambda env, command, **kwargs: {"output": "YES"},
        _vm_fetch=vm_fetch,
    )
    monkeypatch.setenv("GUI_TRAJECTORY_VM_FETCH_RETRIES", "3")
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _install_vm_transport_guard(upstream)

    assert upstream._vm_fetch(object(), "/tmp/agent.log", tmp_path / "agent.log")
    assert len(calls) == 3


def test_vm_transport_guard_retries_artifact_upload(monkeypatch, tmp_path):
    calls = []

    def vm_upload_bytes(env, path, remote, **kwargs):
        calls.append((env, path, remote, kwargs))
        if len(calls) < 3:
            response = requests.Response()
            response.status_code = 500
            raise requests.HTTPError("temporary upload failure", response=response)
        return True

    upstream = SimpleNamespace(
        _vm_exec=lambda env, command, **kwargs: {"output": "YES"},
        _vm_fetch=lambda env, remote, local: True,
        _vm_upload_bytes=vm_upload_bytes,
    )
    monkeypatch.setenv("GUI_TRAJECTORY_VM_UPLOAD_RETRIES", "3")
    monkeypatch.setenv("GUI_TRAJECTORY_VM_UPLOAD_RETRY_BACKOFF_SECONDS", "0.01")
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _install_vm_transport_guard(upstream)

    archive = tmp_path / "openclaw.tar.gz"
    archive.write_bytes(b"archive")
    assert upstream._vm_upload_bytes(object(), archive, "/tmp/openclaw.tar.gz")
    assert len(calls) == 3


def test_claudecode_adapter_writes_transport_failure_status(monkeypatch, tmp_path):
    class FakeClaudeCodeAgent:
        def __init__(self, **kwargs):
            self.timeout = kwargs["timeout"]
            self.gui = kwargs.get("gui", False)
            self.model = kwargs.get("model", "test-model")
            self.litellm_api_key = ""

        def run(self, env, instruction, output_dir, **kwargs):
            raise requests.ConnectionError("relay closed connection")

    upstream = ModuleType("weavebench.agents.claudecode_agent")
    upstream.ClaudeCodeAgent = FakeClaudeCodeAgent
    upstream._vm_exec = lambda env, command, **kwargs: {"returncode": 0}
    upstream._vm_fetch = lambda env, remote, local: True
    weavebench = ModuleType("weavebench")
    agents = ModuleType("weavebench.agents")
    weavebench.agents = agents
    agents.claudecode_agent = upstream
    monkeypatch.setitem(sys.modules, "weavebench", weavebench)
    monkeypatch.setitem(sys.modules, "weavebench.agents", agents)
    monkeypatch.setitem(sys.modules, "weavebench.agents.claudecode_agent", upstream)
    monkeypatch.setenv("GUI_TRAJECTORY_RUNTIME_LOCATION", "guest")
    monkeypatch.setenv("WEAVEBENCH_VM_PROFILE_ID", "profile-a")
    adapter = _install_agent_adapter("claudecode")
    agent = adapter(timeout=3600, gui=True, model="test-model")
    env = SimpleNamespace(
        vm_base_url="http://worker.example:21000",
        provider=SimpleNamespace(
            resource_id="lease-123",
            endpoint_host="worker.example",
            ports={"server": 21000},
        ),
    )
    output_dir = tmp_path / "task-123"

    with pytest.raises(VMTransportUnavailable, match="agent-run"):
        agent.run(env, "instruction", output_dir)

    status = json.loads((output_dir / "execution_status.json").read_text())
    assert status["agent_done"] is False
    assert status["timed_out"] is False
    assert status["transport_failure"] is True
    assert status["vm_transport"]["resource_id"] == "lease-123"
    failure = json.loads((output_dir / "vm_transport_failure.json").read_text())
    assert failure["task_id"] == "task-123"
    fingerprint = json.loads(
        (output_dir / "environment_fingerprint.json").read_text()
    )
    assert fingerprint["transport_failure"]["resource_id"] == "lease-123"


def test_task_endpoint_failover_normalizes_absolute_model_path(monkeypatch, tmp_path):
    pool_path = tmp_path / "kimi.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: /models/moonshotai--Kimi-K3\nbase_urls:\n"
        "  - http://model-a:30000/v1\n",
        encoding="utf-8",
    )
    result_root = tmp_path / "result"
    monkeypatch.setenv("GUI_TRAJECTORY_ENDPOINT_POOL", str(pool_path))
    monkeypatch.setenv("GUI_TRAJECTORY_RESULT_DIR", str(result_root))
    monkeypatch.setenv("GUI_TRAJECTORY_REQUESTED_MODE", "hybrid")
    monkeypatch.setenv("WEAVEBENCH_MODEL", "/models/moonshotai--Kimi-K3")
    monkeypatch.setattr(
        "cuamaster.collection.weavebench_runner._endpoint_pool_instance",
        None,
    )
    runner = SimpleNamespace(
        ds=SimpleNamespace(run_one=lambda *args, **kwargs: {"error": None, "score": 1.0})
    )
    _install_task_endpoint_failover(runner)

    runner.ds.run_one(
        object(),
        SimpleNamespace(litellm_base_url=""),
        {"task_id": "task-1", "category": "OSWORLD_V2"},
        "gui",
        Path("/models/moonshotai--Kimi-K3/OSWORLD_V2/task-1"),
        tmp_path,
    )

    audit_path = (
        result_root
        / "hybrid"
        / "moonshotai--Kimi-K3"
        / "OSWORLD_V2"
        / "task-1"
        / "endpoint_failover.json"
    )
    assert audit_path.is_file()


def test_write_task_metadata_persists_manifest_instruction(tmp_path):
    output_dir = tmp_path / "result"
    _write_task_metadata(
        output_dir,
        {
            "task_id": "task-1",
            "category": "OSWORLD_V2",
            "name": "Example task",
            "prompt": "Complete the desktop task.",
        },
        {
            "id": "task-1",
            "version": "task-version",
            "markdown": "OSWORLD_V2/task-1.md",
            "environment_id": "ubuntu-image",
            "environment_digest": "image-digest",
            "required_capabilities": ["computer"],
            "target_environments": [{"os_type": "Ubuntu"}],
            "runtime_config": [{"type": "osworld_v2_task_class"}],
            "evaluator": {"type": "osworld_v2_task_class"},
        },
    )

    task = json.loads((output_dir / "task.json").read_text())
    assert task["instruction"] == "Complete the desktop task."
    assert task["evaluator"] == {"type": "osworld_v2_task_class"}
    assert task["required_capabilities"] == ["computer"]
    assert task["metadata"]["environment_id"] == "ubuntu-image"


def test_weavebench_compiler_emits_platform_manifest(tmp_path):
    generated_root = tmp_path / "generated"
    task_root = generated_root / "windows-task"
    task_root.mkdir(parents=True)
    (task_root / "generated_task.json").write_text(json.dumps({
        "task": {
            "id": "windows-task",
            "instruction": "Open Notepad and save the file.",
            "required_capabilities": ["computer", "app:notepad"],
            "target_environments": [{
                "os_type": "Windows",
                "os_versions": ["11"],
                "architectures": ["x86_64"],
            }],
            "config": [{"type": "osworld_v2_task_class"}],
            "evaluator": {
                "type": "osworld_v2_task_class",
                "path": "/repo/task.py",
                "repository_root": "/repo",
                "class_name": "Task001",
            },
            "metadata": {"category": "DOC"},
        },
        "recipe": {
            "id": "recipe",
            "anchor_task_id": "source",
            "capability_ids": [],
            "required_vm_capabilities": ["computer", "app:notepad"],
            "source_task_ids": ["source"],
            "environment_id": "windows-image",
            "environment_digest": "environment-digest",
        },
        "evaluator_plugins": [],
        "assets": [],
    }), encoding="utf-8")
    output = tmp_path / "taskset"
    manifest = compile_weavebench_taskset(generated_root, output)
    row = manifest["tasks"][0]
    assert row["required_capabilities"] == ["app:notepad", "computer"]
    assert row["target_environments"][0]["os_type"] == "Windows"
    assert row["environment_id"] == "windows-image"
    assert row["environment_digest"] == "environment-digest"
    assert row["runtime_config"] == [{"type": "osworld_v2_task_class"}]
    assert row["evaluator"]["class_name"] == "Task001"
