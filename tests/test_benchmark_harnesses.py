from __future__ import annotations

import pytest

from cuamaster.benchmarks import (
    CLI_HARNESS,
    GUI_HARNESS,
    HYBRID_HARNESS,
    HYBRID_CANONICAL_SYSTEM_PROMPT,
    HYBRID_QWEN_XML_SYSTEM_PROMPT,
    claude_code_tool_policy,
    get_interaction_harness,
)


@pytest.mark.parametrize(
    ("name", "gui", "cli"),
    [("gui", True, False), ("cli", False, True), ("hybrid", True, True)],
)
def test_interaction_harness_capabilities(name, gui, cli):
    harness = get_interaction_harness(name)

    assert harness.supports_gui is gui
    assert harness.supports_cli is cli
    assert harness.allows_action_type("computer.click") is gui
    assert harness.allows_action_type("shell.exec") is cli
    assert harness.allows_action_type("finish")
    assert not harness.allows_action_type("tool.call")


def test_interaction_harness_registry_returns_stable_instances():
    assert get_interaction_harness(" GUI ") is GUI_HARNESS
    assert get_interaction_harness("cli") is CLI_HARNESS
    assert get_interaction_harness(HYBRID_HARNESS) is HYBRID_HARNESS


def test_claude_guest_hybrid_is_gui_plus_bash_only():
    policy = claude_code_tool_policy("hybrid", runtime_location="guest")

    assert policy.allowed_tools == (
        "mcp__weavebench_computer__computer",
        "Bash",
    )
    assert {"Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit"} <= set(
        policy.disallowed_tools
    )
    assert "TodoWrite" in policy.disallowed_tools


def test_claude_host_hybrid_routes_cli_through_guest_shell_mcp():
    policy = claude_code_tool_policy("hybrid", runtime_location="host")

    assert policy.allowed_tools == (
        "mcp__weavebench_computer__computer",
        "mcp__weavebench_computer__shell",
    )
    assert "Bash" in policy.disallowed_tools


def test_unknown_interaction_harness_is_rejected():
    with pytest.raises(ValueError, match="unknown interaction harness"):
        get_interaction_harness("tools")


def test_hybrid_prompts_advertise_only_gui_and_cli_actions():
    assert "shell.exec" in HYBRID_CANONICAL_SYSTEM_PROMPT
    assert "computer_use" in HYBRID_QWEN_XML_SYSTEM_PROMPT
    assert "shell_exec" in HYBRID_QWEN_XML_SYSTEM_PROMPT
    assert "tool.call(" not in HYBRID_CANONICAL_SYSTEM_PROMPT
    assert "function=tool_call" not in HYBRID_QWEN_XML_SYSTEM_PROMPT
    assert "registered structured tool" not in HYBRID_QWEN_XML_SYSTEM_PROMPT.lower()
