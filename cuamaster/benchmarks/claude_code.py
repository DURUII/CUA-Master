"""Claude Code tool policies for reusable interaction harnesses."""
from __future__ import annotations

from dataclasses import dataclass

from .base import InteractionHarness
from .modes import get_interaction_harness


CLAUDE_INTERNAL_TOOL_NAMES = (
    "Agent",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "Skill",
    "Task",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "ToolSearch",
    "TodoWrite",
)
CLAUDE_STRUCTURED_TOOL_NAMES = (
    "Edit",
    "Glob",
    "Grep",
    "NotebookEdit",
    "Read",
    "Write",
)
NON_ENVIRONMENT_TOOL_NAMES = frozenset(
    name.casefold() for name in CLAUDE_INTERNAL_TOOL_NAMES
)


@dataclass(frozen=True)
class ClaudeCodeToolPolicy:
    allowed_tools: tuple[str, ...]
    disallowed_tools: tuple[str, ...]

    @property
    def allowed_argument(self) -> str:
        return ",".join(self.allowed_tools)

    @property
    def disallowed_argument(self) -> str:
        return ",".join(self.disallowed_tools)


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def claude_code_tool_policy(
    harness: str | InteractionHarness,
    *,
    runtime_location: str,
) -> ClaudeCodeToolPolicy:
    interaction = get_interaction_harness(harness)
    location = str(runtime_location or "").strip().lower()
    if location not in {"guest", "host"}:
        raise ValueError("runtime_location must be guest or host")

    gui_tool = "mcp__weavebench_computer__computer"
    cli_tool = "Bash" if location == "guest" else "mcp__weavebench_computer__shell"
    allowed: list[str] = []
    if interaction.supports_gui:
        allowed.append(gui_tool)
    if interaction.supports_cli:
        allowed.append(cli_tool)

    disallowed = [
        "WebFetch",
        "WebSearch",
        *CLAUDE_STRUCTURED_TOOL_NAMES,
        *CLAUDE_INTERNAL_TOOL_NAMES,
    ]
    if cli_tool != "Bash" or not interaction.supports_cli:
        disallowed.append("Bash")
    return ClaudeCodeToolPolicy(
        allowed_tools=_unique(allowed),
        disallowed_tools=_unique(disallowed),
    )
