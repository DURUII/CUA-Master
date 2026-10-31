"""Reusable interaction harnesses for benchmark evaluation."""

from .base import InteractionCapability, InteractionHarness
from .claude_code import (
    CLAUDE_INTERNAL_TOOL_NAMES,
    CLAUDE_STRUCTURED_TOOL_NAMES,
    NON_ENVIRONMENT_TOOL_NAMES,
    ClaudeCodeToolPolicy,
    claude_code_tool_policy,
)
from .modes import (
    CLI_HARNESS,
    GUI_HARNESS,
    HYBRID_HARNESS,
    INTERACTION_HARNESSES,
    get_interaction_harness,
)
from .prompts import (
    HYBRID_CANONICAL_SYSTEM_PROMPT,
    HYBRID_QWEN_XML_SYSTEM_PROMPT,
    hybrid_system_prompt,
)

__all__ = [
    "CLAUDE_INTERNAL_TOOL_NAMES",
    "CLAUDE_STRUCTURED_TOOL_NAMES",
    "CLI_HARNESS",
    "GUI_HARNESS",
    "HYBRID_HARNESS",
    "HYBRID_CANONICAL_SYSTEM_PROMPT",
    "HYBRID_QWEN_XML_SYSTEM_PROMPT",
    "INTERACTION_HARNESSES",
    "NON_ENVIRONMENT_TOOL_NAMES",
    "ClaudeCodeToolPolicy",
    "InteractionCapability",
    "InteractionHarness",
    "claude_code_tool_policy",
    "get_interaction_harness",
    "hybrid_system_prompt",
]
