"""Reusable canonical CUA runtime.

Benchmarks provide a small :class:`RuntimeEnvironment` adapter. Parsing,
validation, GUI coordinate conversion, guest-shell routing, registered tools,
the OpenAI-compatible model client, and trajectory logging stay shared.
"""

from cuamaster.action_space import CUAAction, CUAResponse, ToolDefinition

from .core import (
    CallbackEnvironment,
    CanonicalActionExecutor,
    CanonicalActionParser,
    RuntimeEnvironment,
    ToolCallError,
    ToolCapabilityEnvironment,
    ToolPathRoot,
    ToolRegistry,
    load_tool_registry,
)
from .episode import (
    CanonicalEpisodeRunner,
    ModelClient,
    ModelResponse,
    OpenAICompatibleModel,
)
from .qwen_xml import (
    ACTION_PROTOCOL_AUTO,
    ACTION_PROTOCOL_CANONICAL,
    ACTION_PROTOCOL_QWEN_XML,
    DEFAULT_ACTION_PROTOCOL,
    SUPPORTED_ACTION_PROTOCOLS,
    CompatibleActionParser,
    QWEN_XML_SYSTEM_PROMPT,
    QwenXmlActionParser,
    action_to_qwen_xml_calls,
    describe_qwen_xml_action,
    format_qwen_xml_action,
    qwen_coordinate,
    qwen_xml_registered_tools_prompt,
    qwen_xml_tool_name,
)
from .tools import CORE_TOOL_NAMES, ToolLimits, build_tool_registry

__all__ = [
    "CUAAction",
    "CUAResponse",
    "ACTION_PROTOCOL_AUTO",
    "ACTION_PROTOCOL_CANONICAL",
    "ACTION_PROTOCOL_QWEN_XML",
    "CallbackEnvironment",
    "CanonicalActionExecutor",
    "CanonicalActionParser",
    "CanonicalEpisodeRunner",
    "CompatibleActionParser",
    "DEFAULT_ACTION_PROTOCOL",
    "ModelClient",
    "ModelResponse",
    "OpenAICompatibleModel",
    "QWEN_XML_SYSTEM_PROMPT",
    "QwenXmlActionParser",
    "RuntimeEnvironment",
    "SUPPORTED_ACTION_PROTOCOLS",
    "ToolCallError",
    "ToolCapabilityEnvironment",
    "ToolDefinition",
    "ToolLimits",
    "ToolPathRoot",
    "ToolRegistry",
    "CORE_TOOL_NAMES",
    "build_tool_registry",
    "action_to_qwen_xml_calls",
    "describe_qwen_xml_action",
    "format_qwen_xml_action",
    "load_tool_registry",
    "qwen_xml_registered_tools_prompt",
    "qwen_xml_tool_name",
    "qwen_coordinate",
]
