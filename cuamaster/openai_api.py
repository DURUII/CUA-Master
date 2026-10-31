"""Shared OpenAI-compatible Chat Completions and Responses helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextResponse:
    text: str
    response: Any


def create_text_response(
    client: Any,
    *,
    wire_api: str,
    model: str,
    content: str | list[dict[str, Any]],
    max_output_tokens: int,
    reasoning_effort: str | None = None,
    system_prompt: str | None = None,
    store: bool = False,
    chat_token_param: str = "max_tokens",
    chat_response_format: dict[str, Any] | None = None,
) -> TextResponse:
    normalized_wire_api = normalize_wire_api(wire_api)
    normalized_effort = str(reasoning_effort or "").strip().lower()
    if normalized_wire_api == "responses":
        kwargs: dict[str, Any] = {
            "model": model,
            "input": _responses_input(content),
            "max_output_tokens": max_output_tokens,
            "store": store,
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt
        if normalized_effort and normalized_effort not in {"none", "off", "false", "0"}:
            kwargs["reasoning"] = {"effort": normalized_effort}
        response = client.responses.create(**kwargs)
        return TextResponse(text=_responses_output_text(response), response=response)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    kwargs = {
        "model": model,
        "stream": False,
        "messages": messages,
        chat_token_param: max_output_tokens,
    }
    if _include_reasoning_effort(model, normalized_effort):
        kwargs["reasoning_effort"] = normalized_effort
    if chat_response_format is not None:
        kwargs["response_format"] = chat_response_format
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    return TextResponse(text=text, response=response)


def normalize_wire_api(value: str | None) -> str:
    normalized = str(value or "chat_completions").strip().lower()
    normalized = {
        "chat": "chat_completions",
        "chat.completions": "chat_completions",
        "chat-completions": "chat_completions",
        "response": "responses",
    }.get(normalized, normalized)
    if normalized not in {"chat_completions", "responses"}:
        raise ValueError("wire_api must be chat_completions or responses")
    return normalized


def _include_reasoning_effort(model: str, normalized_effort: str) -> bool:
    if not normalized_effort:
        return False
    if normalized_effort == "none" and model.lower().startswith("deepseek-v4.1"):
        return True
    return normalized_effort not in {"none", "off", "false", "0"}


def usage_token_counts(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", 0)
    return int(input_tokens or 0), int(output_tokens or 0)


def _responses_input(content: str | list[dict[str, Any]]) -> Any:
    if isinstance(content, str):
        return content
    converted: list[dict[str, Any]] = []
    for item in content:
        item_type = str(item.get("type") or "")
        if item_type in {"text", "input_text"}:
            converted.append({"type": "input_text", "text": str(item.get("text") or "")})
            continue
        if item_type in {"image_url", "input_image"}:
            image_value = item.get("image_url")
            if isinstance(image_value, dict):
                image_value = image_value.get("url")
            converted.append({"type": "input_image", "image_url": str(image_value or "")})
            continue
        converted.append(dict(item))
    return [{"role": "user", "content": converted}]


def _responses_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return str(output_text)
    fragments: list[str] = []
    for output_item in getattr(response, "output", None) or []:
        for content_item in getattr(output_item, "content", None) or []:
            text = getattr(content_item, "text", None)
            if text:
                fragments.append(str(text))
    return "".join(fragments)
