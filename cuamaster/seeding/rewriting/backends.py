"""Model backends for counterfactual action rewriting."""
from __future__ import annotations

import base64
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from cuamaster.endpoints import DynamicEndpointPool, is_endpoint_failure, normalize_base_urls
from cuamaster.openai_api import create_text_response, normalize_wire_api

from .schema import RewriteSpan, RewriteTool


class ActionRewriteBackend(Protocol):
    model: str

    def propose_spans(self, trajectory: dict[str, Any], tools: list[RewriteTool]) -> list[dict[str, Any]]:
        """Return candidate GUI ranges suitable for CLI or tool execution."""

    def rewrite_span(
        self,
        trajectory: dict[str, Any],
        span: RewriteSpan,
        source_steps: list[dict[str, Any]],
        tools: list[RewriteTool],
        image_paths: list[Path],
    ) -> dict[str, Any]:
        """Return one exact canonical action replacing the source steps."""


class OpenAIRewriteBackend:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        base_urls: list[str] | None = None,
        endpoint_pool: str | Path | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        wire_api: str = "chat_completions",
        store: bool = False,
        trust_env: bool = True,
        reasoning_effort: str = "none",
        max_tokens: int = 4096,
        max_retries: int = 3,
        request_delay: float = 0.0,
    ) -> None:
        try:
            import httpx
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"openai package is required for action rewriting: {exc}") from exc

        configured_urls = list(normalize_base_urls(base_urls))
        if base_url:
            configured_urls = list(normalize_base_urls([base_url]))
        if not configured_urls:
            configured_urls = [
                (
                    os.environ.get("ACTION_REWRITE_BASE_URL")
                    or os.environ.get("ACTION_CONSTRUCTION_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL")
                    or "https://api.openai.com/v1"
                ).rstrip("/")
            ]
        self._endpoint_pool = DynamicEndpointPool(
            config_path=endpoint_pool,
            base_urls=configured_urls,
            model=model,
        )
        configured_urls = list(self._endpoint_pool.base_urls)
        self.model = model
        self.wire_api = normalize_wire_api(wire_api)
        self.store = bool(store)
        self.trust_env = bool(trust_env)
        self.base_urls = configured_urls
        self.api_key = (
            api_key
            or (os.environ.get(api_key_env) if api_key_env else None)
            or os.environ.get("ACTION_REWRITE_API_KEY")
            or os.environ.get("ACTION_CONSTRUCTION_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key and all(_allows_empty_api_key(url) for url in configured_urls):
            self.api_key = "EMPTY"
        if not self.api_key:
            raise ValueError("missing API key; set ACTION_REWRITE_API_KEY or OPENAI_API_KEY")
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_delay = request_delay
        self._openai_class = OpenAI
        self._httpx_class = httpx.Client
        self._clients: dict[str, Any] = {}
        self._client_lock = threading.Lock()

    def propose_spans(self, trajectory: dict[str, Any], tools: list[RewriteTool]) -> list[dict[str, Any]]:
        prompt = _build_proposal_prompt(trajectory, tools)
        payload = self._chat_json([{"type": "text", "text": prompt}])
        spans = payload.get("spans")
        return spans if isinstance(spans, list) else []

    def rewrite_span(
        self,
        trajectory: dict[str, Any],
        span: RewriteSpan,
        source_steps: list[dict[str, Any]],
        tools: list[RewriteTool],
        image_paths: list[Path],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": _build_rewrite_prompt(trajectory, span, source_steps, tools)}
        ]
        labels = ["state before the GUI span", "state after the GUI span"]
        for label, path in zip(labels, image_paths):
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
        return self._chat_json(content)

    def _chat_json(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            client, base_url = self._next_client()
            try:
                wire_api = getattr(self, "wire_api", "chat_completions")
                send_reasoning = wire_api == "responses" or _should_send_reasoning_effort(
                    self.reasoning_effort, base_url, self.model
                )
                response = create_text_response(
                    client,
                    wire_api=wire_api,
                    model=self.model,
                    content=content,
                    max_output_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort if send_reasoning else None,
                    store=getattr(self, "store", False),
                    chat_token_param=_completion_token_param(base_url, self.model),
                    chat_response_format=(
                        {"type": "json_object"}
                        if _supports_response_format(base_url)
                        else None
                    ),
                )
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                raw = response.text
                result = _parse_json_object(raw)
                self._endpoint_pool.mark_success(base_url)
                return result
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if is_endpoint_failure(exc):
                    self._endpoint_pool.mark_failure(base_url, exc)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.25))
        raise RuntimeError(f"action rewrite API request failed: {last_error}") from last_error

    def _next_client(self):
        base_url = self._endpoint_pool.acquire()
        self.base_urls = list(self._endpoint_pool.base_urls)
        with self._client_lock:
            client = self._clients.get(base_url)
            if client is None:
                kwargs: dict[str, Any] = {"api_key": self.api_key, "base_url": base_url}
                trust_env = getattr(self, "trust_env", True)
                if not trust_env or _should_bypass_proxy(base_url):
                    kwargs["http_client"] = self._httpx_class(
                        trust_env=trust_env and not _should_bypass_proxy(base_url)
                    )
                client = self._openai_class(**kwargs)
                self._clients[base_url] = client
        return client, base_url


def _build_proposal_prompt(trajectory: dict[str, Any], tools: list[RewriteTool]) -> str:
    compact_steps = [
        {
            "index": step.get("index"),
            "start": step.get("start"),
            "end": step.get("end"),
            "intent": step.get("intent"),
            "action": step.get("action"),
            "evidence": _compact_evidence(step.get("evidence")),
        }
        for step in trajectory.get("steps") or []
        if isinstance(step, dict)
    ]
    tool_payload = [tool.to_dict() for tool in tools]
    return "\n".join(
        [
            "You label counterfactual execution alternatives for a computer-use training trajectory.",
            "Find only contiguous ranges made entirely of computer.* actions that can be replaced by one exact shell.exec or one registered tool.call while reaching the same task-relevant environment state and preserving prerequisites for later steps.",
            "The source observations are retained as video provenance and are not regenerated, so the replacement does not need to produce the same visible UI or terminal rendering.",
            "Propose a rewrite only when it is expected to improve execution success/reliability, reduce action count or execution time, or both.",
            "Be conservative. Do not rewrite visual editing, drawing, layout, drag-and-drop, ambiguous navigation, authentication, CAPTCHAs, or operations requiring hidden identifiers or unknown file paths.",
            "Never propose passive computer.wait ranges. Replacing a wait with shell sleep is not a meaningful CLI alternative.",
            "A CLI candidate is valid only when a complete executable command can be derived without placeholders.",
            "A tool candidate must use one of the registered tools below and its arguments must be fully derivable.",
            "Ranges must not overlap. Prefer the smallest self-contained range. A range may contain one or more steps.",
            "Set improves_success_rate=true only when CLI/Tool removes coordinate sensitivity, UI timing, focus, menu navigation, or another likely failure source.",
            "Set improves_efficiency=true only when the replacement reduces actions, interaction time, or unnecessary application navigation.",
            "Return strict JSON only: {\"spans\":[{\"start_index\":1,\"end_index\":3,\"target_type\":\"cli|tool\",\"tool_name\":null,\"confidence\":0.0,\"reason\":\"goal-state equivalence explanation\",\"improves_success_rate\":true,\"improves_efficiency\":true,\"benefit_reason\":\"why this is more reliable or efficient\"}]}",
            f"Task metadata: {json.dumps(_trajectory_metadata(trajectory), ensure_ascii=False)}",
            f"Registered tools: {json.dumps(tool_payload, ensure_ascii=False)}",
            f"Steps: {json.dumps(compact_steps, ensure_ascii=False)}",
        ]
    )


def _build_rewrite_prompt(
    trajectory: dict[str, Any],
    span: RewriteSpan,
    source_steps: list[dict[str, Any]],
    tools: list[RewriteTool],
) -> str:
    allowed_tools = [tool.to_dict() for tool in tools if span.target_type == "tool" and tool.name == span.tool_name]
    action_shape = (
        '{"type":"shell.exec","args":{"command":"complete command","cwd":"optional","timeout_sec":30}}'
        if span.target_type == "cli"
        else '{"type":"tool.call","args":{"name":"registered_name","args":{}}}'
    )
    return "\n".join(
        [
            "Rewrite the supplied GUI action range as one counterfactual canonical action.",
            "The replacement must produce the same task-relevant environment state as the full source range and preserve all prerequisites needed by later steps.",
            "The original screenshots remain attached as source-video observations and are not regenerated; visible UI equivalence is not required.",
            "The exact replacement must retain the success-rate or efficiency benefit claimed in the proposed range.",
            "Reject the rewrite when any exact command, file path, identifier, argument, or tool input is unknown.",
            "Do not return sleep, timeout, Start-Sleep, or another command whose only effect is waiting.",
            "Never use placeholders, ellipses, comments describing missing values, shell pseudocode, or invented tools.",
            f"The action must have this shape: {action_shape}",
            "Return strict JSON only: {\"equivalent\":true,\"action\":{...},\"intent\":\"atomic intent\",\"confidence\":0.0,\"reason\":\"why it reaches the same task state more reliably or efficiently\"}",
            f"Task metadata: {json.dumps(_trajectory_metadata(trajectory), ensure_ascii=False)}",
            f"Proposed range: {json.dumps(span.to_dict(), ensure_ascii=False)}",
            f"Allowed tools: {json.dumps(allowed_tools, ensure_ascii=False)}",
            f"Source steps: {json.dumps(source_steps, ensure_ascii=False)}",
        ]
    )


def _trajectory_metadata(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        key: trajectory.get(key)
        for key in ("video_id", "application", "category", "subcategory", "task", "source_video")
    }


def _compact_evidence(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if key in {"transcript", "api_boundary", "target", "text"}}


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload


def _allows_empty_api_key(base_url: str) -> bool:
    return any(marker in base_url.lower() for marker in ("localhost", "127.0.0.1", "0.0.0.0", "10.", "192.168."))


def _should_bypass_proxy(base_url: str) -> bool:
    return _allows_empty_api_key(base_url)


def _supports_response_format(base_url: str) -> bool:
    lowered = base_url.lower()
    return "api.openai.com" in lowered or "openrouter.ai" in lowered


def _is_openai_first_party(base_url: str) -> bool:
    return "api.openai.com" in base_url.lower()


def _completion_token_param(base_url: str, model: str) -> str:
    if _is_openai_first_party(base_url) or model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _should_send_reasoning_effort(reasoning_effort: str | None, base_url: str, model: str) -> bool:
    value = str(reasoning_effort or "").strip().lower()
    if not value or value in {"none", "off", "false", "0"}:
        return False
    return _is_openai_first_party(base_url) or model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
