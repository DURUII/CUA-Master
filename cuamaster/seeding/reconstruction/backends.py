"""Backend interface for API-assisted action labeling and verification."""
from __future__ import annotations

import base64
import hashlib
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

from .schema import ActionCandidate, VideoTask


ALLOWED_ACTION_TYPES = frozenset(
    {
        "computer.click",
        "computer.double_click",
        "computer.move",
        "computer.drag",
        "computer.scroll",
        "computer.type",
        "computer.press",
        "computer.wait",
        "shell.exec",
        "tool.call",
    }
)
ALLOWED_ACTION_TYPES_TEXT = ", ".join(sorted(ALLOWED_ACTION_TYPES))


class ActionLabelBackend(Protocol):
    """Refine action candidates and verify assembled steps."""

    def plan_video(
        self,
        task: VideoTask,
        transcript: list[dict[str, Any]],
        overview_frames: list[Path],
    ) -> dict:
        """Return video-level task/range planning metadata."""

    def refine_candidates(self, task: VideoTask, candidates: list[ActionCandidate]) -> list[ActionCandidate]:
        """Return action-labeled, grounded candidates for one video."""

    def refine_boundaries(
        self,
        task: VideoTask,
        transcript: list[dict[str, Any]],
        candidates: list[ActionCandidate],
        frames: list[Path],
        video_plan: dict,
    ) -> tuple[list[ActionCandidate], dict]:
        """Return API-refined action boundaries and diagnostics."""

    def select_trajectory(
        self,
        task: VideoTask,
        candidates: list[ActionCandidate],
        video_plan: dict,
        boundary_info: dict,
    ) -> tuple[list[ActionCandidate], dict]:
        """Return final selected candidates and selection diagnostics."""

    def refine_trajectory(self, task: VideoTask, trajectory: dict) -> dict:
        """Return a screenshot-grounded trajectory."""

    def verify_trajectory(self, task: VideoTask, trajectory: dict) -> dict:
        """Return trajectory-level verification metadata."""


class PassthroughBackend:
    """Deterministic backend used before a SOTA model API is connected."""

    def plan_video(
        self,
        task: VideoTask,
        transcript: list[dict[str, Any]],
        overview_frames: list[Path],
    ) -> dict:
        return {"enabled": False, "backend": "passthrough"}

    def refine_candidates(self, task: VideoTask, candidates: list[ActionCandidate]) -> list[ActionCandidate]:
        return candidates

    def refine_boundaries(
        self,
        task: VideoTask,
        transcript: list[dict[str, Any]],
        candidates: list[ActionCandidate],
        frames: list[Path],
        video_plan: dict,
    ) -> tuple[list[ActionCandidate], dict]:
        return candidates, {"enabled": False, "backend": "passthrough"}

    def select_trajectory(
        self,
        task: VideoTask,
        candidates: list[ActionCandidate],
        video_plan: dict,
        boundary_info: dict,
    ) -> tuple[list[ActionCandidate], dict]:
        return candidates, {"enabled": False, "backend": "passthrough", "selected_count": len(candidates)}

    def refine_trajectory(self, task: VideoTask, trajectory: dict) -> dict:
        return trajectory

    def verify_trajectory(self, task: VideoTask, trajectory: dict) -> dict:
        return {"verified": False, "backend": "passthrough", "reason": "no API backend configured"}


class OpenAIChatBackend:
    """OpenAI-compatible chat backend for screenshot-grounded step labeling.

    This follows the simple WeaveBench GPT-5.5 style: ``OpenAI(api_key,
    base_url).chat.completions.create(..., reasoning_effort=...)`` with image
    content parts. It works with OpenAI-compatible gateways such as
    OpenRouter when the endpoint supports multimodal chat completions.
    """

    def __init__(
        self,
        *,
        model: str = "GPT-5.5",
        base_url: str | None = None,
        base_urls: list[str] | None = None,
        endpoint_pool: str | Path | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        wire_api: str = "chat_completions",
        store: bool = False,
        trust_env: bool = True,
        reasoning_effort: str = "none",
        plan_reasoning_effort: str | None = None,
        boundary_reasoning_effort: str | None = None,
        selection_reasoning_effort: str | None = None,
        step_reasoning_effort: str | None = None,
        verify_reasoning_effort: str | None = None,
        max_tokens: int = 2048,
        plan_max_tokens: int = 4096,
        max_retries: int = 3,
        request_delay: float = 0.0,
        step_batch_size: int = 1,
        step_batch_fallback: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        try:
            from openai import OpenAI
            import httpx
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(f"openai package is required for openai_chat backend: {exc}") from exc

        self.model = model
        self.wire_api = normalize_wire_api(wire_api)
        self.store = bool(store)
        self.trust_env = bool(trust_env)
        configured_base_urls = _normalize_base_urls(base_urls)
        if base_url:
            configured_base_urls = _normalize_base_urls([base_url])
        if not configured_base_urls:
            configured_base_urls = [
                os.environ.get("ACTION_CONSTRUCTION_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENROUTER_BASE_URL")
                or "https://your-model-endpoint.example.com/v1"
            ]
        self._endpoint_pool = DynamicEndpointPool(
            config_path=endpoint_pool,
            base_urls=configured_base_urls,
            model=model,
        )
        configured_base_urls = list(self._endpoint_pool.base_urls)
        self.base_urls = configured_base_urls
        self.base_url = self.base_urls[0]
        self.api_key = (
            api_key
            or (os.environ.get(api_key_env) if api_key_env else None)
            or os.environ.get("ACTION_CONSTRUCTION_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if not self.api_key and all(_allows_empty_api_key(url) for url in self.base_urls):
            self.api_key = "EMPTY"
        if not self.api_key:
            raise ValueError(
                "missing API key for openai_chat backend; set ACTION_CONSTRUCTION_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY"
            )
        self.reasoning_effort = reasoning_effort
        self.stage_reasoning_effort = {
            "plan": plan_reasoning_effort if plan_reasoning_effort is not None else reasoning_effort,
            "boundary": boundary_reasoning_effort if boundary_reasoning_effort is not None else reasoning_effort,
            "selection": selection_reasoning_effort if selection_reasoning_effort is not None else reasoning_effort,
            "step": step_reasoning_effort if step_reasoning_effort is not None else reasoning_effort,
            "verify": verify_reasoning_effort if verify_reasoning_effort is not None else reasoning_effort,
        }
        self.max_tokens = max_tokens
        self.plan_max_tokens = plan_max_tokens
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.step_batch_size = max(1, int(step_batch_size or 1))
        self.step_batch_fallback = bool(step_batch_fallback)
        self.cache_dir = output_dir / ".api_cache" if output_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._openai_class = OpenAI
        self._httpx_class = httpx.Client
        self._clients: dict[str, Any] = {}
        self._client_lock = threading.Lock()

    def plan_video(
        self,
        task: VideoTask,
        transcript: list[dict[str, Any]],
        overview_frames: list[Path],
    ) -> dict:
        content: list[dict[str, Any]] = []
        for index, frame in enumerate(overview_frames, 1):
            if frame.exists():
                content.append({"type": "text", "text": f"Overview frame {index}."})
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(frame)}})
        content.append({"type": "text", "text": _build_video_plan_prompt(task, transcript)})
        payload = self._chat_json(content, max_tokens=self.plan_max_tokens, cache_stage="plan", cache_key=task.record.get("video_id"))
        payload.setdefault("enabled", True)
        payload.setdefault("backend", "openai_chat")
        payload.setdefault("model", self.model)
        return payload

    def refine_candidates(self, task: VideoTask, candidates: list[ActionCandidate]) -> list[ActionCandidate]:
        return candidates

    def refine_boundaries(
        self,
        task: VideoTask,
        transcript: list[dict[str, Any]],
        candidates: list[ActionCandidate],
        frames: list[Path],
        video_plan: dict,
    ) -> tuple[list[ActionCandidate], dict]:
        if not candidates and not transcript:
            return candidates, {"enabled": True, "backend": "openai_chat", "reason": "no candidates or transcript"}
        content: list[dict[str, Any]] = []
        for index, frame in enumerate(frames, 1):
            if frame.exists():
                content.append({"type": "text", "text": f"Boundary frame {index}."})
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(frame)}})
        content.append({"type": "text", "text": _build_boundary_prompt(task, transcript, candidates, video_plan)})
        payload = self._chat_json(content, max_tokens=self.plan_max_tokens, cache_stage="boundary", cache_key=task.record.get("video_id"))
        refined = _candidates_from_boundary_payload(candidates, payload)
        info = {
            "enabled": True,
            "backend": "openai_chat",
            "model": self.model,
            "raw": payload,
            "input_candidates": len(candidates),
            "output_candidates": len(refined),
        }
        return refined, info

    def select_trajectory(
        self,
        task: VideoTask,
        candidates: list[ActionCandidate],
        video_plan: dict,
        boundary_info: dict,
    ) -> tuple[list[ActionCandidate], dict]:
        if not candidates:
            return [], {"enabled": True, "backend": "openai_chat", "selected_count": 0}
        content = [{"type": "text", "text": _build_selection_prompt(task, candidates, video_plan, boundary_info)}]
        payload = self._chat_json(content, max_tokens=self.plan_max_tokens, cache_stage="selection", cache_key=task.record.get("video_id"))
        selected, info = _select_candidates_from_payload(candidates, payload)
        info.update({"enabled": True, "backend": "openai_chat", "model": self.model, "raw": payload})
        return selected, info

    def refine_trajectory(self, task: VideoTask, trajectory: dict) -> dict:
        steps = trajectory.get("steps") or []
        if self.step_batch_size > 1:
            return self._refine_trajectory_batched(task, trajectory, steps)

        refined_steps = 0
        failed_steps = 0
        for step in steps:
            observation = step.get("observation") or {}
            post_observation = step.get("post_observation") or {}
            image_paths = [observation.get("screenshot"), post_observation.get("screenshot")]
            if not all(path and Path(path).exists() for path in image_paths):
                step.setdefault("api_refinement", {})["error"] = "missing_step_screenshots"
                failed_steps += 1
                continue
            try:
                payload = self._refine_step(task, trajectory, step, [Path(p) for p in image_paths])
                _apply_step_refinement(step, payload)
                refined_steps += 1
            except Exception as exc:
                step.setdefault("api_refinement", {})["error"] = f"{type(exc).__name__}: {exc}"
                failed_steps += 1
            if self.request_delay > 0:
                time.sleep(self.request_delay)
        trajectory.setdefault("quality", {})["api_refined_steps"] = refined_steps
        trajectory.setdefault("quality", {})["api_failed_steps"] = failed_steps
        trajectory.setdefault("quality", {})["api_backend"] = "openai_chat"
        trajectory.setdefault("quality", {})["api_model"] = self.model
        trajectory.setdefault("quality", {})["needs_grounding_steps"] = sum(
            1 for step in steps if step.get("needs_grounding")
        )
        return trajectory

    def _refine_trajectory_batched(self, task: VideoTask, trajectory: dict, steps: list[dict]) -> dict:
        refined_steps = 0
        failed_steps = 0
        fallback_steps = 0
        batch_errors = 0
        batch_size = max(1, self.step_batch_size)

        for start in range(0, len(steps), batch_size):
            chunk = steps[start : start + batch_size]
            try:
                payload = self._refine_step_batch(task, trajectory, chunk)
                by_index = _step_batch_payload_by_index(payload)
            except Exception as exc:
                batch_errors += 1
                by_index = {}
                for step in chunk:
                    step.setdefault("api_refinement", {})["batch_error"] = f"{type(exc).__name__}: {exc}"

            for step in chunk:
                result = by_index.get(int(step.get("index") or -1))
                if result and not _step_batch_payload_needs_fallback(step, result):
                    _apply_step_refinement(step, result)
                    refined_steps += 1
                    continue
                if result:
                    step.setdefault("api_refinement", {})["batch_result"] = result
                if self.step_batch_fallback:
                    fallback_steps += 1
                    ok = self._refine_step_with_status(task, trajectory, step)
                    refined_steps += 1 if ok else 0
                    failed_steps += 0 if ok else 1
                else:
                    step.setdefault("api_refinement", {})["error"] = "missing_or_low_quality_batch_result"
                    failed_steps += 1
                if self.request_delay > 0:
                    time.sleep(self.request_delay)

            if self.request_delay > 0:
                time.sleep(self.request_delay)

        trajectory.setdefault("quality", {})["api_refined_steps"] = refined_steps
        trajectory.setdefault("quality", {})["api_failed_steps"] = failed_steps
        trajectory.setdefault("quality", {})["api_batch_errors"] = batch_errors
        trajectory.setdefault("quality", {})["api_batch_fallback_steps"] = fallback_steps
        trajectory.setdefault("quality", {})["api_step_batch_size"] = batch_size
        trajectory.setdefault("quality", {})["api_backend"] = "openai_chat"
        trajectory.setdefault("quality", {})["api_model"] = self.model
        trajectory.setdefault("quality", {})["needs_grounding_steps"] = sum(
            1 for step in steps if step.get("needs_grounding")
        )
        return trajectory

    def _refine_step_with_status(self, task: VideoTask, trajectory: dict, step: dict) -> bool:
        observation = step.get("observation") or {}
        post_observation = step.get("post_observation") or {}
        image_paths = [observation.get("screenshot"), post_observation.get("screenshot")]
        if not all(path and Path(path).exists() for path in image_paths):
            step.setdefault("api_refinement", {})["error"] = "missing_step_screenshots"
            return False
        try:
            payload = self._refine_step(task, trajectory, step, [Path(p) for p in image_paths])
            _apply_step_refinement(step, payload)
            return True
        except Exception as exc:
            step.setdefault("api_refinement", {})["error"] = f"{type(exc).__name__}: {exc}"
            return False

    def verify_trajectory(self, task: VideoTask, trajectory: dict) -> dict:
        if trajectory.get("_api_whole_verify"):
            return self._verify_trajectory_with_api(task, trajectory)
        steps = trajectory.get("steps") or []
        if not steps:
            return {"verified": False, "backend": "openai_chat", "model": self.model, "reason": "no steps"}
        refined = sum(1 for step in steps if (step.get("api_refinement") or {}).get("is_action") is True)
        invalid = sum(1 for step in steps if (step.get("api_refinement") or {}).get("is_action") is False)
        return {
            "verified": refined > 0 and invalid == 0,
            "backend": "openai_chat",
            "model": self.model,
            "refined_steps": refined,
            "rejected_steps": invalid,
            "reason": "step-level screenshot grounding completed" if refined else "no step accepted by API",
        }

    def _refine_step(self, task: VideoTask, trajectory: dict, step: dict, image_paths: list[Path]) -> dict:
        prompt = _build_step_prompt(task, trajectory, step)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Before-state screenshot: the initial state or the previous action's after-state."},
            {"type": "image_url", "image_url": {"url": _image_data_url(image_paths[0])}},
            {"type": "text", "text": "After-state screenshot for this action."},
            {"type": "image_url", "image_url": {"url": _image_data_url(image_paths[1])}},
            {"type": "text", "text": prompt},
        ]
        return self._chat_json(
            content,
            max_tokens=self.max_tokens,
            cache_stage="step",
            cache_key=f"{task.record.get('video_id')}_{step.get('index')}",
        )

    def _refine_step_batch(self, task: VideoTask, trajectory: dict, steps: list[dict]) -> dict:
        if not steps:
            return {"steps": []}
        image_paths = _state_chain_image_paths(steps)
        if not image_paths:
            raise ValueError("missing_step_screenshots")
        content: list[dict[str, Any]] = []
        for index, path in enumerate(image_paths):
            content.append({"type": "text", "text": f"State frame {index}."})
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
        content.append({"type": "text", "text": _build_step_batch_prompt(task, trajectory, steps, len(image_paths))})
        first = steps[0].get("index")
        last = steps[-1].get("index")
        return self._chat_json(
            content,
            max_tokens=max(self.max_tokens, min(self.plan_max_tokens, self.max_tokens * max(1, len(steps)))),
            cache_stage="step_batch",
            cache_key=f"{task.record.get('video_id')}_{first}_{last}",
        )

    def _verify_trajectory_with_api(self, task: VideoTask, trajectory: dict) -> dict:
        steps = trajectory.get("steps") or []
        if not steps:
            return {"verified": False, "backend": "openai_chat", "model": self.model, "reason": "no steps"}
        content: list[dict[str, Any]] = []
        image_count = 0
        for step in steps[:8]:
            for field in ("observation", "post_observation"):
                path = (step.get(field) or {}).get("screenshot")
                if path and Path(path).exists():
                    content.append({"type": "text", "text": f"Step {step.get('index')} {field}."})
                    content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path))}})
                    image_count += 1
        content.append({"type": "text", "text": _build_verify_prompt(task, trajectory, image_count)})
        payload = self._chat_json(content, max_tokens=self.plan_max_tokens, cache_stage="verify", cache_key=task.record.get("video_id"))
        return {
            "verified": bool(payload.get("trajectory_valid")) and bool(payload.get("exportable_for_training")),
            "backend": "openai_chat",
            "model": self.model,
            **payload,
        }

    def _chat(self, content: list[dict[str, Any]], *, cache_stage: str | None = None, cache_key: Any = None) -> str:
        return self._chat_raw(content, max_tokens=self.max_tokens, cache_stage=cache_stage, cache_key=cache_key)

    def _chat_json(self, content: list[dict[str, Any]], *, max_tokens: int, cache_stage: str | None = None, cache_key: Any = None) -> dict:
        raw = self._chat_raw(
            content,
            max_tokens=max_tokens,
            cache_stage=cache_stage,
            cache_key=cache_key,
            require_json=True,
        )
        return _parse_json_object(raw)

    def _chat_raw(
        self,
        content: list[dict[str, Any]],
        *,
        max_tokens: int,
        cache_stage: str | None = None,
        cache_key: Any = None,
        require_json: bool = False,
    ) -> str:
        cache_path = self._cache_path(cache_stage, cache_key, content, max_tokens)
        if cache_path and cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))["raw"]
                if require_json:
                    _parse_json_object(raw)
                return raw
            except Exception:
                cache_path.unlink(missing_ok=True)
        last_error: Exception | None = None
        attempts = max(1, self.max_retries)
        for attempt in range(attempts):
            try:
                base_url, client = self._next_client()
                reasoning_effort = self.stage_reasoning_effort.get(cache_stage or "") or self.reasoning_effort
                wire_api = getattr(self, "wire_api", "chat_completions")
                send_reasoning = wire_api == "responses" or _should_send_reasoning_effort(
                    reasoning_effort, base_url, self.model
                )
                result = create_text_response(
                    client,
                    wire_api=wire_api,
                    model=self.model,
                    content=content,
                    max_output_tokens=max_tokens,
                    reasoning_effort=reasoning_effort if send_reasoning else None,
                    store=getattr(self, "store", False),
                    chat_token_param=_completion_token_param(base_url, self.model),
                    chat_response_format=(
                        {"type": "json_object"}
                        if _supports_response_format(base_url, self.model)
                        else None
                    ),
                )
                raw = result.text.strip()
                if require_json:
                    _parse_json_object(raw)
                if cache_path:
                    _write_cache(cache_path, {"raw": raw, "stage": cache_stage, "model": self.model, "base_url": base_url})
                if hasattr(self, "_endpoint_pool"):
                    self._endpoint_pool.mark_success(base_url)
                return raw
            except Exception as exc:  # pragma: no cover - network/backend specific
                last_error = exc
                if hasattr(self, "_endpoint_pool") and is_endpoint_failure(exc):
                    self._endpoint_pool.mark_failure(base_url, exc)
                if attempt >= attempts - 1:
                    break
                time.sleep(_backoff_delay(attempt))
        raise RuntimeError(f"OpenAI-compatible chat request failed: {last_error}") from last_error

    def _next_client(self):
        if not hasattr(self, "_endpoint_pool"):
            with self._base_url_lock:
                base_url = next(self._base_url_cycle)
            return base_url, self._clients[base_url]
        base_url = self._endpoint_pool.acquire()
        self.base_urls = list(self._endpoint_pool.base_urls)
        with self._client_lock:
            client = self._clients.get(base_url)
            if client is None:
                client_kwargs: dict[str, Any] = {
                    "api_key": self.api_key,
                    "base_url": base_url,
                }
                trust_env = getattr(self, "trust_env", True)
                if not trust_env or _should_bypass_proxy(base_url):
                    client_kwargs["http_client"] = self._httpx_class(
                        trust_env=trust_env and not _should_bypass_proxy(base_url)
                    )
                client = self._openai_class(**client_kwargs)
                self._clients[base_url] = client
        return base_url, client

    def _cache_path(self, stage: str | None, cache_key: Any, content: list[dict[str, Any]], max_tokens: int) -> Path | None:
        if not self.cache_dir or not stage:
            return None
        digest = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model,
                    "base_urls": list(self._endpoint_pool.base_urls)
                    if hasattr(self, "_endpoint_pool")
                    else self.base_urls,
                    "stage": stage,
                    "key": str(cache_key),
                    "max_tokens": max_tokens,
                    "content": content,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        return self.cache_dir / stage / f"{cache_key or 'unknown'}_{digest}.json"


def make_backend(
    name: str | None,
    *,
    output_dir: Path | None = None,
    model: str | None = None,
    base_url: str | None = None,
    base_urls: list[str] | None = None,
    endpoint_pool: str | Path | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    wire_api: str = "chat_completions",
    store: bool = False,
    trust_env: bool = True,
    reasoning_effort: str = "none",
    plan_reasoning_effort: str | None = None,
    boundary_reasoning_effort: str | None = None,
    selection_reasoning_effort: str | None = None,
    step_reasoning_effort: str | None = None,
    verify_reasoning_effort: str | None = None,
    max_tokens: int = 2048,
    plan_max_tokens: int = 4096,
    max_retries: int = 3,
    request_delay: float = 0.0,
    step_batch_size: int = 1,
    step_batch_fallback: bool = True,
) -> ActionLabelBackend:
    if name in {None, "none", "passthrough"}:
        return PassthroughBackend()
    if name in {"openai_chat", "gpt55", "gpt5.5"}:
        return OpenAIChatBackend(
            model=model or os.environ.get("ACTION_CONSTRUCTION_MODEL") or "GPT-5.5",
            base_url=base_url,
            base_urls=base_urls,
            endpoint_pool=endpoint_pool,
            api_key=api_key,
            api_key_env=api_key_env,
            wire_api=wire_api,
            store=store,
            trust_env=trust_env,
            reasoning_effort=reasoning_effort,
            plan_reasoning_effort=plan_reasoning_effort,
            boundary_reasoning_effort=boundary_reasoning_effort,
            selection_reasoning_effort=selection_reasoning_effort,
            step_reasoning_effort=step_reasoning_effort,
            verify_reasoning_effort=verify_reasoning_effort,
            max_tokens=max_tokens,
            plan_max_tokens=plan_max_tokens,
            max_retries=max_retries,
            request_delay=request_delay,
            step_batch_size=step_batch_size,
            step_batch_fallback=step_batch_fallback,
            output_dir=output_dir,
        )
    raise ValueError(f"unsupported action construction backend: {name}")


def _build_step_prompt(task: VideoTask, trajectory: dict, step: dict) -> str:
    evidence = step.get("evidence") or {}
    boundary = evidence.get("api_boundary") if isinstance(evidence, dict) else None
    expected_action_type = (boundary or {}).get("action_type") or (step.get("action") or {}).get("type")
    expected_intent = (boundary or {}).get("intent") or step.get("intent")
    expected_target = (boundary or {}).get("target") or evidence.get("target") or ""
    return "\n".join(
        [
            "You are labeling one atomic CUA action from a tutorial video for Computer-Use Agent training.",
            "Use the before/after screenshots plus transcript evidence to decide the action and parameters.",
            "Screenshots form a state chain: before is the initial state or previous action's after-state; after is the state after this action.",
            "The boundary stage already chose the expected action type, intent, and target. Preserve that action type unless the screenshots make it clearly impossible.",
            "If you change the expected action type, include changed_action_type_reason with concrete visual evidence. Do not convert a type action into a click unless the action is only focusing an input field.",
            "Coordinates must be normalized to 0-1000 screen space, x from left to right and y from top to bottom.",
            "Use shell.exec when the evidence shows one complete terminal command being submitted or executed; include the exact command and optional cwd/timeout_sec. Mere text entry without command submission remains computer.type.",
            "Use tool.call only when the evidence unambiguously identifies a registered structured tool invocation; include name and args.",
            "If the candidate is not a real CUA action, set is_action=false and keep action=null.",
            f"Allowed action types: {ALLOWED_ACTION_TYPES_TEXT}.",
            "For click/double_click/move, include x and y. For drag include x1,y1,x2,y2. For scroll include dx,dy and optional x,y. For type include text. For press include keys. For wait include seconds. For shell.exec include command and optional cwd/timeout_sec. For tool.call include name and args.",
            "Return ONLY a JSON object with this shape:",
            '{"is_action":true,"action":{"type":"computer.click","args":{"x":500,"y":500,"button":"left"}},"intent":"...","confidence":0.0,"target":"...","reason":"...","changed_action_type_reason":""}',
            "",
            f"Video id: {task.record.get('video_id')}",
            f"Application: {task.record.get('application')}",
            f"Task: {trajectory.get('task')}",
            f"Step index: {step.get('index')}",
            f"Time window: {step.get('start')}s to {step.get('end')}s",
            f"Expected action type: {expected_action_type}",
            f"Expected intent: {expected_intent}",
            f"Expected target: {expected_target}",
            f"Draft intent: {step.get('intent')}",
            f"Draft action: {json.dumps(step.get('action'), ensure_ascii=False)}",
            f"Transcript evidence: {evidence.get('transcript') or evidence.get('text') or ''}",
            f"Target phrase: {evidence.get('target') or ''}",
        ]
    )


def _build_step_batch_prompt(task: VideoTask, trajectory: dict, steps: list[dict], image_count: int) -> str:
    rows = []
    for offset, step in enumerate(steps, 1):
        evidence = step.get("evidence") or {}
        boundary = evidence.get("api_boundary") if isinstance(evidence, dict) else None
        rows.append(
            {
                "index": step.get("index"),
                "state_transition": f"state_{offset - 1} -> state_{offset}",
                "start": step.get("start"),
                "end": step.get("end"),
                "expected_action_type": (boundary or {}).get("action_type") or (step.get("action") or {}).get("type"),
                "expected_intent": (boundary or {}).get("intent") or step.get("intent"),
                "expected_target": (boundary or {}).get("target") or evidence.get("target") or "",
                "draft_action": step.get("action"),
                "transcript_evidence": evidence.get("transcript") or evidence.get("text") or "",
                "target_phrase": evidence.get("target") or "",
            }
        )
    return "\n".join(
        [
            "You are labeling a short contiguous chain of atomic CUA actions from a tutorial video for Computer-Use Agent training.",
            "Use the state frames and step drafts to label each action. Step i is the transition from state frame i-1 to state frame i in the provided sequence.",
            "Use the neighboring transitions as context, but return one independent action object for every requested step index.",
            "Preserve the expected action type unless the screenshots make it clearly impossible. If changed, include changed_action_type_reason with concrete visual evidence.",
            "Coordinates must be normalized to 0-1000 screen space, x from left to right and y from top to bottom.",
            "Use shell.exec only for a complete submitted/executed terminal command, not for unsubmitted terminal text. Use tool.call only for an explicit registered structured tool invocation.",
            "If a requested transition is not a real CUA action, set is_action=false and action=null for that step.",
            f"Allowed action types: {ALLOWED_ACTION_TYPES_TEXT}.",
            "For click/double_click/move, include x and y. For drag include x1,y1,x2,y2. For scroll include dx,dy and optional x,y. For type include text. For press include keys. For wait include seconds. For shell.exec include command and optional cwd/timeout_sec. For tool.call include name and args.",
            "Return ONLY a JSON object with this shape:",
            '{"steps":[{"index":1,"is_action":true,"action":{"type":"computer.click","args":{"x":500,"y":500,"button":"left"}},"intent":"...","confidence":0.0,"target":"...","reason":"...","changed_action_type_reason":""}]}',
            "",
            f"Video id: {task.record.get('video_id')}",
            f"Application: {task.record.get('application')}",
            f"Task: {trajectory.get('task')}",
            f"State frames provided: {image_count}",
            f"Requested steps: {json.dumps(rows, ensure_ascii=False)}",
        ]
    )


def _build_video_plan_prompt(task: VideoTask, transcript: list[dict[str, Any]]) -> str:
    transcript_text = _compact_transcript(transcript, max_chars=12000)
    return "\n".join(
        [
            "You are planning how to extract Computer-Use Agent trajectories from a tutorial video.",
            "Identify only ranges that contain real software operation, including GUI interaction, terminal commands, and explicit structured tool calls. Exclude intro, outro, marketing, slide-only, talking-head, and explanation-only ranges.",
            "Choose one primary long-horizon episode: the longest single continuous workflow with a clear user goal, visible start state, many intermediate CUA actions, and visible completion or stable end state. Prefer episodes that can yield 40+ atomic actions; require at least 20 visible actions when possible.",
            "Do not combine independent examples, repeated demonstrations, or unrelated tutorial sections into one episode. Prefer the longest coherent workflow over isolated tips.",
            "Also list every other coherent exportable episode in episode_segments so the pipeline can split one tutorial video into multiple trajectories. Each episode must be independently trainable and continuous.",
            "Return ONLY JSON with this shape:",
            '{"task_title":"...","primary_episode":{"start":1.0,"end":60.0,"task":"complete workflow goal","summary":"why this is one coherent episode","confidence":0.0},"episode_segments":[{"start":1.0,"end":60.0,"task":"complete workflow goal","summary":"single coherent workflow","confidence":0.0}],"ignore_ranges":[{"start":0.0,"end":1.0,"reason":"intro"}],"high_value_ranges":[{"start":1.0,"end":10.0,"reason":"CUA operation"}],"confidence":0.0}',
            "",
            f"Video id: {task.record.get('video_id')}",
            f"Application: {task.record.get('application')}",
            f"Taxonomy: {task.record.get('category')} / {task.record.get('subcategory')}",
            "Transcript:",
            transcript_text,
        ]
    )


def _build_boundary_prompt(
    task: VideoTask,
    transcript: list[dict[str, Any]],
    candidates: list[ActionCandidate],
    video_plan: dict,
) -> str:
    candidate_rows = []
    for index, candidate in enumerate(candidates, 1):
        candidate_rows.append(
            {
                "id": f"c{index}",
                "start": round(candidate.start, 3),
                "end": round(candidate.end, 3),
                "source": candidate.source,
                "action_type": candidate.action_type,
                "args": candidate.args,
                "target": candidate.target,
                "text": candidate.text,
                "confidence": candidate.confidence,
            }
        )
    return "\n".join(
        [
            "You are refining action boundaries for CUA tutorial video mining.",
            "Use transcript, sampled frames, and candidates to produce accurate atomic GUI, CLI, and structured-tool actions.",
            "Delete explanation-only candidates, add missing visible CUA actions, split multi-action candidates, and merge duplicated candidates.",
            "Focus on one continuous CUA episode in the video plan. Prefer the longest reproducible workflow that remains state-continuous and visibly verifiable, with a target of 40+ atomic actions and a minimum long-horizon target of 20 actions when the video supports it.",
            "Add visible missing actions that are necessary for state transitions, including menu opens, dialog opens, focus clicks before typing, submit/save clicks, waits for navigation, and confirmation actions.",
            "Represent a complete submitted/executed terminal command as shell.exec with its exact command; do not emit both shell.exec and the lower-level terminal typing/Enter actions for the same transition. Keep unsubmitted terminal text as computer.type. Emit tool.call only for an explicit registered structured invocation with a stable name and args.",
            "When the same workflow has many visible steps, preserve them as atomic actions instead of collapsing the workflow into only the most important actions.",
            "Reject candidates outside the primary episode unless screenshots/transcript show they are required intermediate actions in the same workflow. Do not bridge unexplained jumps with guessed actions.",
            "Return ONLY JSON with this shape:",
            '{"actions":[{"id":"a1","start":1.0,"end":2.0,"action_type":"shell.exec","args":{"command":"pwd"},"target":"terminal","intent":"print working directory","confidence":0.9,"source_ids":["c1"],"evidence":"..."}],"false_positive_candidate_ids":["c2"],"merge_groups":[["c3","c4"]],"missing_actions":[{"start":3.0,"end":4.0,"action_type":"computer.scroll","args":{"dx":0,"dy":-5},"target":"list","intent":"scroll down","confidence":0.8,"evidence":"..."}]}',
            f"Allowed action types: {ALLOWED_ACTION_TYPES_TEXT}.",
            "",
            f"Video id: {task.record.get('video_id')}",
            f"Application: {task.record.get('application')}",
            f"Video plan: {json.dumps(_small_json(video_plan), ensure_ascii=False)}",
            f"Candidates: {json.dumps(candidate_rows, ensure_ascii=False)}",
            "Transcript:",
            _compact_transcript(transcript, max_chars=12000),
        ]
    )


def _build_selection_prompt(
    task: VideoTask,
    candidates: list[ActionCandidate],
    video_plan: dict,
    boundary_info: dict,
) -> str:
    rows = []
    for index, candidate in enumerate(candidates, 1):
        rows.append(
            {
                "id": f"a{index}",
                "start": round(candidate.start, 3),
                "end": round(candidate.end, 3),
                "action_type": candidate.action_type,
                "args": candidate.args,
                "target": candidate.target,
                "intent": candidate.text or candidate.target,
                "confidence": candidate.confidence,
                "source": candidate.source,
            }
        )
    return "\n".join(
        [
            "You are selecting the final action path for a CUA training trajectory.",
            "Select one coherent workflow or self-contained subtask from the primary episode, not disconnected tutorial fragments.",
            "Prefer 40+ highly continuous atomic CUA actions when the video supports it. If 40+ is not reliable, choose the longest reliable 20+ action chain. Use fewer than 20 actions only when no high-quality long-horizon chain exists.",
            "Keep real, atomic, reproducible GUI, CLI, and structured-tool actions in chronological order, including necessary focus, menu, dialog, submit, navigation, terminal command, and wait actions.",
            "Prioritize state continuity and task/step alignment over length. If the full video task is interrupted, crosses to another app/site, or becomes too broad, choose the longest continuous subtask with a visible start and stable end state.",
            "Do not include competitor research, web search, ChatGPT prompting, documentation lookup, or other side workflows in a task about editing/creating an object unless the transition and return are fully visible and essential.",
            "Do not split a coherent visible workflow into a short subtask just because some later actions are lower confidence; keep lower-confidence but necessary visible actions only when they preserve continuity.",
            "Do not truncate the selected workflow before its visible goal is reached unless the remaining actions are not visible or switch context.",
            "If multiple candidate paths are valid, choose the longest path that preserves visible state continuity and a stable end state. Never reward length by itself; reject or shorten chains with state jumps, task drift, hidden intermediate work, or unrelated side workflows.",
            "Remove duplicated demonstrations, explanation-only actions, repeated examples, unrelated app/site switches, and low-confidence guesses that break continuity.",
            "Return ONLY JSON with this shape:",
            '{"selected_action_ids":["a1","a3"],"rejected_action_ids":["a2"],"quality_score":4,"coverage":"complete|partial","reason":"..."}',
            "",
            f"Video id: {task.record.get('video_id')}",
            f"Application: {task.record.get('application')}",
            f"Video plan: {json.dumps(_small_json(video_plan), ensure_ascii=False)}",
            f"Boundary diagnostics: {json.dumps(_small_json(boundary_info), ensure_ascii=False)}",
            f"Candidate actions: {json.dumps(rows, ensure_ascii=False)}",
        ]
    )


def _build_verify_prompt(task: VideoTask, trajectory: dict, image_count: int) -> str:
    steps = []
    for step in trajectory.get("steps") or []:
        steps.append(
            {
                "index": step.get("index"),
                "start": step.get("start"),
                "end": step.get("end"),
                "intent": step.get("intent"),
                "action": step.get("action"),
                "confidence": step.get("confidence"),
                "rejected_by_api": step.get("rejected_by_api", False),
            }
        )
    return "\n".join(
        [
            "You are verifying a CUA trajectory mined from a tutorial video.",
            "Judge whether the selected steps form a coherent, trainable CUA trajectory or self-contained CUA subtask across GUI, CLI, and structured tools. Use screenshots when provided.",
            "Set trajectory_valid=true when the steps are locally continuous, screenshots align with actions, and the final state is stable, even if the clip is only a useful subtask of a broader video task.",
            "For exportable_for_training=true, require a single continuous workflow or complete subtask with no large unexplained state jumps, enough intermediate actions to reproduce that subtask, aligned pre/post screenshots, and a visible completed or stable end state.",
            "Long-horizon training prefers 20+ actions and ideally 40+ actions, but quality is more important than length. Do not mark a long chain exportable if it contains task drift, hidden work, unrelated side workflows, order errors, or unstable end state.",
            "Set exportable_for_training=false when the task title claims a goal that the selected steps do not perform or visibly complete, even if the individual actions are locally plausible.",
            "Set exportable_for_training=false when steps stitch together unrelated apps, sites, tabs, examples, competitor research, search/ChatGPT side workflows, or page contexts without visible navigation and return actions.",
            "Prefer exportable_for_training=true for coherent trajectories with 3 or more valid steps when they form a stable, verifiable subtask. Do not reward length by itself.",
            "Do not reject a short trajectory solely because it does not complete the entire tutorial-level task; reject it only if it is a mere tip, lacks a stable subtask goal, omits essential intermediate actions for the visible subtask, has task/step mismatch, or stitches unrelated examples.",
            "When the trajectory is a valid fragment but not exportable as a complete workflow/subtask, set trajectory_valid=true and exportable_for_training=false, and explain what is missing.",
            "Return ONLY JSON with this shape:",
            '{"trajectory_valid":true,"exportable_for_training":true,"quality_score":4,"step_validity":[{"index":1,"valid":true,"reason":"..."}],"missing_step_ranges":[],"redundant_steps":[],"order_errors":[],"reason":"..."}',
            "",
            f"Video id: {task.record.get('video_id')}",
            f"Application: {task.record.get('application')}",
            f"Task: {trajectory.get('task')}",
            f"Images provided: {image_count}",
            f"Steps: {json.dumps(steps, ensure_ascii=False)}",
        ]
    )


def _apply_step_refinement(step: dict, payload: dict) -> None:
    step["api_refinement"] = payload
    if payload.get("is_action") is False:
        step["rejected_by_api"] = True
        step["confidence"] = min(float(step.get("confidence") or 0.0), float(payload.get("confidence") or 0.0))
        return
    action = payload.get("action")
    if isinstance(action, dict) and action.get("type") and isinstance(action.get("args"), dict):
        if action.get("type") not in ALLOWED_ACTION_TYPES:
            step["rejected_by_api"] = True
            step["needs_review"] = True
            return
        step["action"] = action
        step["needs_grounding"] = _action_needs_grounding(action)
    if payload.get("intent"):
        step["intent"] = str(payload["intent"])
    if payload.get("confidence") is not None:
        step["confidence"] = max(0.0, min(1.0, float(payload["confidence"])))
    if payload.get("target"):
        step.setdefault("evidence", {})["api_target"] = str(payload["target"])
    _annotate_action_type_consistency(step, payload)


def _annotate_action_type_consistency(step: dict, payload: dict) -> None:
    evidence = step.setdefault("evidence", {})
    expected = (evidence.get("api_boundary") or {}).get("action_type") or payload.get("expected_action_type")
    actual = (step.get("action") or {}).get("type")
    if not expected or not actual:
        return
    if expected == actual:
        evidence["action_type_consistent"] = True
        return
    reason = str(payload.get("changed_action_type_reason") or "").strip()
    evidence["action_type_consistent"] = False
    evidence["action_type_mismatch"] = {
        "expected": expected,
        "actual": actual,
        "reason": reason,
    }
    if not reason or _suspicious_action_type_change(expected, actual, reason):
        step["needs_review"] = True


def _suspicious_action_type_change(expected: str, actual: str, reason: str) -> bool:
    if expected == "computer.type" and actual == "computer.click":
        return "focus" not in reason.lower() and "input" not in reason.lower()
    if expected in {"computer.type", "computer.press"} and actual in {"computer.scroll", "computer.wait"}:
        return True
    return False


def _candidates_from_boundary_payload(candidates: list[ActionCandidate], payload: dict) -> list[ActionCandidate]:
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return candidates
    by_id = {f"c{index}": candidate for index, candidate in enumerate(candidates, 1)}
    refined: list[ActionCandidate] = []
    for index, action in enumerate(actions, 1):
        if not isinstance(action, dict):
            continue
        action_type = action.get("action_type")
        start = _safe_float(action.get("start"), None)
        end = _safe_float(action.get("end"), None)
        if not action_type or str(action_type) not in ALLOWED_ACTION_TYPES or start is None or end is None or end <= start:
            continue
        source_ids = [str(item) for item in action.get("source_ids") or []]
        source_candidates = [by_id[item] for item in source_ids if item in by_id]
        evidence = {
            "api_boundary": action,
            "source_ids": source_ids,
        }
        if source_candidates:
            evidence["source_candidates"] = [candidate.to_dict() for candidate in source_candidates]
        action_args = action.get("args") if isinstance(action.get("args"), dict) else None
        if action_args is None:
            action_args = next(
                (
                    dict(candidate.args)
                    for candidate in source_candidates
                    if candidate.action_type == action_type and candidate.args
                ),
                {},
            )
        refined.append(
            ActionCandidate(
                start=start,
                end=end,
                source="api",
                action_type=str(action_type),
                text=str(action.get("intent") or action.get("evidence") or ""),
                target=str(action.get("target") or ""),
                args=dict(action_args),
                semantic_score=max((c.semantic_score for c in source_candidates), default=0.0),
                visual_score=max((c.visual_score for c in source_candidates), default=0.0),
                confidence=max(0.0, min(1.0, _safe_float(action.get("confidence"), 0.7) or 0.7)),
                evidence=evidence,
            )
        )
    return refined or candidates


def _select_candidates_from_payload(candidates: list[ActionCandidate], payload: dict) -> tuple[list[ActionCandidate], dict]:
    selected_ids = payload.get("selected_action_ids")
    if not isinstance(selected_ids, list):
        return candidates, {"selected_count": len(candidates), "reason": "missing selected_action_ids; kept all"}
    by_id = {f"a{index}": candidate for index, candidate in enumerate(candidates, 1)}
    selected = [by_id[str(item)] for item in selected_ids if str(item) in by_id]
    if not selected:
        selected = candidates
        reason = "API selected no valid ids; kept all"
    else:
        reason = str(payload.get("reason") or "API selected final path")
    return selected, {
        "selected_count": len(selected),
        "input_count": len(candidates),
        "rejected_action_ids": payload.get("rejected_action_ids") or [],
        "quality_score": payload.get("quality_score"),
        "reason": reason,
    }


def _state_chain_image_paths(steps: list[dict]) -> list[Path]:
    if not steps:
        return []
    paths: list[Path] = []
    first_observation = steps[0].get("observation") or {}
    first_path = first_observation.get("screenshot")
    if not first_path or not Path(first_path).exists():
        return []
    paths.append(Path(first_path))
    for step in steps:
        post_path = (step.get("post_observation") or {}).get("screenshot")
        if not post_path or not Path(post_path).exists():
            return []
        paths.append(Path(post_path))
    return paths


def _step_batch_payload_by_index(payload: dict) -> dict[int, dict]:
    items = payload.get("steps")
    if not isinstance(items, list):
        return {}
    by_index: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[index] = item
    return by_index


def _step_batch_payload_needs_fallback(step: dict, payload: dict) -> bool:
    if "is_action" not in payload:
        return True
    confidence = _safe_float(payload.get("confidence"), None)
    if confidence is None or confidence < 0.55:
        return True
    if payload.get("is_action") is False:
        return confidence < 0.85
    action = payload.get("action")
    if not isinstance(action, dict) or not action.get("type") or not isinstance(action.get("args"), dict):
        return True
    expected = ((step.get("evidence") or {}).get("api_boundary") or {}).get("action_type") or (step.get("action") or {}).get("type")
    actual = action.get("type")
    if actual not in ALLOWED_ACTION_TYPES:
        return True
    if expected and actual and expected != actual:
        reason = str(payload.get("changed_action_type_reason") or "").strip()
        if not reason or _suspicious_action_type_change(str(expected), str(actual), reason):
            return True
    if actual in {"computer.click", "computer.double_click", "computer.move"}:
        return not all(key in action["args"] for key in ("x", "y"))
    if actual == "computer.drag":
        return not all(key in action["args"] for key in ("x1", "y1", "x2", "y2"))
    if actual == "computer.type":
        return "text" not in action["args"]
    if actual == "computer.press":
        return "keys" not in action["args"]
    if actual == "shell.exec":
        command = action["args"].get("command")
        return not isinstance(command, str) or not command.strip()
    if actual == "tool.call":
        name = action["args"].get("name")
        return (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(action["args"].get("args"), dict)
        )
    return False


def _action_needs_grounding(action: dict) -> bool:
    action_type = action.get("type")
    args = action.get("args") or {}
    if action_type in {"computer.click", "computer.double_click", "computer.move"}:
        return not all(key in args for key in ("x", "y"))
    if action_type == "computer.drag":
        return not all(key in args for key in ("x1", "y1", "x2", "y2"))
    if action_type == "computer.type":
        return "text" not in args
    if action_type == "computer.press":
        return "keys" not in args
    if action_type == "shell.exec":
        command = args.get("command")
        return not isinstance(command, str) or not command.strip()
    if action_type == "tool.call":
        name = args.get("name")
        return not isinstance(name, str) or not name.strip() or not isinstance(args.get("args"), dict)
    return False


def _compact_transcript(transcript: list[dict[str, Any]], *, max_chars: int) -> str:
    lines = []
    for segment in transcript:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{float(segment.get('start') or 0):.1f}-{float(segment.get('end') or 0):.1f}] {text}")
    compact = "\n".join(lines)
    if len(compact) > max_chars:
        return compact[:max_chars] + "\n...(truncated)"
    return compact


def _small_json(obj: Any, *, max_chars: int = 4000) -> Any:
    text = json.dumps(obj, ensure_ascii=False)
    if len(text) <= max_chars:
        return obj
    return {"truncated": True, "preview": text[:max_chars]}


def _safe_float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    extracted = _extract_balanced_json_object(text)
    if extracted is None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        extracted = match.group(0) if match else None
    if extracted is None:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as exc:
        repaired = _repair_json_trailing_commas(extracted)
        if repaired != extracted:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        repaired = _repair_truncated_json_object(extracted)
        if repaired and repaired != extracted:
            return json.loads(repaired)
        raise exc


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _repair_truncated_json_object(text: str) -> str | None:
    repaired = text.strip()
    if not repaired.startswith("{"):
        return None
    while repaired:
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError as exc:
            if exc.pos < len(repaired) - 2:
                break
            repaired = repaired.rstrip(", \n\r\t")
            if repaired.endswith('"'):
                repaired += "}"
            elif repaired.endswith(":"):
                repaired += "null}"
            elif repaired.endswith("["):
                repaired += "]}"
            else:
                repaired += "}"
    return None


def _repair_json_trailing_commas(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            out.append(char)
            continue
        if char in "}]":
            index = len(out) - 1
            while index >= 0 and out[index] in " \n\r\t":
                index -= 1
            if index >= 0 and out[index] == ",":
                del out[index]
            out.append(char)
            continue
        out.append(char)
    return "".join(out)


def _supports_response_format(base_url: str, model: str = "") -> bool:
    lowered = (base_url or "").lower()
    if "openrouter" in lowered:
        return False
    return "api.openai.com" in lowered or model.lower().startswith("deepseek-v4.1")


def _normalize_base_urls(base_urls: list[str] | tuple[str, ...] | str | None) -> list[str]:
    return list(normalize_base_urls(base_urls))


def _allows_empty_api_key(base_url: str) -> bool:
    return _is_local_endpoint(base_url)


def _should_bypass_proxy(base_url: str) -> bool:
    return _is_local_endpoint(base_url)


def _completion_token_param(base_url: str, model: str) -> str:
    if _is_openai_first_party(base_url) or model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _should_send_reasoning_effort(reasoning_effort: str | None, base_url: str, model: str) -> bool:
    value = str(reasoning_effort or "").strip().lower()
    if not value:
        return False
    if model.lower().startswith("deepseek-v4.1"):
        return value not in {"off", "false", "0"}
    if value == "none":
        return False
    return _is_openai_first_party(base_url) or model.lower().startswith(
        ("gpt-5", "o1", "o3", "o4")
    )


def _is_openai_first_party(base_url: str) -> bool:
    return "api.openai.com" in (base_url or "").lower()


def _is_local_endpoint(base_url: str) -> bool:
    lowered = (base_url or "").lower()
    if "localhost" in lowered or "127.0.0.1" in lowered:
        return True
    if lowered.startswith("http://"):
        return "openrouter" not in lowered and "api.openai.com" not in lowered
    return False


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _backoff_delay(attempt: int) -> float:
    base = 2.0
    cap = 30.0
    ceiling = min(cap, base * (2 ** attempt))
    return random.uniform(base, ceiling) if ceiling > base else base
