"""Generator backends for decomposition, composition, and repair."""
from __future__ import annotations

import json
import hashlib
import ipaddress
import os
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from cuamaster.endpoints import DynamicEndpointPool, is_endpoint_failure, normalize_base_urls
from cuamaster.openai_api import create_text_response, normalize_wire_api, usage_token_counts
from cuamaster.contracts import TargetEnvironment, TaskSpec, canonical_json, content_digest

from ..alignment import evaluator_source_excerpt
from ..evaluators import build_composite_evaluator
from ..ingestion import EpisodeEvidence
from ..platforms import environment_platform
from ..quality import task_class_reference
from ..schemas import AbilityPrimitive, EnvironmentContext, GeneratedTask, SourceEvidence, SourceTask, TaskRecipe
from .variations import variation_directive


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:64] or "operation"


def _target_environments(environment: EnvironmentContext) -> tuple[TargetEnvironment, ...]:
    os_type = str(environment.os.get("type") or "unknown")
    version = str(environment.os.get("version") or "")
    architecture = str(environment.os.get("architecture") or "")
    return (
        TargetEnvironment(
            os_type=os_type,
            os_versions=(version,) if version and version != "unknown" else (),
            architectures=(architecture,) if architecture else (),
        ),
    )


def _target_platform(environment: EnvironmentContext) -> str:
    return environment_platform(environment)


def _source_task_class_evaluator(anchor: SourceTask) -> dict[str, Any] | None:
    reference = task_class_reference(anchor.task)
    if reference is None:
        return None
    if reference.get("type") == "ale_task_module":
        return {
            **reference,
            "type": "task_evaluator",
            "source_type": "ale_task_module",
            "method": "evaluate",
        }
    return {
        "type": "osworld_v2_task_class",
        "task_id": reference.get("task_id"),
        "class_name": reference.get("class_name"),
        "method": "evaluate",
        "path": reference.get("path"),
        "repository_root": reference.get("repository_root"),
    }


def _generated_task_class_evaluator(task: TaskSpec) -> dict[str, Any] | None:
    reference = task_class_reference(task)
    if reference is None:
        return None
    if reference.get("type") == "ale_task_module":
        return {
            **reference,
            "type": "task_evaluator",
            "source_type": "ale_task_module",
            "method": "evaluate",
        }
    return {
        "type": "osworld_v2_task_class",
        "task_id": reference.get("task_id"),
        "class_name": reference.get("class_name"),
        "method": "evaluate",
        "path": reference.get("path"),
        "repository_root": reference.get("repository_root"),
    }


def _clauses(instruction: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", instruction).strip()
    parts = re.split(r"(?:;|\bthen\b|\bafter that\b|\bfinally\b|\n\s*\d+[.)])", normalized, flags=re.I)
    parts = [part.strip(" .:-") for part in parts if len(part.strip(" .:-")) >= 8]
    if len(parts) == 1:
        split = re.split(r"\s+(?:and|并且|然后|再)\s+", normalized, maxsplit=3, flags=re.I)
        if 1 < len(split) <= 4:
            parts = [part.strip(" .:-") for part in split if len(part.strip(" .:-")) >= 8]
    return parts[:4] or [normalized]


class GeneratorBackend(ABC):
    model: str

    def usage_stats(self) -> dict[str, Any]:
        """Per-operation request/token counters; empty when not tracked."""
        return {}

    @abstractmethod
    def decompose(
        self,
        source: SourceTask,
        episodes: list[EpisodeEvidence],
        environment: EnvironmentContext,
    ) -> list[AbilityPrimitive]:
        raise NotImplementedError

    @abstractmethod
    def compose(
        self,
        recipe: TaskRecipe,
        anchor: SourceTask,
        capabilities: list[AbilityPrimitive],
        environment: EnvironmentContext,
    ) -> GeneratedTask:
        raise NotImplementedError

    def repair(
        self,
        generated: GeneratedTask,
        report: Mapping[str, Any],
        environment: EnvironmentContext,
    ) -> GeneratedTask:
        return generated

    def align_check(
        self,
        generated: GeneratedTask,
        evaluator_source: str,
        *,
        judgement_index: int = 0,
    ) -> dict[str, Any] | None:
        """Judge whether the instruction matches what the evaluator scores.

        Returns None when the backend cannot judge (alignment then degrades to
        a warning instead of an error). ``judgement_index`` distinguishes
        repeated judgements of the same candidate so they bypass the response
        cache.
        """
        return None

    def solve_check(
        self,
        generated: GeneratedTask,
        environment: EnvironmentContext,
    ) -> dict[str, Any] | None:
        """Judge solvability from an executor's viewpoint.

        Returns {"solvable": bool, "blockers": [...], "plan": [...]}, or None
        when the backend cannot judge.
        """
        return None


class DeterministicGeneratorBackend(GeneratorBackend):
    def __init__(self, model: str = "deterministic") -> None:
        self.model = model

    def decompose(
        self,
        source: SourceTask,
        episodes: list[EpisodeEvidence],
        environment: EnvironmentContext,
    ) -> list[AbilityPrimitive]:
        successful = [episode for episode in episodes if episode.success]
        failed = [episode for episode in episodes if not episode.success]
        failure_modes = []
        for episode in failed:
            for step in episode.steps:
                execution = step.get("execution") or {}
                if execution.get("ok") is False or execution.get("error"):
                    failure_modes.append(str(execution.get("error") or "action execution failed"))
        fragments = (source.task.evaluator,) if source.task.evaluator else ()
        capabilities: list[AbilityPrimitive] = []
        for index, clause in enumerate(_clauses(source.task.instruction), 1):
            evidence = [SourceEvidence(task_id=source.task.id)]
            for episode in successful[:3]:
                evidence.append(SourceEvidence(
                    task_id=source.task.id,
                    episode_id=episode.episode_id,
                    step_start=1,
                    step_end=max(1, len(episode.steps)),
                    success=True,
                ))
            effect = f"{source.application}:{_slug(clause)}:completed"
            capabilities.append(AbilityPrimitive(
                id=f"{source.application}.{_slug(clause)}.{index}",
                application=source.application,
                intent=clause,
                preconditions=(f"{source.application}_available",),
                effects=(effect,),
                side_effects=("environment_state_change",),
                required_vm_capabilities=source.task.required_capabilities,
                config_dependencies=source.task.config,
                evaluator_fragments=fragments,
                source_evidence=tuple(evidence),
                failure_modes=tuple(sorted(set(failure_modes))[:8]),
                confidence=0.9 if successful else 0.65,
            ))
        return capabilities

    def compose(
        self,
        recipe: TaskRecipe,
        anchor: SourceTask,
        capabilities: list[AbilityPrimitive],
        environment: EnvironmentContext,
    ) -> GeneratedTask:
        ordered = [capability.intent.rstrip(".") for capability in capabilities]
        instruction = "First, " + "; then, ".join(ordered) + "."
        generation_metadata = {
            "anchor_task_id": anchor.task.id,
            "source_task_ids": list(recipe.source_task_ids),
            "source_episode_ids": sorted({
                evidence.episode_id
                for capability in capabilities
                for evidence in capability.source_evidence
                if evidence.episode_id
            }),
            "ability_primitive_ids": list(recipe.primitive_ids),
            "recipe_id": recipe.id,
            "generator_model": self.model,
            "environment_id": environment.image_id,
            "environment_digest": environment.image_digest,
            "variant_index": recipe.variant_index,
            "prompt_hash": content_digest({"recipe": recipe.to_dict(), "backend": self.model}),
            "task_blueprint": {
                "goal": instruction,
                "primitive_ids": list(recipe.primitive_ids),
                "dependency_order": list(recipe.primitive_ids),
                "expected_effects": list(recipe.expected_effects),
                "failure_modes": sorted({
                    failure
                    for capability in capabilities
                    for failure in capability.failure_modes
                }),
            },
        }
        source_evaluator = _source_task_class_evaluator(anchor)
        task = TaskSpec(
            id=f"gen-{anchor.task.id}-{recipe.id[:10]}",
            instruction=instruction,
            snapshot=anchor.task.snapshot,
            config=anchor.task.config,
            evaluator=source_evaluator or build_composite_evaluator(capabilities),
            required_capabilities=recipe.required_vm_capabilities,
            target_environments=_target_environments(environment),
            metadata={
                **anchor.task.metadata,
                "application": recipe.application,
                "category": anchor.category,
                "platform": _target_platform(environment),
                "generation": generation_metadata,
                "expected_effects": list(recipe.expected_effects),
            },
        )
        return GeneratedTask(task=task, recipe=recipe)


class _JSONGeneratorBackend(GeneratorBackend):
    blueprint_first = False

    def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def decompose(self, source, episodes, environment):
        response = self._request("decompose", {
            "source_task": source.to_dict(),
            "episodes": [episode.to_prompt_dict() for episode in episodes],
            "environment": environment.to_dict(),
        })
        return [
            AbilityPrimitive.from_dict(
                _normalize_primitive(item, source, environment)
            )
            for item in response.get("capabilities") or []
            if isinstance(item, Mapping)
        ]

    def compose(self, recipe, anchor, capabilities, environment):
        directive = variation_directive(
            recipe.id,
            recipe.variant_index,
            outcome_preserving_only=recipe.strategy == "source_complete",
            difficulty_target=str(recipe.metadata.get("difficulty_target") or ""),
        )
        payload = {
            "recipe": recipe.to_dict(),
            "anchor": anchor.to_dict(),
            "capabilities": [item.to_dict() for item in capabilities],
            "environment": environment.to_dict(),
            "variation_directive": directive,
        }
        evaluator_source = evaluator_source_excerpt(anchor.task)
        if evaluator_source is not None:
            payload["evaluator_source"] = evaluator_source
        blueprint: dict[str, Any] = {}
        if self.blueprint_first:
            blueprint_response = self._request("blueprint", payload)
            blueprint_value = blueprint_response.get("blueprint", blueprint_response)
            if isinstance(blueprint_value, Mapping):
                blueprint = dict(blueprint_value)
                payload["blueprint"] = blueprint
        response = self._request("compose", payload)
        response_blueprint = response.get("blueprint")
        final_blueprint = (
            dict(response_blueprint)
            if isinstance(response_blueprint, Mapping)
            else blueprint
        )
        task_raw = dict(response.get("task") or {})
        response_metadata = dict(task_raw.get("metadata") or {})
        task_raw.update({
            "id": f"gen-{anchor.task.id}-{recipe.id[:10]}",
            "snapshot": anchor.task.snapshot,
            "config": list(anchor.task.config),
            "required_capabilities": list(recipe.required_vm_capabilities),
            "target_environments": [
                item.to_dict() for item in _target_environments(environment)
            ],
        })
        source_evaluator = _source_task_class_evaluator(anchor)
        if source_evaluator is not None:
            task_raw["evaluator"] = source_evaluator
        elif not task_raw.get("evaluator"):
            task_raw["evaluator"] = build_composite_evaluator(capabilities)
        task_raw["metadata"] = {
            **anchor.task.metadata,
            "application": recipe.application,
            "category": anchor.category,
            "platform": _target_platform(environment),
            "related_apps": list(
                response_metadata.get("related_apps")
                or anchor.task.metadata.get("related_apps")
                or [recipe.application]
            ),
            "generation": {
                "anchor_task_id": anchor.task.id,
                "source_task_ids": list(recipe.source_task_ids),
                "ability_primitive_ids": list(recipe.primitive_ids),
                "recipe_id": recipe.id,
                "generator_model": self.model,
                "environment_id": environment.image_id,
                "environment_digest": environment.image_digest,
                "variant_index": recipe.variant_index,
                "variation_axis": directive["axis"],
                "difficulty_target": directive["difficulty_target"],
                "scaffolding": directive.get("scaffolding", "neutral"),
                "task_blueprint": final_blueprint,
            },
            "expected_effects": list(recipe.expected_effects),
        }
        task = TaskSpec.from_dict(task_raw)
        return GeneratedTask(
            task=task,
            recipe=recipe,
            evaluator_plugins=tuple(
                dict(item)
                for item in response.get("evaluator_plugins") or []
                if isinstance(item, Mapping)
            ),
            assets=tuple(
                dict(item)
                for item in response.get("assets") or []
                if isinstance(item, Mapping)
            ),
        )

    def align_check(self, generated, evaluator_source, *, judgement_index=0):
        response = self._request("align_check", {
            "instruction": generated.task.instruction,
            "evaluator_source": evaluator_source,
            "judgement_index": judgement_index,
        })
        if not isinstance(response, Mapping):
            return None
        return {
            "aligned": bool(response.get("aligned")),
            "unverified_requirements": [
                str(item) for item in response.get("unverified_requirements") or []
                if str(item).strip()
            ],
            "missing_requirements": [
                str(item) for item in response.get("missing_requirements") or []
                if str(item).strip()
            ],
        }

    def solve_check(self, generated, environment):
        response = self._request("solve_check", {
            "task": generated.task.to_dict(),
            "environment": environment.to_dict(),
        })
        if not isinstance(response, Mapping):
            return None
        return {
            "solvable": bool(response.get("solvable")),
            "blockers": [
                str(item) for item in response.get("blockers") or []
                if str(item).strip()
            ],
            "plan": [
                str(item) for item in response.get("plan") or []
                if str(item).strip()
            ],
        }

    def repair(self, generated, report, environment):
        repair_focus, focused_errors = _repair_focus(report)
        response = self._request("repair", {
            "generated_task": generated.to_dict(),
            "validation_report": dict(report),
            "repair_focus": repair_focus,
            "focused_errors": focused_errors,
            "environment": environment.to_dict(),
        })
        task_raw = generated.task.to_dict()
        task_patch = response.get("task")
        if isinstance(task_patch, Mapping):
            task_raw.update(task_patch)
        if not str(task_raw.get("instruction") or "").strip():
            task_raw["instruction"] = generated.task.instruction
        task_raw.update({
            "id": generated.task.id,
            "snapshot": generated.task.snapshot,
            "config": list(generated.task.config),
            "required_capabilities": list(generated.task.required_capabilities),
            "target_environments": [
                item.to_dict() for item in generated.task.target_environments
            ],
        })
        source_evaluator = _generated_task_class_evaluator(generated.task)
        if source_evaluator is not None:
            task_raw["evaluator"] = source_evaluator
        elif not task_raw.get("evaluator"):
            task_raw["evaluator"] = generated.task.evaluator
        task_raw["metadata"] = {
            **generated.task.metadata,
            "platform": _target_platform(environment),
        }
        task = TaskSpec.from_dict(task_raw)
        return GeneratedTask(
            task=task,
            recipe=generated.recipe,
            evaluator_plugins=tuple(
                dict(item)
                for item in response.get("evaluator_plugins") or generated.evaluator_plugins
                if isinstance(item, Mapping)
            ),
            assets=tuple(
                dict(item)
                for item in response.get("assets") or generated.assets
                if isinstance(item, Mapping)
            ),
        )


def _repair_focus(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    errors = [str(item) for item in report.get("errors") or []]
    categories = (
        ("alignment", ("alignment", "unverified requirement", "missing requirement")),
        ("evaluator", ("evaluator", "judge", "score")),
        ("platform", ("platform", "windows", "linux", "macos", "path", "shortcut")),
        ("assets", ("asset", "sha-256", "domain")),
        ("sandbox", ("sandbox", "reset", "agent_solve", "negative")),
        ("instruction", ("instruction", "procedural", "too long")),
    )
    for category, markers in categories:
        focused = [
            error for error in errors if any(marker in error.casefold() for marker in markers)
        ]
        if focused:
            return category, focused
    return "general", errors


def _normalize_primitive(
    raw: Mapping[str, Any],
    source: SourceTask,
    environment: EnvironmentContext,
) -> dict[str, Any]:
    value = dict(raw)
    raw_application = str(value.get("application") or "").strip().casefold()
    application_aliases = {
        "chrome": "google_chrome",
        "google chrome": "google_chrome",
        "writer": "libreoffice_writer",
        "calc": "libreoffice_calc",
        "impress": "libreoffice_impress",
    }
    candidate_application = application_aliases.get(raw_application, raw_application)
    primitive_prefix = str(value.get("id") or "").split(".", 1)[0].casefold()
    prefix_application = application_aliases.get(primitive_prefix, primitive_prefix)
    available = set(environment.capabilities)
    if candidate_application in available:
        value["application"] = candidate_application
    elif prefix_application in available:
        value["application"] = prefix_application
    else:
        value["application"] = source.application
    value["inputs"] = dict(value.get("inputs") or {}) if isinstance(
        value.get("inputs"), Mapping
    ) else {}
    value["config_dependencies"] = [
        dict(item) if isinstance(item, Mapping) else {"description": str(item)}
        for item in value.get("config_dependencies") or []
        if isinstance(item, Mapping) or str(item).strip()
    ]
    value["evaluator_fragments"] = [
        dict(item)
        for item in value.get("evaluator_fragments") or []
        if isinstance(item, Mapping)
    ]
    evidence = []
    for item in value.get("source_evidence") or []:
        if not isinstance(item, Mapping):
            continue
        normalized = dict(item)
        normalized["task_id"] = source.task.id
        evidence.append(normalized)
    if not evidence:
        evidence.append({"task_id": source.task.id})
    value["source_evidence"] = evidence
    value["required_vm_capabilities"] = [
        str(item)
        for item in value.get("required_vm_capabilities") or []
        if str(item) in available
    ]
    value["confidence"] = min(1.0, max(0.0, float(value.get("confidence") or 0.0)))
    return value


class CommandGeneratorBackend(_JSONGeneratorBackend):
    def __init__(self, command: tuple[str, ...], model: str) -> None:
        if not command:
            raise ValueError("command generator backend requires generation.command")
        self.command = list(command)
        self.model = model

    def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        completed = subprocess.run(
            self.command,
            input=json.dumps({"operation": operation, "payload": payload}),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"generator command failed: {completed.stderr[-1000:]}")
        value = json.loads(completed.stdout)
        if not isinstance(value, dict):
            raise ValueError("generator command must return a JSON object")
        return value


class OpenAICompatibleGeneratorBackend(_JSONGeneratorBackend):
    def __init__(
        self,
        *,
        model: str,
        base_urls: tuple[str, ...],
        api_key_env: str,
        reasoning_effort: str,
        max_tokens: int,
        max_retries: int,
        request_delay: float,
        timeout_seconds: float,
        wire_api: str = "chat_completions",
        store: bool = False,
        trust_env: bool = True,
        endpoint_pool: str | Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        try:
            import httpx
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError(f"openai and httpx are required for API generation: {exc}") from exc

        self.model = model
        self._endpoint_pool = DynamicEndpointPool(
            config_path=endpoint_pool,
            base_urls=base_urls,
            model=model,
        )
        self.base_urls = self._endpoint_pool.base_urls
        if not self.base_urls:
            raise ValueError("API generator requires generation.base_url or generation.base_urls")
        self.api_key_env = api_key_env
        self.wire_api = normalize_wire_api(wire_api)
        self.store = bool(store)
        self.trust_env = bool(trust_env)
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key and all(_is_local_endpoint(url) for url in self.base_urls):
            api_key = "EMPTY"
        if not api_key:
            raise ValueError(f"missing generator API key env: {api_key_env}")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._openai_class = OpenAI
        self._httpx_class = httpx.Client
        self._clients: dict[str, Any] = {}
        self._client_lock = threading.Lock()
        self._cache_index_lock = threading.Lock()
        self._observed_cache_base_urls: dict[str, tuple[str, ...]] = {}
        self._usage_lock = threading.Lock()
        self._usage: dict[str, dict[str, int]] = {}

    def usage_stats(self) -> dict[str, Any]:
        with self._usage_lock:
            return {
                operation: dict(counters)
                for operation, counters in sorted(self._usage.items())
            }

    def _record_usage(self, operation: str, response: Any, *, cached: bool) -> None:
        with self._usage_lock:
            counters = self._usage.setdefault(operation, {
                "requests": 0,
                "cached": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            })
            if cached:
                counters["cached"] += 1
                return
            counters["requests"] += 1
            input_tokens, output_tokens = usage_token_counts(response)
            counters["prompt_tokens"] += input_tokens
            counters["completion_tokens"] += output_tokens

    def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        cache_path = self._cache_path(operation, payload)
        for candidate_path in self._cache_read_paths(operation, payload):
            try:
                cached = json.loads(candidate_path.read_text(encoding="utf-8"))
                result = _parse_json_object(str(cached["raw"]))
                if isinstance(result, dict):
                    if cache_path is not None and candidate_path != cache_path:
                        _write_cache(cache_path, cached)
                    self._record_usage(operation, None, cached=True)
                    return result
            except Exception:
                candidate_path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                base_url, client = self._next_client()
                send_reasoning = self.wire_api == "responses" or _should_send_reasoning_effort(
                    self.reasoning_effort, base_url, self.model
                )
                response_result = create_text_response(
                    client,
                    wire_api=self.wire_api,
                    model=self.model,
                    content=_operation_prompt(operation, payload),
                    max_output_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort if send_reasoning else None,
                    system_prompt=_SYSTEM_PROMPT,
                    store=self.store,
                    chat_response_format=(
                        {"type": "json_object"}
                        if _supports_response_format(base_url)
                        else None
                    ),
                )
                response = response_result.response
                raw = response_result.text.strip()
                result = _parse_json_object(raw)
                self._endpoint_pool.mark_success(base_url)
                self._record_usage(operation, response, cached=False)
                if cache_path:
                    _write_cache(cache_path, {
                        "raw": raw,
                        "operation": operation,
                        "model": self.model,
                        "base_url": base_url,
                    })
                return result
            except Exception as exc:  # pragma: no cover - backend specific
                last_error = exc
                if is_endpoint_failure(exc):
                    self._endpoint_pool.mark_failure(base_url, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"Kimi task-generation request failed: {last_error}") from last_error

    def _next_client(self):
        base_url = self._endpoint_pool.acquire()
        self.base_urls = self._endpoint_pool.base_urls
        with self._client_lock:
            client = self._clients.get(base_url)
            if client is None:
                kwargs: dict[str, Any] = {
                    "api_key": self._api_key,
                    "base_url": base_url,
                    "timeout": self._timeout_seconds,
                    "max_retries": 0,
                }
                if not self.trust_env or _is_local_endpoint(base_url):
                    kwargs["http_client"] = self._httpx_class(
                        trust_env=self.trust_env and not _is_local_endpoint(base_url)
                    )
                client = self._openai_class(**kwargs)
                self._clients[base_url] = client
        return base_url, client

    def _cache_path(self, operation: str, payload: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(canonical_json({
            "model": self.model,
            "operation": operation,
            "payload": payload,
        }).encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / operation / f"{digest}.json"

    def _legacy_cache_path(
        self,
        operation: str,
        payload: dict[str, Any],
        base_urls: tuple[str, ...],
    ) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(canonical_json({
            "model": self.model,
            "base_urls": list(base_urls),
            "operation": operation,
            "payload": payload,
        }).encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / operation / f"{digest}.json"

    def _cache_read_paths(
        self,
        operation: str,
        payload: dict[str, Any],
    ):
        cache_path = self._cache_path(operation, payload)
        if cache_path is None:
            return
        if cache_path.is_file():
            yield cache_path

        current_urls = tuple(self._endpoint_pool.base_urls)
        observed_urls = self._cached_response_base_urls(operation)
        historical_urls = tuple(
            url for url in observed_urls if url not in current_urls
        )
        seen: set[tuple[str, ...]] = set()

        def legacy_candidates(urls: tuple[str, ...]):
            if not urls or urls in seen:
                return
            seen.add(urls)
            path = self._legacy_cache_path(operation, payload, urls)
            if path is not None and path.is_file():
                yield path

        yield from legacy_candidates(current_urls)
        if 0 < len(historical_urls) <= 6:
            for urls in permutations(historical_urls):
                yield from legacy_candidates(urls)
        for size in range(len(current_urls) - 1, 0, -1):
            for urls in combinations(current_urls, size):
                yield from legacy_candidates(urls)
        if 0 < len(observed_urls) <= 4:
            for size in range(len(observed_urls), 0, -1):
                for urls in permutations(observed_urls, size):
                    yield from legacy_candidates(urls)

    def _cached_response_base_urls(self, operation: str) -> tuple[str, ...]:
        with self._cache_index_lock:
            cached = self._observed_cache_base_urls.get(operation)
            if cached is not None:
                return cached
            operation_dir = self.cache_dir / operation if self.cache_dir else None
            values: set[str] = set()
            if operation_dir and operation_dir.is_dir():
                for path in operation_dir.glob("*.json"):
                    try:
                        value = json.loads(path.read_text(encoding="utf-8"))
                        base_url = str(value.get("base_url") or "").strip()
                        if base_url:
                            values.add(base_url)
                    except Exception:
                        continue
            result = tuple(sorted(values))
            self._observed_cache_base_urls[operation] = result
            return result


class KimiGeneratorBackend(OpenAICompatibleGeneratorBackend):
    """Kimi labeling backend using the action-construction API behavior."""


_SYSTEM_PROMPT = """You are Kimi, an expert GUI task-data labeler.
Return exactly one JSON object and no prose or Markdown fences.
Use only applications, capabilities, files, snapshots, and evaluator mechanisms present in the supplied VM environment and source evidence.
Never require an external account, unknown online service, host access, or an unavailable application.
Preserve source evidence and produce deterministic, machine-verifiable labels.
Treat the supplied environment OS as authoritative. Adapt paths, shortcuts, application names, and desktop workflows to that OS instead of copying source-platform details."""


def _operation_prompt(operation: str, payload: dict[str, Any]) -> str:
    instructions = {
        "decompose": """Decompose the source task into minimal reusable GUI atomic capabilities.
Each capability must describe one independently meaningful user-visible state transition, not low-level clicks.
Use successful episodes as positive evidence and failed episodes for failure_modes.
Every capability must include source_evidence with the source task_id.
When episodes are supplied, cite the episode_id and exact inclusive step_start/step_end that demonstrate the capability. Inspect the complete ordered action sequence rather than only the task instruction.
Return {\"capabilities\":[{\"id\":\"app.semantic_name.1\",\"application\":\"...\",\"intent\":\"...\",\"preconditions\":[],\"inputs\":{},\"effects\":[],\"side_effects\":[],\"required_vm_capabilities\":[],\"config_dependencies\":[],\"evaluator_fragments\":[],\"source_evidence\":[{\"task_id\":\"...\",\"episode_id\":null,\"step_start\":null,\"step_end\":null,\"success\":null}],\"failure_modes\":[],\"confidence\":0.0}]}.
Do not invent evaluator fragments or VM capabilities.""",
        "blueprint": """Design a grounded task blueprint before writing the final TaskSpec.
Use only the supplied capabilities, source evidence, evaluator fragments, and target environment.
List capability IDs in executable dependency order. Every expected effect must be observable by the supplied evaluator or explicitly identified as unsupported.
When evaluator_source is supplied it is the exact code that scores the task and the only reward signal. Derive expected_effects and evaluator_coverage from the conditions that code actually scores; do not plan outcomes it cannot observe.
Separate exact scored requirements from merely helpful context. Reject any plan that needs direct edits to hidden application state, service/process control, browser developer tools, or package installation instead of normal visible application interactions.
Do not write the final user instruction in this operation.
Return {\"blueprint\":{\"goal\":\"...\",\"dependency_order\":[\"capability-id\"],\"scored_requirements\":[\"...\"],\"expected_effects\":[\"...\"],\"evaluator_coverage\":[\"...\"],\"visible_apps\":[\"...\"],\"failure_modes\":[\"...\"]}}.""",
        "compose": """Create one new solvable task by composing the supplied atomic capabilities in dependency order.
The anchor owns snapshot, config, workspace assumptions, application, category, and target environment; preserve them.
The task must be meaningfully different from the source wording while remaining feasible in the supplied VM.
Apply the supplied variation_directive to this variant. When the directive is outcome-preserving, keep every scored requirement identical and vary only framing and phrasing; otherwise the directed change must remain observable by the supplied capabilities and evaluator.
Variants must not differ only by paraphrasing. Change an executable dimension only when the supplied capabilities and evaluator can observe it; for source_complete task-class recipes, preserve the scored source outcome instead of inventing variation.
Express the user-visible goal and required outcomes concisely. Do not turn the instruction into a replay of low-level trajectory steps when multiple valid GUI approaches exist.
Prefer 60-220 words and at most 8 sentences. Avoid role-play filler, repeated motivation, implementation advice, and long enumerations unless exact values are required by the evaluator.
The instruction must cover exactly the supplied capabilities in dependency order. Do not add goals, fields, or success criteria unsupported by the selected capabilities.
When evaluator_source is supplied it is the exact code that scores the task and the only reward signal. Every requirement the instruction states must influence the score that code returns, and every condition that code scores must be stated in the instruction. Copy gold values (names, ids, paths, numbers, exact strings) from evaluator_source exactly; never invent unscored requirements such as cleanup steps, saving intermediate files, or restarting services. Keep the instruction a natural user request; do not mention the evaluator, scoring, or code internals.
Use the target environment's native paths, shortcuts, application names, and desktop conventions. Never mention a source OS convention that is unavailable on the target OS.
Do not ask the agent to install dependencies, manipulate hidden application databases, use developer tools, or bypass the GUI unless an explicitly supplied capability requires that action.
State-changing work must be completed through the visible target application UI. A terminal may inspect information or create an explicitly requested code/file deliverable, but must not directly rewrite another application's private storage or restart its background services.
When the anchor is an OSWorld V2 task class, preserve its trusted evaluator directly; do not wrap it in a composite evaluator or invent additional evaluator syntax.
List only newly generated or downloaded files in assets, each with path and SHA-256. Do not list fixtures that already belong to the VM or source task.
The generated task must have a non-empty evaluator and required_capabilities.
Use the supplied blueprint when present. Otherwise produce a blueprint with goal, dependency_order, expected_effects, evaluator_coverage, and failure_modes before writing the task. The final task must implement that blueprint exactly; do not expose the blueprint as extra user-facing instructions.
Return {\"blueprint\":{\"goal\":\"...\",\"dependency_order\":[\"capability-id\"],\"expected_effects\":[\"...\"],\"evaluator_coverage\":[\"...\"],\"failure_modes\":[\"...\"]},\"task\":{\"id\":\"gen-...\",\"instruction\":\"...\",\"snapshot\":\"...\",\"config\":[],\"evaluator\":{},\"required_capabilities\":[],\"target_environments\":[],\"metadata\":{}},\"evaluator_plugins\":[],\"assets\":[]}.
Do not generate executable evaluator code unless no trusted declarative evaluator can express the checks.""",
        "repair": """Repair the generated task using every validation error while preserving its recipe and intended atomic capabilities.
Prioritize repair_focus and focused_errors first. Make the smallest change that resolves that category, then preserve all already valid fields.
Only make changes supported by the supplied VM environment and source evidence.
Shorten overly procedural instructions, replace source-platform conventions with target-platform conventions, and preserve trusted task-class evaluators exactly.
Remove any instruction or implied solution that relies on direct edits to hidden application state, browser developer tools, package installation, or process/service control. Keep state-changing work inside the visible target application UI.
When an error reports instruction/evaluator misalignment: remove or soften instruction requirements the evaluator does not score (listed as unverified requirements), add evaluator-scored requirements the instruction omits (listed as missing requirements), and never modify the evaluator itself.
Return a complete task object with a non-empty instruction. Never delete or blank an existing required task field.
Return the same shape as compose: {\"task\":{...},\"evaluator_plugins\":[],\"assets\":[]}.""",
        "align_check": """Compare the task instruction against the evaluator source code that scores the task.
The evaluator is the only reward signal: anything it does not measure earns no credit, and anything it measures that the instruction omits silently fails the agent.
Classify every user-facing requirement stated in the instruction:
- unverified_requirements: concrete instruction requirements whose completion the evaluator can NEVER observe in its final score (e.g. cleanup steps, restarting services, intermediate verification the evaluator does not inspect). Ignore requirements that merely help reach a scored end state.
- missing_requirements: conditions the evaluator scores that the instruction never asks for.
Judge only observable end state. Setup/config code and helper utilities in the source are not scoring logic; only what influences the returned score matters.
Set aligned=true only when both lists are empty.
Return {\"aligned\":true|false,\"unverified_requirements\":[\"...\"],\"missing_requirements\":[\"...\"]}.""",
        "solve_check": """Act as the GUI agent that must execute the task in the supplied VM environment and produce a human-demonstrated desktop trajectory.
Write a short step plan achieving the instruction through visible application interactions using only applications and capabilities present in the environment.
State-changing actions must use the target application's normal UI. Do not directly edit hidden databases, profile files, calendar stores, browser DOM/state, or registry-backed application data; do not stop/restart services or install packages to make the task pass. A terminal is acceptable only when the user explicitly requests command-line/code work or the scored deliverable is itself a file produced through that workflow.
Report blockers: any step that needs an application, file, account, network service, or capability the environment does not provide, or any requirement that is ambiguous or contradictory.
Judge feasibility, not difficulty; a long but achievable task is solvable.
Set solvable=true only when the blockers list is empty.
Return {\"solvable\":true|false,\"blockers\":[\"...\"],\"plan\":[\"...\"]}.""",
    }
    if operation not in instructions:
        raise ValueError(f"unsupported generator operation: {operation}")
    return "\n\n".join((
        instructions[operation],
        "Input JSON:",
        canonical_json({"operation": operation, "payload": payload}),
    ))


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    end = None
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
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
                end = index + 1
                break
    if end is None:
        raise ValueError(f"incomplete JSON object in response: {raw[:200]!r}")
    value = json.loads(text[start:end])
    if not isinstance(value, dict):
        raise ValueError("generator API must return a JSON object")
    return value


def _normalize_base_urls(values: tuple[str, ...]) -> tuple[str, ...]:
    return normalize_base_urls(values)


def _is_local_endpoint(base_url: str) -> bool:
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname in {"localhost", "host.docker.internal"} or hostname.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _supports_response_format(base_url: str) -> bool:
    return "api.openai.com" in base_url.lower()


def _should_send_reasoning_effort(value: str, base_url: str, model: str) -> bool:
    normalized = value.strip().lower()
    if not normalized or normalized in {"none", "off", "false", "0"}:
        return False
    return "api.openai.com" in base_url.lower() or model.lower().startswith(
        ("gpt-5", "o1", "o3", "o4")
    )


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def create_generator_backend(config, *, cache_dir: Path | None = None) -> GeneratorBackend:
    if config.backend == "deterministic":
        return DeterministicGeneratorBackend(config.model)
    if config.backend == "command":
        backend = CommandGeneratorBackend(config.command, config.model)
        backend.blueprint_first = bool(getattr(config, "blueprint_first", False))
        return backend
    if config.backend in {"kimi", "openai", "openai-compatible"}:
        base_urls = config.base_urls or ((config.base_url,) if config.base_url else ())
        backend_type = KimiGeneratorBackend if config.backend == "kimi" else OpenAICompatibleGeneratorBackend
        backend = backend_type(
            model=config.model,
            base_urls=base_urls,
            endpoint_pool=getattr(config, "endpoint_pool", None),
            api_key_env=config.api_key_env,
            wire_api=config.wire_api,
            store=config.store,
            trust_env=config.trust_env,
            reasoning_effort=config.reasoning_effort,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
            request_delay=config.request_delay,
            timeout_seconds=config.timeout_seconds,
            cache_dir=cache_dir,
        )
        backend.blueprint_first = bool(getattr(config, "blueprint_first", False))
        return backend
    raise ValueError(f"unknown generator backend: {config.backend}")
