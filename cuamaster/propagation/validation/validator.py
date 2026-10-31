"""Static and sandbox-backed validation for generated tasks."""
from __future__ import annotations

import hashlib
import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..evaluators import inspect_evaluator_code
from ..quality import generated_task_quality
from ..schemas import EnvironmentContext, GeneratedTask, ValidationReport


class VerifierBackend(ABC):
    @abstractmethod
    def verify(self, generated: GeneratedTask, environment: EnvironmentContext) -> dict[str, Any]:
        raise NotImplementedError


class StaticVerifier(VerifierBackend):
    def verify(self, generated: GeneratedTask, environment: EnvironmentContext) -> dict[str, Any]:
        return {
            "ran": False,
            "judge_only": False,
            "initial_negative": False,
            "partial_negative": False,
            "reset_determinism": False,
            "agent_solve": False,
            "repeated_score": False,
            "reason": "no sandbox verifier configured",
        }


class CommandVerifier(VerifierBackend):
    def __init__(self, command: tuple[str, ...], *, reset_trials: int, solve_attempts: int) -> None:
        if not command:
            raise ValueError("command verifier requires validation.verifier_command")
        self.command = list(command)
        self.reset_trials = reset_trials
        self.solve_attempts = solve_attempts

    def verify(self, generated: GeneratedTask, environment: EnvironmentContext) -> dict[str, Any]:
        completed = subprocess.run(
            self.command,
            input=json.dumps({
                "operation": "verify_task",
                "generated_task": generated.to_dict(),
                "environment": environment.to_dict(),
                "modes": ["judge_only", "negative_control", "agent_solve"],
                "reset_trials": self.reset_trials,
                "solve_attempts": self.solve_attempts,
            }),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            return {
                "ran": True,
                "judge_only": False,
                "initial_negative": False,
                "partial_negative": False,
                "reset_determinism": False,
                "agent_solve": False,
                "repeated_score": False,
                "error": completed.stderr[-2000:],
            }
        value = json.loads(completed.stdout)
        if not isinstance(value, dict):
            raise ValueError("verifier command must return a JSON object")
        value["ran"] = True
        return value


def create_verifier(config) -> VerifierBackend:
    if config.verifier_command:
        return CommandVerifier(
            config.verifier_command,
            reset_trials=config.reset_trials,
            solve_attempts=config.solve_attempts,
        )
    return StaticVerifier()


def _asset_errors(generated: GeneratedTask, allowed_domains: set[str]) -> list[str]:
    errors: list[str] = []
    for asset in generated.assets:
        path = str(asset.get("path") or "")
        digest = str(asset.get("sha256") or "")
        source = str(asset.get("source") or "generated")
        if not path or len(digest) != 64:
            errors.append("asset requires path and SHA-256")
            continue
        if source.startswith(("http://", "https://")):
            domain = (urlparse(source).hostname or "").lower()
            if domain not in allowed_domains:
                errors.append(f"asset domain is not allowed: {domain}")
        local = Path(path)
        if local.is_file():
            actual = hashlib.sha256(local.read_bytes()).hexdigest()
            if actual != digest:
                errors.append(f"asset hash mismatch: {path}")
    return errors


class TaskValidator:
    def __init__(
        self,
        environment: EnvironmentContext,
        verifier: VerifierBackend,
        *,
        require_sandbox: bool,
        max_instruction_chars: int = 4000,
        max_instruction_words: int = 700,
        require_platform_consistency: bool = True,
        require_supported_evaluator: bool = True,
        require_gui_realism: bool = True,
    ) -> None:
        self.environment = environment
        self.verifier = verifier
        self.require_sandbox = require_sandbox
        self.max_instruction_chars = max_instruction_chars
        self.max_instruction_words = max_instruction_words
        self.require_platform_consistency = require_platform_consistency
        self.require_supported_evaluator = require_supported_evaluator
        self.require_gui_realism = require_gui_realism

    def validate(self, generated: GeneratedTask) -> ValidationReport:
        task = generated.task
        errors: list[str] = []
        warnings: list[str] = []
        checks = {
            "schema": True,
            "environment": self.environment.supports(list(task.required_capabilities)),
            "evaluator": bool(task.evaluator),
            "assets": True,
            "evaluator_safety": True,
        }
        quality_checks, quality_errors, quality_warnings = generated_task_quality(
            generated,
            self.environment,
            max_instruction_chars=self.max_instruction_chars,
            max_instruction_words=self.max_instruction_words,
            require_platform_consistency=self.require_platform_consistency,
            require_supported_evaluator=self.require_supported_evaluator,
            require_gui_realism=self.require_gui_realism,
        )
        checks.update(quality_checks)
        errors.extend(quality_errors)
        warnings.extend(quality_warnings)
        inventory_verified = self.environment.policy.get("inventory_verified")
        checks["environment_inventory_verified"] = inventory_verified is not False
        if inventory_verified is False:
            warnings.append(
                "environment capabilities come from configuration overrides, not verified inventory"
            )
        if not checks["environment"]:
            missing = sorted(set(task.required_capabilities) - set(self.environment.capabilities))
            errors.append(f"environment missing capabilities: {', '.join(missing)}")
        if not task.evaluator:
            errors.append("generated task has no evaluator")
        asset_errors = _asset_errors(
            generated,
            set(self.environment.policy.get("allowed_download_domains") or []),
        )
        if asset_errors:
            checks["assets"] = False
            errors.extend(asset_errors)
        for plugin in generated.evaluator_plugins:
            report = inspect_evaluator_code(
                str(plugin.get("code") or ""),
                source_trust=str(plugin.get("source_trust") or "generated"),
            )
            if not report["safe"]:
                checks["evaluator_safety"] = False
                errors.extend(report["errors"])
        sandbox = self.verifier.verify(generated, self.environment)
        sandbox_checks = {
            "sandbox_judge_only": bool(sandbox.get("judge_only")),
            "sandbox_initial_negative": bool(sandbox.get("initial_negative")),
            "sandbox_partial_negative": bool(sandbox.get("partial_negative")),
            "sandbox_reset_determinism": bool(sandbox.get("reset_determinism")),
            "sandbox_agent_solve": bool(sandbox.get("agent_solve")),
            "sandbox_repeated_score": bool(sandbox.get("repeated_score")),
        }
        checks.update(sandbox_checks)
        if self.require_sandbox:
            failed = [name for name, passed in sandbox_checks.items() if not passed]
            if failed:
                errors.append(f"sandbox validation failed: {', '.join(failed)}")
        elif not sandbox.get("ran"):
            warnings.append("sandbox validation was not run")
        return ValidationReport(
            task_id=task.id,
            valid=not errors,
            checks=checks,
            errors=tuple(errors),
            warnings=tuple(warnings),
            sandbox=sandbox,
        )
