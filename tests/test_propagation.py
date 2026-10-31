from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuamaster.collection.schemas import TaskSpec as GuiTaskSpec
from cuamaster.propagation.cli import _apply_run_overrides, parse_args
from cuamaster.propagation.primitives import PrimitiveLibrary
from cuamaster.propagation.config import ReleaseConfig, load_config
from cuamaster.propagation.diversity import score_candidate
from cuamaster.propagation.generation import PrimitiveComposer
from cuamaster.propagation.compilers import compile_osworld_taskset, compile_weavebench_taskset
from cuamaster.propagation.evaluators import inspect_evaluator_code
from cuamaster.propagation.generation.backends import (
    DeterministicGeneratorBackend,
    KimiGeneratorBackend,
    _JSONGeneratorBackend,
    _normalize_primitive,
    _parse_json_object,
)
from cuamaster.propagation.ingestion import load_episodes, load_source_tasks
from cuamaster.propagation.pipeline import TaskGenerationPipeline
from cuamaster.propagation.quality import gui_realism_errors, solve_plan_quality_errors
from cuamaster.propagation.release import promote_candidates
from cuamaster.propagation.schemas import AbilityPrimitive, CandidateQualityScore, EnvironmentContext, GeneratedTask, SourceEvidence, SourceTask, TaskRecipe
from cuamaster.propagation.validation import StaticVerifier, TaskValidator
from cuamaster.contracts import TaskSpec


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _image_store(tmp_path: Path) -> Path:
    root = tmp_path / "vm_images"
    descriptor = {
        "schema_version": 1,
        "image_id": "ubuntu-test",
        "os": {"type": "Ubuntu", "version": "22.04", "architecture": "x86_64"},
        "inventory_status": "complete",
        "apps": [
            {
                "name": "LibreOffice Writer",
                "desktop_id": "libreoffice-writer.desktop",
                "executable": "/usr/bin/libreoffice",
                "version": "7.3",
            }
        ],
    }
    _write_json(root / "linux" / "ubuntu-test" / "descriptor.json", descriptor)
    second_descriptor = {
        **descriptor,
        "image_id": "ubuntu-test-v2",
        "apps": [
            {
                **descriptor["apps"][0],
                "version": "7.6",
            }
        ],
    }
    _write_json(
        root / "linux" / "ubuntu-test-v2" / "descriptor.json",
        second_descriptor,
    )
    pending_descriptor = {
        "schema_version": 1,
        "image_id": "windows-pending",
        "os": {"type": "Windows", "version": "10", "architecture": "x86_64"},
        "inventory_status": "pending",
        "apps": [],
    }
    _write_json(
        root / "windows" / "windows-pending" / "descriptor.json",
        pending_descriptor,
    )
    macos_descriptor = {
        "schema_version": 1,
        "image_id": "macos-pending",
        "os": {"type": "Darwin", "version": "14.6", "architecture": "x86_64"},
        "inventory_status": "pending",
        "apps": [],
    }
    _write_json(
        root / "macos" / "macos-pending" / "descriptor.json",
        macos_descriptor,
    )
    _write_json(root / "catalog.json", {
        "schema_version": 1,
        "images": {
            "ubuntu-test": {
                "path": "linux/ubuntu-test/disk.qcow2",
                "descriptor": "linux/ubuntu-test/descriptor.json",
                "format": "qcow2",
                "os": descriptor["os"],
                "software": {"libreoffice": "7.3"},
                "size_bytes": 123,
            },
            "ubuntu-test-v2": {
                "path": "linux/ubuntu-test-v2/disk.qcow2",
                "descriptor": "linux/ubuntu-test-v2/descriptor.json",
                "format": "qcow2",
                "os": second_descriptor["os"],
                "software": {"libreoffice": "7.6"},
                "size_bytes": 456,
            },
            "windows-pending": {
                "path": "windows/windows-pending/disk.qcow2",
                "descriptor": "windows/windows-pending/descriptor.json",
                "format": "qcow2",
                "os": pending_descriptor["os"],
                "software": {},
                "size_bytes": 789,
            },
            "macos-pending": {
                "path": "macos/macos-pending/disk.qcow2",
                "descriptor": "macos/macos-pending/descriptor.json",
                "format": "qcow2",
                "os": macos_descriptor["os"],
                "software": {},
                "size_bytes": 987,
            },
        },
    })
    return root


def _source_tasks(tmp_path: Path) -> Path:
    root = tmp_path / "tasks" / "DOC"
    for task_id, instruction in (
        (
            "writer-format",
            "Open the Writer document; then make the title bold; then save the document.",
        ),
        (
            "writer-export",
            "Open the Writer document; then export it as PDF; then verify the PDF exists.",
        ),
    ):
        _write_json(root / f"{task_id}.json", {
            "id": task_id,
            "instruction": instruction,
            "snapshot": "clean-v1",
            "config": [],
            "evaluator": {"type": "file_exists", "path": "/tmp/result"},
            "required_capabilities": ["computer", "filesystem", "libreoffice_writer"],
            "metadata": {"category": "DOC", "application": "libreoffice_writer"},
        })
    return tmp_path / "tasks"


def _episodes(tmp_path: Path, task_ids: tuple[str, ...]) -> Path:
    root = tmp_path / "episodes" / "model" / "run"
    for task_id in task_ids:
        task_root = root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        (task_root / "traj.jsonl").write_text(
            json.dumps({
                "step_num": 1,
                "response": "done",
                "tool_call": {"name": "computer", "arguments": {}},
            })
            + "\n",
            encoding="utf-8",
        )
        (task_root / "result.txt").write_text("1.0\n", encoding="utf-8")
    return tmp_path / "episodes"


def _config(tmp_path: Path, *, require_sandbox: bool = False) -> Path:
    path = tmp_path / "cuamaster.propagation.json"
    _write_json(path, {
        "schema_version": 1,
        "sources": {"tasks": [str(_source_tasks(tmp_path))], "episodes": []},
        "environment": {
            "image_store": str(_image_store(tmp_path)),
            "image_id": "ubuntu-test",
        },
        "generation": {
            "backend": "deterministic",
            "model": "deterministic-test",
            "variants_per_anchor": 2,
            "min_primitives": 2,
            "max_primitives": 3,
        },
        "validation": {
            "require_sandbox": require_sandbox,
            "max_repairs": 1,
        },
        "output_root": str(tmp_path / "runs"),
    })
    return path


def test_shared_task_contract_is_reexported():
    assert GuiTaskSpec is TaskSpec


def test_repository_root_module_entrypoint_uses_src_pythonpath():
    repository_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuamaster.propagation.generate",
            "--help",
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--trajectory-dir" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_cli_matches_action_construction_override_style(tmp_path):
    config = load_config(_config(tmp_path))
    task_root = tmp_path / "custom-tasks"
    trajectory_root = tmp_path / "custom-trajectories"
    vm_images = tmp_path / "custom-vm-images"
    output_dir = tmp_path / "custom-output"
    args = parse_args([
        "--config",
        str(config.path),
        "--input",
        str(task_root),
        "--trajectory-dir",
        str(trajectory_root),
        "--output-dir",
        str(output_dir),
        "--vm-images-dir",
        str(vm_images),
        "--image-id",
        "custom-image",
        "custom-image-v2",
        "--vm-selection",
        "random",
        "--vm-sample-size",
        "2",
        "--vm-seed",
        "99",
        "--include-pending-vms",
        "--vm-assignment",
        "balanced_random",
        "--vms-per-task",
        "2",
        "--platform-policy",
        "primitive_compatible",
        "--task-id",
        "001",
        "013",
        "--require-trajectories",
        "--concurrency",
        "8",
        "--variants-per-anchor",
        "5",
        "--allow-cross-source",
    ])

    overridden = _apply_run_overrides(config, args)

    assert args.command == "run"
    assert args.stage == "validate"
    assert args.output_dir == str(output_dir)
    assert overridden.sources.task_roots == (str(task_root.resolve()),)
    assert overridden.sources.episode_roots == (str(trajectory_root.resolve()),)
    assert overridden.sources.task_ids == ("001", "013")
    assert overridden.environment.image_store == str(vm_images.resolve())
    assert overridden.environment.image_id == ""
    assert overridden.environment.image_ids == ("custom-image", "custom-image-v2")
    assert overridden.environment.selection == "random"
    assert overridden.environment.sample_size == 2
    assert overridden.environment.random_seed == 99
    assert overridden.environment.include_pending is True
    assert overridden.environment.assignment == "balanced_random"
    assert overridden.environment.environments_per_task == 2
    assert overridden.environment.platform_policy == "primitive_compatible"
    assert overridden.sources.require_episodes is True
    assert overridden.generation.concurrency == 8
    assert overridden.generation.variants_per_anchor == 5
    assert overridden.generation.allow_cross_source is True
    assert config.generation.variants_per_anchor == 2
    assert config.generation.concurrency == 1
    assert config.generation.allow_cross_source is False


def test_pipeline_runs_decomposition_and_composition_concurrently(
    tmp_path,
    monkeypatch,
):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"]["concurrency"] = 2
    raw["generation"]["variants_per_anchor"] = 1
    _write_json(path, raw)

    class TrackingBackend(DeterministicGeneratorBackend):
        def __init__(self):
            super().__init__("tracking")
            self.lock = threading.Lock()
            self.active = {"decompose": 0, "compose": 0}
            self.peak = {"decompose": 0, "compose": 0}

        def _track(self, operation, callback):
            with self.lock:
                self.active[operation] += 1
                self.peak[operation] = max(
                    self.peak[operation],
                    self.active[operation],
                )
            try:
                time.sleep(0.05)
                return callback()
            finally:
                with self.lock:
                    self.active[operation] -= 1

        def decompose(self, source, episodes, environment):
            return self._track(
                "decompose",
                lambda: super(TrackingBackend, self).decompose(
                    source,
                    episodes,
                    environment,
                ),
            )

        def compose(self, recipe, anchor, capabilities, environment):
            return self._track(
                "compose",
                lambda: super(TrackingBackend, self).compose(
                    recipe,
                    anchor,
                    capabilities,
                    environment,
                ),
            )

    backend = TrackingBackend()
    monkeypatch.setattr(
        "cuamaster.propagation.pipeline.create_generator_backend",
        lambda *_args, **_kwargs: backend,
    )

    result = TaskGenerationPipeline(load_config(path)).run(
        stop_after="compose",
        run_id="concurrent-generation",
    )

    assert backend.peak == {"decompose": 2, "compose": 2}
    assert result["stats"]["concurrency"] == 2
    assert result["stats"]["candidates"] == 2


def test_pipeline_persists_generated_task_before_compose_batch_finishes(
    tmp_path,
    monkeypatch,
):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"]["concurrency"] = 2
    raw["generation"]["variants_per_anchor"] = 1
    _write_json(path, raw)

    first_compose_finished = threading.Event()
    release_second_compose = threading.Event()

    class BlockingBackend(DeterministicGeneratorBackend):
        def __init__(self):
            super().__init__("blocking")
            self.lock = threading.Lock()
            self.compose_calls = 0

        def compose(self, recipe, anchor, capabilities, environment):
            with self.lock:
                self.compose_calls += 1
                call_number = self.compose_calls
            generated = super().compose(recipe, anchor, capabilities, environment)
            if call_number == 1:
                first_compose_finished.set()
                return generated
            if not release_second_compose.wait(timeout=5):
                raise TimeoutError("second compose was not released")
            return generated

    backend = BlockingBackend()
    monkeypatch.setattr(
        "cuamaster.propagation.pipeline.create_generator_backend",
        lambda *_args, **_kwargs: backend,
    )
    output_dir = tmp_path / "incremental-output"
    results = []
    errors = []

    def run_pipeline():
        try:
            results.append(
                TaskGenerationPipeline(load_config(path)).run(
                    stop_after="compose",
                    output_dir=output_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    pipeline_thread = threading.Thread(target=run_pipeline)
    pipeline_thread.start()
    try:
        assert first_compose_finished.wait(timeout=2)
        deadline = time.monotonic() + 2
        candidate_files = []
        while time.monotonic() < deadline:
            candidate_files = list(
                (output_dir / "candidates").glob("*/generated_task.json")
            )
            if candidate_files:
                break
            time.sleep(0.01)
        assert len(candidate_files) == 1
        assert pipeline_thread.is_alive()
    finally:
        release_second_compose.set()
        pipeline_thread.join(timeout=5)

    assert not pipeline_thread.is_alive()
    assert errors == []
    assert results[0]["stats"]["candidates"] == 2


def test_pipeline_reuses_existing_candidates_when_expanding_variants(
    tmp_path,
    monkeypatch,
):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"]["variants_per_anchor"] = 1
    _write_json(path, raw)
    for task_id, instruction in (
        (
            "writer-format",
            "Create a Writer document; then make the title bold; then save the document.",
        ),
        (
            "writer-export",
            "Load an existing Writer report; then export it as PDF; then verify the PDF exists.",
        ),
    ):
        task_path = tmp_path / "tasks" / "DOC" / f"{task_id}.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["instruction"] = instruction
        _write_json(task_path, task)
    output_dir = tmp_path / "resumable-output"

    first_result = TaskGenerationPipeline(load_config(path)).run(
        stop_after="compose",
        output_dir=output_dir,
    )
    existing_files = {
        candidate.parent.name: candidate.read_bytes()
        for candidate in (output_dir / "candidates").glob("*/generated_task.json")
    }
    assert first_result["stats"]["candidates"] == 2

    raw["generation"]["variants_per_anchor"] = 2
    _write_json(path, raw)

    class CountingBackend(DeterministicGeneratorBackend):
        def __init__(self):
            super().__init__("counting")
            self.compose_calls = 0

        def compose(self, recipe, anchor, capabilities, environment):
            self.compose_calls += 1
            return super().compose(recipe, anchor, capabilities, environment)

    backend = CountingBackend()
    monkeypatch.setattr(
        "cuamaster.propagation.pipeline.create_generator_backend",
        lambda *_args, **_kwargs: backend,
    )

    second_result = TaskGenerationPipeline(load_config(path)).run(
        stop_after="compose",
        output_dir=output_dir,
    )
    final_files = {
        candidate.parent.name: candidate.read_bytes()
        for candidate in (output_dir / "candidates").glob("*/generated_task.json")
    }

    assert second_result["stats"]["candidates"] == 4
    assert second_result["stats"]["reused_candidates"] == 2
    assert second_result["stats"]["generated_candidates"] == 2
    assert backend.compose_calls == 2
    assert final_files.keys() > existing_files.keys()
    assert all(final_files[task_id] == payload for task_id, payload in existing_files.items())


def test_pipeline_fresh_run_clears_existing_candidates(tmp_path):
    config = load_config(_config(tmp_path))
    output_dir = tmp_path / "fresh-output"
    TaskGenerationPipeline(config).run(
        stop_after="compose",
        output_dir=output_dir,
    )
    assert list((output_dir / "candidates").glob("*/generated_task.json"))

    TaskGenerationPipeline(config).run(
        stop_after="index",
        output_dir=output_dir,
        fresh=True,
    )

    assert not (output_dir / "candidates").exists()


def test_cli_can_disable_cross_source_without_overriding_variant_count(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"]["allow_cross_source"] = True
    _write_json(path, raw)
    config = load_config(path)
    args = parse_args([
        "compose",
        "--config",
        str(config.path),
        "--no-allow-cross-source",
    ])

    overridden = _apply_run_overrides(config, args)

    assert overridden.generation.variants_per_anchor == 2
    assert overridden.generation.allow_cross_source is False


def test_pipeline_output_dir_is_exact_run_directory(tmp_path):
    config = load_config(_config(tmp_path))
    output_dir = tmp_path / "exact-run-output"

    result = TaskGenerationPipeline(config).run(
        stop_after="index",
        output_dir=output_dir,
    )

    assert Path(result["run_dir"]) == output_dir.resolve()
    assert (output_dir / "source_tasks.jsonl").is_file()


def test_pipeline_reports_progress_to_stderr(tmp_path, capsys):
    config = load_config(_config(tmp_path))

    TaskGenerationPipeline(config).run(run_id="progress-run")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[task-generation] run: start stage=validate" in captured.err
    assert "[task-generation] index: loaded sources=2" in captured.err
    assert "[task-generation] decompose: start total=2" in captured.err
    assert "start stage=decompose task=" in captured.err
    assert "record=" in captured.err
    assert "stage=decompose throughput=" in captured.err
    assert " elapsed=" in captured.err
    assert " eta=" in captured.err
    assert "ok task=" in captured.err
    assert "decompose: done" in captured.err
    assert "counts={'ok': 2}" in captured.err
    assert "[task-generation] compose: prepared" in captured.err
    assert "start stage=compose task=" in captured.err
    assert "compose: done" in captured.err
    assert "start stage=validate task=" in captured.err
    assert "validate: done" in captured.err
    assert "[task-generation] run: complete stage=validate" in captured.err


def test_pipeline_filters_to_trajectory_tasks_and_balances_vm_assignment(tmp_path):
    path = _config(tmp_path)
    task_root = _source_tasks(tmp_path)
    _write_json(task_root / "DOC" / "writer-without-trajectory.json", {
        "id": "writer-without-trajectory",
        "instruction": "Open Writer and save an empty document.",
        "evaluator": {"type": "file_exists", "path": "/tmp/empty"},
        "required_capabilities": ["computer", "filesystem", "libreoffice_writer"],
        "metadata": {"category": "DOC", "application": "libreoffice_writer"},
    })
    episode_root = _episodes(tmp_path, ("writer-format", "writer-export"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sources"] = {
        "tasks": [str(task_root)],
        "episodes": [str(episode_root)],
        "require_episodes": True,
    }
    raw["environment"].update({
        "image_ids": ["*"],
        "selection": "all",
        "assignment": "balanced_random",
        "environments_per_task": 1,
    })
    raw["environment"].pop("image_id", None)
    _write_json(path, raw)

    result = TaskGenerationPipeline(load_config(path)).run(
        stop_after="index",
        run_id="balanced-assignment",
    )
    matrix = [
        json.loads(line)
        for line in (Path(result["run_dir"]) / "source_environment_matrix.jsonl")
        .read_text()
        .splitlines()
    ]

    assert result["stats"]["sources"] == 2
    assert result["stats"]["compatible_source_environment_pairs"] == 4
    assert result["stats"]["assigned_source_environment_pairs"] == 2
    assert sorted(result["stats"]["assignments_per_environment"].values()) == [1, 1]
    assert sum(row["assigned"] for row in matrix) == 2
    assert {row["task_id"] for row in matrix} == {"writer-format", "writer-export"}


def test_platform_balanced_assignment_prefers_distinct_os_families(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["environment"].update({
        "image_ids": ["*"],
        "selection": "all",
        "include_pending": True,
        "primitive_overrides": {
            "windows-pending": ["libreoffice_writer"],
            "macos-pending": ["libreoffice_writer"],
        },
        "assignment": "platform_balanced",
        "environments_per_task": 3,
        "platform_policy": "primitive_compatible",
    })
    raw["environment"].pop("image_id", None)
    _write_json(path, raw)

    result = TaskGenerationPipeline(load_config(path)).run(
        stop_after="index",
        run_id="platform-balanced-assignment",
    )
    run_dir = Path(result["run_dir"])
    contexts = json.loads((run_dir / "environment_contexts.json").read_text())
    platform_by_environment = {
        item["image_id"]: str(item["os"]["type"]).lower()
        for item in contexts["selected"]
    }
    matrix = [
        json.loads(line)
        for line in (run_dir / "source_environment_matrix.jsonl").read_text().splitlines()
    ]
    assigned_by_task = {}
    for row in matrix:
        if row["assigned"]:
            assigned_by_task.setdefault(row["task_id"], set()).add(
                platform_by_environment[row["environment_id"]]
            )

    assert result["stats"]["assigned_source_environment_pairs"] == 6
    assert all(
        platforms == {"ubuntu", "windows", "darwin"}
        for platforms in assigned_by_task.values()
    )


def test_capability_compatible_policy_allows_cross_os_assignment(tmp_path):
    path = _config(tmp_path)
    for task_path in (tmp_path / "tasks" / "DOC").glob("*.json"):
        task = json.loads(task_path.read_text())
        task["metadata"]["platform"] = "linux"
        _write_json(task_path, task)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["environment"] = {
        "image_store": raw["environment"]["image_store"],
        "image_ids": ["windows-pending"],
        "selection": "fixed",
        "include_pending": True,
        "primitive_overrides": {
            "windows-pending": ["libreoffice_writer"],
        },
        "assignment": "balanced_random",
        "environments_per_task": 1,
        "platform_policy": "strict",
    }
    _write_json(path, raw)

    strict = TaskGenerationPipeline(load_config(path)).run(
        stop_after="index",
        run_id="strict-platform",
    )

    raw["environment"]["platform_policy"] = "primitive_compatible"
    _write_json(path, raw)
    compatible = TaskGenerationPipeline(load_config(path)).run(
        stop_after="index",
        run_id="capability-platform",
    )

    assert strict["stats"]["assigned_source_environment_pairs"] == 0
    assert compatible["stats"]["assigned_source_environment_pairs"] == 2
    assert compatible["stats"]["assignments_per_environment"] == {
        "windows-pending": 2
    }


def test_evaluator_safety_rejects_generated_network_and_subprocess():
    report = inspect_evaluator_code(
        "import requests\nimport subprocess\nsubprocess.run(['rm', '-rf', '/tmp/x'])"
    )
    assert report["safe"] is False
    assert "banned import: requests" in report["errors"]
    assert "banned call: subprocess.run" in report["errors"]


def test_task_generation_pipeline_and_compilers(tmp_path):
    config = load_config(_config(tmp_path))
    result = TaskGenerationPipeline(config).run(run_id="test-run")
    run_dir = Path(result["run_dir"])
    assert result["stats"]["sources"] == 2
    assert result["stats"]["capabilities"] >= 4
    assert result["stats"]["candidates"] >= 1
    assert result["stats"]["valid_candidates"] == result["stats"]["candidates"]
    assert result["stats"]["promoted_candidates"] == result["stats"]["valid_candidates"]
    assert (run_dir / "promoted" / "promotion_manifest.json").is_file()

    candidate_root = run_dir / "candidates"
    weavebench = compile_weavebench_taskset(
        candidate_root,
        tmp_path / "compiled-weavebench",
        sources=load_source_tasks([tmp_path / "tasks"]),
    )
    osworld = compile_osworld_taskset(candidate_root, tmp_path / "compiled-osworld")
    assert weavebench["tasks"]
    assert osworld["tasks"]
    compiled_task = next((tmp_path / "compiled-weavebench" / "DOC").glob("*.md"))
    text = compiled_task.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "gui: true" in text
    assert "## Automated Checks" in text
    assert "def grade(" in text


def test_weavebench_compiler_marks_legacy_osworld_tasks(tmp_path):
    generated_root = tmp_path / "candidates"
    task_dir = generated_root / "legacy-task"
    _write_json(task_dir / "generated_task.json", {
        "task": {
            "id": "legacy-task",
            "instruction": "Open the supplied file.",
            "config": [{"type": "open", "parameters": {"path": "/tmp/a.txt"}}],
            "evaluator": {
                "func": "exact_match",
                "result": {"type": "vm_command_line", "command": "true"},
                "expected": {"type": "rule", "rules": {"expected": ""}},
            },
            "metadata": {"category": "os", "legacy_osworld": {"proxy": False}},
            "target_environments": [{"os_type": "Linux"}],
        },
        "recipe": {
            "id": "recipe",
            "anchor_task_id": "source",
        },
    })

    manifest = compile_weavebench_taskset(generated_root, tmp_path / "compiled")

    assert manifest["tasks"][0]["execution_adapter"] == "legacy_osworld"
    assert manifest["tasks"][0]["proxy"] is False
    workspace = tmp_path / "compiled/workspace/os/legacy-task"
    assert all((workspace / name).is_dir() for name in ("exec", "gt", "tmp"))
    markdown = (tmp_path / "compiled/os/legacy-task.md").read_text()
    assert "open /tmp/a.txt" not in markdown
    assert "def grade(" not in markdown


def test_weavebench_compiler_skips_grader_for_osworld_v2_task_class(tmp_path):
    generated_root = tmp_path / "candidates"
    task_dir = generated_root / "osworld-v2-task"
    _write_json(task_dir / "generated_task.json", {
        "task": {
            "id": "osworld-v2-task",
            "instruction": "Complete the desktop task.",
            "config": [{"type": "osworld_v2_task_class"}],
            "evaluator": {
                "type": "osworld_v2_task_class",
                "path": "/repo/task.py",
                "repository_root": "/repo",
                "class_name": "Task001",
            },
            "metadata": {"category": "osworld_v2"},
            "target_environments": [{"os_type": "Linux"}],
        },
        "recipe": {
            "id": "recipe",
            "anchor_task_id": "source",
        },
    })

    compile_weavebench_taskset(generated_root, tmp_path / "compiled")

    markdown = (tmp_path / "compiled/osworld_v2/osworld-v2-task.md").read_text()
    automated_checks = markdown.split(
        "## Automated Checks", 1
    )[1].split("## Workspace Path", 1)[0]
    assert not automated_checks.strip()
    assert "def grade(" not in markdown


def test_task_generation_randomly_samples_multiple_compatible_vms(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["environment"] = {
        "image_store": raw["environment"]["image_store"],
        "image_ids": ["*"],
        "selection": "random",
        "sample_size": 2,
        "random_seed": 17,
    }
    raw["generation"]["variants_per_anchor"] = 1
    _write_json(path, raw)

    result = TaskGenerationPipeline(load_config(path)).run(run_id="multi-vm")
    run_dir = Path(result["run_dir"])
    selection = json.loads((run_dir / "environment_contexts.json").read_text())
    selected_ids = {item["image_id"] for item in selection["selected"]}
    generated = [
        json.loads(path.read_text())
        for path in (run_dir / "candidates").glob("*/generated_task.json")
    ]
    generated_environment_ids = {
        item["task"]["metadata"]["generation"]["environment_id"]
        for item in generated
    }

    assert selected_ids == {"ubuntu-test", "ubuntu-test-v2"}
    assert "windows-pending" not in selected_ids
    assert "macos-pending" not in selected_ids
    assert result["stats"]["environments"] == 2
    assert result["stats"]["compatible_source_environment_pairs"] == 4
    assert generated_environment_ids == selected_ids
    assert len({item["task"]["id"] for item in generated}) == len(generated)
    assert all(item["recipe"]["environment_id"] in selected_ids for item in generated)
    assert all(
        item["task"]["target_environments"][0]["os_type"] == "Ubuntu"
        for item in generated
    )


def test_task_generation_includes_pending_windows_and_macos_with_overrides(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["environment"] = {
        "image_store": raw["environment"]["image_store"],
        "image_ids": ["*"],
        "selection": "all",
        "sample_size": 4,
        "include_pending": True,
        "primitive_overrides": {
            "windows-pending": ["libreoffice_writer"],
            "macos-pending": ["libreoffice_writer"],
        },
    }
    raw["generation"]["variants_per_anchor"] = 1
    _write_json(path, raw)

    result = TaskGenerationPipeline(load_config(path)).run(run_id="cross-platform")
    run_dir = Path(result["run_dir"])
    selection = json.loads((run_dir / "environment_contexts.json").read_text())
    selected = {item["image_id"]: item for item in selection["selected"]}
    generated = [
        json.loads(candidate.read_text())
        for candidate in (run_dir / "candidates").glob("*/generated_task.json")
    ]
    generated_environment_ids = {
        item["task"]["metadata"]["generation"]["environment_id"]
        for item in generated
    }

    assert set(selected) == {
        "ubuntu-test",
        "ubuntu-test-v2",
        "windows-pending",
        "macos-pending",
    }
    assert selected["windows-pending"]["policy"]["inventory_verified"] is False
    assert selected["macos-pending"]["policy"]["inventory_verified"] is False
    assert selected["windows-pending"]["policy"]["primitive_overrides"] == [
        "libreoffice_writer"
    ]
    assert generated_environment_ids == set(selected)
    assert result["stats"]["compatible_source_environment_pairs"] == 8
    assert {
        item["task"]["target_environments"][0]["os_type"]
        for item in generated
    } == {"Ubuntu", "Windows", "Darwin"}


def test_sandbox_required_rejects_static_verification(tmp_path):
    config = load_config(_config(tmp_path, require_sandbox=True))
    result = TaskGenerationPipeline(config).run(run_id="sandbox-required")
    assert result["stats"]["valid_candidates"] == 0
    assert result["stats"]["rejected_candidates"] >= 1


def test_release_gate_automatically_promotes_valid_candidates(tmp_path):
    config = load_config(_config(tmp_path))
    result = TaskGenerationPipeline(config).run(run_id="release-run")
    run_dir = Path(result["run_dir"])
    task_ids = sorted(path.name for path in (run_dir / "candidates").iterdir() if path.is_dir())
    manifest = promote_candidates(run_dir)
    assert manifest["promoted"] == task_ids
    assert manifest["policy"] == "automatic_after_validation"
    assert (run_dir / "promoted" / "promotion_manifest.json").is_file()


def test_release_gate_promotes_nested_candidate_ids(tmp_path):
    run_dir = tmp_path / "run"
    task_id = "gen-business_finance/example-task"
    _write_json(
        run_dir / "candidates" / task_id / "generated_task.json",
        {"task": {"id": task_id}},
    )
    _write_json(run_dir / "validation" / f"{task_id}.json", {"valid": True})
    _write_json(
        run_dir / "quality" / f"{task_id}.json",
        CandidateQualityScore(
            task_id=task_id,
            total_score=1.0,
            dimensions={},
            hard_gate_passed=True,
            eligible=True,
        ).to_dict(),
    )

    manifest = promote_candidates(run_dir)

    assert manifest["promoted"] == [task_id]
    assert (run_dir / "promoted" / task_id / "generated_task.json").is_file()


def test_kimi_config_matches_action_construction_api_controls(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"].update({
        "backend": "kimi",
        "model": "moonshotai--Kimi-K3",
        "base_urls": ["http://127.0.0.1:30000/v1"],
        "api_key_env": "KIMI_API_KEY",
        "reasoning_effort": "none",
        "max_tokens": 16384,
        "max_retries": 20,
        "request_delay": 0.1,
        "timeout_seconds": 7200,
    })
    _write_json(path, raw)
    generation = load_config(path).generation
    assert generation.backend == "kimi"
    assert generation.base_urls == ("http://127.0.0.1:30000/v1",)
    assert generation.max_retries == 20
    assert generation.max_tokens == 16384


def test_kimi_config_loads_shared_endpoint_pool(tmp_path):
    path = _config(tmp_path)
    pool_path = tmp_path / "kimi.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: kimi-k3\nbase_urls:\n  - model-a:30000\n",
        encoding="utf-8",
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"].update({
        "backend": "kimi",
        "endpoint_pool": "kimi.yaml",
    })
    raw["generation"].pop("base_url", None)
    raw["generation"].pop("base_urls", None)
    raw["generation"].pop("model", None)
    _write_json(path, raw)

    generation = load_config(path).generation

    assert generation.model == "kimi-k3"
    assert generation.base_urls == ("http://model-a:30000/v1",)
    assert generation.endpoint_pool == str(pool_path.resolve())


def test_generation_inherits_responses_api_defaults_from_endpoint_pool(tmp_path):
    path = _config(tmp_path)
    pool_path = tmp_path / "gpt.yaml"
    pool_path.write_text(
        """
schema_version: 1
model: GPT-5.6-sol
base_urls:
  - https://llm-gateway.example.com/v1
api:
  wire_api: responses
  api_key_env: LLM_GATEWAY_API_KEY
  reasoning_effort: xhigh
  store: false
  trust_env: false
""".strip(),
        encoding="utf-8",
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"].update({
        "backend": "openai",
        "endpoint_pool": "gpt.yaml",
    })
    for key in ("base_url", "base_urls", "model", "api_key_env", "reasoning_effort"):
        raw["generation"].pop(key, None)
    _write_json(path, raw)

    generation = load_config(path).generation

    assert generation.model == "GPT-5.6-sol"
    assert generation.wire_api == "responses"
    assert generation.api_key_env == "LLM_GATEWAY_API_KEY"
    assert generation.reasoning_effort == "xhigh"
    assert generation.store is False
    assert generation.trust_env is False


def test_kimi_backend_retries_and_parses_fenced_json(tmp_path, monkeypatch):
    calls = []
    client_options = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary failure")
            message = SimpleNamespace(content='```json\n{"capabilities": []}\n```')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            client_options.append(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=FakeHTTPClient))
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = KimiGeneratorBackend(
        model="moonshotai--Kimi-K3",
        base_urls=("http://10.0.0.1:30000",),
        api_key_env="KIMI_API_KEY",
        reasoning_effort="none",
        max_tokens=16384,
        max_retries=2,
        request_delay=0,
        timeout_seconds=30,
        cache_dir=tmp_path / "cache",
    )
    result = backend._request("decompose", {"source": {"task": {"id": "t1"}}})
    assert result == {"capabilities": []}
    assert len(calls) == 2
    assert client_options[0]["api_key"] == "EMPTY"
    assert calls[-1]["max_tokens"] == 16384
    assert "response_format" not in calls[-1]
    assert "reasoning_effort" not in calls[-1]
    assert _parse_json_object("prefix {\"ok\": true} suffix") == {"ok": True}


def test_kimi_cache_key_does_not_depend_on_endpoint(tmp_path, monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=FakeHTTPClient))
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    options = {
        "model": "moonshotai--Kimi-K3",
        "api_key_env": "KIMI_API_KEY",
        "reasoning_effort": "none",
        "max_tokens": 16384,
        "max_retries": 2,
        "request_delay": 0,
        "timeout_seconds": 30,
        "cache_dir": tmp_path / "cache",
    }
    first = KimiGeneratorBackend(
        base_urls=("http://10.0.0.1:30000/v1",),
        **options,
    )
    second = KimiGeneratorBackend(
        base_urls=("http://10.0.0.2:30000/v1",),
        **options,
    )
    payload = {"source": {"task": {"id": "t1"}}}

    assert first._cache_path("decompose", payload) == second._cache_path(
        "decompose", payload
    )


def test_kimi_backend_migrates_endpoint_aware_cache(tmp_path, monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("legacy cache should avoid an API request")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=FakeHTTPClient))
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = KimiGeneratorBackend(
        model="moonshotai--Kimi-K3",
        base_urls=("http://10.0.0.2:30000/v1",),
        api_key_env="KIMI_API_KEY",
        reasoning_effort="none",
        max_tokens=16384,
        max_retries=2,
        request_delay=0,
        timeout_seconds=30,
        cache_dir=tmp_path / "cache",
    )
    payload = {"source": {"task": {"id": "t1"}}}
    legacy_path = backend._legacy_cache_path(
        "decompose",
        payload,
        ("http://10.0.0.1:30000/v1",),
    )
    assert legacy_path is not None
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        json.dumps({
            "raw": '{"capabilities": []}',
            "operation": "decompose",
            "model": backend.model,
            "base_url": "http://10.0.0.1:30000/v1",
        }),
        encoding="utf-8",
    )

    result = backend._request("decompose", payload)

    assert result == {"capabilities": []}
    assert not calls
    assert backend._cache_path("decompose", payload).is_file()
    assert backend.usage_stats()["decompose"]["cached"] == 1


def test_kimi_capability_labels_are_normalized_to_contract():
    source = SourceTask(
        task=TaskSpec(id="task-1", instruction="Save a Writer document."),
        source_path="task.json",
        format="json",
        application="libreoffice_writer",
        category="DOC",
    )
    environment = EnvironmentContext(
        image_id="ubuntu-test",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "filesystem", "libreoffice_writer"),
    )
    raw = {
        "id": "writer.save.1",
        "application": "Writer",
        "intent": "Save the document.",
        "inputs": {"path": "/tmp/result.odt"},
        "config_dependencies": ["Output path must be writable."],
        "evaluator_fragments": [{"type": "file_exists", "path": "/tmp/result.odt"}],
        "required_vm_capabilities": ["computer", "invented_capability"],
        "source_evidence": [],
        "confidence": 1.4,
    }
    capability = AbilityPrimitive.from_dict(
        _normalize_primitive(raw, source, environment)
    )
    assert capability.application == "libreoffice_writer"
    assert capability.config_dependencies == ({"description": "Output path must be writable."},)
    assert capability.required_vm_capabilities == ("computer",)
    assert capability.source_evidence[0].task_id == "task-1"
    assert capability.confidence == 1.0


@pytest.mark.parametrize(
    "task_patch",
    (
        {"metadata": {"repair": "partial"}},
        {"instruction": "   ", "metadata": {"repair": "blank-instruction"}},
    ),
)
def test_json_backend_repair_preserves_instruction_for_partial_or_blank_patch(
    task_patch,
):
    class StubJSONBackend(_JSONGeneratorBackend):
        model = "stub"

        def _request(self, operation, payload):
            assert operation == "repair"
            return {"task": task_patch}

    generated = GeneratedTask(
        task=TaskSpec(
            id="gen-task",
            instruction="Keep the original user-visible goal.",
            evaluator={"type": "file_exists", "path": "/tmp/result"},
            required_capabilities=("computer",),
            metadata={"application": "libreoffice_writer"},
        ),
        recipe=TaskRecipe(
            id="recipe-1",
            anchor_task_id="source-1",
            primitive_ids=("capability-1",),
            application="libreoffice_writer",
            strategy="compose",
            expected_effects=("result exists",),
            required_vm_capabilities=("computer",),
            source_task_ids=("source-1",),
        ),
    )
    environment = EnvironmentContext(
        image_id="ubuntu-test",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "libreoffice_writer"),
    )

    repaired = StubJSONBackend().repair(generated, {"valid": False}, environment)

    assert repaired.task.instruction == generated.task.instruction
    assert "repair" not in repaired.task.metadata


def test_osworld_v2_task_class_and_raw_trajectory_ingestion(tmp_path):
    task_root = tmp_path / "OSWorld-V2" / "evaluation_examples" / "task_class"
    task_path = task_root / "task_001.py"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        """
TASK_ID = "001"
class Task001(BaseTask):
    id = TASK_ID
    snapshot = "thunderbird"
    _base_instruction = "Read the message and update the calendar."
    instruction = _base_instruction
    related_apps = ["thunderbird", "calendar"]
    def setup(self, controller):
        pass
    def evaluate(self, env):
        return 1.0
""".strip(),
        encoding="utf-8",
    )
    episode_dir = (
        tmp_path
        / "verified"
        / "claude_computer_use"
        / "screenshot"
        / "claude-opus"
        / "tasks"
        / "001"
    )
    episode_dir.mkdir(parents=True)
    (episode_dir / "result.txt").write_text("1.0\n", encoding="utf-8")
    (episode_dir / "traj.jsonl").write_text(
        "\n".join((
            json.dumps({
                "step_num": 1,
                "response": "Open Thunderbird.",
                "tool_call": {"name": "computer", "input": {"action": "left_click"}},
                "screenshot_file": "step_1.png",
            }),
            json.dumps({
                "step_num": 2,
                "response": "Create the calendar event.",
                "tool_call": {"name": "computer", "input": {"action": "type"}},
                "screenshot_file": "step_2.png",
            }),
        )) + "\n",
        encoding="utf-8",
    )

    tasks = load_source_tasks([task_root])
    episodes = load_episodes([tmp_path / "verified"])
    assert len(tasks) == 1
    assert tasks[0].task.id == "001"
    assert tasks[0].task.required_capabilities == (
        "calendar",
        "computer",
        "filesystem",
        "thunderbird",
    )
    assert tasks[0].task.config[0]["type"] == "osworld_v2_task_class"
    episode = episodes["001"][0]
    assert episode.success is True
    assert len(episode.steps) == 2
    assert episode.steps[0]["screenshot"].endswith("step_1.png")
    assert episode.to_prompt_dict()["steps"][1]["response"] == "Create the calendar event."


def test_ale_task_card_and_collection_manifest_trajectory_ingestion(tmp_path):
    task_root = tmp_path / "agents-last-exam" / "tasks"
    task_dir = task_root / "business_finance" / "american_option_pricing_ls"
    task_dir.mkdir(parents=True)
    (task_dir / "main.py").write_text("def load():\n    return []\n", encoding="utf-8")
    _write_json(task_dir / "task_card.json", {
        "taskId": "business_finance/american_option_pricing_ls",
        "category": "business_finance",
        "title": "American Option Pricing",
        "summary": "Produce the required pricing outputs.",
        "taskPrompt": "Read the problem specification and produce the required output files.",
        "software": ["Python", "NumPy"],
        "evaluation": "The output files are scored by the task evaluator.",
        "vm": {"snapshot": "cpu-free-ubuntu"},
        "inputFiles": [{"path": "input/problem_spec.md"}],
        "referenceFiles": [{"path": "output/results.json"}],
    })

    episode_root = tmp_path / "traj_ale_success"
    episode_dir = (
        episode_root
        / "openclaw_cli"
        / "screenshot"
        / "deepseek-v4-1-flash-a"
        / "tasks"
        / "business_finance__american_option_pricing_ls__v0"
    )
    episode_dir.mkdir(parents=True)
    (episode_dir / "result.txt").write_text("1.0\n", encoding="utf-8")
    (episode_dir / "traj.jsonl").write_text(
        json.dumps({
            "step_num": 1,
            "response": "Inspect the task inputs.",
            "tool_call": {"name": "exec", "input": {"command": "ls input"}},
            "screenshot_file": None,
        }) + "\n",
        encoding="utf-8",
    )
    (episode_root / "collection_manifest.jsonl").write_text(
        json.dumps({
            "episode": "business_finance__american_option_pricing_ls__v0",
            "episode_id": "episode-1",
            "task_path": "business_finance/american_option_pricing_ls",
            "model": "deepseek-v4-1-flash-a",
        }) + "\n",
        encoding="utf-8",
    )

    tasks = load_source_tasks([task_root])
    episodes = load_episodes([episode_root])

    assert len(tasks) == 1
    assert tasks[0].task.id == "business_finance/american_option_pricing_ls"
    assert tasks[0].format == "ale_task_card"
    assert tasks[0].task.config[0]["type"] == "ale_task_module"
    assert tasks[0].task.evaluator["type"] == "task_evaluator"
    assert tasks[0].task.target_environments[0].os_type == "Ubuntu"
    episode = episodes["business_finance/american_option_pricing_ls"][0]
    assert episode.episode_id == "episode-1"
    assert episode.success is True
    assert episode.steps[0]["tool_call"]["name"] == "exec"


def test_config_supports_source_task_id_filter(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sources"]["task_ids"] = ["writer-format"]
    _write_json(path, raw)
    config = load_config(path)
    assert config.sources.task_ids == ("writer-format",)
    result = TaskGenerationPipeline(config).run(run_id="filtered-run")
    assert result["stats"]["sources"] == 1


def test_osworld_compiler_emits_v2_task_class_wrapper(tmp_path):
    generated_root = tmp_path / "promoted"
    candidate = generated_root / "gen-task"
    source_path = tmp_path / "OSWorld-V2" / "evaluation_examples" / "task_class" / "task_001.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("TASK_CLASS = object\n", encoding="utf-8")
    _write_json(candidate / "generated_task.json", {
        "task": {
            "id": "gen-task",
            "instruction": "Open the message and save its attachment.",
            "snapshot": "thunderbird",
            "config": [{
                "type": "osworld_v2_task_class",
                "task_id": "001",
                "class_name": "Task001",
                "path": str(source_path),
                "repository_root": str(source_path.parents[2]),
            }],
            "evaluator": {
                "type": "task_evaluator",
                "class_name": "Task001",
                "method": "evaluate",
                "path": str(source_path),
            },
            "metadata": {"related_apps": ["thunderbird"]},
        },
        "recipe": {},
    })
    manifest = compile_osworld_taskset(generated_root, tmp_path / "compiled")
    assert manifest["tasks"][0]["format"] == "osworld_v2_task_class"
    wrapper = tmp_path / "compiled" / manifest["tasks"][0]["path"]
    assert "class GeneratedTask_gen_task(_SourceTask):" in wrapper.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "compiled" / "test_v2.json").read_text()) == {
        "tasks": ["gen-task"]
    }


def test_config_rejects_invalid_capability_limits(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"]["min_primitives"] = 4
    raw["generation"]["max_primitives"] = 2
    _write_json(path, raw)
    with pytest.raises(ValueError, match="capability limits"):
        load_config(path)


def test_pipeline_rejects_missing_required_episode_roots(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sources"]["episodes"] = [str(tmp_path / "missing-episodes")]
    raw["sources"]["require_episodes"] = True
    _write_json(path, raw)

    with pytest.raises(ValueError, match="no episodes were loaded"):
        TaskGenerationPipeline(load_config(path)).run(
            stop_after="index",
            run_id="missing-required-episodes",
        )


def test_task_class_recipes_cover_all_source_capabilities_for_each_variant():
    source = SourceTask(
        task=TaskSpec.from_dict({
            "id": "001",
            "instruction": "Complete the prepared workflow.",
            "config": [{
                "type": "osworld_v2_task_class",
                "task_id": "001",
                "class_name": "Task001",
                "path": "/repo/task_001.py",
            }],
            "evaluator": {
                "type": "task_evaluator",
                "class_name": "Task001",
                "method": "evaluate",
                "path": "/repo/task_001.py",
            },
            "required_capabilities": ["computer", "filesystem"],
        }),
        source_path="/repo/task_001.py",
        format="osworld_v2_task_class",
        application="google_chrome",
        category="OSWORLD_V2",
    )
    capabilities = [
        AbilityPrimitive(
            id=f"001::capability.{index}",
            application=application,
            intent=f"Complete outcome {index}",
            preconditions=(),
            effects=(f"effect-{index}",),
            required_vm_capabilities=("computer", "filesystem"),
            source_evidence=(SourceEvidence(task_id="001"),),
        )
        for index, application in enumerate(("google_chrome", "terminal", "calendar"), 1)
    ]
    environment = EnvironmentContext(
        image_id="windows-test",
        image_digest="digest",
        os={"type": "Windows", "version": "10", "architecture": "x86_64"},
        applications={},
        capabilities=("computer", "filesystem", "google_chrome", "terminal", "calendar"),
    )

    recipes = PrimitiveComposer(
        min_primitives=2,
        max_primitives=2,
        variants_per_anchor=3,
    ).recipes([source], capabilities, environment)

    assert len(recipes) == 3
    assert {recipe.variant_index for recipe in recipes} == {0, 1, 2}
    assert all(recipe.primitive_ids == tuple(item.id for item in capabilities) for recipe in recipes)
    assert len({recipe.id for recipe in recipes}) == 3


def test_ale_task_class_recipes_bypass_application_filter():
    source = SourceTask(
        task=TaskSpec.from_dict({
            "id": "business_finance/example",
            "instruction": "Complete the prepared ALE workflow.",
            "config": [{
                "type": "ale_task_module",
                "task_id": "business_finance/example",
                "path": "/repo/tasks/business_finance/example/main.py",
            }],
            "evaluator": {
                "type": "task_evaluator",
                "source_type": "ale_task_module",
                "task_id": "business_finance/example",
                "method": "evaluate",
                "path": "/repo/tasks/business_finance/example/main.py",
            },
            "required_capabilities": ["computer", "filesystem", "shell"],
        }),
        source_path="/repo/tasks/business_finance/example/task_card.json",
        format="ale_task_card",
        application="desktop_os",
        category="business_finance",
    )
    capabilities = [
        AbilityPrimitive(
            id=f"business_finance/example::{application}.operation.{index}",
            application=application,
            intent=f"Complete ALE operation {index}",
            effects=(f"effect-{index}",),
            required_vm_capabilities=(application,),
            source_evidence=(
                SourceEvidence(
                    task_id="business_finance/example",
                    step_start=index,
                ),
            ),
        )
        for index, application in enumerate(("shell", "filesystem"), 1)
    ]
    environment = EnvironmentContext(
        image_id="ale-ubuntu",
        image_digest="digest",
        os={"type": "Ubuntu", "version": "22.04", "architecture": "x86_64"},
        applications={},
        capabilities=("computer", "desktop_os", "filesystem", "shell"),
    )

    recipes = PrimitiveComposer(
        min_primitives=2,
        max_primitives=2,
        variants_per_anchor=73,
        source_complete_variants_per_anchor=3,
        allow_cross_app=False,
    ).recipes([source], capabilities, environment)

    assert len(recipes) == 3
    assert all(recipe.strategy == "source_complete" for recipe in recipes)
    assert all(
        recipe.primitive_ids == tuple(item.id for item in capabilities)
        for recipe in recipes
    )
    assert all(recipe.source_task_ids == (source.task.id,) for recipe in recipes)


def test_static_quality_rejects_cross_platform_instruction_and_composite_class_evaluator():
    environment = EnvironmentContext(
        image_id="windows-test",
        image_digest="digest",
        os={"type": "Windows", "version": "10", "architecture": "x86_64"},
        applications={},
        capabilities=("computer", "filesystem"),
    )
    generated = GeneratedTask.from_dict({
        "task": {
            "id": "gen-001",
            "instruction": "Open /home/user/result.txt from the GNOME desktop.",
            "config": [{
                "type": "osworld_v2_task_class",
                "task_id": "001",
                "class_name": "Task001",
                "path": "/repo/task_001.py",
            }],
            "evaluator": {
                "type": "composite",
                "components": [{
                    "type": "task_evaluator",
                    "class_name": "Task001",
                    "method": "evaluate",
                    "path": "/repo/task_001.py",
                }],
            },
            "required_capabilities": ["computer", "filesystem"],
            "target_environments": [{"os_type": "Windows"}],
            "metadata": {"category": "OSWORLD_V2", "platform": "linux"},
        },
        "recipe": {"id": "recipe-1"},
    })

    report = TaskValidator(
        environment,
        StaticVerifier(),
        require_sandbox=False,
    ).validate(generated)

    assert report.valid is False
    assert report.checks["platform_consistency"] is False
    assert report.checks["evaluator_contract"] is False
    assert any("Linux /home path" in error for error in report.errors)
    assert any("directly reuse" in error for error in report.errors)


def test_osworld_compiler_rejects_unsupported_generated_evaluator(tmp_path):
    generated_root = tmp_path / "promoted"
    _write_json(generated_root / "gen-task" / "generated_task.json", {
        "task": {
            "id": "gen-task",
            "instruction": "Complete the prepared task.",
            "snapshot": "clean",
            "config": [],
            "evaluator": {"type": "composite", "components": []},
            "metadata": {"category": "OSWORLD_V2", "platform": "windows"},
        },
        "recipe": {},
    })

    with pytest.raises(ValueError, match="unsupported OSWorld evaluator schema"):
        compile_osworld_taskset(generated_root, tmp_path / "compiled")


def test_release_gate_limits_promotion_to_current_candidates(tmp_path):
    run_dir = tmp_path / "run"
    for task_id in ("current", "stale"):
        _write_json(run_dir / "candidates" / task_id / "generated_task.json", {"task": {}})
        _write_json(run_dir / "validation" / f"{task_id}.json", {"valid": True})
        _write_json(run_dir / "promoted" / task_id / "generated_task.json", {"task": {}})

    manifest = promote_candidates(run_dir, candidate_ids={"current"})

    assert manifest["promoted"] == ["current"]
    assert (run_dir / "promoted" / "current").is_dir()
    assert not (run_dir / "promoted" / "stale").exists()


def test_capability_index_merges_semantic_duplicates_across_sources():
    first = AbilityPrimitive(
        id="task-a::make-report",
        application="libreoffice_writer",
        intent="Create report 12 at /tmp/report.txt",
        effects=("report saved at /tmp/report.txt",),
        source_evidence=(SourceEvidence(task_id="task-a", success=True),),
        confidence=0.8,
    )
    second = AbilityPrimitive(
        id="task-b::make-report",
        application="libreoffice_writer",
        intent=r"Generate report 42 at C:\\tmp\\report.txt",
        effects=(r"report saved at C:\\tmp\\report.txt",),
        source_evidence=(SourceEvidence(task_id="task-b", success=False),),
        confidence=0.9,
    )

    index = PrimitiveLibrary(similarity_threshold=0.85)
    index.add(first)
    merged = index.add(second)
    distinct_same_source = AbilityPrimitive(
        id="task-a::make-summary",
        application="libreoffice_writer",
        intent="Create report summary at /tmp/summary.txt",
        effects=("report summary saved at /tmp/summary.txt",),
        source_evidence=(SourceEvidence(task_id="task-a", success=True),),
        confidence=0.9,
    )
    index.add(distinct_same_source)

    assert len(index.all()) == 2
    assert merged.evidence_count == 2
    assert merged.successful_evidence_count == 1
    assert merged.failed_evidence_count == 1
    assert {item.task_id for item in merged.source_evidence} == {"task-a", "task-b"}


def test_composer_generates_only_explicit_cross_app_handoffs():
    source = SourceTask(
        task=TaskSpec.from_dict({
            "id": "handoff-source",
            "instruction": "Create and send a report.",
            "config": [],
            "evaluator": {"type": "file_exists", "path": "/tmp/report"},
            "required_capabilities": ["computer", "writer", "mail"],
            "metadata": {"application": "writer"},
        }),
        source_path="/tmp/source.json",
        format="json",
        application="writer",
        category="DOC",
    )
    capabilities = [
        AbilityPrimitive(
            id="handoff-source::create",
            application="writer",
            intent="Create the report",
            effects=("artifact:report",),
            required_vm_capabilities=("computer", "writer"),
            source_evidence=(
                SourceEvidence(task_id="handoff-source", step_start=1, success=True),
            ),
            confidence=0.9,
        ),
        AbilityPrimitive(
            id="handoff-source::send",
            application="mail",
            intent="Send the report",
            preconditions=("requires:artifact:report",),
            effects=("report:sent",),
            required_vm_capabilities=("computer", "mail"),
            source_evidence=(
                SourceEvidence(task_id="handoff-source", step_start=2, success=True),
            ),
            confidence=0.9,
        ),
    ]
    environment = EnvironmentContext(
        image_id="handoff-vm",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "writer", "mail"),
    )

    disabled = PrimitiveComposer(
        min_primitives=2,
        max_primitives=2,
        variants_per_anchor=2,
        allow_cross_app=False,
    ).recipes([source], capabilities, environment)
    enabled = PrimitiveComposer(
        min_primitives=2,
        max_primitives=2,
        variants_per_anchor=2,
        allow_cross_app=True,
    ).recipes([source], capabilities, environment)

    assert disabled == []
    assert len(enabled) == 1
    assert enabled[0].strategy == "cross_app_handoff"
    assert enabled[0].metadata["application_count"] == 2
    assert enabled[0].metadata["dependency_depth"] == 1


def test_candidate_quality_score_enforces_alignment_threshold():
    environment = EnvironmentContext(
        image_id="quality-vm",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "filesystem"),
    )
    generated = GeneratedTask.from_dict({
        "task": {
            "id": "quality-task",
            "instruction": "Save the prepared result.",
            "config": [],
            "evaluator": {"type": "file_exists", "path": "/tmp/result"},
            "required_capabilities": ["computer", "filesystem"],
            "metadata": {"application": "files"},
        },
        "recipe": {
            "id": "quality-recipe",
            "anchor_task_id": "source-task",
            "primitive_ids": ["source-task::save"],
            "application": "files",
            "strategy": "sequential",
            "expected_effects": ["result:saved"],
            "required_vm_capabilities": ["computer", "filesystem"],
            "source_task_ids": ["source-task"],
            "environment_id": "quality-vm",
        },
    })
    report = {
        "valid": True,
        "checks": {
            "evaluator": True,
            "evaluator_contract": False,
            "instruction_quality": True,
            "platform_consistency": False,
        },
        "sandbox": {"ran": False},
    }

    score = score_candidate(
        generated,
        environment,
        report,
        minimum_total_score=0.0,
        minimum_grounding=0.0,
        minimum_alignment=0.8,
        minimum_solvability=0.0,
    )

    assert score.eligible is False
    assert score.dimensions["alignment"] == 0.5
    assert any("alignment score" in reason for reason in score.rejection_reasons)


def test_diversity_release_keeps_higher_quality_duplicate(tmp_path):
    run_dir = tmp_path / "run"
    for task_id, total_score in (("high", 0.95), ("low", 0.8)):
        _write_json(run_dir / "candidates" / task_id / "generated_task.json", {
            "task": {
                "id": task_id,
                "instruction": "Create the quarterly report and save it as report.pdf.",
                "config": [],
                "evaluator": {"type": "file_exists", "path": "/tmp/report.pdf"},
                "metadata": {"application": "writer"},
            },
            "recipe": {
                "id": f"recipe-{task_id}",
                "anchor_task_id": f"source-{task_id}",
                "primitive_ids": [f"capability-{task_id}"],
                "application": "writer",
                "strategy": "sequential",
                "expected_effects": ["report:saved"],
                "source_task_ids": [f"source-{task_id}"],
                "metadata": {"semantic_primitive_ids": [f"semantic-{task_id}"]},
            },
        })
        _write_json(run_dir / "validation" / f"{task_id}.json", {"valid": True})
        _write_json(
            run_dir / "quality" / f"{task_id}.json",
            CandidateQualityScore(
                task_id=task_id,
                total_score=total_score,
                dimensions={"novelty": 1.0},
                hard_gate_passed=True,
                eligible=True,
                coverage_bucket={
                    "application": "writer",
                    "strategy": "sequential",
                    "difficulty": "easy",
                },
            ).to_dict(),
        )

    manifest = promote_candidates(
        run_dir,
        config=ReleaseConfig(
            diversity_enabled=True,
            semantic_similarity_threshold=0.85,
            max_source_fraction=1.0,
            max_primitive_fraction=1.0,
            difficulty_targets={"easy": 1.0},
        ),
    )

    assert manifest["promoted"] == ["high"]
    assert "low" in manifest["diversity_rejections"]


def test_pipeline_writes_quality_and_feedback_reports(tmp_path):
    result = TaskGenerationPipeline(load_config(_config(tmp_path))).run(
        run_id="quality-report-run"
    )
    run_dir = Path(result["run_dir"])

    dataset_report = json.loads(
        (run_dir / "dataset_report.json").read_text(encoding="utf-8")
    )
    feedback = json.loads(
        (run_dir / "trajectory_feedback.json").read_text(encoding="utf-8")
    )
    generated = json.loads(next(
        (run_dir / "candidates").glob("*/generated_task.json")
    ).read_text(encoding="utf-8"))

    assert dataset_report["quality"]["scored_candidates"] > 0
    assert dataset_report["quality"]["average_total_score"] > 0
    assert feedback["schema_version"] == "propagation.trajectory_feedback.v1"
    assert generated["task"]["metadata"]["generation"]["task_blueprint"]


def test_config_loads_search_scoring_and_release_policy(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["generation"]["search"] = {
        "candidate_pool_multiplier": 5,
        "beam_width": 17,
        "primitive_similarity_threshold": 0.9,
        "recipe_similarity_threshold": 0.8,
        "random_seed": 42,
    }
    raw["generation"]["blueprint_first"] = True
    raw["validation"]["scoring"] = {
        "minimum_total_score": 0.7,
        "minimum_grounding": 0.75,
        "minimum_alignment": 0.85,
        "minimum_solvability": 0.65,
    }
    raw["validation"]["require_gui_realism"] = False
    raw["release"] = {
        "diversity_enabled": True,
        "semantic_similarity_threshold": 0.82,
        "target_promoted_tasks": 100,
        "max_promotions_per_anchor": 2,
        "platform_targets": {"linux": 0.5, "darwin": 0.25, "windows": 0.25},
    }
    _write_json(path, raw)

    config = load_config(path)

    assert config.generation.search.candidate_pool_multiplier == 5
    assert config.generation.search.beam_width == 17
    assert config.generation.blueprint_first is True
    assert config.validation.scoring.minimum_alignment == 0.85
    assert config.validation.require_gui_realism is False
    assert config.release.diversity_enabled is True
    assert config.release.target_promoted_tasks == 100
    assert config.release.max_promotions_per_anchor == 2
    assert config.release.platform_targets["windows"] == 0.25


def test_promote_cli_accepts_release_config():
    args = parse_args([
        "promote",
        "--run-dir",
        "/tmp/run",
        "--config",
        "/tmp/task-generation.yaml",
    ])

    assert args.command == "promote"
    assert args.config == "/tmp/task-generation.yaml"


def test_composer_is_bounded_and_deterministic_for_large_capability_pool():
    source = SourceTask(
        task=TaskSpec.from_dict({
            "id": "large-source",
            "instruction": "Complete several independent document operations.",
            "config": [],
            "evaluator": {"type": "file_exists", "path": "/tmp/result"},
            "required_capabilities": ["computer", "writer"],
            "metadata": {"application": "writer"},
        }),
        source_path="/tmp/large.json",
        format="json",
        application="writer",
        category="DOC",
    )
    capabilities = [
        AbilityPrimitive(
            id=f"large-source::operation-{index}",
            application="writer",
            intent=f"Apply document operation {index}",
            effects=(f"operation:{index}:complete",),
            required_vm_capabilities=("computer", "writer"),
            source_evidence=(
                SourceEvidence(
                    task_id="large-source",
                    step_start=index,
                    success=True,
                ),
            ),
            confidence=0.9,
        )
        for index in range(200)
    ]
    environment = EnvironmentContext(
        image_id="large-vm",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "writer"),
    )
    composer = PrimitiveComposer(
        min_primitives=2,
        max_primitives=4,
        variants_per_anchor=5,
        candidate_pool_multiplier=4,
        beam_width=16,
        random_seed=7,
    )

    first = composer.recipes([source], capabilities, environment)
    second = composer.recipes([source], capabilities, environment)

    assert len(first) <= 5
    assert [recipe.id for recipe in first] == [recipe.id for recipe in second]
    assert all(len(recipe.primitive_ids) <= 4 for recipe in first)


def test_cross_source_composer_bounds_large_external_pool():
    sources = [
        SourceTask(
            task=TaskSpec.from_dict({
                "id": task_id,
                "instruction": f"Complete {task_id} document operations.",
                "config": [],
                "evaluator": {"type": "file_exists", "path": f"/tmp/{task_id}"},
                "required_capabilities": ["computer", "writer"],
                "metadata": {"application": "writer"},
            }),
            source_path=f"/tmp/{task_id}.json",
            format="json",
            application="writer",
            category="DOC",
        )
        for task_id in ("anchor", "external")
    ]
    capabilities = [
        AbilityPrimitive(
            id=f"anchor::{index}",
            application="writer",
            intent=f"Apply anchor operation {index}",
            effects=(f"anchor:{index}:complete",),
            required_vm_capabilities=("computer", "writer"),
            source_evidence=(SourceEvidence(task_id="anchor", success=True),),
            confidence=0.9,
        )
        for index in range(8)
    ] + [
        AbilityPrimitive(
            id=f"external::{index}",
            application="writer",
            intent=f"Apply external operation {index}",
            effects=(f"external:{index}:complete",),
            required_vm_capabilities=("computer", "writer"),
            source_evidence=(SourceEvidence(task_id="external", success=True),),
            confidence=0.9,
        )
        for index in range(500)
    ]
    environment = EnvironmentContext(
        image_id="large-cross-source-vm",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "writer"),
    )
    composer = PrimitiveComposer(
        min_primitives=2,
        max_primitives=4,
        variants_per_anchor=2,
        allow_cross_source=True,
        cross_source_fraction=0.5,
        candidate_pool_multiplier=4,
        beam_width=16,
        random_seed=7,
    )

    recipes = composer.recipes(sources, capabilities, environment)

    cross_source = [
        recipe for recipe in recipes if recipe.metadata["cross_source"]
    ]
    assert cross_source
    assert all("anchor" in recipe.source_task_ids for recipe in cross_source)


def test_json_backend_runs_blueprint_before_compose():
    calls = []

    class BlueprintBackend(_JSONGeneratorBackend):
        model = "blueprint-test"
        blueprint_first = True

        def _request(self, operation, payload):
            calls.append((operation, payload))
            if operation == "blueprint":
                return {
                    "blueprint": {
                        "goal": "Save the report",
                        "dependency_order": ["source::save"],
                        "expected_effects": ["report:saved"],
                        "evaluator_coverage": ["report:saved"],
                        "failure_modes": [],
                    }
                }
            assert operation == "compose"
            assert payload["blueprint"]["goal"] == "Save the report"
            return {
                "task": {
                    "instruction": "Save the prepared report.",
                    "evaluator": {"type": "file_exists", "path": "/tmp/report"},
                    "metadata": {"untrusted_model_field": "discard me"},
                }
            }

    source = SourceTask(
        task=TaskSpec.from_dict({
            "id": "source",
            "instruction": "Save the report.",
            "config": [],
            "evaluator": {"type": "file_exists", "path": "/tmp/report"},
            "required_capabilities": ["computer", "filesystem"],
        }),
        source_path="/tmp/source.json",
        format="json",
        application="files",
        category="DOC",
    )
    capability = AbilityPrimitive(
        id="source::save",
        application="files",
        intent="Save the report",
        effects=("report:saved",),
        required_vm_capabilities=("computer", "filesystem"),
        source_evidence=(SourceEvidence(task_id="source", success=True),),
    )
    recipe = TaskRecipe(
        id="recipe",
        anchor_task_id="source",
        primitive_ids=(capability.id,),
        application="files",
        strategy="sequential",
        expected_effects=("report:saved",),
        required_vm_capabilities=("computer", "filesystem"),
        source_task_ids=("source",),
        environment_id="vm",
    )
    environment = EnvironmentContext(
        image_id="vm",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "filesystem"),
    )

    generated = BlueprintBackend().compose(
        recipe,
        source,
        [capability],
        environment,
    )

    assert [operation for operation, _ in calls] == ["blueprint", "compose"]
    assert generated.task.metadata["generation"]["task_blueprint"]["goal"] == (
        "Save the report"
    )
    assert "untrusted_model_field" not in generated.task.metadata


def test_validation_parallelizes_across_vms_but_serializes_each_vm(
    tmp_path,
    monkeypatch,
):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["environment"]["image_ids"] = ["ubuntu-test", "ubuntu-test-v2"]
    raw["environment"]["selection"] = "all"
    raw["generation"]["concurrency"] = 4
    raw["generation"]["variants_per_anchor"] = 1
    _write_json(path, raw)

    class TrackingVerifier:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.peak = 0
            self.active_by_environment = {}
            self.peak_by_environment = {}

        def verify(self, generated, environment):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.active_by_environment[environment.image_id] = (
                    self.active_by_environment.get(environment.image_id, 0) + 1
                )
                self.peak_by_environment[environment.image_id] = max(
                    self.peak_by_environment.get(environment.image_id, 0),
                    self.active_by_environment[environment.image_id],
                )
            try:
                time.sleep(0.05)
                return {
                    "ran": True,
                    "judge_only": True,
                    "initial_negative": True,
                    "partial_negative": True,
                    "reset_determinism": True,
                    "agent_solve": True,
                    "repeated_score": True,
                }
            finally:
                with self.lock:
                    self.active -= 1
                    self.active_by_environment[environment.image_id] -= 1

    verifier = TrackingVerifier()
    monkeypatch.setattr(
        "cuamaster.propagation.pipeline.create_verifier",
        lambda config: verifier,
    )

    TaskGenerationPipeline(load_config(path)).run(run_id="parallel-validation")

    assert verifier.peak == 2
    assert set(verifier.peak_by_environment.values()) == {1}


def test_variation_directive_is_deterministic_and_distinct_per_variant():
    from cuamaster.propagation.generation.variations import variation_directive

    first = variation_directive("recipe-a", 0)
    again = variation_directive("recipe-a", 0)
    assert first == again
    directives = {
        variation_directive("recipe-a", index)["axis"] for index in range(8)
    }
    assert len(directives) > 1
    preserving = variation_directive("recipe-a", 1, outcome_preserving_only=True)
    assert preserving["outcome_preserving"] is True


def _cross_source_fixture():
    def _source(task_id):
        return SourceTask(
            task=TaskSpec.from_dict({
                "id": task_id,
                "instruction": f"Task {task_id}.",
                "config": [],
                "evaluator": {"type": "file_exists", "path": "/tmp/x"},
                "required_capabilities": ["computer"],
            }),
            source_path=f"/repo/{task_id}.md",
            format="weavebench",
            application="writer",
            category="test",
        )

    def _capability(task_id, index):
        return AbilityPrimitive(
            id=f"{task_id}::capability.{index}",
            application="writer",
            intent=f"{task_id} {'export the report' if index == 1 else 'archive old drafts'}",
            preconditions=(),
            effects=(f"{task_id}:{'report-exported' if index == 1 else 'drafts-archived'}",),
            required_vm_capabilities=("computer",),
            source_evidence=(
                SourceEvidence(task_id=task_id, step_start=index, success=True),
            ),
            confidence=0.9,
        )

    sources = [_source("alpha"), _source("beta")]
    capabilities = [
        _capability("alpha", 1),
        _capability("alpha", 2),
        _capability("beta", 1),
        _capability("beta", 2),
    ]
    environment = EnvironmentContext(
        image_id="cross-vm",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "writer"),
    )
    return sources, capabilities, environment


def test_cross_source_fraction_adds_marked_cross_source_recipes():
    sources, capabilities, environment = _cross_source_fixture()
    baseline = PrimitiveComposer(
        min_primitives=2,
        max_primitives=2,
        variants_per_anchor=2,
    ).recipes(sources, capabilities, environment)
    assert all(not recipe.metadata["cross_source"] for recipe in baseline)

    expanded = PrimitiveComposer(
        min_primitives=2,
        max_primitives=2,
        variants_per_anchor=2,
        cross_source_fraction=0.5,
    ).recipes(sources, capabilities, environment)
    cross = [recipe for recipe in expanded if recipe.metadata["cross_source"]]
    assert cross
    assert all(len(set(recipe.source_task_ids)) > 1 for recipe in cross)
    assert len(expanded) > len(baseline)


def test_anchor_difficulty_target_recorded_in_recipe_metadata():
    sources, capabilities, environment = _cross_source_fixture()
    recipes = PrimitiveComposer(
        min_primitives=2,
        max_primitives=2,
        variants_per_anchor=1,
        anchor_difficulty_targets={"alpha": "harder"},
    ).recipes(sources, capabilities, environment)
    by_anchor = {recipe.anchor_task_id: recipe for recipe in recipes}
    assert by_anchor["alpha"].metadata["difficulty_target"] == "harder"
    assert "difficulty_target" not in by_anchor["beta"].metadata


def test_release_gate_enforces_application_and_strategy_quotas(tmp_path):
    from cuamaster.propagation.release.gate import _select_diverse_candidates

    def _record(index, application, strategy):
        generated = GeneratedTask.from_dict({
            "task": {
                "id": f"task-{index}",
                "instruction": f"Do distinct work item {index} in {application} regarding topic {chr(97 + index)}.",
                "config": [],
                "evaluator": {"type": "file_exists", "path": f"/tmp/{index}"},
                "required_capabilities": ["computer"],
            },
            "recipe": {
                "id": f"recipe-{index}",
                "anchor_task_id": f"anchor-{index}",
                "primitive_ids": [f"anchor-{index}::c1"],
                "application": application,
                "strategy": strategy,
                "expected_effects": [f"effect-{index}"],
                "required_vm_capabilities": ["computer"],
                "source_task_ids": [f"anchor-{index}"],
            },
        })
        score = CandidateQualityScore(
            task_id=f"task-{index}",
            total_score=1.0 - index * 0.01,
            dimensions={},
            hard_gate_passed=True,
            eligible=True,
            coverage_bucket={
                "application": application,
                "strategy": strategy,
                "difficulty": "medium",
            },
        )
        return (tmp_path / f"task-{index}", generated, score)

    records = [_record(index, "writer", "sequential") for index in range(6)]
    config = ReleaseConfig(
        diversity_enabled=True,
        target_promoted_tasks=6,
        max_source_fraction=1.0,
        max_primitive_fraction=1.0,
        max_application_fraction=0.5,
        max_strategy_fraction=1.0,
        difficulty_targets={"easy": 0.2, "medium": 0.6, "hard": 0.2},
    )
    selected, rejected = _select_diverse_candidates(records, config)
    assert len(selected) == 3
    assert any("application quota" in reason for reason in rejected.values())


def test_release_gate_balances_platforms_and_caps_each_anchor(tmp_path):
    from cuamaster.propagation.release.gate import _select_diverse_candidates

    records = []
    platforms = ("linux", "darwin", "windows")
    topics = (
        "budget forecast",
        "museum catalog",
        "network diagram",
        "research bibliography",
        "travel itinerary",
        "inventory ledger",
        "garden schedule",
        "audio transcript",
        "photo contact sheet",
    )
    for index in range(9):
        platform = platforms[index % len(platforms)]
        anchor = f"anchor-{index if index < 6 else index - 6}"
        generated = GeneratedTask.from_dict({
            "task": {
                "id": f"task-{index}",
                "instruction": (
                    f"Create the requested {topics[index]} with its distinct final layout."
                ),
                "config": [],
                "evaluator": {"type": "file_exists", "path": f"/tmp/{index}"},
                "required_capabilities": ["computer"],
                "metadata": {"platform": platform},
            },
            "recipe": {
                "id": f"recipe-{index}",
                "anchor_task_id": anchor,
                "primitive_ids": [f"{anchor}::c{index}"],
                "application": "writer",
                "strategy": "source_complete",
                "expected_effects": [f"effect-{index}"],
                "required_vm_capabilities": ["computer"],
                "source_task_ids": [anchor],
            },
        })
        score = CandidateQualityScore(
            task_id=generated.task.id,
            total_score=1.0 - index * 0.01,
            dimensions={},
            hard_gate_passed=True,
            eligible=True,
            coverage_bucket={
                "application": "writer",
                "strategy": "source_complete",
                "difficulty": "medium",
                "platform": platform,
            },
        )
        records.append((tmp_path / generated.task.id, generated, score))

    selected, rejected = _select_diverse_candidates(
        records,
        ReleaseConfig(
            diversity_enabled=True,
            target_promoted_tasks=9,
            max_promotions_per_anchor=1,
            max_source_fraction=1.0,
            max_primitive_fraction=1.0,
            max_application_fraction=1.0,
            max_strategy_fraction=1.0,
            difficulty_targets={"medium": 1.0},
            platform_targets={"linux": 1 / 3, "darwin": 1 / 3, "windows": 1 / 3},
        ),
    )

    selected_platforms = [
        score.coverage_bucket["platform"] for _, _, score in selected
    ]
    selected_anchors = [generated.recipe.anchor_task_id for _, generated, _ in selected]
    assert selected_platforms.count("linux") == 2
    assert selected_platforms.count("darwin") == 2
    assert selected_platforms.count("windows") == 2
    assert len(selected_anchors) == len(set(selected_anchors))
    assert any("source task promotion maximum" in reason for reason in rejected.values())


def test_gui_realism_rejects_hidden_state_shortcuts():
    assert gui_realism_errors(
        "Edit calendar.ics directly and restart the calendar service."
    )
    assert solve_plan_quality_errors([
        "Open Terminal and modify the application's internal database.",
    ])
    assert solve_plan_quality_errors([
        "Open Calendar and edit the event through its visible form.",
    ]) == []


def test_release_gate_dedups_against_prior_runs(tmp_path):
    from cuamaster.propagation.release.gate import _select_diverse_candidates

    prior_run = tmp_path / "prior"
    promoted = prior_run / "promoted" / "task-old"
    promoted.mkdir(parents=True)
    (promoted / "generated_task.json").write_text(json.dumps({
        "task": {"id": "task-old", "instruction": "Save the report as PDF."},
        "recipe": {"id": "old"},
    }), encoding="utf-8")

    generated = GeneratedTask.from_dict({
        "task": {
            "id": "task-new",
            "instruction": "Save the report as PDF.",
            "config": [],
            "evaluator": {"type": "file_exists", "path": "/tmp/x"},
            "required_capabilities": ["computer"],
        },
        "recipe": {
            "id": "recipe-new",
            "anchor_task_id": "anchor",
            "primitive_ids": ["anchor::c1"],
            "application": "writer",
            "strategy": "sequential",
            "expected_effects": ["saved"],
            "required_vm_capabilities": ["computer"],
            "source_task_ids": ["anchor"],
        },
    })
    score = CandidateQualityScore(
        task_id="task-new",
        total_score=1.0,
        dimensions={},
        hard_gate_passed=True,
        eligible=True,
        coverage_bucket={"difficulty": "medium"},
    )
    config = ReleaseConfig(
        diversity_enabled=True,
        max_source_fraction=1.0,
        max_primitive_fraction=1.0,
        dedup_against=(str(prior_run),),
    )
    selected, rejected = _select_diverse_candidates(
        [(tmp_path / "task-new", generated, score)],
        config,
    )
    assert selected == []
    assert rejected["task-new"] == "exact normalized instruction duplicate"


def test_solve_check_verdict_drives_solvability_dimension():
    environment = EnvironmentContext(
        image_id="solve-vm",
        image_digest="digest",
        os={"type": "Ubuntu"},
        applications={},
        capabilities=("computer", "filesystem"),
    )
    generated = GeneratedTask.from_dict({
        "task": {
            "id": "solve-task",
            "instruction": "Save the prepared result.",
            "config": [],
            "evaluator": {"type": "file_exists", "path": "/tmp/result"},
            "required_capabilities": ["computer"],
            "metadata": {"application": "files"},
        },
        "recipe": {
            "id": "solve-recipe",
            "anchor_task_id": "anchor",
            "primitive_ids": ["anchor::save"],
            "application": "files",
            "strategy": "sequential",
            "expected_effects": ["saved"],
            "required_vm_capabilities": ["computer"],
            "source_task_ids": ["anchor"],
            "environment_id": "solve-vm",
        },
    })
    base_report = {
        "valid": True,
        "checks": {},
        "sandbox": {"ran": False},
    }
    blocked = score_candidate(
        generated,
        environment,
        {**base_report, "solve_check": {"solvable": False, "blockers": ["missing app"]}},
        minimum_total_score=0.0,
        minimum_grounding=0.0,
        minimum_alignment=0.0,
        minimum_solvability=0.8,
    )
    assert blocked.dimensions["solvability"] == 0.0
    assert not blocked.eligible

    solvable = score_candidate(
        generated,
        environment,
        {**base_report, "solve_check": {"solvable": True, "blockers": []}},
        minimum_total_score=0.0,
        minimum_grounding=0.0,
        minimum_alignment=0.0,
        minimum_solvability=0.8,
    )
    assert solvable.dimensions["solvability"] == 1.0


def test_alignment_report_requires_every_judgement_to_agree():
    from cuamaster.propagation.alignment import instruction_alignment_report

    class SplitBackend:
        def __init__(self):
            self.calls = 0

        def align_check(self, generated, source, *, judgement_index=0):
            self.calls += 1
            if judgement_index == 0:
                return {"aligned": True}
            return {
                "aligned": False,
                "unverified_requirements": ["restart the service"],
            }

    generated = GeneratedTask.from_dict({
        "task": {
            "id": "align-task",
            "instruction": "Do the thing and restart the service.",
            "config": [{
                "type": "osworld_v2_task_class",
                "task_id": "001",
                "class_name": "Task001",
                "path": __file__,
            }],
            "evaluator": {
                "type": "task_evaluator",
                "class_name": "Task001",
                "method": "evaluate",
                "path": __file__,
            },
            "required_capabilities": ["computer"],
        },
        "recipe": {"id": "align-recipe"},
    })
    backend = SplitBackend()
    errors, warnings = instruction_alignment_report(
        generated,
        backend,
        judgements=2,
    )
    assert backend.calls == 2
    assert any("unverified requirement" in error for error in errors)


def test_release_gate_min_promotions_per_anchor_overrides_quotas(tmp_path):
    from cuamaster.propagation.release.gate import _select_diverse_candidates

    def _record(index, anchor, difficulty, total_score):
        generated = GeneratedTask.from_dict({
            "task": {
                "id": f"task-{anchor}-{index}",
                "instruction": (
                    f"Perform distinct anchor {anchor} objective number {index} "
                    f"covering subject {chr(97 + index)} in the workspace."
                ),
                "config": [],
                "evaluator": {"type": "file_exists", "path": f"/tmp/{anchor}-{index}"},
                "required_capabilities": ["computer"],
            },
            "recipe": {
                "id": f"recipe-{anchor}-{index}",
                "anchor_task_id": anchor,
                "primitive_ids": [f"{anchor}::c{index}"],
                "application": "writer",
                "strategy": "source_complete",
                "expected_effects": [f"effect-{anchor}-{index}"],
                "required_vm_capabilities": ["computer"],
                "source_task_ids": [anchor],
            },
        })
        score = CandidateQualityScore(
            task_id=f"task-{anchor}-{index}",
            total_score=total_score,
            dimensions={},
            hard_gate_passed=True,
            eligible=True,
            coverage_bucket={
                "application": "writer",
                "strategy": "source_complete",
                "difficulty": difficulty,
            },
        )
        return (tmp_path / f"task-{anchor}-{index}", generated, score)

    # anchor-b candidates are all hard and lower-scored: without the floor the
    # hard difficulty quota (filled by anchor-a first) would zero out anchor-b.
    records = [
        _record(index, "anchor-a", "hard", 1.0 - index * 0.01) for index in range(4)
    ] + [
        _record(index, "anchor-b", "hard", 0.8 - index * 0.01) for index in range(4)
    ]
    config = ReleaseConfig(
        diversity_enabled=True,
        target_promoted_tasks=4,
        min_promotions_per_anchor=1,
        max_source_fraction=1.0,
        max_primitive_fraction=1.0,
        difficulty_targets={"easy": 0.3, "medium": 0.4, "hard": 0.3},
    )
    selected, rejected = _select_diverse_candidates(records, config)
    selected_anchors = {generated.recipe.anchor_task_id for _, generated, _ in selected}
    assert "anchor-a" in selected_anchors
    assert "anchor-b" in selected_anchors

    without_floor = ReleaseConfig(
        diversity_enabled=True,
        target_promoted_tasks=4,
        min_promotions_per_anchor=0,
        max_source_fraction=1.0,
        max_primitive_fraction=1.0,
        difficulty_targets={"easy": 0.3, "medium": 0.4, "hard": 0.3},
    )
    baseline, _ = _select_diverse_candidates(records, without_floor)
    baseline_anchors = {generated.recipe.anchor_task_id for _, generated, _ in baseline}
    assert baseline_anchors == {"anchor-a"}


def test_variation_directive_rotates_axes_and_tiers_scaffolding():
    from cuamaster.propagation.generation.variations import (
        OUTCOME_PRESERVING_AXES,
        variation_directive,
    )

    pool_size = len(OUTCOME_PRESERVING_AXES)
    axes = [
        variation_directive("recipe-a", index, outcome_preserving_only=True)["axis"]
        for index in range(pool_size)
    ]
    assert len(set(axes)) == pool_size

    easier = variation_directive(
        "recipe-a", 0, outcome_preserving_only=True, difficulty_target="easier"
    )
    assert easier["scaffolding"] == "high"
    assert easier["axis"] in {"verbose_novice", "context_motivation"}
    assert "orienting guidance" in easier["directive"]

    harder = variation_directive(
        "recipe-a", 0, outcome_preserving_only=True, difficulty_target="harder"
    )
    assert harder["scaffolding"] == "low"
    assert harder["axis"] in {"terse_expert", "outcome_first"}
    assert "no procedural hints" in harder["directive"]
