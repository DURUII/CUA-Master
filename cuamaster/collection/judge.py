"""Post-collection model judging for trajectory quality and task success."""

from __future__ import annotations

import base64
import json
import math
import os
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cuamaster.endpoints import DynamicEndpointPool, load_endpoint_pool_config


JUDGE_SCHEMA_VERSION = "trajectory_model_judge.v1"
JUDGE_METHOD = "trajectory_model_judge"
_TEXT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".log",
    ".md",
    ".py",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_SYSTEM_PROMPT = """You are a strict trajectory evaluator.

Judge whether the computer-use agent completed the requested task and whether
the recorded trajectory is useful training data. Use only the supplied task,
trajectory, deliverable, and screenshot evidence. Do not assume hidden state.

Return exactly one JSON object with these fields:
{
  "score": number from 0.0 to 1.0,
  "task_success": boolean,
  "trajectory_usable": boolean,
  "confidence": number from 0.0 to 1.0,
  "reason": concise string,
  "evidence": [concise strings],
  "failure_category": "none|agent|environment|incomplete|incorrect|uncertain"
}

Set trajectory_usable=false for empty, incoherent, corrupted, or infrastructure-
dominated traces. Set task_success=true only when visible or textual evidence
supports completion of the requested deliverables. Be conservative when the
evidence is incomplete."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = max(1, limit * 2 // 5)
    tail = max(1, limit - head)
    return value[:head] + "\n... [truncated] ...\n" + value[-tail:]


def _trajectory_evidence(path: Path, limit: int) -> str:
    rows: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"trajectory read failed: {type(exc).__name__}: {exc}"
    for index, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            rows.append(f"step={index} invalid_json={line[:300]}")
            continue
        actions = event.get("actions") if isinstance(event, dict) else []
        executions = event.get("executions") if isinstance(event, dict) else []
        action_rows = []
        for action_index, action in enumerate(actions or []):
            if not isinstance(action, dict):
                continue
            execution = (
                executions[action_index]
                if isinstance(executions, list) and action_index < len(executions)
                else {}
            )
            execution = execution if isinstance(execution, dict) else {}
            args = json.dumps(
                action.get("args") or {}, ensure_ascii=False, separators=(",", ":")
            )
            output = json.dumps(
                execution.get("output") or "",
                ensure_ascii=False,
                separators=(",", ":"),
            )
            action_rows.append(
                f"{action.get('type')} args={_clip(args, 500)} "
                f"ok={execution.get('ok')} output={_clip(output, 700)}"
            )
        if action_rows:
            rows.append(f"step={event.get('step', index)} " + " | ".join(action_rows))
    return _clip("\n".join(rows), limit)


def _chat_tail(path: Path, limit: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return _clip("\n".join(lines[-30:]), limit)


def _deliverable_evidence(path: Path, limit: int) -> str:
    if not path.is_file():
        return "results.tar.gz is missing"
    rows: list[str] = []
    remaining = limit
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            rows.append(
                "deliverable files: "
                + ", ".join(member.name for member in members[:200])
            )
            preferred = sorted(
                members,
                key=lambda member: (
                    "results.json" not in member.name.casefold(),
                    member.size,
                    member.name,
                ),
            )
            for member in preferred:
                if remaining <= 0 or member.size > 65536:
                    continue
                if Path(member.name).suffix.casefold() not in _TEXT_ARTIFACT_SUFFIXES:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                text = extracted.read(min(member.size, remaining)).decode(
                    "utf-8", errors="replace"
                )
                snippet = f"\n--- {member.name} ---\n{text}"
                rows.append(snippet)
                remaining -= len(snippet)
    except (OSError, tarfile.TarError) as exc:
        return f"results.tar.gz read failed: {type(exc).__name__}: {exc}"
    return _clip("\n".join(rows), limit)


def _final_screenshot(task_dir: Path) -> Path | None:
    preferred = task_dir / "final_screenshot.png"
    if preferred.is_file():
        return preferred
    screenshots = [
        path for path in (task_dir / "screenshots").glob("*.png") if path.is_file()
    ]
    return (
        max(screenshots, key=lambda path: path.stat().st_mtime_ns)
        if screenshots
        else None
    )


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_prompt(task_dir: Path, max_evidence_chars: int) -> str:
    task = _read_json(task_dir / "task.json")
    score = _read_json(task_dir / "score.json")
    instruction = str(task.get("instruction") or "").strip()
    evaluator = task.get("evaluator") or {}
    budget = max(4000, max_evidence_chars)
    trajectory_budget = budget * 3 // 5
    artifact_budget = budget // 4
    chat_budget = budget - trajectory_budget - artifact_budget
    return "\n\n".join(
        (
            f"TASK ID: {task.get('id') or score.get('task_id') or task_dir.name}",
            f"TASK INSTRUCTION:\n{instruction}",
            "EVALUATOR METADATA:\n"
            + _clip(json.dumps(evaluator, ensure_ascii=False, default=str), 4000),
            "RECORDED TRAJECTORY:\n"
            + _trajectory_evidence(task_dir / "trajectory.jsonl", trajectory_budget),
            "FINAL CHAT EXCERPT:\n" + _chat_tail(task_dir / "chat.jsonl", chat_budget),
            "ARCHIVED DELIVERABLE EVIDENCE:\n"
            + _deliverable_evidence(task_dir / "results.tar.gz", artifact_budget),
        )
    )


def _response_text(payload: dict[str, Any], wire_api: str) -> str:
    if wire_api == "chat_completions":
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            return str(message.get("content") or "")
        return ""
    if payload.get("output_text"):
        return str(payload["output_text"])
    fragments: list[str] = []
    for output in payload.get("output") or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content") or []:
            if isinstance(content, dict) and content.get("text"):
                fragments.append(str(content["text"]))
    return "".join(fragments)


def _empty_response_detail(payload: dict[str, Any], wire_api: str) -> str:
    if wire_api == "chat_completions":
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            reasoning = str(message.get("reasoning_content") or "")
            return (
                f"finish_reason={choices[0].get('finish_reason')!r} "
                f"reasoning_chars={len(reasoning)}"
            )
    return f"response_keys={sorted(payload)}"


def _parse_judgment(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        candidate = candidate.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            raise ValueError("judge response does not contain a JSON object")
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    try:
        score = float(value["score"])
        confidence = float(value.get("confidence", 0.5))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("judge response has invalid score/confidence") from exc
    if not math.isfinite(score) or not math.isfinite(confidence):
        raise ValueError("judge response score/confidence must be finite")
    evidence = value.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [evidence]
    return {
        "score": min(1.0, max(0.0, score)),
        "task_success": value.get("task_success") is True,
        "trajectory_usable": value.get("trajectory_usable") is True,
        "confidence": min(1.0, max(0.0, confidence)),
        "reason": str(value.get("reason") or "").strip()[:2000],
        "evidence": [str(item)[:1000] for item in evidence[:20]],
        "failure_category": str(value.get("failure_category") or "uncertain")[:100],
    }


def _request_judgment(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    wire_api: str,
    trust_env: bool,
    prompt: str,
    screenshot: Path | None,
    max_output_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], str]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if screenshot is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(screenshot)}}
        )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if wire_api == "responses":
        url = endpoint.rstrip("/") + "/responses"
        response_content = []
        for item in content:
            if item["type"] == "text":
                response_content.append({"type": "input_text", "text": item["text"]})
            else:
                response_content.append(
                    {"type": "input_image", "image_url": item["image_url"]["url"]}
                )
        payload = {
            "model": model,
            "instructions": _SYSTEM_PROMPT,
            "input": [{"role": "user", "content": response_content}],
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
    else:
        url = endpoint.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_output_tokens,
            "temperature": 0,
        }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    opener = (
        urllib.request.build_opener()
        if trust_env
        else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"judge HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"judge request failed: {exc}") from exc
    text = _response_text(response_payload, wire_api)
    if not text.strip():
        raise RuntimeError(
            "judge returned no text: "
            + _empty_response_detail(response_payload, wire_api)
        )
    return _parse_judgment(text), text


def _agent_collection_succeeded(score: dict[str, Any]) -> bool:
    if score.get("agent_done") is False:
        return False
    if score.get("agent_exit") not in (None, 0, "0"):
        return False
    status = str(score.get("agent_status") or "").strip().casefold()
    return status not in {
        "empty_run",
        "error",
        "failed",
        "retry_exhausted",
        "timed_out",
        "timeout",
    }


def _apply_judgment(
    task_dir: Path,
    judgment: dict[str, Any],
    *,
    model: str,
    endpoint: str,
    threshold: float,
    raw_response: str,
) -> bool:
    score_path = task_dir / "score.json"
    original = _read_json(score_path)
    backup_path = task_dir / "score.collection.json"
    if original and not backup_path.exists():
        _atomic_write_json(backup_path, original)
    accepted = bool(
        judgment["task_success"]
        and judgment["trajectory_usable"]
        and judgment["score"] >= threshold
    )
    judged_at = datetime.now(timezone.utc).isoformat()
    judge_record = {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "judged_at": judged_at,
        "model": model,
        "endpoint": endpoint,
        "threshold": threshold,
        "accepted": accepted,
        **judgment,
        "raw_response": raw_response,
    }
    _atomic_write_json(task_dir / "model_judge.json", judge_record)
    updated = dict(original)
    original_scores = (
        original.get("scores") if isinstance(original.get("scores"), dict) else {}
    )
    updated.update(
        {
            "score": judgment["score"],
            "success": accepted,
            "error": None,
            "evaluator": JUDGE_METHOD,
            "evaluator_version": JUDGE_SCHEMA_VERSION,
            "scores": {
                "judge_method": JUDGE_METHOD,
                "judge_model": model,
                "judge_score": judgment["score"],
                "judge_success": judgment["task_success"],
                "trajectory_usable": judgment["trajectory_usable"],
                "accepted": accepted,
                "threshold": threshold,
                "confidence": judgment["confidence"],
                "reason": judgment["reason"],
                "evidence": judgment["evidence"],
                "failure_category": judgment["failure_category"],
                "collection_scores": original_scores,
                "collection_error": original.get("error"),
                "judged_at": judged_at,
            },
        }
    )
    _atomic_write_json(score_path, updated)
    return accepted


def score_collection(
    collection_root: str | Path,
    *,
    endpoint_pool_path: str | Path,
    model: str | None = None,
    api_key_env: str | None = None,
    allow_unauthenticated: bool = False,
    concurrency: int = 4,
    threshold: float = 0.6,
    attempts: int = 3,
    timeout: float = 300.0,
    max_output_tokens: int = 4096,
    max_evidence_chars: int = 24000,
    include_final_screenshot: bool = True,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    root = Path(collection_root).expanduser().resolve()
    endpoint_config = load_endpoint_pool_config(endpoint_pool_path)
    pool = DynamicEndpointPool(config_path=endpoint_pool_path)
    resolved_model = str(model or endpoint_config.model or "").strip()
    if not resolved_model:
        raise ValueError("model judge requires a model name")
    resolved_api_key_env = str(api_key_env or endpoint_config.api_key_env or "").strip()
    api_key = os.environ.get(resolved_api_key_env, "") if resolved_api_key_env else ""
    if not api_key and not allow_unauthenticated:
        raise ValueError(
            "model judge API key is missing; configure scoring.api_key_env or "
            "scoring.allow_unauthenticated"
        )
    api_key = api_key or "EMPTY"
    candidates = []
    for trajectory_path in root.rglob("trajectory.jsonl"):
        relative = trajectory_path.relative_to(root)
        if not relative.parts or relative.parts[0].startswith("."):
            continue
        if trajectory_path.stat().st_size > 0:
            candidates.append(trajectory_path.parent)

    def score_one(task_dir: Path) -> dict[str, Any]:
        existing = _read_json(task_dir / "score.json")
        existing_scores = (
            existing.get("scores") if isinstance(existing.get("scores"), dict) else {}
        )
        if not force and existing_scores.get("judge_method") == JUDGE_METHOD:
            return {
                "task_dir": str(task_dir),
                "status": "skipped_existing",
                "accepted": bool(existing.get("success")),
            }
        if not _agent_collection_succeeded(existing):
            return {
                "task_dir": str(task_dir),
                "status": "skipped_collection_failure",
                "accepted": False,
            }
        prompt = _build_prompt(task_dir, max_evidence_chars)
        screenshot = _final_screenshot(task_dir) if include_final_screenshot else None
        attempted: set[str] = set()
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            if len(attempted) >= len(pool.base_urls):
                attempted.clear()
            endpoint = pool.acquire(exclude=attempted)
            attempted.add(endpoint)
            try:
                judgment, raw_response = _request_judgment(
                    endpoint=endpoint,
                    api_key=api_key,
                    model=resolved_model,
                    wire_api=endpoint_config.wire_api,
                    trust_env=endpoint_config.trust_env,
                    prompt=prompt,
                    screenshot=screenshot,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                pool.mark_failure(endpoint, exc)
                if attempt < attempts:
                    time.sleep(min(5.0, max(0.0, pool.retry_after())))
                continue
            pool.mark_success(endpoint)
            accepted = _apply_judgment(
                task_dir,
                judgment,
                model=resolved_model,
                endpoint=endpoint,
                threshold=threshold,
                raw_response=raw_response,
            )
            return {
                "task_dir": str(task_dir),
                "status": "scored",
                "accepted": accepted,
                "score": judgment["score"],
            }
        error_record = {
            "schema_version": JUDGE_SCHEMA_VERSION,
            "judged_at": datetime.now(timezone.utc).isoformat(),
            "model": resolved_model,
            "status": "failed",
            "error": f"{type(last_error).__name__}: {last_error}",
        }
        _atomic_write_json(task_dir / "model_judge.json", error_record)
        return {
            "task_dir": str(task_dir),
            "status": "failed",
            "accepted": False,
            "error": error_record["error"],
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(score_one, task_dir): task_dir for task_dir in candidates
        }
        for future in as_completed(futures):
            rows.append(future.result())
            if progress is not None:
                row = rows[-1]
                progress(
                    f"model scoring {len(rows)}/{len(candidates)} "
                    f"status={row['status']} task={Path(row['task_dir']).name}"
                )
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "schema_version": "trajectory_model_judge.summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": resolved_model,
        "endpoint_pool": str(Path(endpoint_pool_path).expanduser().resolve()),
        "threshold": threshold,
        "candidates": len(candidates),
        "accepted": sum(1 for row in rows if row.get("accepted")),
        "counts": dict(sorted(counts.items())),
        "results": sorted(rows, key=lambda row: row["task_dir"]),
    }
    _atomic_write_json(root / "model_scoring_summary.json", summary)
    eligible = [row for row in rows if row["status"] != "skipped_collection_failure"]
    if eligible and not any(
        row["status"] in {"scored", "skipped_existing"} for row in eligible
    ):
        raise RuntimeError("model judge failed to score every collected trajectory")
    return summary
