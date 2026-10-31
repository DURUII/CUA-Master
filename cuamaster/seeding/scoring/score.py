"""Rate downloaded videos for Computer-Use-Agent (CUA) data suitability.

For each mp4 under the downloaded video tree, sample a few frames with ffmpeg,
optionally attach its faster-whisper transcript, and ask a vision model to rate
0-5 how suitable the video is as CUA training data. One `.score.json` is written
per video (mirroring the taxonomy layout), so re-runs skip finished videos.

Backends are pluggable (`--backend`): `anthropic` (Claude via the configured
proxy, needs no local GPU) or `qwen_vl` (local Qwen3-VL via vLLM).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import make_backend
from .frames import extract_frames, load_transcript_text, probe_duration
from ..common.utils import (
    format_duration,
    label_matches,
    parse_stem,
    read_failure_records,
    write_failure_records,
)

logger = logging.getLogger("video2traj.score")

DEFAULT_INPUT_DIR = "data/videos"
DEFAULT_TRANSCRIPT_DIR = "data/transcripts"

_worker_state = threading.local()


def _scan_tasks(
    input_dir: Path,
    category: str | None,
    subcategory: str | None,
    app: str | None,
    limit: int | None,
) -> list[dict]:
    tasks: list[dict] = []
    for video_path in sorted(input_dir.rglob("*.mp4")):
        relative = video_path.relative_to(input_dir)
        parts = relative.parts
        application, video_id = parse_stem(video_path.stem)
        record = {
            "video_id": video_id,
            "category": parts[0] if len(parts) >= 3 else "",
            "subcategory": parts[1] if len(parts) >= 3 else "",
            "application": application,
        }
        if category and not label_matches(record["category"], category):
            continue
        if subcategory and not label_matches(record["subcategory"], subcategory):
            continue
        if app and not label_matches(record["application"], app):
            continue
        tasks.append({"record": record, "video_path": video_path, "relative_path": relative})
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def _score_output_path(relative: Path, output_dir: Path) -> Path:
    return (output_dir / relative).with_suffix(".score.json")


def _transcript_path(relative: Path, transcript_dir: Path) -> Path:
    return (transcript_dir / relative).with_suffix(".segments.json")


def _get_backend(args: argparse.Namespace, gpu_index: int = 0):
    key = (args.backend, args.model, args.base_url, gpu_index)
    if getattr(_worker_state, "backend_key", None) != key:
        _worker_state.backend = make_backend(
            args.backend, model=args.model, gpu_index=gpu_index, base_url=args.base_url
        )
        _worker_state.backend_key = key
    return _worker_state.backend


def _score_one(
    task: dict, args: argparse.Namespace, output_dir: Path, transcript_dir: Path, gpu_index: int = 0
) -> tuple[str, Path, str | None]:
    record = task["record"]
    video_path = Path(task["video_path"])
    relative = task["relative_path"]
    video_id = record["video_id"]
    output_path = _score_output_path(relative, output_dir)

    if not video_path.exists():
        return video_id, video_path, "missing"
    if output_path.exists() and not args.overwrite:
        return video_id, video_path, "skipped"

    duration = probe_duration(video_path)
    if duration <= 0:
        return video_id, video_path, "bad_video: no duration"

    transcript_file = _transcript_path(relative, transcript_dir)
    if args.require_transcript and not transcript_file.exists():
        return video_id, video_path, "no_transcript"
    transcript = load_transcript_text(transcript_file)
    backend = _get_backend(args, gpu_index)

    with tempfile.TemporaryDirectory(prefix="score_frames_") as tmp:
        frames = extract_frames(
            video_path, Path(tmp), args.num_frames, args.frame_width, duration=duration
        )
        rating = backend.score(frames, transcript, record)

    payload = {
        "video_id": video_id,
        "source_video": str(video_path),
        "category": record["category"],
        "subcategory": record["subcategory"],
        "application": record["application"],
        "duration_sec": duration,
        "has_transcript": transcript is not None,
        "num_frames": len(frames),
        **rating,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_name(f".{output_path.name}.tmp")
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_out, output_path)
    return video_id, video_path, None


def _summarize(output_dir: Path) -> Path:
    """Aggregate every .score.json under output_dir into scores.jsonl."""
    summary_path = output_dir / "scores.jsonl"
    rows = []
    for score_path in sorted(output_dir.rglob("*.score.json")):
        try:
            with open(score_path, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "video_id": obj.get("video_id"),
                "path": obj.get("source_video"),
                "category": obj.get("category"),
                "subcategory": obj.get("subcategory"),
                "application": obj.get("application"),
                "score": obj.get("score"),
                "is_cua": obj.get("is_cua", obj.get("is_gui")),
                "has_trajectory": obj.get("has_trajectory"),
                "clarity": obj.get("clarity"),
                "has_transcript": obj.get("has_transcript"),
                "reason": obj.get("reason"),
                "backend": obj.get("backend"),
            }
        )
    tmp = summary_path.with_name(f".{summary_path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, summary_path)
    logger.info("summarized %d scores -> %s", len(rows), summary_path)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rate downloaded videos 0-5 for CUA data suitability."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="downloaded mp4 root to scan")
    parser.add_argument("--output-dir", default=None, help="score output root; default writes next to each mp4")
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR, help="root of .segments.json transcripts")
    parser.add_argument("--category", default=None, help="only this top-level category (case-insensitive)")
    parser.add_argument("--subcategory", default=None, help="only this subcategory (case-insensitive)")
    parser.add_argument("--app", default=None, help="only this application (case-insensitive)")
    parser.add_argument("--limit", type=int, default=None, help="score at most N videos")
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing .score.json files")
    parser.add_argument("--require-transcript", action=argparse.BooleanOptionalAction, default=True, help="skip videos whose .segments.json transcript file is missing (default on); pass --no-require-transcript to score from frames only")
    parser.add_argument("--backend", choices=["anthropic", "openai", "qwen_vl"], default="anthropic", help="scoring backend")
    parser.add_argument("--model", default=None, help="model name/path for the backend (backend-specific default)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for the openai backend (e.g. http://HOST:8000/v1)")
    parser.add_argument("--num-frames", type=int, default=8, help="frames sampled per video")
    parser.add_argument("--frame-width", type=int, default=768, help="sampled frame width in px")
    parser.add_argument("--concurrency", type=int, default=4, help="worker threads (anthropic); 1 recommended for qwen_vl")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs for qwen_vl parallelism (one backend per GPU)")
    parser.add_argument("--summarize", action="store_true", help="only (re)build scores.jsonl from existing .score.json and exit")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    transcript_dir = Path(args.transcript_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        _summarize(output_dir)
        return

    failures_path = output_dir / "_score_failures.jsonl"
    failure_records = read_failure_records(failures_path)

    tasks = _scan_tasks(input_dir, args.category, args.subcategory, args.app, args.limit)
    num_gpus = max(1, args.num_gpus) if args.backend == "qwen_vl" else 1

    logger.info(
        "tasks=%d input_dir=%s output_dir=%s backend=%s model=%s num_gpus=%d concurrency=%d",
        len(tasks), input_dir, output_dir, args.backend, args.model or "<default>", num_gpus, args.concurrency,
    )
    logger.info("pending_failures=%d -> %s", len(failure_records), failures_path)

    scored = skipped = missing = no_transcript = failed = 0
    started_at = time.monotonic()
    failure_lock = threading.Lock()

    def _record_result(video_id: str, video_path: Path, error: str | None) -> str:
        nonlocal scored, skipped, missing, no_transcript, failed
        key = video_id or str(video_path)
        if error is None:
            scored += 1
            status = "ok"
        elif error == "skipped":
            skipped += 1
            status = "skip"
        elif error == "missing":
            missing += 1
            status = "missing"
        elif error == "no_transcript":
            no_transcript += 1
            status = "no_transcript"
        else:
            failed += 1
            status = "FAIL"
        with failure_lock:
            if error is None or error in {"skipped", "missing", "no_transcript"}:
                failure_records.pop(key, None)
            else:
                failure_records[key] = {
                    "video_id": video_id,
                    "source_video": str(video_path),
                    "error": error,
                }
            write_failure_records(failures_path, failure_records)
        return status

    workers_per_gpu = max(1, args.concurrency)
    pools = [ThreadPoolExecutor(max_workers=workers_per_gpu) for _ in range(num_gpus)]
    futures: dict = {}
    for i, task in enumerate(tasks):
        gpu_idx = i % num_gpus
        fut = pools[gpu_idx].submit(_score_one, task, args, output_dir, transcript_dir, gpu_idx)
        futures[fut] = task

    completed = 0
    total = len(futures)
    while futures:
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for fut in done:
            task = futures.pop(fut)
            completed += 1
            try:
                video_id, video_path, error = fut.result()
            except Exception as e:
                video_path = Path(task["video_path"])
                video_id = task["record"]["video_id"]
                error = str(e)
            status = _record_result(video_id, video_path, error)
            elapsed = time.monotonic() - started_at
            throughput = completed / elapsed if elapsed > 0 else 0.0
            eta = (total - completed) / throughput if throughput > 0 else 0.0
            logger.info(
                "[%d/%d %.1f/min eta=%s] %s %s",
                completed, total, throughput * 60, format_duration(eta), status, video_id,
            )
    for pool in pools:
        pool.shutdown()

    logger.info(
        "done total=%d scored=%d skipped=%d missing=%d no_transcript=%d failed=%d",
        total, scored, skipped, missing, no_transcript, failed,
    )
    if scored or skipped:
        _summarize(output_dir)


if __name__ == "__main__":
    main()
