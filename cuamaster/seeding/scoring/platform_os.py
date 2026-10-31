"""Classify each video record with platform and operating-system labels.

Input is a JSONL metadata list such as ``data/raw_data_list.jsonl``. For each
record, the classifier locates the downloaded mp4 in the taxonomy video tree,
samples frames, optionally attaches the matching transcript, and writes a new
JSONL row with two additional fields:

    platform: desktop | mobile | website
    os: Windows | MacOS&iOS | Android | Linux

Runs are resumable: existing rows in the output JSONL are treated as completed
for their exact metadata key. When ``--require-transcript`` is enabled, records
whose transcript file does not exist are deferred and not written, so a later
rerun can classify them after transcription catches up. Existing but empty
transcript files are treated as completed transcripts and the video is still
classified from frames.
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import make_backend
from .frames import extract_frames, load_transcript_text, probe_duration
from ..common.core import target_path
from ..common.utils import format_duration

logger = logging.getLogger("video2traj.platform_os")

DEFAULT_INPUT = "data/raw_data_list.jsonl"
DEFAULT_OUTPUT = "data/raw_data_list_platform_os.jsonl"
DEFAULT_VIDEO_DIR = "data/videos"
DEFAULT_TRANSCRIPT_DIR = "data/transcripts"

_worker_state = threading.local()


def _record_key(record: dict) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("video_id") or ""),
        str(record.get("source") or ""),
        str(record.get("category") or ""),
        str(record.get("subcategory") or ""),
        str(record.get("application") or ""),
    )


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("video_id"):
                raise ValueError(f"missing video_id at {path}:{line_no}")
            records.append(obj)
    return records


def _load_done_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    done: set[tuple[str, str, str, str, str]] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            done.add(_record_key(obj))
    return done


def _transcript_path(video_path: Path, video_dir: Path, transcript_dir: Path) -> Path:
    try:
        relative = video_path.relative_to(video_dir)
    except ValueError:
        relative = video_path.name
    return (transcript_dir / relative).with_suffix(".segments.json")


def _get_backend(args: argparse.Namespace, gpu_index: int = 0):
    key = (args.backend, args.model, args.base_url, gpu_index)
    if getattr(_worker_state, "backend_key", None) != key:
        _worker_state.backend = make_backend(
            args.backend, model=args.model, gpu_index=gpu_index, base_url=args.base_url
        )
        _worker_state.backend_key = key
    return _worker_state.backend


def _empty_result(
    record: dict,
    error: str,
    include_diagnostics: bool = False,
    video_path: Path | None = None,
) -> dict:
    row = dict(record)
    row["platform"] = None
    row["os"] = None
    if include_diagnostics:
        row["platform_os_reason"] = ""
        row["platform_os_error"] = error
        if video_path is not None:
            row["source_video"] = str(video_path)
    return row


def _classify_one(
    record: dict,
    args: argparse.Namespace,
    video_dir: Path,
    transcript_dir: Path,
    gpu_index: int = 0,
) -> tuple[dict | None, str]:
    video_path = target_path(record, video_dir)
    transcript_file = _transcript_path(video_path, video_dir, transcript_dir)
    if args.require_transcript and not transcript_file.exists():
        return None, "deferred_transcript"

    if not video_path.exists():
        return _empty_result(record, "missing_video", args.include_diagnostics, video_path), "missing"

    duration = probe_duration(video_path)
    if duration <= 0:
        return _empty_result(record, "bad_video: no duration", args.include_diagnostics, video_path), "bad_video"

    transcript = load_transcript_text(transcript_file)
    backend = _get_backend(args, gpu_index)

    with tempfile.TemporaryDirectory(prefix="platform_os_frames_") as tmp:
        frames = extract_frames(
            video_path, Path(tmp), args.num_frames, args.frame_width, duration=duration
        )
        result = backend.classify_platform_os(frames, transcript, record)

    row = dict(record)
    row["platform"] = result["platform"]
    row["os"] = result["os"]
    if args.include_diagnostics:
        row["platform_os_reason"] = result.get("reason", "")
        row["platform_os_backend"] = result.get("backend")
        row["platform_os_model"] = result.get("model")
        row["source_video"] = str(video_path)
        row["has_transcript"] = transcript is not None
        row["num_frames"] = len(frames)
    return row, "ok"


def _write_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add platform and OS labels to a video metadata JSONL."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="input video metadata JSONL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="augmented output JSONL")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="downloaded mp4 root")
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR, help="root of .segments.json transcripts")
    parser.add_argument("--limit", type=int, default=None, help="classify at most N pending records")
    parser.add_argument("--overwrite", action="store_true", help="rewrite output from scratch")
    parser.add_argument("--require-transcript", action=argparse.BooleanOptionalAction, default=False, help="defer records whose matching transcript file does not exist; existing empty transcripts are still classified from frames")
    parser.add_argument("--backend", choices=["anthropic", "openai", "qwen_vl"], default="anthropic", help="vision backend")
    parser.add_argument("--model", default=None, help="model name/path for the backend")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for the openai backend")
    parser.add_argument("--num-frames", type=int, default=6, help="frames sampled per video")
    parser.add_argument("--frame-width", type=int, default=768, help="sampled frame width in px")
    parser.add_argument("--concurrency", type=int, default=4, help="worker threads")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs for qwen_vl parallelism")
    parser.add_argument("--include-diagnostics", action="store_true", help="also write reason/backend/model/source_video/debug fields")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_path = Path(args.input)
    output_path = Path(args.output)
    video_dir = Path(args.video_dir)
    transcript_dir = Path(args.transcript_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    records = _load_records(input_path)
    done = _load_done_keys(output_path)
    pending = [r for r in records if _record_key(r) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    num_gpus = max(1, args.num_gpus) if args.backend == "qwen_vl" else 1
    logger.info(
        "records=%d done=%d pending=%d output=%s backend=%s model=%s num_gpus=%d concurrency=%d",
        len(records), len(done), len(pending), output_path, args.backend,
        args.model or "<default>", num_gpus, args.concurrency,
    )

    if not pending:
        return

    started_at = time.monotonic()
    counts = {"ok": 0, "missing": 0, "deferred_transcript": 0, "bad_video": 0, "failed": 0}
    workers_per_gpu = max(1, args.concurrency)
    max_workers = workers_per_gpu * num_gpus
    max_in_flight = max_workers * 4

    def submit(pool: ThreadPoolExecutor, futures: dict, index: int) -> None:
        record = pending[index]
        gpu_index = index % num_gpus
        futures[pool.submit(_classify_one, record, args, video_dir, transcript_dir, gpu_index)] = (index, record)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict = {}
        buffered: dict[int, tuple[dict | None, str]] = {}
        next_index = 0
        next_write = 0
        while next_index < len(pending) and len(futures) < max_in_flight:
            submit(pool, futures, next_index)
            next_index += 1

        completed = 0
        total = len(pending)
        with open(output_path, "a", encoding="utf-8") as out:
            while futures:
                done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done_futures:
                    index, record = futures.pop(fut)
                    error_text = None
                    try:
                        row, status = fut.result()
                    except Exception as exc:
                        row = None
                        status = "failed"
                        error_text = str(exc)
                    buffered[index] = (row, status)

                    completed += 1
                    elapsed = time.monotonic() - started_at
                    throughput = completed / elapsed if elapsed > 0 else 0.0
                    eta = (total - completed) / throughput if throughput > 0 else 0.0
                    logger.info(
                        "[%d/%d %.1f/min eta=%s] %s %s",
                        completed, total, throughput * 60, format_duration(eta),
                        status, record.get("video_id"),
                    )
                    if error_text:
                        logger.warning("failed %s: %s", record.get("video_id"), error_text)

                    while next_write in buffered:
                        row_to_write, status_to_write = buffered.pop(next_write)
                        counts[status_to_write] = counts.get(status_to_write, 0) + 1
                        if row_to_write is not None:
                            _write_row(out, row_to_write)
                        next_write += 1

                    while next_index < len(pending) and len(futures) < max_in_flight:
                        submit(pool, futures, next_index)
                        next_index += 1

    logger.info("done %s", counts)


if __name__ == "__main__":
    main()
