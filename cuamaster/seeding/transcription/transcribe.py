"""Generate sentence-level transcript segments for downloaded mp4 files.

This is the first Video to CUA step after download: mp4 files are transcribed
with faster-whisper, and one JSON file is written next to each video.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


def _ensure_cublas() -> None:
    """Add nvidia-cublas-cu12 lib path to LD_LIBRARY_PATH if libcublas.so.12 is missing."""
    if ctypes.util.find_library("cublas"):
        return
    try:
        import nvidia.cublas.lib
        lib_dir = nvidia.cublas.lib.__path__[0]
    except (ImportError, AttributeError):
        return
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld_path if ld_path else "")
    ctypes.CDLL(os.path.join(lib_dir, "libcublas.so.12"))


_ensure_cublas()

from ..common.core import load_records, target_path  # noqa: E402
from ..common.utils import (  # noqa: E402
    filter_records,
    format_duration,
    label_matches,
    parse_stem,
    read_failure_records,
    video_id_from_path,
    write_failure_records,
)

logger = logging.getLogger("video2traj.transcribe")

DEFAULT_INPUTS = ["data/after/video_data1.jsonl", "data/after/video_data2.jsonl"]
DEFAULT_INPUT_DIR = "data/videos"

_worker_state = threading.local()


def _output_path(video_path: Path) -> Path:
    return video_path.with_suffix(".segments.json")


def _output_path_for_task(task: dict, output_dir: Path | None) -> Path:
    video_path = Path(task["video_path"])
    if output_dir is None:
        return _output_path(video_path)
    relative = task.get("relative_path") or Path(video_path.name)
    return (output_dir / relative).with_suffix(".segments.json")


def _scan_record(out_dir: Path, video_path: Path) -> dict:
    relative = video_path.relative_to(out_dir)
    parts = relative.parts
    app, video_id = parse_stem(video_path.stem)
    return {
        "video_id": video_id,
        "category": parts[0] if len(parts) >= 3 else "",
        "subcategory": parts[1] if len(parts) >= 3 else "",
        "application": app,
    }


def _scan_root(input_dir: Path, category: str | None, subcategory: str | None) -> Path:
    """Narrow the rglob start to input_dir/category[/subcategory] when possible.

    Avoids walking the whole ~68k-file tree on NFS just to keep one category.
    Falls back to input_dir if the exact-case subpath does not exist.
    """
    root = input_dir
    for part in (category, subcategory):
        if not part:
            break
        candidate = root / part
        if not candidate.is_dir():
            return input_dir
        root = candidate
    return root


def _scan_tasks(
    out_dir: Path,
    category: str | None,
    subcategory: str | None,
    app: str | None,
    limit: int | None,
) -> list[dict]:
    videos = sorted(_scan_root(out_dir, category, subcategory).rglob("*.mp4"))
    tasks: list[dict] = []
    for video_path in videos:
        record = _scan_record(out_dir, video_path)
        if category and not label_matches(record["category"], category):
            continue
        if subcategory and not label_matches(record["subcategory"], subcategory):
            continue
        if app and not label_matches(record["application"], app):
            continue
        tasks.append(
            {
                "record": record,
                "video_path": video_path,
                "relative_path": video_path.relative_to(out_dir),
            }
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def _input_mp4_task(input_mp4: str | Path) -> dict:
    video_path = Path(input_mp4)
    return {
        "record": {"video_id": video_id_from_path(video_path)},
        "video_path": video_path,
        "relative_path": Path(video_path.name),
    }


def _metadata_tasks(
    inputs: list[str],
    out_dir: Path,
    category: str | None,
    subcategory: str | None,
    app: str | None,
    limit: int | None,
) -> list[dict]:
    records = filter_records(
        load_records(inputs), category=category, subcategory=subcategory, app=app, limit=limit
    )
    tasks: list[dict] = []
    for record in records:
        video_path = target_path(record, out_dir)
        tasks.append(
            {
                "record": record,
                "video_path": video_path,
                "relative_path": video_path.relative_to(out_dir),
            }
        )
    return tasks


def _model_key(args: argparse.Namespace, device_index: int = 0) -> tuple[str, str, str, int]:
    return (args.model, args.device, args.compute_type, device_index)


def _use_batched_pipeline(args: argparse.Namespace) -> bool:
    return getattr(args, "batch_size", 0) > 0


def _get_model(args: argparse.Namespace, device_index: int = 0) -> Any:
    key = _model_key(args, device_index)
    if getattr(_worker_state, "model_key", None) != key:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed; run `pip install 'cuamaster[seeding]'`"
            ) from e
        _worker_state.model = WhisperModel(
            args.model,
            device=args.device,
            device_index=device_index,
            compute_type=args.compute_type,
        )
        _worker_state.model_key = key
    return _worker_state.model


def _get_transcriber(args: argparse.Namespace, device_index: int = 0) -> Any:
    model = _get_model(args, device_index)
    if not _use_batched_pipeline(args):
        return model

    if getattr(_worker_state, "pipeline_model_key", None) != _model_key(args, device_index):
        try:
            from faster_whisper import BatchedInferencePipeline
        except ImportError as e:
            raise RuntimeError(
                "BatchedInferencePipeline is not available; upgrade faster-whisper"
            ) from e
        _worker_state.pipeline = BatchedInferencePipeline(model=model)
        _worker_state.pipeline_model_key = _model_key(args, device_index)
    return _worker_state.pipeline


def _segment_to_dict(segment: Any) -> dict:
    return {
        "id": int(segment.id),
        "start": float(segment.start),
        "end": float(segment.end),
        "text": segment.text.strip(),
    }


def _write_transcript(
    output_path: Path,
    video_path: Path,
    record: dict,
    args: argparse.Namespace,
    info: Any,
    segments: list[dict],
) -> None:
    payload = {
        "video_id": record.get("video_id") or video_id_from_path(video_path),
        "source_video": str(video_path),
        "title": record.get("title"),
        "category": record.get("category", ""),
        "subcategory": record.get("subcategory", ""),
        "application": record.get("application", ""),
        "model": args.model,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration_sec": getattr(info, "duration", None),
        "segments": segments,
    }
    tmp = output_path.with_name(f".{output_path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, output_path)


def _transcribe_one(
    task: dict, args: argparse.Namespace, output_dir: Path | None, device_index: int = 0
) -> tuple[str, Path, str | None]:
    record = task["record"]
    video_path = Path(task["video_path"])
    video_id = record.get("video_id") or video_id_from_path(video_path)
    output_path = _output_path_for_task(task, output_dir)

    if not video_path.exists():
        return video_id, video_path, "missing"
    if output_path.exists() and not args.overwrite:
        return video_id, video_path, "skipped"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transcriber = _get_transcriber(args, device_index)
    kwargs: dict[str, Any] = {"vad_filter": args.vad_filter}
    if args.language:
        kwargs["language"] = args.language
    if _use_batched_pipeline(args):
        kwargs["batch_size"] = args.batch_size
        # BatchedInferencePipeline defaults to no timestamps; keep segment timestamps.
        kwargs["without_timestamps"] = False
    segments_iter, info = transcriber.transcribe(str(video_path), **kwargs)
    segments = [_segment_to_dict(segment) for segment in segments_iter]
    _write_transcript(output_path, video_path, record, args, info, segments)
    return video_id, video_path, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate faster-whisper transcript segment JSON for downloaded mp4 files."
    )
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="metadata JSONL file(s)")
    parser.add_argument("--input-mp4", default=None, help="transcribe one explicit mp4 file")
    parser.add_argument("--input-dir", default=None, help="downloaded mp4 root directory to scan")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="transcript output root; default writes next to each input mp4",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="downloaded video root directory for metadata mode; deprecated alias for --input-dir with --scan",
    )
    parser.add_argument("--category", default=None, help="only this top-level category (case-insensitive)")
    parser.add_argument("--subcategory", default=None, help="only this subcategory (case-insensitive)")
    parser.add_argument("--app", default=None, help="only this application (case-insensitive)")
    parser.add_argument("--limit", type=int, default=None, help="transcribe at most N videos")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="scan --input-dir/--out-dir recursively for mp4 files instead of reading metadata",
    )
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing .segments.json files")
    parser.add_argument("--model", default="medium", help="faster-whisper model name or local path")
    parser.add_argument("--device", default="auto", help="faster-whisper device, e.g. auto, cuda, cpu")
    parser.add_argument("--compute-type", default="default", help="faster-whisper compute type")
    parser.add_argument("--language", default=None, help="force language code; default auto-detect")
    parser.add_argument("--concurrency", type=int, default=1, help="number of worker threads")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="use faster-whisper BatchedInferencePipeline with this batch size; 0 disables batching",
    )
    parser.add_argument("--vad-filter", action="store_true", help="enable faster-whisper VAD filtering")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of GPUs for parallel transcription; 0 auto-detects all CUDA GPUs (only with --device cuda)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_mp4 = Path(args.input_mp4) if args.input_mp4 else None
    input_dir = input_mp4.parent if input_mp4 else Path(args.input_dir or args.out_dir or DEFAULT_INPUT_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_dir:
        failure_dir = output_dir
    elif input_mp4:
        failure_dir = input_mp4.parent
    else:
        failure_dir = input_dir
    failure_dir.mkdir(parents=True, exist_ok=True)
    failures_path = failure_dir / "_transcribe_failures.jsonl"
    failure_records = read_failure_records(failures_path)

    if input_mp4:
        tasks = [_input_mp4_task(input_mp4)]
    elif args.input_dir or args.scan or not args.out_dir:
        tasks = _scan_tasks(input_dir, args.category, args.subcategory, args.app, args.limit)
    else:
        tasks = _metadata_tasks(
            args.inputs, input_dir, args.category, args.subcategory, args.app, args.limit
        )

    # Determine number of GPUs for parallel transcription.
    num_gpus = 1
    if args.device == "cuda":
        if args.num_gpus > 0:
            num_gpus = args.num_gpus
        else:
            try:
                import ctranslate2
                num_gpus = ctranslate2.get_cuda_device_count()
            except Exception:
                num_gpus = 1
        num_gpus = max(1, num_gpus)

    logger.info(
        "tasks=%d  input_dir=%s  output_dir=%s  model=%s  device=%s  num_gpus=%d  concurrency=%d",
        len(tasks), input_dir, output_dir or "<input directories>",
        args.model, args.device, num_gpus, args.concurrency,
    )
    logger.info("pending_failures=%d -> %s", len(failure_records), failures_path)

    transcribed = skipped = missing = failed = 0
    started_at = time.monotonic()
    failure_lock = threading.Lock()

    def _record_result(task: dict, video_id: str, video_path: Path, error: str | None) -> str:
        nonlocal transcribed, skipped, missing, failed
        key = video_id or str(video_path)
        if error is None:
            transcribed += 1
            status = "ok"
        elif error == "skipped":
            skipped += 1
            status = "skip"
        elif error == "missing":
            missing += 1
            status = "missing"
        else:
            failed += 1
            status = "FAIL"

        with failure_lock:
            if error is None or error in {"skipped", "missing"}:
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

    if num_gpus <= 1:
        with ThreadPoolExecutor(max_workers=workers_per_gpu) as pool:
            futures = {
                pool.submit(_transcribe_one, task, args, output_dir, 0): task
                for task in tasks
            }
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
                        video_id = task["record"].get("video_id") or video_id_from_path(video_path)
                        error = str(e)
                    status = _record_result(task, video_id, video_path, error)
                    elapsed = time.monotonic() - started_at
                    throughput = completed / elapsed if elapsed > 0 else 0.0
                    eta = (total - completed) / throughput if throughput > 0 else 0.0
                    logger.info(
                        "[%d/%d throughput=%.2f/min eta=%s] %s %s",
                        completed, total, throughput * 60, format_duration(eta), status, video_id,
                    )
    else:
        # One pool per GPU so each thread stays bound to its GPU via threading.local().
        gpu_pools = [ThreadPoolExecutor(max_workers=workers_per_gpu) for _ in range(num_gpus)]
        futures: dict = {}
        for i, task in enumerate(tasks):
            gpu_idx = i % num_gpus
            fut = gpu_pools[gpu_idx].submit(_transcribe_one, task, args, output_dir, gpu_idx)
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
                    video_id = task["record"].get("video_id") or video_id_from_path(video_path)
                    error = str(e)
                status = _record_result(task, video_id, video_path, error)
                elapsed = time.monotonic() - started_at
                throughput = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / throughput if throughput > 0 else 0.0
                logger.info(
                    "[%d/%d throughput=%.2f/min eta=%s] %s %s",
                    completed, total, throughput * 60, format_duration(eta), status, video_id,
                )

        for p in gpu_pools:
            p.shutdown(wait=False)

    logger.info(
        "done: total=%d transcribed=%d skipped=%d missing=%d failed=%d pending_failures=%d elapsed=%s",
        len(tasks), transcribed, skipped, missing, failed, len(failure_records),
        format_duration(time.monotonic() - started_at),
    )


if __name__ == "__main__":
    main()
