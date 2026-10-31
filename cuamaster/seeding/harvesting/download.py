"""CLI for large-scale YouTube video download (pipeline step 1.4).

Reads metadata JSONL (from the search step), optionally filters by
category/app/limit, then downloads each video concurrently into a
taxonomy-organized tree. Safe to re-run: existing files are skipped.

    PYTHONPATH=src python -m video2traj.download --limit 3 --concurrency 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from ..common.core import download_one, load_records
from ..common.utils import (
    filter_records,
    format_duration,
    read_failure_records,
    write_failure_records,
)

logger = logging.getLogger("video2traj.download")

DEFAULT_INPUTS = ["data/after/video_data1.jsonl", "data/after/video_data2.jsonl"]
ACCESS_BLOCK_FAILURES = {"rate_limited", "bot_check"}
ANTI_BOT_FAILURES = ACCESS_BLOCK_FAILURES
RECOVERABLE_FAILURES = {
    "rate_limited",
    "bot_check",
    "proxy_503",
    "download_transport",
    "bad_download",
}
PERMANENT_FAILURES = {"account_terminated", "policy_removed", "uploader_removed"}
DEFAULT_FAILURE_CLASSES = RECOVERABLE_FAILURES
DEFAULT_ROTATE_COOKIE_ON = sorted({"bad_cookiefile"} | ACCESS_BLOCK_FAILURES)

RATE_LIMIT_MARKERS = (
    "rate-limited by youtube",
    "rate limited by youtube",
    "rate-limit",
    "rate limit",
    "ratelimit",
    "too many requests",
    "http error 429",
    "429: too many requests",
    "temporarily blocked",
    "unusual traffic",
)
BOT_CHECK_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "confirm you're not a bot",
    "confirm you are not a bot",
)


def _clean_error(error: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", error or "")


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def classify_failure(error: str) -> str:
    """Return a coarse failure class for retry decisions."""
    text = _clean_error(error)
    lower_text = text.lower()
    if _contains_any(lower_text, RATE_LIMIT_MARKERS):
        return "rate_limited"
    if "Tunnel connection failed: 503" in text:
        return "proxy_503"
    if "does not look like a Netscape format cookies file" in text:
        return "bad_cookiefile"
    if _contains_any(lower_text, BOT_CHECK_MARKERS):
        return "bot_check"
    if "account associated with this video has been terminated" in text:
        return "account_terminated"
    if "removed for violating" in text or "violating YouTube" in text:
        return "policy_removed"
    if "removed by the uploader" in text:
        return "uploader_removed"
    if "Got error:" in text:
        return "download_transport"
    if "downloaded file is not decodable" in text or "output file is empty" in text:
        return "bad_download"
    if "Private video" in text:
        return "private"
    if "Video unavailable" in text:
        return "other_unavailable"
    return "other"


def _load_failure_ids(paths: list[str], classes: set[str] | None) -> tuple[set[str], dict[str, int]]:
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                failure_class = classify_failure(obj.get("error", ""))
                counts[failure_class] = counts.get(failure_class, 0) + 1
                if classes is not None and failure_class not in classes:
                    continue
                vid = obj.get("video_id")
                if vid:
                    ids.add(vid)
    return ids, counts


def _load_cookiefiles_from_dir(path: str | Path) -> list[str]:
    cookie_dir = Path(path)
    if not cookie_dir.exists():
        raise FileNotFoundError(f"cookie directory does not exist: {cookie_dir}")
    if not cookie_dir.is_dir():
        raise NotADirectoryError(f"cookie path is not a directory: {cookie_dir}")
    files = [
        p for p in sorted(cookie_dir.iterdir())
        if p.is_file() and not p.name.startswith(".")
    ]
    return [str(p) for p in files]


def _cookiefile_validation_error(path: str | Path) -> str | None:
    """Return why a cookie file is unusable, or None when it looks valid."""
    cookie_path = Path(path)
    if not cookie_path.exists():
        return "file does not exist"
    if not cookie_path.is_file():
        return "not a regular file"
    try:
        with open(cookie_path, encoding="utf-8") as f:
            head = [f.readline() for _ in range(5)]
    except OSError as exc:
        return f"cannot read file: {exc}"
    except UnicodeDecodeError:
        return "not a UTF-8 text file"

    if any("Netscape HTTP Cookie File" in line for line in head):
        return None
    return "missing Netscape HTTP Cookie File header"


def _filter_valid_cookiefiles(paths: list[str], strict: bool) -> list[str]:
    valid: list[str] = []
    invalid: list[tuple[str, str]] = []
    for path in paths:
        error = _cookiefile_validation_error(path)
        if error is None:
            valid.append(path)
        else:
            invalid.append((path, error))

    for path, error in invalid:
        logger.warning("ignoring invalid cookiefile %s: %s", path, error)

    if strict and invalid:
        details = "; ".join(f"{path} ({error})" for path, error in invalid)
        raise SystemExit(f"invalid --cookiefile: {details}")
    if paths and not valid:
        details = "; ".join(f"{path} ({error})" for path, error in invalid)
        raise SystemExit(f"no valid Netscape cookie files found: {details}")
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos by id (metadata -> mp4 files)."
    )
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="metadata JSONL file(s)")
    parser.add_argument("--out-dir", default="data/videos", help="output root directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download over existing target files instead of skipping them (e.g. to replace truncated downloads)",
    )
    parser.add_argument("--category", default=None, help="only this top-level category (case-insensitive)")
    parser.add_argument("--app", default=None, help="only this application (case-insensitive)")
    parser.add_argument("--limit", type=int, default=None, help="download at most N videos (after filtering)")
    parser.add_argument("--concurrency", type=int, default=4, help="number of parallel downloads")
    parser.add_argument("--max-height", type=int, default=1080, help="max video height (px)")
    parser.add_argument("--delay-min", type=float, default=0.0, help="min random delay between downloads (s)")
    parser.add_argument("--delay-max", type=float, default=0.0, help="max random delay between downloads (s)")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle records before downloading to spread requests across apps/categories",
    )
    parser.add_argument(
        "--cooldown-on-antibot",
        dest="cooldown_on_access_block",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cooldown-on-access-block",
        dest="cooldown_on_access_block",
        type=float,
        default=0.0,
        help="global sleep after a rate-limit or bot-check access block (s)",
    )
    parser.add_argument(
        "--stop-after-antibot",
        dest="stop_after_access_block",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stop-after-access-block",
        dest="stop_after_access_block",
        type=int,
        default=0,
        help="stop submitting new downloads after this many consecutive rate-limit/bot-check access blocks; 0 disables",
    )
    parser.add_argument(
        "--inflight-factor",
        type=int,
        default=1,
        help="submit at most concurrency*N futures at once so stop conditions react quickly",
    )
    parser.add_argument(
        "--sleep-requests",
        type=float,
        default=0.0,
        help="yt-dlp sleep between HTTP requests inside one video download (s)",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=0.0,
        help="yt-dlp minimum sleep before each download (s)",
    )
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=0.0,
        help="yt-dlp maximum sleep before each download (s), used with --sleep-interval",
    )
    parser.add_argument(
        "--throttled-rate",
        default=None,
        help="yt-dlp treats rates below this as throttled, e.g. 100K or 1M",
    )
    parser.add_argument("--proxy", default=None, help="proxy URL (default: inherit https_proxy env)")
    parser.add_argument("--cookiefile", default=None, help="Netscape cookies.txt to bypass anti-bot")
    parser.add_argument(
        "--cookiefiles",
        nargs="+",
        default=None,
        help="ordered fallback Netscape cookies.txt files; switch only after cookie-related failures",
    )
    parser.add_argument(
        "--cookie-dir",
        default=None,
        help="directory of Netscape cookie files, loaded in filename order when --cookiefiles is not passed",
    )
    parser.add_argument(
        "--rotate-cookie-on",
        nargs="+",
        default=DEFAULT_ROTATE_COOKIE_ON,
        help="failure classes that trigger switching to the next --cookiefiles entry",
    )
    parser.add_argument(
        "--rotate-cookie-every",
        type=int,
        default=0,
        help="switch to the next --cookiefiles entry after every N successful downloads; 0 disables",
    )
    parser.add_argument("--cookies-from-browser", default=None, help="read cookies from a local browser, e.g. chrome")
    parser.add_argument(
        "--player-client",
        dest="player_clients",
        nargs="+",
        default=["android"],
        help="yt-dlp youtube player_client(s) to try in order; 'android' avoids the "
        "JS n-challenge so no JS runtime is needed",
    )
    parser.add_argument(
        "--concurrent-fragments",
        type=int,
        default=10,
        help="parallel fragment downloads per video (yt-dlp -N); 1 disables",
    )
    parser.add_argument(
        "--validate-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify each downloaded mp4 can decode at least one frame before marking it successful",
    )
    parser.add_argument(
        "--js-runtime",
        default="none",
        help="JS runtime for solving YouTube's n-challenge (needs yt-dlp-ejs); "
        "only used by the 'web' player_client, 'none' to disable",
    )
    parser.add_argument(
        "--retry-failures",
        nargs="+",
        default=None,
        help="retry only video_ids listed in one or more failure JSONL files; also merges them into the pending failure ledger",
    )
    parser.add_argument(
        "--retry-classes",
        nargs="+",
        default=sorted(DEFAULT_FAILURE_CLASSES),
        help="failure classes to retry; use 'all' to retry every class",
    )
    parser.add_argument(
        "--append-failures",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-backup-failures",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_path = out_dir / "_failures.jsonl"
    failure_records = read_failure_records(failures_path)

    records = load_records(args.inputs)
    if args.retry_failures:
        retry_classes = None if "all" in args.retry_classes else set(args.retry_classes)
        retry_ids, failure_counts = _load_failure_ids(args.retry_failures, retry_classes)
        for path in args.retry_failures:
            for vid, obj in read_failure_records(Path(path)).items():
                if vid in retry_ids:
                    failure_records[vid] = obj
        records = [r for r in records if r.get("video_id") in retry_ids]
        logger.info(
            "retry_failures=%d matched_records=%d classes=%s counts=%s",
            len(retry_ids), len(records),
            "all" if retry_classes is None else sorted(retry_classes),
            dict(sorted(failure_counts.items())),
        )
        write_failure_records(failures_path, failure_records)
    records = filter_records(records, args.category, args.app, args.limit)
    if args.shuffle:
        random.shuffle(records)

    proxy = args.proxy or os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    js_runtime = None if args.js_runtime.lower() == "none" else args.js_runtime
    if args.cookiefile and args.cookiefiles:
        logger.warning("both --cookiefile and --cookiefiles were passed; using --cookiefiles")
    if args.cookiefiles:
        cookiefiles = _filter_valid_cookiefiles(args.cookiefiles, strict=False)
    elif args.cookie_dir:
        cookiefiles = _filter_valid_cookiefiles(
            _load_cookiefiles_from_dir(args.cookie_dir), strict=False
        )
    elif args.cookiefile:
        cookiefiles = _filter_valid_cookiefiles([args.cookiefile], strict=True)
    else:
        cookiefiles = []
    if len(cookiefiles) > 1:
        random.shuffle(cookiefiles)
    if cookiefiles:
        logger.info("cookiefiles=%d first=%s", len(cookiefiles), cookiefiles[0])
    elif args.cookie_dir:
        logger.warning("cookie directory is empty: %s", args.cookie_dir)
    cookie_lock = threading.Lock()
    current_cookie_index = 0
    rotate_cookie_on = set(args.rotate_cookie_on)

    logger.info(
        "records=%d  out_dir=%s  concurrency=%d  max_height=%d",
        len(records), out_dir, args.concurrency, args.max_height,
    )

    downloaded = skipped = failed = 0
    downloads_since_cookie_rotate = 0
    counter_lock = threading.Lock()
    failure_lock = threading.Lock()
    throttle_lock = threading.Lock()
    next_allowed_at = 0.0
    consecutive_access_blocks = 0
    stop_requested = False
    logger.info("pending_failures=%d -> %s", len(failure_records), failures_path)

    def _current_cookiefile() -> str | None:
        with cookie_lock:
            if not cookiefiles:
                return None
            return cookiefiles[current_cookie_index]

    def _rotate_cookie(reason: str, expected_cookiefile: str | None = None) -> bool:
        nonlocal current_cookie_index
        with cookie_lock:
            if len(cookiefiles) <= 1:
                return False
            current_cookiefile = cookiefiles[current_cookie_index]
            if expected_cookiefile and expected_cookiefile != current_cookiefile:
                logger.info(
                    "cookie rotate request came from an older cookiefile; current remains %s",
                    current_cookiefile,
                )
                return False
            previous = cookiefiles[current_cookie_index]
            current_cookie_index = (current_cookie_index + 1) % len(cookiefiles)
            logger.warning(
                "switched cookiefile from %s to %s (%s)",
                previous, cookiefiles[current_cookie_index], reason,
            )
            return True

    def _maybe_rotate_cookie(failure_class: str | None, used_cookiefile: str | None) -> bool:
        if failure_class not in rotate_cookie_on or not used_cookiefile:
            return False
        if not _rotate_cookie(f"failure={failure_class}", expected_cookiefile=used_cookiefile):
            logger.warning(
                "cookie failure (%s) on %s; no alternate cookiefile available",
                failure_class, used_cookiefile,
            )
            return False
        return True

    def _task(record: dict) -> tuple[str, str | None, str | None, float]:
        nonlocal next_allowed_at
        task_started_at = time.monotonic()
        while True:
            with throttle_lock:
                wait_until = next_allowed_at
            delay = wait_until - time.monotonic()
            if delay <= 0:
                break
            time.sleep(min(delay, 5.0))
        if args.delay_max > 0:
            time.sleep(random.uniform(args.delay_min, args.delay_max))
        cookiefile = _current_cookiefile()
        vid, error = download_one(
            record,
            args.out_dir,
            max_height=args.max_height,
            proxy=proxy,
            cookiefile=cookiefile,
            cookies_from_browser=args.cookies_from_browser,
            js_runtime=js_runtime,
            sleep_requests=args.sleep_requests,
            sleep_interval=args.sleep_interval,
            max_sleep_interval=args.max_sleep_interval,
            throttled_rate=args.throttled_rate,
            player_clients=args.player_clients,
            concurrent_fragments=args.concurrent_fragments,
            overwrite=args.overwrite,
            validate_output=args.validate_output,
        )
        return vid, error, cookiefile, time.monotonic() - task_started_at

    def _record_result(
        record: dict,
        vid: str,
        error: str | None,
        used_cookiefile: str | None,
    ) -> str:
        nonlocal downloaded, skipped, failed, downloads_since_cookie_rotate
        nonlocal consecutive_access_blocks, next_allowed_at, stop_requested
        failure_class = classify_failure(error or "") if error and error != "skipped" else None
        if error is None:
            status = "ok"
            with counter_lock:
                downloaded += 1
                downloads_since_cookie_rotate += 1
                should_rotate = (
                    args.rotate_cookie_every > 0
                    and downloads_since_cookie_rotate >= args.rotate_cookie_every
                )
                if should_rotate:
                    downloads_since_cookie_rotate = 0
            if should_rotate:
                _rotate_cookie(f"{args.rotate_cookie_every} successful downloads")
        elif error == "skipped":
            status = "skip"
            with counter_lock:
                skipped += 1
        else:
            status = f"FAIL:{failure_class}"
            with counter_lock:
                failed += 1

        with failure_lock:
            if error is None or error == "skipped":
                failure_records.pop(vid, None)
            else:
                failure_records[vid] = {
                    "video_id": vid,
                    "url": record.get("url"),
                    "error": error,
                }
            write_failure_records(failures_path, failure_records)

        if failure_class in ACCESS_BLOCK_FAILURES:
            consecutive_access_blocks += 1
            if args.cooldown_on_access_block > 0:
                with throttle_lock:
                    next_allowed_at = max(
                        next_allowed_at,
                        time.monotonic() + args.cooldown_on_access_block,
                    )
                logger.warning(
                    "access block failure (%s); cooling down %.1fs",
                    failure_class, args.cooldown_on_access_block,
                )
            _maybe_rotate_cookie(failure_class, used_cookiefile)
            if (
                args.stop_after_access_block
                and consecutive_access_blocks >= args.stop_after_access_block
            ):
                stop_requested = True
                logger.warning(
                    "stopping after %d consecutive access block failures",
                    consecutive_access_blocks,
                )
        elif error is None or error == "skipped" or failure_class is not None:
            consecutive_access_blocks = 0

        return status

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        total = len(records)
        started_at = time.monotonic()
        max_inflight = max(args.concurrency, args.concurrency * max(args.inflight_factor, 1))
        iterator = iter(records)
        futures = {}
        submitted = 0
        completed = 0
        total_task_elapsed = 0.0

        def _submit_until_full() -> None:
            nonlocal submitted
            while not stop_requested and len(futures) < max_inflight:
                try:
                    record = next(iterator)
                except StopIteration:
                    return
                futures[pool.submit(_task, record)] = record
                submitted += 1

        _submit_until_full()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                record = futures.pop(fut)
                completed += 1
                vid, error, used_cookiefile, task_elapsed = fut.result()
                total_task_elapsed += task_elapsed
                status = _record_result(record, vid, error, used_cookiefile)
                elapsed = time.monotonic() - started_at
                avg_task = total_task_elapsed / completed
                throughput = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / throughput if throughput > 0 else 0.0
                logger.info(
                    "[%d/%d submitted=%d avg_task=%s throughput=%.2f/min eta=%s] %s %s",
                    completed,
                    total,
                    submitted,
                    format_duration(avg_task),
                    throughput * 60,
                    format_duration(eta),
                    status,
                    vid,
                )
            _submit_until_full()

    logger.info(
        "done: total=%d downloaded=%d skipped=%d failed=%d pending_failures=%d elapsed=%s  (failures -> %s)",
        len(records), downloaded, skipped, failed, len(failure_records),
        format_duration(time.monotonic() - started_at), failures_path,
    )


if __name__ == "__main__":
    main()
