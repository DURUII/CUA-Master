"""Download YouTube videos by video_id using yt-dlp (pipeline step 1.4 core).

Given metadata records (as produced by the search step), download each video
as an mp4 (video+audio) capped at a max height. Files are laid out by taxonomy
so downstream steps can browse them by category:

    <out_dir>/<category>/<subcategory>/<app>_<video_id>.mp4

Re-running is safe: a record whose target file already exists is skipped.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize(name: str) -> str:
    """Reduce a taxonomy label to a safe path segment.

    Keeps unicode letters/digits (so Chinese app names survive) and collapses
    every other run of characters to a single underscore. Empty input maps to
    "unknown" so we never emit an empty path segment.
    """
    if not name:
        return "unknown"
    # "&" maps to the word "and" so labels like "Science & Research" match the
    # existing on-disk tree ("Science_and_Research") instead of splitting into a
    # separate "Science_Research" directory.
    name = name.replace("&", " and ")
    cleaned = re.sub(r"[^\w]+", "_", name, flags=re.UNICODE).strip("_")
    return cleaned or "unknown"


def target_path(record: dict, out_dir: str | Path) -> Path:
    """Build <out_dir>/<category>/<subcategory>/<app>_<video_id>.mp4.

    A record may carry `_dest_relpath` to pin the on-disk path verbatim (used to
    overwrite pre-existing files whose directory names predate the current
    sanitize() rules); when present it wins over the taxonomy-derived layout.
    """
    dest_relpath = record.get("_dest_relpath")
    if dest_relpath:
        return Path(out_dir) / dest_relpath
    category = sanitize(record.get("category", ""))
    subcategory = sanitize(record.get("subcategory", ""))
    app = sanitize(record.get("application", ""))
    vid = record["video_id"]
    return Path(out_dir) / category / subcategory / f"{app}_{vid}.mp4"


def load_records(paths: list[str | Path]) -> list[dict]:
    """Read JSONL files and dedup by video_id (first occurrence wins)."""
    seen: set[str] = set()
    records: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                vid = obj.get("video_id")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                records.append(obj)
    return records


def build_opts(
    dest: Path,
    max_height: int,
    proxy: str | None = None,
    cookiefile: str | None = None,
    cookies_from_browser: str | None = None,
    js_runtime: str | None = "node",
    sleep_requests: float = 0.0,
    sleep_interval: float = 0.0,
    max_sleep_interval: float = 0.0,
    throttled_rate: str | None = None,
    player_clients: list[str] | None = None,
    concurrent_fragments: int = 1,
) -> dict:
    opts: dict = {
        # Prefer H.264 (avc1) so the merged file needs no re-encode, then fall
        # back to any codec within the height cap.
        "format": (
            f"bv*[height<={max_height}][vcodec^=avc1]+ba/"
            f"bv*[height<={max_height}]+ba/"
            f"b[height<={max_height}]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(dest.with_suffix("")) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
        "fragment_retries": 5,
        "ignoreerrors": False,
    }
    if concurrent_fragments > 1:
        opts["concurrent_fragment_downloads"] = concurrent_fragments
    if sleep_requests > 0:
        opts["sleep_interval_requests"] = sleep_requests
    if sleep_interval > 0:
        opts["sleep_interval"] = sleep_interval
    if max_sleep_interval > 0:
        opts["max_sleep_interval"] = max_sleep_interval
    if throttled_rate:
        opts["throttledratelimit"] = throttled_rate
    if player_clients:
        opts["extractor_args"] = {"youtube": {"player_client": player_clients}}
    # The web player_client requires solving a JS "n challenge" to expose real
    # formats (needs a JS runtime + yt-dlp-ejs). Clients like "android" return
    # formats directly, so a runtime is only needed when one is configured.
    if js_runtime:
        opts["js_runtimes"] = {js_runtime: {}}
    if proxy:
        opts["proxy"] = proxy
    if cookiefile:
        opts["cookiefile"] = cookiefile
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def validate_video_file(path: Path, frame_width: int = 160) -> str | None:
    """Return None when ffmpeg can decode at least one frame from `path`."""
    if not path.exists():
        return "output file is missing"
    if path.stat().st_size <= 0:
        return "output file is empty"

    with tempfile.TemporaryDirectory(prefix="download_validate_") as tmp:
        frame_path = Path(tmp) / "frame.jpg"
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={frame_width}:-2",
                str(frame_path),
            ],
            capture_output=True,
            text=True,
        )
        if frame_path.exists() and frame_path.stat().st_size > 0:
            return None

    stderr = " ".join(result.stderr.strip().split())
    if stderr:
        return f"downloaded file is not decodable: {stderr[:500]}"
    return f"downloaded file is not decodable: ffmpeg exited {result.returncode} without a frame"


def download_one(
    record: dict,
    out_dir: str | Path,
    max_height: int = 1080,
    proxy: str | None = None,
    cookiefile: str | None = None,
    cookies_from_browser: str | None = None,
    js_runtime: str | None = "node",
    sleep_requests: float = 0.0,
    sleep_interval: float = 0.0,
    max_sleep_interval: float = 0.0,
    throttled_rate: str | None = None,
    player_clients: list[str] | None = None,
    concurrent_fragments: int = 1,
    overwrite: bool = False,
    validate_output: bool = True,
) -> tuple[str, str | None]:
    """Download one video. Returns (video_id, error). error is None on success.

    A special error string "skipped" is returned when the target file already
    exists, so callers can distinguish resume-skips from real downloads. Pass
    overwrite=True to re-download over an existing file (e.g. a truncated one)
    instead of skipping it.
    """
    vid = record["video_id"]
    dest = target_path(record, out_dir)
    final_dest = dest
    if final_dest.exists():
        if not overwrite:
            return vid, "skipped"

        # Download into a sibling path and atomically replace only after yt-dlp
        # succeeds. This preserves the previous file when a repair download fails.
        dest = final_dest.with_name(f".{final_dest.stem}.repair{final_dest.suffix}")
        if dest.exists():
            dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = record.get("url") or f"https://www.youtube.com/watch?v={vid}"
    opts = build_opts(
        dest,
        max_height,
        proxy,
        cookiefile,
        cookies_from_browser,
        js_runtime,
        sleep_requests,
        sleep_interval,
        max_sleep_interval,
        throttled_rate,
        player_clients,
        concurrent_fragments,
    )
    try:
        from yt_dlp import YoutubeDL

        with YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:  # yt-dlp raises many error types
        if dest != final_dest and dest.exists():
            dest.unlink()
        return vid, str(e)

    if validate_output:
        validation_error = validate_video_file(dest)
        if validation_error:
            if dest != final_dest and dest.exists():
                dest.unlink()
            return vid, validation_error
    elif not dest.exists():
        return vid, "download reported success but output file is missing"
    if dest != final_dest:
        dest.replace(final_dest)
    return vid, None
