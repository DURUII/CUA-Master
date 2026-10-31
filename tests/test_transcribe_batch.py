from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

sys.modules.setdefault("yt_dlp", types.SimpleNamespace(YoutubeDL=object))

from cuamaster.seeding.transcription import transcribe  # noqa: E402


class _FakeSegment:
    id = 0
    start = 1.0
    end = 2.0
    text = " batched text "


class _FakeInfo:
    language = "en"
    language_probability = 0.99
    duration = 2.0


class _FakeWhisperModel:
    def __init__(self, model: str, device: str, compute_type: str, device_index: int = 0) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.device_index = device_index


class _FakeBatchedInferencePipeline:
    last_kwargs = None

    def __init__(self, model: _FakeWhisperModel) -> None:
        self.model = model

    def transcribe(self, audio: str, **kwargs):
        _FakeBatchedInferencePipeline.last_kwargs = kwargs
        return [_FakeSegment()], _FakeInfo()


def test_transcribe_one_uses_batched_pipeline(tmp_path: Path, monkeypatch) -> None:
    fake_module = types.SimpleNamespace(
        WhisperModel=_FakeWhisperModel,
        BatchedInferencePipeline=_FakeBatchedInferencePipeline,
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    transcribe._worker_state.__dict__.clear()

    video_path = tmp_path / "App_abc123.mp4"
    video_path.write_bytes(b"not a real mp4; fake transcriber does not read it")
    args = argparse.Namespace(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        language="en",
        vad_filter=True,
        overwrite=False,
        batch_size=16,
    )
    task = {
        "record": {"video_id": "abc123", "application": "App"},
        "video_path": video_path,
        "relative_path": video_path.name,
    }

    video_id, _, error = transcribe._transcribe_one(task, args, tmp_path / "out")

    assert video_id == "abc123"
    assert error is None
    assert _FakeBatchedInferencePipeline.last_kwargs == {
        "vad_filter": True,
        "language": "en",
        "batch_size": 16,
        "without_timestamps": False,
    }
    assert (tmp_path / "out" / "App_abc123.segments.json").exists()
