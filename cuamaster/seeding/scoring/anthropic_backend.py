"""Anthropic (Claude) scoring backend.

Uses ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN from the environment. Frames are
sent as base64 JPEG image blocks alongside the rubric + transcript text.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import (
    PLATFORM_OS_RUBRIC,
    RUBRIC,
    build_platform_os_user_text,
    build_user_text,
    parse_platform_os,
    parse_rating,
)
from .frames import encode_frame

DEFAULT_MODEL = "Claude-Opus-4.8-hq"


class AnthropicBackend:
    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed; run `pip install anthropic`"
            ) from e
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self._client = anthropic.Anthropic(
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
            auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        )

    def score(self, frames: list[Path], transcript: str | None, meta: dict) -> dict:
        return self._classify(
            frames,
            system=RUBRIC,
            user_text=build_user_text(meta, transcript),
            parser=parse_rating,
            max_tokens=512,
        )

    def classify_platform_os(self, frames: list[Path], transcript: str | None, meta: dict) -> dict:
        return self._classify(
            frames,
            system=PLATFORM_OS_RUBRIC,
            user_text=build_platform_os_user_text(meta, transcript),
            parser=parse_platform_os,
            max_tokens=256,
        )

    def _classify(
        self,
        frames: list[Path],
        system: str,
        user_text: str,
        parser,
        max_tokens: int,
    ) -> dict:
        content: list[dict] = []
        for frame in frames:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encode_frame(frame),
                    },
                }
            )
        content.append({"type": "text", "text": user_text})
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        raw = "".join(block.text for block in resp.content if block.type == "text")
        rating = parser(raw)
        rating["backend"] = "anthropic"
        rating["model"] = self.model
        return rating
