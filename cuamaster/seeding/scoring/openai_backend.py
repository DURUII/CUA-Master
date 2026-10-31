"""OpenAI-compatible scoring backend (e.g. a self-hosted vLLM server).

Points at any OpenAI-compatible `/v1` endpoint via OPENAI_BASE_URL / --base-url
(such as the multi-instance Qwen vLLM deploy behind Nginx). Frames are sent as
base64 JPEG data URLs alongside the shared rubric + transcript text, so scores
stay comparable with the `anthropic` and `qwen_vl` backends.
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

DEFAULT_MODEL = "Qwen3_6-27B-FP8"


class OpenAIBackend:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        try:
            import httpx
            import openai
        except ImportError as e:
            raise RuntimeError("openai SDK not installed; run `pip install openai`") from e
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        # trust_env=False so the SDK ignores http_proxy/https_proxy: a self-hosted
        # vLLM endpoint is reached directly, not through the outbound web proxy.
        self._client = openai.OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            http_client=httpx.Client(trust_env=False, timeout=600.0),
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
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_frame(frame)}"},
                }
            )
        content.append({"type": "text", "text": user_text})
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        rating = parser(raw)
        rating["backend"] = "openai"
        rating["model"] = self.model
        return rating
