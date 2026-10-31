"""Local Qwen3-VL scoring backend via vLLM offline inference.

Loads a local Qwen3-VL checkpoint once per process (one process/worker per GPU
in the CLI) and scores each video from its sampled frames + transcript.
"""
from __future__ import annotations

from pathlib import Path

from .base import (
    PLATFORM_OS_RUBRIC,
    RUBRIC,
    build_platform_os_user_text,
    build_user_text,
    parse_platform_os,
    parse_rating,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class QwenVLBackend:
    def __init__(self, model: str | None = None, gpu_index: int = 0):
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "vllm/transformers not installed; run `pip install vllm transformers`"
            ) from e
        self.model = model or DEFAULT_MODEL
        self._processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        self._llm = LLM(
            model=self.model,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 16},
            gpu_memory_utilization=0.9,
        )
        self._sampling = SamplingParams(temperature=0.0, max_tokens=512)

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
        from qwen_vl_utils import process_vision_info

        content: list[dict] = [{"type": "image", "image": str(f)} for f in frames]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        self._sampling.max_tokens = max_tokens
        outputs = self._llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_inputs},
                }
            ],
            self._sampling,
        )
        raw = outputs[0].outputs[0].text
        rating = parser(raw)
        rating["backend"] = "qwen_vl"
        rating["model"] = self.model
        return rating
