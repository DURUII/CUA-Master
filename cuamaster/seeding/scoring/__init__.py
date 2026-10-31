"""Pluggable scoring backends for CUA-suitability rating."""
from __future__ import annotations


def make_backend(name: str, model: str | None = None, gpu_index: int = 0, base_url: str | None = None):
    """Instantiate a scoring backend by name."""
    if name == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model)
    if name == "openai":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(model=model, base_url=base_url)
    if name == "qwen_vl":
        from .qwen_vl_backend import QwenVLBackend

        return QwenVLBackend(model=model, gpu_index=gpu_index)
    raise ValueError(f"unknown backend: {name}")
