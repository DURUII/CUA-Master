"""Backend registry for trajectory collection."""

from __future__ import annotations

import importlib

from .base import CollectionBackend, RunSpec
from .command import CommandBackend
from .weavebench import WeaveBenchBackend

_BACKENDS = {
    "command": CommandBackend,
    "weavebench": WeaveBenchBackend,
}


def create_backend(name: str) -> CollectionBackend:
    backend = _BACKENDS.get(name)
    if backend is None and ":" in name:
        module_name, class_name = name.split(":", 1)
        module = importlib.import_module(module_name)
        backend = getattr(module, class_name, None)
    if backend is None:
        raise ValueError(
            f"unknown backend {name!r}; use command, weavebench, or module.path:ClassName"
        )
    if not isinstance(backend, type) or not issubclass(backend, CollectionBackend):
        raise TypeError(f"backend {name!r} must inherit CollectionBackend")
    return backend()


__all__ = ["CollectionBackend", "RunSpec", "create_backend"]
