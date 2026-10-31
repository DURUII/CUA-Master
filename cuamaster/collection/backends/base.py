"""Backend contract for arbitrary VM/task runtimes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..profiles import MatrixConfig, VMProfile


@dataclass(frozen=True)
class RunSpec:
    command: list[str]
    env: dict[str, str]
    cwd: Path
    metadata: dict[str, Any]


class CollectionBackend(ABC):
    """Translate a generic VM profile into one executable collection run."""

    @abstractmethod
    def prepare(
        self,
        *,
        config: MatrixConfig,
        profile: VMProfile,
        output_dir: Path,
        cli: Any,
        base_env: Mapping[str, str],
    ) -> RunSpec:
        raise NotImplementedError
