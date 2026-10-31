"""Atomic artifact writes and ScaleCUA-inspired trace indexes."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class TraceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(value: str) -> str:
        return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)

    def index_episode(
        self,
        episode_dir: str | Path,
        *,
        episode_id: str,
        task_id: str,
        agent_id: str,
        run_id: str,
        created_at: str | None = None,
    ) -> dict[str, str]:
        target = Path(episode_dir).resolve()
        timestamp = datetime.fromisoformat(created_at) if created_at else datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        partitions = {
            "by-time": self.root / "by-time" / timestamp.strftime("%Y/%m/%d/%H") / episode_id,
            "by-task": self.root / "by-task" / self._safe(task_id) / episode_id,
            "by-agent": self.root / "by-agent" / self._safe(agent_id) / episode_id,
            "by-run": self.root / "by-run" / self._safe(run_id) / episode_id,
        }
        for link in partitions.values():
            link.parent.mkdir(parents=True, exist_ok=True)
            relative = os.path.relpath(target, link.parent)
            temporary = link.with_name(f".{link.name}.tmp")
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            temporary.symlink_to(relative)
            os.replace(temporary, link)
        return {name: str(path) for name, path in partitions.items()}
