"""SQLite-backed run and episode state ledger."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .schemas import EpisodeStatus


_TERMINAL = {
    EpisodeStatus.SUCCEEDED.value,
    EpisodeStatus.TASK_FAILED.value,
    EpisodeStatus.AGENT_FAILED.value,
    EpisodeStatus.INFRA_FAILED.value,
    EpisodeStatus.TIMED_OUT.value,
    EpisodeStatus.CANCELLED.value,
    EpisodeStatus.CORRUPT.value,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    config_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    task_id TEXT,
                    agent_id TEXT,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    worker_id TEXT,
                    output_dir TEXT,
                    error TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    heartbeat_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS episodes_run_status
                    ON episodes(run_id, status);
                CREATE INDEX IF NOT EXISTS episodes_profile
                    ON episodes(profile_id, task_id, agent_id);
            """)

    def create_run(
        self,
        run_id: str,
        *,
        config_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, config_path, status, metadata_json, created_at, updated_at)
                VALUES (?, ?, 'running', ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status='running', metadata_json=excluded.metadata_json, updated_at=excluded.updated_at
                """,
                (run_id, config_path, json.dumps(metadata or {}), timestamp, timestamp),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
                (status, _now(), run_id),
            )

    def register_episode(
        self,
        episode_id: str,
        *,
        run_id: str,
        profile_id: str,
        task_id: str | None = None,
        agent_id: str | None = None,
        attempt: int = 1,
        output_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO episodes(
                    episode_id, run_id, profile_id, task_id, agent_id, status, attempt,
                    output_dir, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    output_dir=COALESCE(excluded.output_dir, episodes.output_dir),
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    episode_id, run_id, profile_id, task_id, agent_id, attempt,
                    output_dir, json.dumps(metadata or {}), timestamp, timestamp,
                ),
            )

    def episode(self, episode_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def reset_episode(self, episode_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE episodes SET status='pending', worker_id=NULL, error=NULL,
                    started_at=NULL, heartbeat_at=NULL, finished_at=NULL, updated_at=?
                WHERE episode_id=?
                """,
                (_now(), episode_id),
            )

    def mark_running(self, episode_id: str, *, worker_id: str | None = None) -> None:
        timestamp = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
            if row is None:
                raise KeyError(episode_id)
            if row["status"] in _TERMINAL:
                raise ValueError(f"cannot restart terminal episode {episode_id}: {row['status']}")
            connection.execute(
                """
                UPDATE episodes SET status='running', worker_id=?, started_at=COALESCE(started_at, ?),
                    heartbeat_at=?, updated_at=? WHERE episode_id=?
                """,
                (worker_id, timestamp, timestamp, timestamp, episode_id),
            )

    def heartbeat(self, episode_id: str) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE episodes SET heartbeat_at=?, updated_at=? WHERE episode_id=?",
                (timestamp, timestamp, episode_id),
            )

    def finish_episode(
        self,
        episode_id: str,
        status: EpisodeStatus | str,
        *,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        value = status.value if isinstance(status, EpisodeStatus) else str(status)
        if value not in _TERMINAL:
            raise ValueError(f"episode terminal status required, got {value}")
        timestamp = _now()
        with self._connect() as connection:
            if metadata is None:
                connection.execute(
                    """
                    UPDATE episodes SET status=?, error=?, finished_at=?, updated_at=?
                    WHERE episode_id=?
                    """,
                    (value, error, timestamp, timestamp, episode_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE episodes SET status=?, error=?, metadata_json=?, finished_at=?, updated_at=?
                    WHERE episode_id=?
                    """,
                    (value, error, json.dumps(metadata), timestamp, timestamp, episode_id),
                )

    def recover_stale(self, stale_after_seconds: int) -> list[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT episode_id FROM episodes
                WHERE status='running' AND COALESCE(heartbeat_at, started_at, updated_at) < ?
                """,
                (cutoff,),
            ).fetchall()
            episode_ids = [str(row["episode_id"]) for row in rows]
            if episode_ids:
                connection.executemany(
                    """
                    UPDATE episodes SET status='pending', worker_id=NULL,
                        error='recovered stale running episode', updated_at=?
                    WHERE episode_id=?
                    """,
                    [(_now(), episode_id) for episode_id in episode_ids],
                )
            return episode_ids

    def rows(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM episodes"
        values: Iterable[Any] = ()
        if run_id is not None:
            query += " WHERE run_id=?"
            values = (run_id,)
        query += " ORDER BY created_at, episode_id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, tuple(values)).fetchall()]
