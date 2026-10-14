"""Small append-only SQLite event log for local coding runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import uuid


@dataclass(frozen=True, slots=True)
class Event:
    sequence: int
    event_id: str
    run_id: str
    task_id: str | None
    event_type: str
    actor: str
    timestamp: str
    payload: dict[str, Any]


class EventStore:
    """Append-only event store; mutation and deletion are blocked by SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=15)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL,
                task_id TEXT,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS events_no_update
            BEFORE UPDATE ON events BEGIN
                SELECT RAISE(ABORT, 'event log is append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete
            BEFORE DELETE ON events BEGIN
                SELECT RAISE(ABORT, 'event log is append-only');
            END;
            CREATE INDEX IF NOT EXISTS events_run_sequence ON events(run_id, sequence);
            CREATE INDEX IF NOT EXISTS events_task_sequence ON events(run_id, task_id, sequence);
            """
        )
        self.connection.commit()

    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        task_id: str | None = None,
        actor: str = "system",
    ) -> Event:
        if not run_id or not event_type or not actor:
            raise ValueError("run_id, event_type, and actor must be non-empty")
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        timestamp = datetime.now(timezone.utc).isoformat()
        event_id = str(uuid.uuid4())
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO events(event_id, run_id, task_id, event_type, actor, timestamp, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, run_id, task_id, event_type, actor, timestamp, payload_json),
            )
            sequence = int(cursor.lastrowid)
        return Event(sequence, event_id, run_id, task_id, event_type, actor, timestamp, payload)

    def events(self, run_id: str | None = None, task_id: str | None = None) -> tuple[Event, ...]:
        clauses: list[str] = []
        values: list[str] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            values.append(task_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            "SELECT * FROM events" + where + " ORDER BY sequence", values
        ).fetchall()
        return tuple(
            Event(
                sequence=row["sequence"], event_id=row["event_id"], run_id=row["run_id"],
                task_id=row["task_id"], event_type=row["event_type"], actor=row["actor"],
                timestamp=row["timestamp"], payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )

    def latest_by_type(self, run_id: str, task_id: str, event_types: Iterable[str]) -> Event | None:
        wanted = set(event_types)
        for event in reversed(self.events(run_id, task_id)):
            if event.event_type in wanted:
                return event
        return None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
