"""SqliteStore: stdlib `sqlite3` + FTS5 implementation of `Store`.

FTS5 external-content triggers keep `documents_fts` in sync with `documents` on
every insert/update/delete, so both a full `build` (bulk insert) and an
incremental `update` (targeted insert/delete per changed file) keep the search
index consistent without any separate reindex step.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from thimiko.dto import SearchDocument
from thimiko.models import Session

from .base import Store

SCHEMA_VERSION = "thimiko/v2"

_DROP_SQL = """
DROP TABLE IF EXISTS documents_fts;
DROP TABLE IF EXISTS embeddings;
DROP TABLE IF EXISTS documents;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS indexed_files;
DROP TABLE IF EXISTS metadata;
"""

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_stream_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    parent_session_id TEXT,
    agent_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    title TEXT,
    cwd TEXT,
    git_branch TEXT,
    model TEXT,
    event_count INTEGER NOT NULL,
    searchable_event_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    source TEXT NOT NULL,
    title TEXT,
    cwd TEXT,
    git_branch TEXT,
    started_at TEXT,
    ended_at TEXT,
    text TEXT NOT NULL,
    provenance_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_session_idx ON documents(session_id);
CREATE INDEX IF NOT EXISTS documents_turn_idx ON documents(turn_id);
CREATE INDEX IF NOT EXISTS documents_source_time_idx ON documents(source, started_at);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    text,
    content='documents',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO documents_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Reserved for a later local or API embedding backend. Kept separate from
-- `documents` so re-embedding never requires rebuilding canonical documents.
CREATE TABLE IF NOT EXISTS embeddings (
    document_id TEXT NOT NULL REFERENCES documents(id),
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (document_id, model)
);

-- `session_ids` is a JSON array: one file can hold many sessions (Cursor).
CREATE TABLE IF NOT EXISTS indexed_files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    session_ids TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
"""


def _match_expression(query: str) -> str:
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    if not terms:
        raise ValueError("Search query has no indexable words")
    return " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _row_with_provenance(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["provenance"] = json.loads(data.pop("provenance_json"))
    return data


def _group_by_turn(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        data = _row_with_provenance(row)
        turn_id = str(data["turn_id"])
        if turn_id not in grouped:
            grouped[turn_id] = []
            order.append(turn_id)
        grouped[turn_id].append(data)
    return [{"turn_id": turn_id, "chunks": grouped[turn_id]} for turn_id in order]


class SqliteStore(Store):
    """SQLite + FTS5 backend. The swap point for a future libSQL/Turso store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(db_path)
        self._connection.row_factory = sqlite3.Row

    def create_schema(self, *, reset: bool) -> None:
        if reset or self._stale_schema():
            self._connection.executescript(_DROP_SQL)
        self._connection.executescript(_SCHEMA_SQL)
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )
        self._connection.commit()

    def _stale_schema(self) -> bool:
        """Whether an existing index predates `SCHEMA_VERSION` and must be rebuilt.

        The index is a derived cache, so a version bump drops and recreates it
        rather than migrating; the next `build`/`update` refills it.
        """
        try:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.DatabaseError:
            return False
        return row is not None and str(row["value"]) != SCHEMA_VERSION

    def upsert_session(self, session: Session, documents: list[SearchDocument]) -> None:
        header = session.header()
        self._connection.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                header["id"],
                header["source"],
                header["source_session_id"],
                header["source_stream_id"],
                header["source_path"],
                header["parent_session_id"],
                header["agent_id"],
                header["started_at"],
                header["ended_at"],
                header["title"],
                header["cwd"],
                header["git_branch"],
                header["model"],
                header["event_count"],
                header["searchable_event_count"],
            ),
        )
        self._connection.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    document.id,
                    document.session_id,
                    document.turn_id,
                    document.chunk_index,
                    document.source,
                    document.title,
                    document.cwd,
                    document.git_branch,
                    document.started_at,
                    document.ended_at,
                    document.text,
                    json.dumps(document.provenance, ensure_ascii=False),
                )
                for document in documents
            ],
        )
        self._connection.commit()

    def delete_session(self, session_id: str) -> None:
        self._connection.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
        self._connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._connection.commit()

    def file_state(self, path: str) -> tuple[float, int] | None:
        row = self._connection.execute(
            "SELECT mtime, size FROM indexed_files WHERE path = ?", (path,)
        ).fetchone()
        return (row["mtime"], row["size"]) if row else None

    def record_file(self, path: str, mtime: float, size: int, session_ids: list[str]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO indexed_files VALUES (?, ?, ?, ?, ?)",
            (path, mtime, size, json.dumps(session_ids), datetime.now(UTC).isoformat()),
        )
        self._connection.commit()

    def forget_file(self, path: str) -> None:
        self._connection.execute("DELETE FROM indexed_files WHERE path = ?", (path,))
        self._connection.commit()

    def known_files(self) -> dict[str, list[str]]:
        rows = self._connection.execute("SELECT path, session_ids FROM indexed_files").fetchall()
        return {str(row["path"]): list(json.loads(row["session_ids"])) for row in rows}

    def session_counts(self) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT source, COUNT(*) AS n FROM sessions GROUP BY source"
        ).fetchall()
        return {str(row["source"]): int(row["n"]) for row in rows}

    def search(
        self,
        query: str,
        *,
        source: str | None,
        limit: int,
        raw_fts: bool,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        match = query if raw_fts else _match_expression(query)
        source_clause = "AND d.source = ?" if source else ""
        since_clause = "AND d.started_at >= ?" if since else ""
        parameters: list[Any] = [match]
        if source:
            parameters.append(source)
        if since:
            parameters.append(since)
        parameters.append(limit)
        rows = self._connection.execute(
            f"""
            SELECT
                d.id,
                d.session_id,
                d.turn_id,
                d.source,
                d.title,
                d.cwd,
                d.git_branch,
                d.started_at,
                d.ended_at,
                s.model AS model,
                snippet(documents_fts, 0, '[', ']', ' ... ', 24) AS snippet,
                bm25(documents_fts) AS score,
                d.provenance_json
            FROM documents_fts
            JOIN documents AS d ON d.rowid = documents_fts.rowid
            JOIN sessions AS s ON s.id = d.session_id
            WHERE documents_fts MATCH ? {source_clause} {since_clause}
            ORDER BY score, d.started_at DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return [_row_with_provenance(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        header = dict(row)
        doc_rows = self._connection.execute(
            "SELECT * FROM documents WHERE session_id = ? ORDER BY turn_id, chunk_index",
            (session_id,),
        ).fetchall()
        header["turns"] = _group_by_turn(doc_rows)
        return header

    def get_turn(self, session_id: str, turn_id: str, neighbors: int) -> dict[str, Any] | None:
        doc_rows = self._connection.execute(
            "SELECT * FROM documents WHERE session_id = ? ORDER BY turn_id, chunk_index",
            (session_id,),
        ).fetchall()
        turn_order = list(dict.fromkeys(str(row["turn_id"]) for row in doc_rows))
        if turn_id not in turn_order:
            return None
        index = turn_order.index(turn_id)
        start = max(0, index - neighbors)
        end = min(len(turn_order), index + neighbors + 1)
        window = set(turn_order[start:end])
        windowed_rows = [row for row in doc_rows if str(row["turn_id"]) in window]
        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "turns": _group_by_turn(windowed_rows),
        }

    def close(self) -> None:
        self._connection.close()
