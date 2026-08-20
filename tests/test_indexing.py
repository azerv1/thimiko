from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from thimiko.indexing import Indexer, documents_for_session
from thimiko.models import Session
from thimiko.search import KeywordRetriever
from thimiko.sources.codex import CodexSource
from thimiko.sources.opencode import OpenCodeSource
from thimiko.storage import SqliteStore


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _codex_records(user_text: str, assistant_text: str) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"session_id": "search1", "cwd": "C:/repo"},
        },
        {
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "turn_context",
            "payload": {"turn_id": "turn1"},
        },
        {
            "timestamp": "2026-01-01T00:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        },
        {
            "timestamp": "2026-01-01T00:00:03Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            },
        },
    ]


def test_documents_for_session_indexes_whole_turn(tmp_path: Path) -> None:
    path = tmp_path / "codex.jsonl"
    write_jsonl(
        path, _codex_records("Where is frobnication handled?", "The schema adapter handles it.")
    )
    session = CodexSource().parse(path)
    documents = documents_for_session(session)

    assert len(documents) == 1
    assert "[USER]" in documents[0].text
    assert "[ASSISTANT]" in documents[0].text


def test_build_then_search_returns_provenance(tmp_path: Path) -> None:
    path = tmp_path / "codex.jsonl"
    write_jsonl(
        path, _codex_records("Where is frobnication handled?", "The schema adapter handles it.")
    )
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        result = Indexer(store).build([path], forced_source="codex")
        assert (result.sessions, result.documents) == (1, 1)

        hits = KeywordRetriever(store).search("frobnication schema", limit=5)
        assert len(hits) == 1
        assert hits[0].session_id == "codex:search1"
        assert hits[0].provenance[0]["path"] == str(path)
    finally:
        store.close()


def test_update_skips_unchanged_and_reindexes_modified_file(tmp_path: Path) -> None:
    records = _codex_records("first question", "first answer")
    path = tmp_path / "codex.jsonl"
    write_jsonl(path, records)
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        indexer = Indexer(store)
        built = indexer.build([path], forced_source="codex")
        assert built.sessions == 1

        unchanged = indexer.update([path], forced_source="codex")
        assert (unchanged.added, unchanged.updated, unchanged.skipped) == (0, 0, 1)

        records.append(
            {
                "timestamp": "2026-01-01T00:00:04Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "second question"}],
                },
            }
        )
        write_jsonl(path, records)
        modified = indexer.update([path], forced_source="codex")
        assert (modified.added, modified.updated, modified.skipped) == (0, 1, 0)

        hits = KeywordRetriever(store).search("second question", limit=5)
        assert len(hits) == 1
    finally:
        store.close()


def test_update_prune_removes_deleted_files_session(tmp_path: Path) -> None:
    path = tmp_path / "codex.jsonl"
    write_jsonl(path, _codex_records("first question", "first answer"))
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        indexer = Indexer(store)
        indexer.build([path], forced_source="codex")

        path.unlink()
        result = indexer.update([path.parent], forced_source="codex", prune=True)
        assert result.pruned == 1

        hits = KeywordRetriever(store).search("first question", limit=5)
        assert hits == []
    finally:
        store.close()


def test_update_adds_new_file(tmp_path: Path) -> None:
    first_path = tmp_path / "codex-a.jsonl"
    write_jsonl(first_path, _codex_records("first question", "first answer"))
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        indexer = Indexer(store)
        indexer.build([first_path], forced_source="codex")

        second_records = _codex_records("second question", "second answer")
        second_records[0]["payload"] = {"session_id": "search2", "cwd": "C:/repo"}
        second_path = tmp_path / "codex-b.jsonl"
        write_jsonl(second_path, second_records)

        result = indexer.update([tmp_path], forced_source="codex")
        assert result.added == 1
        assert result.skipped == 1

        hits = KeywordRetriever(store).search("second answer", limit=5)
        assert len(hits) == 1
    finally:
        store.close()


def _write_cursor_db(path: Path, rows: dict[str, Any]) -> None:
    """A Cursor-shaped `cursorDiskKV` database: one file holding many sessions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
        connection.executemany(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in rows.items()],
        )
        connection.commit()
    finally:
        connection.close()


def _cursor_chat(composer_id: str, bubble_id: str, text: str) -> dict[str, Any]:
    return {
        f"composerData:{composer_id}": {
            "name": f"chat {composer_id}",
            "createdAt": 1767225600000,
            "fullConversationHeadersOnly": [{"bubbleId": bubble_id, "type": 1}],
        },
        f"bubbleId:{composer_id}:{bubble_id}": {"type": 1, "text": text},
    }


def _create_opencode_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY, parent_id TEXT, directory TEXT, title TEXT,
            time_created INTEGER, time_updated INTEGER, workspace_id TEXT,
            agent TEXT, model TEXT
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT
        );
        """
    )


def _insert_opencode_chat(
    connection: sqlite3.Connection, session_id: str, message_id: str, text: str, created: int
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO session VALUES (?, NULL, ?, ?, ?, ?, NULL, NULL, NULL)",
        (session_id, "C:/repo", f"chat {session_id}", created, created + 1),
    )
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?)",
        (message_id, session_id, created, json.dumps({"role": "user", "agent": "build"})),
    )
    connection.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?)",
        (
            f"part-{message_id}",
            message_id,
            session_id,
            json.dumps({"type": "text", "text": text}),
        ),
    )


def _write_opencode_db(path: Path, chats: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        _create_opencode_schema(connection)
        for index, (session_id, text) in enumerate(chats):
            _insert_opencode_chat(
                connection, session_id, f"message-{index}", text, 1767225600000 + index
            )
        connection.commit()
    finally:
        connection.close()


def test_build_indexes_every_session_in_a_multi_session_file(tmp_path: Path) -> None:
    path = tmp_path / "state.vscdb"
    _write_cursor_db(
        path, _cursor_chat("c1", "b1", "alpha question") | _cursor_chat("c2", "b2", "beta question")
    )
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        result = Indexer(store).build([path], forced_source="cursor")
        assert (result.sessions, result.documents) == (2, 2)

        retriever = KeywordRetriever(store)
        assert retriever.search("alpha question", limit=5)[0].session_id == "cursor:c1"
        assert retriever.search("beta question", limit=5)[0].session_id == "cursor:c2"
    finally:
        store.close()


def test_update_replaces_all_sessions_of_a_changed_multi_session_file(tmp_path: Path) -> None:
    path = tmp_path / "state.vscdb"
    _write_cursor_db(
        path, _cursor_chat("c1", "b1", "alpha question") | _cursor_chat("c2", "b2", "beta question")
    )
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        indexer = Indexer(store)
        indexer.build([path], forced_source="cursor")

        path.unlink()
        _write_cursor_db(path, _cursor_chat("c1", "b1", "gamma question"))
        result = indexer.update([path], forced_source="cursor")
        assert (result.added, result.updated) == (0, 1)

        retriever = KeywordRetriever(store)
        assert retriever.search("gamma question", limit=5)[0].session_id == "cursor:c1"
        # The dropped chat's session went with it — no orphan documents.
        assert retriever.search("beta question", limit=5) == []
    finally:
        store.close()


def test_opencode_build_and_update_replace_every_database_session(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    _write_opencode_db(path, [("s1", "alpha opencode"), ("s2", "beta opencode")])
    store = SqliteStore(tmp_path / "thimiko.sqlite")
    try:
        indexer = Indexer(store)
        built = indexer.build([path], forced_source="opencode")
        assert (built.sessions, built.documents) == (2, 2)

        path.unlink()
        _write_opencode_db(path, [("s1", "gamma opencode")])
        updated = indexer.update([path], forced_source="opencode")
        assert (updated.added, updated.updated, updated.skipped) == (0, 1, 0)

        retriever = KeywordRetriever(store)
        assert retriever.search("gamma opencode", limit=5)[0].session_id == "opencode:s1"
        assert retriever.search("beta opencode", limit=5) == []
    finally:
        store.close()


def test_opencode_wal_only_change_triggers_update(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    writer = sqlite3.connect(path)
    store = SqliteStore(tmp_path / "thimiko.sqlite")
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        _create_opencode_schema(writer)
        _insert_opencode_chat(writer, "s1", "m1", "first wal message", 1767225600000)
        writer.commit()

        source = OpenCodeSource()
        indexer = Indexer(store)
        before = source.fingerprint(path)
        main_before = (path.stat().st_mtime_ns, path.stat().st_size)
        indexer.build([path], forced_source="opencode")
        assert store.file_state(str(path)) == before

        _insert_opencode_chat(writer, "s1", "m2", "second wal message", 1767225601000)
        writer.commit()
        after = source.fingerprint(path)
        main_after = (path.stat().st_mtime_ns, path.stat().st_size)

        assert main_after == main_before
        assert after != before
        updated = indexer.update([path], forced_source="opencode")
        assert (updated.added, updated.updated, updated.skipped) == (0, 1, 0)
        assert store.file_state(str(path)) == after
        assert len(KeywordRetriever(store).search("second wal message", limit=5)) == 1
    finally:
        store.close()
        writer.close()


def test_indexer_records_the_fingerprint_captured_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "codex.jsonl"
    write_jsonl(path, _codex_records("before parse", "answer"))

    class MutatingCodexSource(CodexSource):
        name = "mutating-codex"

        def parse_all(self, source_path: Path) -> list[Session]:
            sessions = super().parse_all(source_path)
            with source_path.open("a", encoding="utf-8") as handle:
                handle.write("not-json\n")
            return sessions

    source = MutatingCodexSource()
    captured = source.fingerprint(path)
    store = SqliteStore(tmp_path / "thimiko.sqlite")
    try:
        Indexer(store, [source]).build([path], forced_source=source.name)
        assert store.file_state(str(path)) == captured
        assert source.fingerprint(path) != captured
    finally:
        store.close()
