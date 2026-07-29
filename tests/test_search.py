from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from thimiko.dto import iso_days_ago
from thimiko.indexing import Indexer
from thimiko.search import KeywordRetriever
from thimiko.storage import SqliteStore


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_expand_returns_neighboring_turns(tmp_path: Path) -> None:
    records: list[dict[str, object]] = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"session_id": "expand1", "cwd": "C:/repo"},
        },
    ]
    for index, turn in enumerate(("alpha", "bravo", "charlie"), start=1):
        records.append(
            {
                "timestamp": f"2026-01-01T00:0{index}:00Z",
                "type": "turn_context",
                "payload": {"turn_id": turn},
            }
        )
        records.append(
            {
                "timestamp": f"2026-01-01T00:0{index}:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"question about {turn}"}],
                },
            }
        )
        records.append(
            {
                "timestamp": f"2026-01-01T00:0{index}:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"answer about {turn}"}],
                },
            }
        )

    path = tmp_path / "codex.jsonl"
    write_jsonl(path, records)
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        Indexer(store).build([path], forced_source="codex")
        retriever = KeywordRetriever(store)

        hits = retriever.search("bravo", limit=5)
        assert len(hits) == 1
        middle_hit = hits[0]

        context = retriever.expand(middle_hit.session_id, middle_hit.turn_id, neighbors=1)
        assert context is not None
        turn_ids = [turn["turn_id"] for turn in context["turns"]]
        assert turn_ids == [
            "codex:expand1:turn:alpha",
            "codex:expand1:turn:bravo",
            "codex:expand1:turn:charlie",
        ]

        edge_context = retriever.expand("codex:expand1", "codex:expand1:turn:alpha", neighbors=0)
        assert edge_context is not None
        assert [turn["turn_id"] for turn in edge_context["turns"]] == ["codex:expand1:turn:alpha"]

        assert retriever.expand("codex:expand1", "does-not-exist", neighbors=1) is None
    finally:
        store.close()


def test_since_filters_out_old_turns(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    recent = _iso(now - timedelta(days=1))
    old = _iso(now - timedelta(days=40))
    records: list[dict[str, Any]] = [
        {"timestamp": old, "type": "session_meta", "payload": {"session_id": "s1"}},
        {"timestamp": recent, "type": "turn_context", "payload": {"turn_id": "recent"}},
        {
            "timestamp": recent,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "alpha recent question"}],
            },
        },
        {"timestamp": old, "type": "turn_context", "payload": {"turn_id": "old"}},
        {
            "timestamp": old,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "alpha old question"}],
            },
        },
    ]
    path = tmp_path / "codex.jsonl"
    write_jsonl(path, records)
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        Indexer(store).build([path], forced_source="codex")
        retriever = KeywordRetriever(store)

        assert len(retriever.search("alpha", limit=10)) == 2

        recent_only = retriever.search("alpha", limit=10, since=iso_days_ago(10))
        assert len(recent_only) == 1
        assert recent_only[0].turn_id == "codex:s1:turn:recent"
    finally:
        store.close()
