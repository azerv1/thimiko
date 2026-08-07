"""Regression benchmarks for the ingest/search hot paths (see bench/README.md).

Runs with the normal suite; `pytest-benchmark` reports timings without
failing the build, so these guard against reintroducing per-row commits or
similar regressions without pinning a machine-dependent threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thimiko.indexing import Indexer
from thimiko.search import KeywordRetriever
from thimiko.storage import SqliteStore

_SESSION_COUNT = 20


def _session_records(index: int) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"session_id": f"bench-{index}", "cwd": "C:/repo"},
        },
        {
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"question about topic {index}"}],
            },
        },
        {
            "timestamp": "2026-01-01T00:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"answer about topic {index}"}],
            },
        },
    ]


def _write_fixtures(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        path = directory / f"rollout-bench-{index}.jsonl"
        records = _session_records(index)
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _write_large_session(directory: Path, turn_count: int) -> None:
    """One session with many turns, to benchmark `get_turn` windowing."""
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"session_id": "large", "cwd": "C:/repo"},
        }
    ]
    for turn in range(turn_count):
        records.append(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "type": "turn_context",
                "payload": {"turn_id": f"turn{turn}"},
            }
        )
        records.append(
            {
                "timestamp": "2026-01-01T00:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"question {turn}"}],
                },
            }
        )
        records.append(
            {
                "timestamp": "2026-01-01T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"answer {turn}"}],
                },
            }
        )
    path = directory / "rollout-large.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_build_benchmark(tmp_path: Path, benchmark: Any) -> None:
    fixtures_dir = tmp_path / "fixtures"
    _write_fixtures(fixtures_dir, _SESSION_COUNT)
    db_path = tmp_path / "thimiko.sqlite"

    def build() -> None:
        store = SqliteStore(db_path)
        try:
            Indexer(store).build([fixtures_dir], forced_source="codex")
        finally:
            store.close()

    benchmark(build)


def test_noop_update_benchmark(tmp_path: Path, benchmark: Any) -> None:
    fixtures_dir = tmp_path / "fixtures"
    _write_fixtures(fixtures_dir, _SESSION_COUNT)
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        Indexer(store).build([fixtures_dir], forced_source="codex")
    finally:
        store.close()

    def update() -> None:
        store = SqliteStore(db_path)
        try:
            Indexer(store).update([fixtures_dir], forced_source="codex")
        finally:
            store.close()

    benchmark(update)


def test_search_benchmark(tmp_path: Path, benchmark: Any) -> None:
    fixtures_dir = tmp_path / "fixtures"
    _write_fixtures(fixtures_dir, _SESSION_COUNT)
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        Indexer(store).build([fixtures_dir], forced_source="codex")
        retriever = KeywordRetriever(store)
        benchmark(lambda: retriever.search("question topic", limit=10))
    finally:
        store.close()


def test_get_turn_benchmark(tmp_path: Path, benchmark: Any) -> None:
    fixtures_dir = tmp_path / "fixtures"
    _write_large_session(fixtures_dir, turn_count=500)
    db_path = tmp_path / "thimiko.sqlite"

    store = SqliteStore(db_path)
    try:
        Indexer(store).build([fixtures_dir], forced_source="codex")
        retriever = KeywordRetriever(store)
        benchmark(lambda: retriever.expand("codex:large", "codex:large:turn:turn5", neighbors=1))
    finally:
        store.close()
