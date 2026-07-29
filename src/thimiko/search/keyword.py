"""KeywordRetriever: BM25 full-text search over a `Store`."""

from __future__ import annotations

from typing import Any

from thimiko.dto import SearchResult
from thimiko.storage import Store

from .base import Retriever


class KeywordRetriever(Retriever):
    """Ranks turn documents with SQLite FTS5 / BM25 through the `Store` interface."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def search(
        self,
        query: str,
        *,
        source: str | None = None,
        limit: int = 10,
        raw_fts: bool = False,
        since: str | None = None,
    ) -> list[SearchResult]:
        rows = self.store.search(query, source=source, limit=limit, raw_fts=raw_fts, since=since)
        return [
            SearchResult(
                id=str(row["id"]),
                session_id=str(row["session_id"]),
                turn_id=str(row["turn_id"]),
                source=str(row["source"]),
                title=row["title"],
                cwd=row["cwd"],
                git_branch=row["git_branch"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                snippet=str(row["snippet"]),
                score=float(row["score"]),
                model=row["model"],
                provenance=row["provenance"],
            )
            for row in rows
        ]

    def expand(self, session_id: str, turn_id: str, neighbors: int = 1) -> dict[str, Any] | None:
        return self.store.get_turn(session_id, turn_id, neighbors)
