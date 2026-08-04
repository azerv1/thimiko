"""Store: the pluggable persistence-backend interface.

Everything above this layer (indexer, retriever, CLI, MCP) depends only on this
ABC, never on `sqlite3` directly. Swapping in a libSQL-backed store later means
implementing this interface once; nothing else in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from thimiko.dto import SearchDocument
from thimiko.models import Session


class Store(ABC):
    """Persistence backend for canonical sessions and their search documents."""

    @abstractmethod
    def create_schema(self, *, reset: bool) -> None:
        """Create tables/indexes. `reset=True` drops and recreates everything."""

    @abstractmethod
    def upsert_session(self, session: Session, documents: list[SearchDocument]) -> None:
        """Insert or replace a session header and its search documents."""

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Remove a session, its documents, and their search index rows."""

    @abstractmethod
    def file_state(self, path: str) -> tuple[float, int] | None:
        """Return the (mtime, size) this store last indexed `path` at, if any."""

    @abstractmethod
    def record_file(self, path: str, mtime: float, size: int, session_ids: list[str]) -> None:
        """Remember that `path` was indexed as `session_ids` at (mtime, size).

        A file usually holds one session; Cursor's `state.vscdb` holds many.
        """

    @abstractmethod
    def forget_file(self, path: str) -> None:
        """Remove `path` from the incremental-update index (for pruning)."""

    @abstractmethod
    def known_files(self) -> dict[str, list[str]]:
        """Map of indexed file path -> its session ids, for pruning deleted files."""

    @abstractmethod
    def session_counts(self) -> dict[str, int]:
        """Number of indexed sessions per source name."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        source: str | None,
        limit: int,
        raw_fts: bool,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ranked full-text search over indexed documents.

        `since`, when given, is an ISO-8601 cutoff; only documents whose
        `started_at` is at or after it are returned.
        """

    @abstractmethod
    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Session header plus its documents, grouped by turn."""

    @abstractmethod
    def get_turn(self, session_id: str, turn_id: str, neighbors: int) -> dict[str, Any] | None:
        """A turn's chunks plus up to `neighbors` turns before/after it."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection."""
