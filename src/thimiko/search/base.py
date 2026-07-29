"""Retriever: the pluggable search interface.

Adding a hybrid or vector retriever later means implementing this ABC once;
the CLI and MCP server depend only on it, never on a specific ranking method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from thimiko.dto import SearchResult


class Retriever(ABC):
    """Turns a query into ranked results, and expands a hit into neighboring context."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        source: str | None = None,
        limit: int = 10,
        raw_fts: bool = False,
        since: str | None = None,
    ) -> list[SearchResult]:
        """Ranked search over indexed turn documents.

        `since`, when set, is an ISO-8601 cutoff; only turns at or after it match.
        """

    @abstractmethod
    def expand(self, session_id: str, turn_id: str, neighbors: int = 1) -> dict[str, Any] | None:
        """A turn's chunks plus up to `neighbors` turns before/after it."""
