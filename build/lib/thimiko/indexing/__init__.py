"""Indexing pipeline: turns sessions into stored, searchable documents."""

from __future__ import annotations

from .chunking import documents_for_session
from .indexer import BuildResult, Indexer, UpdateResult

__all__ = ["BuildResult", "Indexer", "UpdateResult", "documents_for_session"]
