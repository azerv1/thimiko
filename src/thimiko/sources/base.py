"""ChatSource: the pluggable ingestion-adapter interface.

Add a new chat-history provider by implementing this ABC and registering an
instance in :mod:`thimiko.sources` — nothing else in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from thimiko.models import Session
from thimiko.utils import discover_jsonl_files


class ChatSource(ABC):
    """One provider's dialect: where to look, how to recognize it, how to parse it."""

    name: str

    @abstractmethod
    def default_roots(self) -> list[Path]:
        """Default directories to scan for this source's session files."""

    def discover(self, root: Path) -> list[Path]:
        """Session files under `root` for this source.

        Defaults to recursive `*.jsonl` discovery; override for providers whose
        history is stored elsewhere or in another shape (e.g. VSCode's mixed
        `.json`/`.jsonl` chat sessions).
        """
        return discover_jsonl_files(root)

    @abstractmethod
    def matches(self, path: Path) -> bool:
        """Whether `path` looks like a session file written by this source."""

    @abstractmethod
    def parse(self, path: Path) -> Session:
        """Normalize one session file into the canonical domain model."""
