"""Typed event hierarchy for one atomic, ordered occurrence within a session.

Replaces a single tagged dataclass (a `kind` string field) with a small class
per event category, so each carries only the fields relevant to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from .provenance import Provenance


@dataclass(kw_only=True)
class Event(ABC):
    """Base fields shared by every event category."""

    id: str
    session_id: str
    source: str
    sequence: int
    timestamp: str | None
    turn_id: str | None
    parent_id: str | None
    role: str | None
    provenance: Provenance

    @property
    @abstractmethod
    def kind(self) -> str:
        """Event category, e.g. ``message``, ``tool_call``, ``reasoning``."""

    @property
    def searchable(self) -> bool:
        """Whether this event belongs in the default search corpus."""
        return False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind
        payload["searchable"] = self.searchable
        return payload


@dataclass(kw_only=True)
class Message(Event):
    """An ordinary chat message, or a fragmented content block from one."""

    text: str | None = None
    is_searchable: bool = False
    data: dict[str, Any] | None = None

    @property
    def kind(self) -> str:
        return "message"

    @property
    def searchable(self) -> bool:
        return bool(self.is_searchable and self.text)


@dataclass(kw_only=True)
class ToolCall(Event):
    """An assistant-initiated tool invocation."""

    tool_call_id: str | None = None
    tool_name: str | None = None
    data: Any = None

    @property
    def kind(self) -> str:
        return "tool_call"


@dataclass(kw_only=True)
class ToolResult(Event):
    """The output of a tool invocation, linked back to its `ToolCall`."""

    tool_call_id: str | None = None
    tool_name: str | None = None
    data: Any = None

    @property
    def kind(self) -> str:
        return "tool_result"


@dataclass(kw_only=True)
class Reasoning(Event):
    """Hidden model reasoning/thinking. Never searchable; size only."""

    data: Any = None

    @property
    def kind(self) -> str:
        return "reasoning"


@dataclass(kw_only=True)
class Attachment(Event):
    """A non-text payload (image, file snapshot, ...)."""

    data: Any = None

    @property
    def kind(self) -> str:
        return "attachment"


__all__ = [
    "Attachment",
    "Event",
    "Message",
    "Reasoning",
    "ToolCall",
    "ToolResult",
]
