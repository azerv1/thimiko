"""Session and Turn: the aggregate roots of the domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import Event, Message


@dataclass(kw_only=True)
class Turn:
    """One user/assistant exchange: events sharing a `turn_id`, in source order."""

    id: str
    events: list[Event] = field(default_factory=list)

    def searchable_messages(self) -> list[Message]:
        return [event for event in self.events if isinstance(event, Message) and event.searchable]


@dataclass(kw_only=True)
class Session:
    """One canonical, provider-namespaced conversation stream."""

    id: str
    source: str
    source_session_id: str
    source_stream_id: str
    source_path: str
    parent_session_id: str | None = None
    agent_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    title: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    model: str | None = None
    events: list[Event] = field(default_factory=list)

    def turns(self) -> list[Turn]:
        """Group events into ordered turns, preserving first-seen order.

        Events without a `turn_id` each become their own singleton turn, matching
        the fallback-turn behavior of the source adapters.
        """
        grouped: dict[str, Turn] = {}
        order: list[str] = []
        for event in self.events:
            key = event.turn_id or f"{self.id}:turn:event-{event.sequence}"
            if key not in grouped:
                grouped[key] = Turn(id=key)
                order.append(key)
            grouped[key].events.append(event)
        return [grouped[key] for key in order]

    def searchable_messages(self) -> list[Message]:
        return [event for event in self.events if isinstance(event, Message) and event.searchable]

    def header(self) -> dict[str, Any]:
        """Session metadata without the event list, for storage rows and API responses."""
        return {
            "id": self.id,
            "source": self.source,
            "source_session_id": self.source_session_id,
            "source_stream_id": self.source_stream_id,
            "source_path": self.source_path,
            "parent_session_id": self.parent_session_id,
            "agent_id": self.agent_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "title": self.title,
            "cwd": self.cwd,
            "git_branch": self.git_branch,
            "model": self.model,
            "event_count": len(self.events),
            "searchable_event_count": len(self.searchable_messages()),
        }


__all__ = ["Session", "Turn"]
