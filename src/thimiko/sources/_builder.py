"""Sequencing helper: assigns event ids and resolves parent/tool linkage as a
source adapter walks a provider's records in order.
"""

from __future__ import annotations

from typing import Any

from thimiko.models import Attachment, Message, Provenance, Reasoning, Session, ToolCall, ToolResult


class EventBuilder:
    """Builds `Event`s for one `Session`, wiring up parent and tool-call links."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._sequence = 0
        self._call_event_ids: dict[str, str] = {}
        self._call_names: dict[str, str] = {}
        self._native_record_ids: dict[str, str] = {}

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _event_id(self, sequence: int) -> str:
        return f"{self.session.id}:event:{sequence:06d}"

    def _parent_id(self, parent_native_id: str | None) -> str | None:
        return self._native_record_ids.get(parent_native_id or "")

    def _remember(self, native_id: str | None, event_id: str) -> None:
        if native_id:
            self._native_record_ids[native_id] = event_id

    def add_message(
        self,
        *,
        line: int,
        native_type: str,
        native_id: str | None,
        part: int | None,
        timestamp: str | None,
        turn_id: str | None,
        parent_native_id: str | None,
        role: str | None,
        text: str | None,
        searchable: bool,
        data: dict[str, Any] | None = None,
    ) -> Message:
        sequence = self._next_sequence()
        event = Message(
            id=self._event_id(sequence),
            session_id=self.session.id,
            source=self.session.source,
            sequence=sequence,
            timestamp=timestamp,
            turn_id=turn_id,
            parent_id=self._parent_id(parent_native_id),
            role=role,
            provenance=Provenance(self.session.source_path, line, native_type, native_id, part),
            text=text,
            is_searchable=searchable,
            data=data,
        )
        self.session.events.append(event)
        self._remember(native_id, event.id)
        return event

    def add_tool_call(
        self,
        *,
        line: int,
        native_type: str,
        native_id: str | None,
        part: int | None = None,
        timestamp: str | None,
        turn_id: str | None,
        parent_native_id: str | None,
        tool_call_id: str | None,
        tool_name: str | None,
        data: Any,
    ) -> ToolCall:
        sequence = self._next_sequence()
        event = ToolCall(
            id=self._event_id(sequence),
            session_id=self.session.id,
            source=self.session.source,
            sequence=sequence,
            timestamp=timestamp,
            turn_id=turn_id,
            parent_id=self._parent_id(parent_native_id),
            role="assistant",
            provenance=Provenance(self.session.source_path, line, native_type, native_id, part),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            data=data,
        )
        self.session.events.append(event)
        self._remember(native_id, event.id)
        if tool_call_id:
            self._call_event_ids[tool_call_id] = event.id
            if tool_name:
                self._call_names[tool_call_id] = tool_name
        return event

    def add_tool_result(
        self,
        *,
        line: int,
        native_type: str,
        native_id: str | None,
        part: int | None = None,
        timestamp: str | None,
        turn_id: str | None,
        parent_native_id: str | None,
        tool_call_id: str | None,
        data: Any,
    ) -> ToolResult:
        sequence = self._next_sequence()
        linked_parent = self._call_event_ids.get(tool_call_id or "")
        event = ToolResult(
            id=self._event_id(sequence),
            session_id=self.session.id,
            source=self.session.source,
            sequence=sequence,
            timestamp=timestamp,
            turn_id=turn_id,
            parent_id=linked_parent or self._parent_id(parent_native_id),
            role="tool",
            provenance=Provenance(self.session.source_path, line, native_type, native_id, part),
            tool_call_id=tool_call_id,
            tool_name=self._call_names.get(tool_call_id or ""),
            data=data,
        )
        self.session.events.append(event)
        self._remember(native_id, event.id)
        return event

    def add_reasoning(
        self,
        *,
        line: int,
        native_type: str,
        native_id: str | None,
        part: int | None = None,
        timestamp: str | None,
        turn_id: str | None,
        parent_native_id: str | None,
        data: Any,
    ) -> Reasoning:
        sequence = self._next_sequence()
        event = Reasoning(
            id=self._event_id(sequence),
            session_id=self.session.id,
            source=self.session.source,
            sequence=sequence,
            timestamp=timestamp,
            turn_id=turn_id,
            parent_id=self._parent_id(parent_native_id),
            role="assistant",
            provenance=Provenance(self.session.source_path, line, native_type, native_id, part),
            data=data,
        )
        self.session.events.append(event)
        self._remember(native_id, event.id)
        return event

    def add_attachment(
        self,
        *,
        line: int,
        native_type: str,
        native_id: str | None,
        part: int | None = None,
        timestamp: str | None,
        turn_id: str | None,
        parent_native_id: str | None,
        role: str | None = None,
        data: Any,
    ) -> Attachment:
        sequence = self._next_sequence()
        event = Attachment(
            id=self._event_id(sequence),
            session_id=self.session.id,
            source=self.session.source,
            sequence=sequence,
            timestamp=timestamp,
            turn_id=turn_id,
            parent_id=self._parent_id(parent_native_id),
            role=role,
            provenance=Provenance(self.session.source_path, line, native_type, native_id, part),
            data=data,
        )
        self.session.events.append(event)
        self._remember(native_id, event.id)
        return event
