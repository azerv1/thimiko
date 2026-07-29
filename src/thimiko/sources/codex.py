"""Codex CLI session adapter.

Codex writes a uniform envelope: ``{"timestamp", "type", "payload"}``, with the
meaningful sub-type in ``payload["type"]``. Codex also emits duplicate display
events (`event_msg`) alongside authoritative `response_item` messages; the
duplicates are dropped whenever `response_item` messages are present.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from thimiko.models import Session
from thimiko.utils import sniff_file_source

from . import _parsing as p
from ._builder import EventBuilder
from .base import ChatSource

CODEX_ROOT_ENV = "THIMIKO_CODEX_ROOT"


def _content_parts(content: Any) -> Iterable[tuple[int, str, str | None]]:
    for index, item in enumerate(p.as_list(content)):
        if isinstance(item, str):
            yield index, "text", item or None
            continue
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "text")
        yield index, item_type, p.content_text(item)


class CodexSource(ChatSource):
    name = "codex"

    def default_roots(self) -> list[Path]:
        override = os.environ.get(CODEX_ROOT_ENV)
        root = Path(override).expanduser() if override else Path.home() / ".codex" / "sessions"
        return [root]

    def matches(self, path: Path) -> bool:
        return sniff_file_source(path) == self.name

    def parse(self, path: Path) -> Session:
        records = p.read_records(path)
        return _normalize(path, records)


def _normalize(file_path: Path, records: list[tuple[int, dict[str, Any]]]) -> Session:
    source_session_id = p.fallback_session_id(file_path)
    source_stream_id = source_session_id
    session_meta: dict[str, Any] = {}
    for _, record in records:
        if record.get("type") == "session_meta":
            session_meta = p.as_dict(record.get("payload"))
            source_session_id = str(
                session_meta.get("session_id") or session_meta.get("id") or source_session_id
            )
            source_stream_id = str(session_meta.get("id") or source_session_id)
            break

    started_at, ended_at = p.timestamp_bounds(records)
    is_subagent = source_stream_id != source_session_id
    session = Session(
        id=f"codex:{source_stream_id}",
        source="codex",
        source_session_id=source_session_id,
        source_stream_id=source_stream_id,
        source_path=str(file_path),
        parent_session_id=(
            f"codex:{session_meta.get('parent_thread_id') or source_session_id}"
            if is_subagent
            else None
        ),
        agent_id=(
            (
                p.as_optional_str(session_meta.get("agent_path"))
                or p.as_optional_str(session_meta.get("agent_nickname"))
                or source_stream_id
            )
            if is_subagent
            else None
        ),
        started_at=started_at,
        ended_at=ended_at,
    )
    builder = EventBuilder(session)
    current_turn: str | None = None
    fallback_turn = 0
    has_response_messages = any(
        record.get("type") == "response_item"
        and p.as_dict(record.get("payload")).get("type") == "message"
        for _, record in records
    )

    for line, record in records:
        top_type = str(record.get("type") or "unknown")
        payload = p.as_dict(record.get("payload"))
        payload_type = str(payload.get("type") or "")
        timestamp = p.as_optional_str(record.get("timestamp"))
        native_id = p.as_optional_str(payload.get("id"))

        if top_type == "session_meta":
            session.cwd = p.as_optional_str(payload.get("cwd")) or session.cwd
            continue

        if top_type == "turn_context":
            native_turn = p.as_optional_str(payload.get("turn_id"))
            if native_turn:
                current_turn = p.turn_id(session.id, native_turn)
            session.cwd = p.as_optional_str(payload.get("cwd")) or session.cwd
            session.model = p.as_optional_str(payload.get("model")) or session.model
            continue

        if top_type == "event_msg" and payload_type == "task_started":
            native_turn = p.as_optional_str(payload.get("turn_id"))
            if native_turn:
                current_turn = p.turn_id(session.id, native_turn)
            continue

        if top_type == "response_item" and payload_type in {"message", "agent_message"}:
            role = str(payload.get("role") or payload.get("author") or "assistant")
            if role == "user" and current_turn is None:
                fallback_turn += 1
                current_turn = p.turn_id(session.id, f"fallback-{fallback_turn}")
            for part, content_type, text in _content_parts(payload.get("content")):
                is_searchable = role in {"user", "assistant"}
                event_data: dict[str, Any] = {}
                if content_type != "text":
                    event_data["content_type"] = content_type
                if not is_searchable and text:
                    event_data["characters"] = len(text)
                builder.add_message(
                    line=line,
                    native_type=p.native_type(record, payload),
                    native_id=native_id,
                    part=part,
                    timestamp=timestamp,
                    turn_id=current_turn,
                    parent_native_id=None,
                    role=role,
                    text=text if is_searchable else None,
                    searchable=is_searchable,
                    data=event_data or None,
                )
            continue

        if top_type == "response_item" and payload_type == "reasoning":
            summary = p.content_text(payload.get("summary"))
            builder.add_reasoning(
                line=line,
                native_type=p.native_type(record, payload),
                native_id=native_id,
                timestamp=timestamp,
                turn_id=current_turn,
                parent_native_id=None,
                data={"summary_characters": len(summary)} if summary else None,
            )
            continue

        if top_type == "response_item" and payload_type in {
            "function_call",
            "custom_tool_call",
            "web_search_call",
            "tool_search_call",
        }:
            call_id = p.as_optional_str(payload.get("call_id")) or native_id
            name = p.as_optional_str(payload.get("name")) or payload_type.removesuffix("_call")
            tool_data = payload.get("arguments", payload.get("input", payload.get("action")))
            builder.add_tool_call(
                line=line,
                native_type=p.native_type(record, payload),
                native_id=native_id,
                timestamp=timestamp,
                turn_id=current_turn,
                parent_native_id=None,
                tool_call_id=call_id,
                tool_name=name,
                data=p.compact_data(tool_data),
            )
            continue

        if top_type == "response_item" and payload_type in {
            "function_call_output",
            "custom_tool_call_output",
            "tool_search_output",
        }:
            call_id = p.as_optional_str(payload.get("call_id"))
            output = payload.get("output", payload.get("content"))
            builder.add_tool_result(
                line=line,
                native_type=p.native_type(record, payload),
                native_id=native_id,
                timestamp=timestamp,
                turn_id=current_turn,
                parent_native_id=None,
                tool_call_id=call_id,
                data=p.content_stats(output),
            )
            continue

        if top_type == "event_msg" and payload_type in {
            "user_message",
            "agent_message",
            "task_complete",
        }:
            if has_response_messages:
                continue
            role = "user" if payload_type == "user_message" else "assistant"
            text = p.as_optional_str(payload.get("message")) or p.as_optional_str(
                payload.get("last_agent_message")
            )
            if role == "user":
                fallback_turn += 1
                current_turn = p.turn_id(session.id, f"fallback-{fallback_turn}")
            builder.add_message(
                line=line,
                native_type=p.native_type(record, payload),
                native_id=native_id,
                part=None,
                timestamp=timestamp,
                turn_id=current_turn,
                parent_native_id=None,
                role=role,
                text=text,
                searchable=True,
            )

    return session
