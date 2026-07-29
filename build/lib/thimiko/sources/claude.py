"""Claude Code session adapter.

Claude writes flat records with a large top-level key set; the nested object is
``message`` (``role`` + ``content[]`` blocks), plus non-message record types
(``attachment``, ``file-history-snapshot``, ``ai-title``, ...). Claude's
fragmented content blocks are kept in source order, and each ``tool_result``
links back to its ``tool_call``.
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

CLAUDE_ROOT_ENV = "THIMIKO_CLAUDE_ROOT"


def _content_parts(content: Any) -> Iterable[tuple[int, str, dict[str, Any]]]:
    for index, item in enumerate(p.as_list(content)):
        if isinstance(item, str):
            yield index, "text", {"type": "text", "text": item}
        elif isinstance(item, dict):
            yield index, str(item.get("type") or "text"), item


class ClaudeSource(ChatSource):
    name = "claude"

    def default_roots(self) -> list[Path]:
        override = os.environ.get(CLAUDE_ROOT_ENV)
        root = Path(override).expanduser() if override else Path.home() / ".claude" / "projects"
        return [root]

    def matches(self, path: Path) -> bool:
        return sniff_file_source(path) == self.name

    def parse(self, path: Path) -> Session:
        records = p.read_records(path)
        return _normalize(path, records)


def _normalize(file_path: Path, records: list[tuple[int, dict[str, Any]]]) -> Session:
    source_session_id = p.fallback_session_id(file_path)
    for _, record in records:
        if record.get("sessionId"):
            source_session_id = str(record["sessionId"])
            break

    is_subagent = file_path.parent.name.lower() == "subagents"
    agent_id = next(
        (
            str(record["agentId"])
            for _, record in records
            if isinstance(record.get("agentId"), str) and record["agentId"]
        ),
        file_path.stem if is_subagent else None,
    )
    root_session_id = f"claude:{source_session_id}"
    canonical_session_id = f"{root_session_id}:agent:{agent_id}" if is_subagent else root_session_id
    started_at, ended_at = p.timestamp_bounds(records)
    session = Session(
        id=canonical_session_id,
        source="claude",
        source_session_id=source_session_id,
        source_stream_id=agent_id if is_subagent and agent_id else source_session_id,
        source_path=str(file_path),
        parent_session_id=root_session_id if is_subagent else None,
        agent_id=agent_id if is_subagent else None,
        started_at=started_at,
        ended_at=ended_at,
    )
    builder = EventBuilder(session)
    current_turn: str | None = None
    fallback_turn = 0

    for line, record in records:
        top_type = str(record.get("type") or "unknown")
        timestamp = p.as_optional_str(record.get("timestamp"))
        record_uuid = p.as_optional_str(record.get("uuid"))
        parent_uuid = p.as_optional_str(record.get("parentUuid"))
        session.cwd = p.as_optional_str(record.get("cwd")) or session.cwd
        session.git_branch = p.as_optional_str(record.get("gitBranch")) or session.git_branch

        if top_type == "ai-title":
            session.title = p.as_optional_str(record.get("aiTitle")) or session.title
            continue

        message = p.as_dict(record.get("message"))
        if message:
            role = str(message.get("role") or top_type or "assistant")
            native_turn = p.as_optional_str(record.get("promptId"))
            parts = list(_content_parts(message.get("content")))
            has_tool_result = any(kind == "tool_result" for _, kind, _ in parts)
            if native_turn:
                current_turn = p.turn_id(session.id, native_turn)
            elif role == "user" and not has_tool_result:
                fallback_turn += 1
                current_turn = p.turn_id(session.id, f"fallback-{fallback_turn}")
            elif current_turn is None:
                fallback_turn += 1
                current_turn = p.turn_id(session.id, f"fallback-{fallback_turn}")
            session.model = p.as_optional_str(message.get("model")) or session.model
            message_id = p.as_optional_str(message.get("id")) or record_uuid

            for part, content_type, item in parts:
                native_id = record_uuid or message_id
                if content_type == "tool_use":
                    builder.add_tool_call(
                        line=line,
                        native_type=f"{top_type}/{role}/{content_type}",
                        native_id=native_id,
                        timestamp=timestamp,
                        turn_id=current_turn,
                        parent_native_id=parent_uuid,
                        tool_call_id=p.as_optional_str(item.get("id")),
                        tool_name=p.as_optional_str(item.get("name")),
                        data=p.compact_data(item.get("input")),
                    )
                elif content_type == "tool_result":
                    output = item.get("content")
                    builder.add_tool_result(
                        line=line,
                        native_type=f"{top_type}/{role}/{content_type}",
                        native_id=native_id,
                        part=part,
                        timestamp=timestamp,
                        turn_id=current_turn,
                        parent_native_id=parent_uuid,
                        tool_call_id=p.as_optional_str(item.get("tool_use_id")),
                        data=p.content_stats(output),
                    )
                elif content_type == "thinking":
                    builder.add_reasoning(
                        line=line,
                        native_type=f"{top_type}/{role}/{content_type}",
                        native_id=native_id,
                        part=part,
                        timestamp=timestamp,
                        turn_id=current_turn,
                        parent_native_id=parent_uuid,
                        data=p.content_stats(item.get("thinking")),
                    )
                elif content_type == "image":
                    builder.add_attachment(
                        line=line,
                        native_type=f"{top_type}/{role}/{content_type}",
                        native_id=native_id,
                        part=part,
                        timestamp=timestamp,
                        turn_id=current_turn,
                        parent_native_id=parent_uuid,
                        role=role,
                        data={"media_type": p.as_dict(item.get("source")).get("media_type")},
                    )
                else:
                    text = p.content_text(item)
                    is_searchable = role in {"user", "assistant"} and not bool(record.get("isMeta"))
                    event_data: dict[str, Any] = {}
                    if content_type != "text":
                        event_data["content_type"] = content_type
                    if not is_searchable and text:
                        event_data["characters"] = len(text)
                    builder.add_message(
                        line=line,
                        native_type=f"{top_type}/{role}/{content_type}",
                        native_id=native_id,
                        part=part,
                        timestamp=timestamp,
                        turn_id=current_turn,
                        parent_native_id=parent_uuid,
                        role=role,
                        text=text if is_searchable else None,
                        searchable=is_searchable,
                        data=event_data or None,
                    )
            continue

        if top_type == "attachment":
            attachment = p.as_dict(record.get("attachment"))
            builder.add_attachment(
                line=line,
                native_type=top_type,
                native_id=record_uuid,
                timestamp=timestamp,
                turn_id=current_turn,
                parent_native_id=parent_uuid,
                data=p.scalar_metadata(attachment, {"content"}),
            )

    return session
