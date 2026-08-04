"""Google Gemini CLI chat-history adapter.

Gemini stores project-scoped sessions below
``~/.gemini/tmp/<project-hash>/chats``. Legacy ``.json`` files are complete
conversation snapshots. Current ``.jsonl`` files are append-only logs made of
metadata, message records, ``$set`` checkpoints, and ``$rewindTo`` records.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from thimiko.models import Session

from . import _parsing as p
from ._builder import EventBuilder
from .base import ChatSource

GEMINI_ROOT_ENV = "THIMIKO_GEMINI_ROOT"
GEMINI_HOME_ENV = "GEMINI_CLI_HOME"

_SESSION_SUFFIXES = {".json", ".jsonl"}
_MESSAGE_TYPES = {"user", "gemini", "info", "error", "warning"}


class GeminiSource(ChatSource):
    name = "gemini"

    def default_roots(self) -> list[Path]:
        override = os.environ.get(GEMINI_ROOT_ENV)
        if override:
            return [Path(override).expanduser()]
        configured_home = os.environ.get(GEMINI_HOME_ENV)
        home = Path(configured_home).expanduser() if configured_home else Path.home()
        return [home / ".gemini" / "tmp"]

    def discover(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in _SESSION_SUFFIXES else []
        if not root.is_dir():
            return []
        chat_dirs = [root] if root.name == "chats" else list(root.rglob("chats"))
        return sorted(
            path
            for chat_dir in chat_dirs
            if chat_dir.is_dir()
            for path in chat_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _SESSION_SUFFIXES
        )

    def matches(self, path: Path) -> bool:
        return _is_gemini_metadata(_read_head(path))

    def parse(self, path: Path) -> Session:
        metadata, messages = _load_session(path)
        return _normalize(path, metadata, messages)


def _read_head(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    if raw.strip():
                        return p.as_dict(json.loads(raw))
            return {}
        return p.as_dict(json.loads(path.read_text(encoding="utf-8", errors="replace")))
    except (OSError, json.JSONDecodeError):
        return {}


def _is_gemini_metadata(record: dict[str, Any]) -> bool:
    return (
        isinstance(record.get("sessionId"), str)
        and isinstance(record.get("projectHash"), str)
        and any(key in record for key in ("startTime", "lastUpdated", "messages"))
    )


def _is_message(record: dict[str, Any]) -> bool:
    return isinstance(record.get("id"), str) and record.get("type") in _MESSAGE_TYPES


def _load_session(path: Path) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    if path.suffix.lower() == ".json":
        snapshot = _read_head(path)
        raw_messages = snapshot.get("messages")
        messages = (
            [
                (1, message)
                for message in raw_messages
                if isinstance(message, dict) and _is_message(message)
            ]
            if isinstance(raw_messages, list)
            else []
        )
        return snapshot, messages
    return _reconstruct_jsonl(path)


def _reconstruct_jsonl(
    path: Path,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    metadata: dict[str, Any] = {}
    messages: dict[str, tuple[int, dict[str, Any]]] = {}
    try:
        records = p.read_records(path)
    except OSError:
        return metadata, []

    for line, record in records:
        rewind_to = p.as_optional_str(record.get("$rewindTo"))
        if rewind_to:
            _rewind_messages(messages, rewind_to)
            continue

        updates = record.get("$set")
        if isinstance(updates, dict):
            checkpoint = updates.get("messages")
            if isinstance(checkpoint, list):
                messages.clear()
                _add_messages(messages, checkpoint, line)
            metadata.update({key: value for key, value in updates.items() if key != "messages"})
            continue

        if _is_message(record):
            messages[str(record["id"])] = (line, record)
            continue

        if "sessionId" in record or "projectHash" in record:
            metadata.update({key: value for key, value in record.items() if key != "messages"})
            snapshot_messages = record.get("messages")
            if isinstance(snapshot_messages, list):
                messages.clear()
                _add_messages(messages, snapshot_messages, line)

    return metadata, list(messages.values())


def _add_messages(
    target: dict[str, tuple[int, dict[str, Any]]], messages: list[Any], line: int
) -> None:
    for message in messages:
        if isinstance(message, dict) and _is_message(message):
            target[str(message["id"])] = (line, message)


def _rewind_messages(messages: dict[str, tuple[int, dict[str, Any]]], rewind_to: str) -> None:
    ids = list(messages)
    try:
        start = ids.index(rewind_to)
    except ValueError:
        messages.clear()
        return
    for message_id in ids[start:]:
        del messages[message_id]


def _normalize(
    file_path: Path,
    metadata: dict[str, Any],
    messages: list[tuple[int, dict[str, Any]]],
) -> Session:
    source_session_id = p.as_optional_str(metadata.get("sessionId")) or file_path.stem
    parent_source_id = _subagent_parent(file_path, metadata)
    if parent_source_id:
        canonical_id = f"gemini:{parent_source_id}:agent:{source_session_id}"
        parent_session_id = f"gemini:{parent_source_id}"
        agent_id = source_session_id
    else:
        canonical_id = f"gemini:{source_session_id}"
        parent_session_id = None
        agent_id = None

    session = Session(
        id=canonical_id,
        source="gemini",
        source_session_id=source_session_id,
        source_stream_id=source_session_id,
        source_path=str(file_path),
        parent_session_id=parent_session_id,
        agent_id=agent_id,
        started_at=p.as_optional_str(metadata.get("startTime")),
        ended_at=p.as_optional_str(metadata.get("lastUpdated")),
        title=p.as_optional_str(metadata.get("summary")),
        cwd=_project_root(file_path),
    )
    builder = EventBuilder(session)
    current_turn: str | None = None
    fallback_turn = 0
    for line, message in messages:
        message_type = str(message.get("type") or "")
        message_id = p.as_optional_str(message.get("id"))
        if message_type == "user":
            current_turn = p.turn_id(session.id, message_id)
        elif current_turn is None:
            fallback_turn += 1
            current_turn = p.turn_id(session.id, f"fallback-{fallback_turn}")
        _add_message_record(builder, session, line, current_turn, message)
    return session


def _subagent_parent(file_path: Path, metadata: dict[str, Any]) -> str | None:
    if metadata.get("kind") != "subagent" or file_path.parent.parent.name != "chats":
        return None
    return file_path.parent.name


def _project_root(file_path: Path) -> str | None:
    for parent in file_path.parents:
        if parent.name != "chats":
            continue
        marker = parent.parent / ".project_root"
        try:
            value = marker.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        return value or None
    return None


def _add_message_record(
    builder: EventBuilder,
    session: Session,
    line: int,
    turn_id: str | None,
    message: dict[str, Any],
) -> None:
    message_type = str(message.get("type") or "")
    message_id = p.as_optional_str(message.get("id"))
    timestamp = p.as_optional_str(message.get("timestamp"))
    session.model = p.as_optional_str(message.get("model")) or session.model

    if message_type == "gemini":
        _add_thoughts(builder, line, turn_id, message_id, timestamp, message.get("thoughts"))
        _add_tool_calls(builder, line, turn_id, message_id, timestamp, message.get("toolCalls"))

    text = p.content_text(message.get("displayContent")) or p.content_text(message.get("content"))
    if text is None:
        return
    role = "assistant" if message_type == "gemini" else message_type
    builder.add_message(
        line=line,
        native_type=f"gemini/{message_type}",
        native_id=message_id,
        part=None,
        timestamp=timestamp,
        turn_id=turn_id,
        parent_native_id=None,
        role=role,
        text=text,
        searchable=message_type in {"user", "gemini"},
    )


def _add_thoughts(
    builder: EventBuilder,
    line: int,
    turn_id: str | None,
    message_id: str | None,
    timestamp: str | None,
    raw_thoughts: Any,
) -> None:
    if not isinstance(raw_thoughts, list):
        return
    for index, thought in enumerate(raw_thoughts):
        if not isinstance(thought, dict):
            continue
        thought_timestamp = p.as_optional_str(thought.get("timestamp")) or timestamp
        content = [thought.get("subject"), thought.get("description"), thought.get("text")]
        builder.add_reasoning(
            line=line,
            native_type="gemini/gemini/thought",
            native_id=f"{message_id}:thought:{index}" if message_id else None,
            part=index,
            timestamp=thought_timestamp,
            turn_id=turn_id,
            parent_native_id=message_id,
            data=p.content_stats(content),
        )


def _add_tool_calls(
    builder: EventBuilder,
    line: int,
    turn_id: str | None,
    message_id: str | None,
    timestamp: str | None,
    raw_calls: Any,
) -> None:
    if not isinstance(raw_calls, list):
        return
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            continue
        call_id = p.as_optional_str(call.get("id"))
        call_timestamp = p.as_optional_str(call.get("timestamp")) or timestamp
        native_id = f"{message_id}:tool:{call_id or index}" if message_id else call_id
        builder.add_tool_call(
            line=line,
            native_type="gemini/gemini/tool_call",
            native_id=native_id,
            timestamp=call_timestamp,
            turn_id=turn_id,
            parent_native_id=message_id,
            tool_call_id=call_id,
            tool_name=p.as_optional_str(call.get("name")),
            data=p.compact_data(call.get("args")),
        )
        if call.get("result") is not None:
            builder.add_tool_result(
                line=line,
                native_type="gemini/gemini/tool_result",
                native_id=f"{native_id}:result" if native_id else None,
                timestamp=call_timestamp,
                turn_id=turn_id,
                parent_native_id=native_id,
                tool_call_id=call_id,
                data=p.content_stats(call.get("result")),
            )


__all__ = ["GeminiSource"]
