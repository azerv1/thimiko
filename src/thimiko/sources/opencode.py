"""Read-only OpenCode SQLite adapter.

OpenCode materializes conversations in ``opencode.db`` as ``session``,
``message``, and ``part`` rows. This adapter reads only those tables plus the
optional ``workspace`` table for branch metadata. One database yields one
canonical :class:`Session` per OpenCode session.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from thimiko.models import Event, Session
from thimiko.utils import iso_from_epoch_ms

from . import _parsing as p
from ._builder import EventBuilder
from .base import ChatSource

OPENCODE_ROOT_ENV = "THIMIKO_OPENCODE_ROOT"

_DATABASE_NAME = "opencode.db"
_TableName = Literal["session", "message", "part", "workspace"]
_REQUIRED_COLUMNS: dict[_TableName, set[str]] = {
    "session": {
        "id",
        "parent_id",
        "directory",
        "title",
        "time_created",
        "time_updated",
        "workspace_id",
        "agent",
        "model",
    },
    "message": {"id", "session_id", "time_created", "data"},
    "part": {"id", "message_id", "session_id", "data"},
}


@dataclass(frozen=True, slots=True)
class _PartRow:
    rowid: int
    id: str
    ordinal: int
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _MessageRow:
    id: str
    time_created: Any
    data: dict[str, Any]
    parts: list[_PartRow]


class OpenCodeSource(ChatSource):
    """OpenCode's materialized SQLite conversation store."""

    name = "opencode"

    def default_roots(self) -> list[Path]:
        override = os.environ.get(OPENCODE_ROOT_ENV)
        if override:
            return [Path(override).expanduser()]
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            return [Path(xdg_data_home).expanduser() / "opencode"]
        user_profile = os.environ.get("USERPROFILE")
        home = Path(user_profile).expanduser() if user_profile else Path.home()
        return [home / ".local" / "share" / "opencode"]

    def discover(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root]
        if not root.is_dir():
            return []
        database = root / _DATABASE_NAME
        return [database] if database.is_file() else []

    def matches(self, path: Path) -> bool:
        try:
            with closing(_connect(path)) as connection:
                connection.execute("BEGIN")
                try:
                    return _has_required_schema(connection)
                finally:
                    connection.rollback()
        except (sqlite3.DatabaseError, OSError):
            return False

    def parse(self, path: Path) -> Session:
        sessions = self.parse_all(path)
        return sessions[0] if sessions else _empty_session(path)

    def parse_all(self, path: Path) -> list[Session]:
        try:
            with closing(_connect(path)) as connection:
                connection.execute("BEGIN")
                try:
                    if not _has_required_schema(connection):
                        return []
                    branches = _workspace_branches(connection)
                    rows = connection.execute(
                        "SELECT id, parent_id, directory, title, time_created, time_updated, "
                        "workspace_id, agent, model FROM session ORDER BY time_created, id"
                    ).fetchall()
                    return [_normalize(path, connection, row, branches) for row in rows]
                finally:
                    connection.rollback()
        except (sqlite3.DatabaseError, OSError):
            return []

    def fingerprint(self, path: Path) -> tuple[float, int]:
        database_stat = path.stat()
        wal_path = Path(f"{path}-wal")
        if not wal_path.is_file():
            return (database_stat.st_mtime, database_stat.st_size)
        wal_stat = wal_path.stat()
        return (
            max(database_stat.st_mtime, wal_stat.st_mtime),
            database_stat.st_size + wal_stat.st_size,
        )


def _connect(path: Path) -> sqlite3.Connection:
    """Open OpenCode's live database without write or migration privileges."""
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _table_columns(connection: sqlite3.Connection, table: _TableName) -> set[str]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row["name"]) for row in rows}


def _has_required_schema(connection: sqlite3.Connection) -> bool:
    return all(
        required.issubset(_table_columns(connection, table))
        for table, required in _REQUIRED_COLUMNS.items()
    )


def _workspace_branches(connection: sqlite3.Connection) -> dict[str, str]:
    if not {"id", "branch"}.issubset(_table_columns(connection, "workspace")):
        return {}
    rows = connection.execute("SELECT id, branch FROM workspace").fetchall()
    return {
        workspace_id: branch
        for row in rows
        if (workspace_id := p.as_optional_str(row["id"]))
        and (branch := p.as_optional_str(row["branch"]))
    }


def _empty_session(path: Path) -> Session:
    return Session(
        id=f"opencode:{path.stem}",
        source="opencode",
        source_session_id=path.stem,
        source_stream_id=path.stem,
        source_path=str(path),
    )


def _normalize(
    path: Path,
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    branches: dict[str, str],
) -> Session:
    native_id = str(row["id"])
    parent_id = p.as_optional_str(row["parent_id"])
    workspace_id = p.as_optional_str(row["workspace_id"])
    session = Session(
        id=f"opencode:{native_id}",
        source="opencode",
        source_session_id=native_id,
        source_stream_id=native_id,
        source_path=str(path),
        parent_session_id=f"opencode:{parent_id}" if parent_id else None,
        agent_id=p.as_optional_str(row["agent"]),
        started_at=iso_from_epoch_ms(row["time_created"]),
        ended_at=iso_from_epoch_ms(row["time_updated"]),
        title=p.as_optional_str(row["title"]),
        cwd=p.as_optional_str(row["directory"]),
        git_branch=branches.get(workspace_id or ""),
        model=_session_model(row["model"]),
    )
    messages = _messages(connection, native_id)
    _add_messages(session, messages)
    return session


def _messages(connection: sqlite3.Connection, session_id: str) -> list[_MessageRow]:
    message_rows = connection.execute(
        "SELECT id, time_created, data FROM message WHERE session_id = ? "
        "ORDER BY time_created, id",
        (session_id,),
    ).fetchall()
    decoded_messages: list[tuple[str, Any, dict[str, Any]]] = []
    for row in message_rows:
        message_id = p.as_optional_str(row["id"])
        data = _json_dict(row["data"])
        if message_id and data.get("role") in {"user", "assistant"}:
            decoded_messages.append((message_id, row["time_created"], data))

    part_rows = connection.execute(
        "SELECT rowid, id, message_id, data FROM part WHERE session_id = ? "
        "ORDER BY message_id, id",
        (session_id,),
    ).fetchall()
    raw_parts: dict[str, list[sqlite3.Row]] = {}
    for row in part_rows:
        message_id = p.as_optional_str(row["message_id"])
        if message_id:
            raw_parts.setdefault(message_id, []).append(row)

    messages: list[_MessageRow] = []
    for message_id, time_created, data in decoded_messages:
        parts: list[_PartRow] = []
        for ordinal, row in enumerate(raw_parts.get(message_id, [])):
            part_id = p.as_optional_str(row["id"])
            part_data = _json_dict(row["data"])
            if part_id and part_data:
                parts.append(
                    _PartRow(
                        rowid=int(row["rowid"]),
                        id=part_id,
                        ordinal=ordinal,
                        data=part_data,
                    )
                )
        messages.append(
            _MessageRow(id=message_id, time_created=time_created, data=data, parts=parts)
        )
    return messages


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        return p.as_dict(json.loads(value))
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return {}


def _add_messages(session: Session, messages: list[_MessageRow]) -> None:
    builder = EventBuilder(session)
    turns = _message_turns(session.id, messages)
    events_by_message: dict[str, list[Event]] = {}

    for message in messages:
        role = str(message.data["role"])
        timestamp = iso_from_epoch_ms(message.time_created) or iso_from_epoch_ms(
            p.as_dict(message.data.get("time")).get("created")
        )
        session.model = session.model or _message_model(message.data)
        session.agent_id = session.agent_id or p.as_optional_str(message.data.get("agent"))
        if session.cwd is None:
            session.cwd = p.as_optional_str(p.as_dict(message.data.get("path")).get("cwd"))
        message_events: list[Event] = []
        for part in message.parts:
            message_events.extend(
                _add_part(
                    builder,
                    part,
                    role=role,
                    timestamp=timestamp,
                    turn_id=turns[message.id],
                )
            )
        events_by_message[message.id] = message_events

    for message in messages:
        parent_native_id = p.as_optional_str(message.data.get("parentID"))
        parent_events = events_by_message.get(parent_native_id or "", [])
        parent_event_id = parent_events[0].id if parent_events else None
        if parent_event_id:
            for event in events_by_message[message.id]:
                event.parent_id = event.parent_id or parent_event_id


def _message_turns(session_id: str, messages: list[_MessageRow]) -> dict[str, str]:
    by_id = {message.id: message for message in messages}
    resolved: dict[str, str] = {}

    def resolve(message_id: str, visiting: set[str]) -> str:
        existing = resolved.get(message_id)
        if existing:
            return existing
        message = by_id[message_id]
        if message.data.get("role") == "user" or message_id in visiting:
            root_id = message_id
        else:
            parent_id = p.as_optional_str(message.data.get("parentID"))
            if parent_id in by_id:
                turn_id = resolve(parent_id, visiting | {message_id})
                resolved[message_id] = turn_id
                return turn_id
            root_id = parent_id or message_id
        turn_id = p.turn_id(session_id, root_id) or f"{session_id}:turn:{message_id}"
        resolved[message_id] = turn_id
        return turn_id

    for message in messages:
        resolve(message.id, set())
    return resolved


def _add_part(
    builder: EventBuilder,
    part: _PartRow,
    *,
    role: str,
    timestamp: str | None,
    turn_id: str,
) -> list[Event]:
    data = part.data
    part_type = p.as_optional_str(data.get("type")) or "unknown"
    native_type = f"opencode/part/{part_type}"

    if part_type == "text":
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return []
        searchable = role in {"user", "assistant"} and not data.get("ignored") and not data.get(
            "synthetic"
        )
        message_event = builder.add_message(
            line=part.rowid,
            native_type=native_type,
            native_id=part.id,
            part=part.ordinal,
            timestamp=timestamp,
            turn_id=turn_id,
            parent_native_id=None,
            role=role,
            text=text,
            searchable=bool(searchable),
            data=_text_metadata(data),
        )
        return [message_event]

    if part_type == "reasoning":
        text = data.get("text")
        metadata = _metadata_without(data, {"text"}) or {}
        if isinstance(text, str):
            metadata["characters"] = len(text)
        reasoning_event = builder.add_reasoning(
            line=part.rowid,
            native_type=native_type,
            native_id=part.id,
            part=part.ordinal,
            timestamp=timestamp,
            turn_id=turn_id,
            parent_native_id=None,
            data=metadata or None,
        )
        return [reasoning_event]

    if part_type == "tool":
        return _add_tool_part(builder, part, timestamp=timestamp, turn_id=turn_id)

    attachment_event = builder.add_attachment(
        line=part.rowid,
        native_type=native_type,
        native_id=part.id,
        part=part.ordinal,
        timestamp=timestamp,
        turn_id=turn_id,
        parent_native_id=None,
        role=role,
        data=_metadata_without(data, set()),
    )
    return [attachment_event]


def _add_tool_part(
    builder: EventBuilder,
    part: _PartRow,
    *,
    timestamp: str | None,
    turn_id: str,
) -> list[Event]:
    data = part.data
    state = p.as_dict(data.get("state"))
    status = p.as_optional_str(state.get("status"))
    call_id = p.as_optional_str(data.get("callID"))
    tool_name = p.as_optional_str(data.get("tool"))
    call = builder.add_tool_call(
        line=part.rowid,
        native_type="opencode/part/tool",
        native_id=part.id,
        part=part.ordinal,
        timestamp=timestamp,
        turn_id=turn_id,
        parent_native_id=None,
        tool_call_id=call_id,
        tool_name=tool_name,
        data=p.compact_data(
            {
                "status": status,
                "input": state.get("input"),
                "title": state.get("title"),
                "metadata": data.get("metadata"),
            }
        ),
    )
    events: list[Event] = [call]
    if status not in {"completed", "error"}:
        return events
    result_data: dict[str, Any] = {"status": status, "metadata": state.get("metadata")}
    if status == "completed":
        result_data["output"] = state.get("output")
        result_data["attachments"] = state.get("attachments")
    else:
        result_data["error"] = state.get("error")
    result = builder.add_tool_result(
        line=part.rowid,
        native_type="opencode/part/tool/result",
        native_id=part.id,
        part=part.ordinal,
        timestamp=timestamp,
        turn_id=turn_id,
        parent_native_id=None,
        tool_call_id=call_id,
        data=p.compact_data(result_data),
    )
    events.append(result)
    return events


def _text_metadata(data: dict[str, Any]) -> dict[str, Any] | None:
    return _metadata_without(data, {"text", "type"})


def _metadata_without(data: dict[str, Any], excluded: set[str]) -> dict[str, Any] | None:
    compact = p.compact_data({key: value for key, value in data.items() if key not in excluded})
    return compact if isinstance(compact, dict) and compact else None


def _session_model(value: Any) -> str | None:
    if isinstance(value, str):
        decoded = _json_dict(value)
        if decoded:
            return _qualified_model(decoded.get("providerID"), decoded.get("id"))
        return p.as_optional_str(value)
    if isinstance(value, dict):
        return _qualified_model(value.get("providerID"), value.get("id"))
    return None


def _message_model(data: dict[str, Any]) -> str | None:
    model = p.as_dict(data.get("model"))
    if model:
        return _qualified_model(model.get("providerID"), model.get("modelID"))
    return _qualified_model(data.get("providerID"), data.get("modelID"))


def _qualified_model(provider: Any, model: Any) -> str | None:
    provider_id = p.as_optional_str(provider)
    model_id = p.as_optional_str(model)
    if provider_id and model_id:
        return f"{provider_id}/{model_id}"
    return model_id or provider_id


__all__ = ["OPENCODE_ROOT_ENV", "OpenCodeSource"]
