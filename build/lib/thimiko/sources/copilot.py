"""GitHub Copilot / VSCode chat adapter.

VSCode stores Copilot chat under
``workspaceStorage/<hash>/chatSessions/<id>.{json,jsonl}`` (and
``globalStorage/emptyWindowChatSessions``). Two on-disk shapes:

- ``.json`` — one full snapshot object
  (``{sessionId, creationDate, requests:[...], ...}``).
- ``.jsonl`` — an event log: line 1 is ``{"kind":0,"v":<snapshot>}``; later
  lines are ``{"kind":1,"k":<json-path>,"v":<value>}`` patches (``kind:2`` ends
  the stream). The session's final state is the base snapshot with every patch
  applied.

Each ``requests[]`` entry is one turn: ``message.text`` is the user prompt and
``response`` is a list of parts (bare ``{"value": ...}`` / ``markdownContent``
carry the assistant text; ``thinking`` is reasoning; tool invocations stay out
of the search corpus).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from thimiko.models import Session
from thimiko.utils import iso_from_epoch_ms

from . import _parsing as p
from ._builder import EventBuilder
from .base import ChatSource

COPILOT_ROOT_ENV = "THIMIKO_COPILOT_ROOT"

_SESSION_SUFFIXES = {".json", ".jsonl"}


class CopilotSource(ChatSource):
    name = "copilot"

    def default_roots(self) -> list[Path]:
        override = os.environ.get(COPILOT_ROOT_ENV)
        if override:
            return [Path(override).expanduser()]
        appdata = os.environ.get("APPDATA")
        return [Path(appdata) / "Code" / "User"] if appdata else []

    def discover(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix in _SESSION_SUFFIXES else []
        if not root.is_dir():
            return []
        patterns = [
            "workspaceStorage/*/chatSessions/*",
            "globalStorage/emptyWindowChatSessions/*",
            "*/chatSessions/*",
        ]
        if root.name == "chatSessions":
            patterns.append("*")
        found = {
            match
            for pattern in patterns
            for match in root.glob(pattern)
            if match.is_file() and match.suffix in _SESSION_SUFFIXES
        }
        return sorted(found)

    def matches(self, path: Path) -> bool:
        return _is_copilot(_snapshot_of(_read_head(path)))

    def parse(self, path: Path) -> Session:
        return _normalize(path, _load_session(path))


def _read_head(path: Path) -> dict[str, Any]:
    """First JSON object of a session file (line 1 for ``.jsonl``)."""
    try:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    stripped = raw.strip()
                    if stripped:
                        return p.as_dict(json.loads(stripped))
            return {}
        return p.as_dict(json.loads(path.read_text(encoding="utf-8", errors="replace")))
    except (OSError, json.JSONDecodeError):
        return {}


def _snapshot_of(record: dict[str, Any]) -> dict[str, Any]:
    """The full snapshot: a ``kind:0`` wrapper's ``v``, else the record itself."""
    if record.get("kind") == 0:
        return p.as_dict(record.get("v"))
    return record


def _is_copilot(snapshot: dict[str, Any]) -> bool:
    if "requests" in snapshot and "sessionId" in snapshot:
        return True
    return snapshot.get("responderUsername") == "GitHub Copilot"


def _load_session(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return _reconstruct(path)
    try:
        return p.as_dict(json.loads(path.read_text(encoding="utf-8", errors="replace")))
    except (OSError, json.JSONDecodeError):
        return {}


def _reconstruct(path: Path) -> dict[str, Any]:
    """Apply the ``kind:1`` patch log onto the ``kind:0`` base snapshot."""
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    records.append(p.as_dict(json.loads(stripped)))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return {}

    snapshot: dict[str, Any] = {}
    for record in records:
        if record.get("kind") == 0 and isinstance(record.get("v"), dict):
            snapshot = record["v"]
            break
    for record in records:
        if record.get("kind") == 1 and isinstance(record.get("k"), list):
            _apply_patch(snapshot, record["k"], record.get("v"))
    return snapshot


def _apply_patch(snapshot: dict[str, Any], keys: list[Any], value: Any) -> None:
    """Set ``value`` at the ``keys`` json-path (appending to a list at its end)."""
    target: Any = snapshot
    try:
        for key in keys[:-1]:
            target = target[key]
        last = keys[-1]
        if isinstance(target, list) and isinstance(last, int) and last == len(target):
            target.append(value)
        else:
            target[last] = value
    except (KeyError, IndexError, TypeError):
        pass


def _normalize(file_path: Path, snapshot: dict[str, Any]) -> Session:
    source_session_id = p.as_optional_str(snapshot.get("sessionId")) or file_path.stem
    session = Session(
        id=f"copilot:{source_session_id}",
        source="copilot",
        source_session_id=source_session_id,
        source_stream_id=source_session_id,
        source_path=str(file_path),
        started_at=iso_from_epoch_ms(snapshot.get("creationDate")),
        ended_at=iso_from_epoch_ms(snapshot.get("lastMessageDate")),
        title=p.as_optional_str(snapshot.get("customTitle")),
        cwd=_workspace_path(snapshot),
    )
    builder = EventBuilder(session)
    requests = snapshot.get("requests")
    for index, request in enumerate(requests if isinstance(requests, list) else []):
        if isinstance(request, dict):
            _add_request(builder, session, index, request)
    return session


def _workspace_path(snapshot: dict[str, Any]) -> str | None:
    location = p.as_optional_str(snapshot.get("initialLocation"))
    if location and ("/" in location or "\\" in location):
        return location
    return None


def _add_request(
    builder: EventBuilder, session: Session, index: int, request: dict[str, Any]
) -> None:
    request_id = p.as_optional_str(request.get("requestId"))
    turn = p.turn_id(session.id, request_id or f"request-{index}")
    timestamp = iso_from_epoch_ms(request.get("timestamp"))
    session.model = p.as_optional_str(request.get("modelId")) or session.model

    user_text = _message_text(request.get("message"))
    if user_text is not None:
        builder.add_message(
            line=1,
            native_type="copilot/request/message",
            native_id=f"{request_id}:user" if request_id else None,
            part=index,
            timestamp=timestamp,
            turn_id=turn,
            parent_native_id=None,
            role="user",
            text=user_text,
            searchable=True,
        )

    assistant_parts: list[str] = []
    response = request.get("response")
    for part_index, item in enumerate(response if isinstance(response, list) else []):
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind == "thinking":
            builder.add_reasoning(
                line=1,
                native_type="copilot/response/thinking",
                native_id=f"{request_id}:think:{part_index}" if request_id else None,
                part=part_index,
                timestamp=timestamp,
                turn_id=turn,
                parent_native_id=None,
                data=p.content_stats(item.get("value")),
            )
        elif kind == "toolInvocationSerialized":
            builder.add_tool_call(
                line=1,
                native_type="copilot/response/toolInvocationSerialized",
                native_id=f"{request_id}:tool:{part_index}" if request_id else None,
                timestamp=timestamp,
                turn_id=turn,
                parent_native_id=None,
                tool_call_id=None,
                tool_name=p.as_optional_str(item.get("toolName")),
                data=p.compact_data(item.get("invocationMessage") or item.get("pastTenseMessage")),
            )
        else:
            text = _part_text(item)
            if text:
                assistant_parts.append(text)

    assistant_text = "".join(assistant_parts) or None
    if assistant_text is not None:
        builder.add_message(
            line=1,
            native_type="copilot/response/markdown",
            native_id=f"{request_id}:assistant" if request_id else None,
            part=index,
            timestamp=timestamp,
            turn_id=turn,
            parent_native_id=None,
            role="assistant",
            text=assistant_text,
            searchable=True,
        )


def _message_text(message: Any) -> str | None:
    if isinstance(message, dict):
        return p.as_optional_str(message.get("text"))
    if isinstance(message, str):
        return message or None
    return None


def _part_text(item: dict[str, Any]) -> str | None:
    """Assistant text from a response part: bare ``{"value"}`` or ``markdownContent``."""
    kind = item.get("kind")
    if kind == "markdownContent":
        content = item.get("content")
        if isinstance(content, dict):
            return p.as_optional_str(content.get("value"))
        return p.as_optional_str(content)
    if kind is None:
        return p.as_optional_str(item.get("value"))
    return None
