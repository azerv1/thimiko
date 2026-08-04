"""Cursor chat adapter.

Cursor is the one provider that does not write per-session files: every chat
lives as rows in a SQLite database at
``<root>/globalStorage/state.vscdb``, table ``cursorDiskKV(key, value)``:

- ``composerData:<composerId>`` — one chat's metadata: ``name``, ``createdAt``,
  ``lastUpdatedAt``, and ``fullConversationHeadersOnly: [{bubbleId, type}, ...]``,
  which is the authoritative message order.
- ``bubbleId:<composerId>:<bubbleId>`` — one message: ``type`` (``1`` = user,
  ``2`` = assistant), ``text``, ``thinking``, ``toolFormerData``.

So one file yields *many* sessions — hence :meth:`CursorSource.parse_all`.
``Provenance.line`` is the ``cursorDiskKV`` rowid and ``native_id`` the full key,
so every event still points back at its exact raw record.

Per-workspace databases (``workspaceStorage/<hash>/state.vscdb``) hold no chat
bodies; they are read only for their ``composer.composerData`` composer lists,
which map a chat back to that workspace's folder (a session's ``cwd``).

Bubbles carry no reliable timestamp of their own, so events fall back to the
composer's ``createdAt`` — otherwise `--days` and time ordering would ignore
every Cursor result.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from thimiko.models import Session
from thimiko.utils import iso_from_epoch_ms

from . import _parsing as p
from ._builder import EventBuilder
from .base import ChatSource

CURSOR_ROOT_ENV = "THIMIKO_CURSOR_ROOT"

_COMPOSER_PREFIX = "composerData:"
_GLOBAL_DB = "globalStorage/state.vscdb"


class CursorSource(ChatSource):
    name = "cursor"

    def default_roots(self) -> list[Path]:
        override = os.environ.get(CURSOR_ROOT_ENV)
        if override:
            return [Path(override).expanduser()]
        appdata = os.environ.get("APPDATA")
        return [Path(appdata) / "Cursor" / "User"] if appdata else []

    def discover(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix == ".vscdb" else []
        if not root.is_dir():
            return []
        global_db = root / _GLOBAL_DB
        return [global_db] if global_db.is_file() else []

    def matches(self, path: Path) -> bool:
        if path.suffix != ".vscdb":
            return False
        try:
            with closing(_connect(path)) as connection:
                row = connection.execute(
                    f"SELECT 1 FROM cursorDiskKV WHERE key LIKE '{_COMPOSER_PREFIX}%' LIMIT 1"
                ).fetchone()
        except (sqlite3.DatabaseError, OSError):
            return False
        return row is not None

    def parse(self, path: Path) -> Session:
        sessions = self.parse_all(path)
        return sessions[0] if sessions else _empty_session(path)

    def parse_all(self, path: Path) -> list[Session]:
        cwds = _workspace_cwds(path)
        try:
            with closing(_connect(path)) as connection:
                rows = connection.execute(
                    "SELECT rowid, key, value FROM cursorDiskKV "
                    f"WHERE key LIKE '{_COMPOSER_PREFIX}%' ORDER BY rowid"
                ).fetchall()
                return [
                    _normalize(path, connection, composer_id, composer, cwds)
                    for composer_id, composer in _decoded(rows)
                ]
        except (sqlite3.DatabaseError, OSError):
            return []


def _connect(path: Path) -> sqlite3.Connection:
    """Read-only connection: Cursor keeps the database open while it runs."""
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _decoded(rows: list[sqlite3.Row]) -> list[tuple[str, dict[str, Any]]]:
    """(composerId, composer) for every parsable `composerData:` row."""
    decoded: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        composer_id = str(row["key"]).removeprefix(_COMPOSER_PREFIX)
        try:
            composer = p.as_dict(json.loads(row["value"]))
        except (json.JSONDecodeError, TypeError):
            continue
        if composer_id:
            decoded.append((composer_id, composer))
    return decoded


def _empty_session(path: Path) -> Session:
    return Session(
        id=f"cursor:{path.stem}",
        source="cursor",
        source_session_id=path.stem,
        source_stream_id=path.stem,
        source_path=str(path),
    )


def _bubbles(
    connection: sqlite3.Connection, composer_id: str
) -> dict[str, tuple[int, dict[str, Any]]]:
    """Every stored bubble for one composer, keyed by bubbleId."""
    rows = connection.execute(
        "SELECT rowid, key, value FROM cursorDiskKV WHERE key LIKE ?",
        (f"bubbleId:{composer_id}:%",),
    ).fetchall()
    bubbles: dict[str, tuple[int, dict[str, Any]]] = {}
    for row in rows:
        bubble_id = str(row["key"]).rsplit(":", 1)[-1]
        try:
            bubbles[bubble_id] = (int(row["rowid"]), p.as_dict(json.loads(row["value"])))
        except (json.JSONDecodeError, TypeError):
            continue
    return bubbles


def _normalize(
    path: Path,
    connection: sqlite3.Connection,
    composer_id: str,
    composer: dict[str, Any],
    cwds: dict[str, str],
) -> Session:
    started_at = iso_from_epoch_ms(composer.get("createdAt"))
    session = Session(
        id=f"cursor:{composer_id}",
        source="cursor",
        source_session_id=composer_id,
        source_stream_id=composer_id,
        source_path=str(path),
        started_at=started_at,
        ended_at=iso_from_epoch_ms(composer.get("lastUpdatedAt")),
        title=p.as_optional_str(composer.get("name")),
        cwd=cwds.get(composer_id),
        model=_model(composer),
    )
    builder = EventBuilder(session)
    bubbles = _bubbles(connection, composer_id)
    headers = composer.get("fullConversationHeadersOnly")
    turn: str | None = None

    for index, header in enumerate(headers if isinstance(headers, list) else []):
        bubble_id = p.as_optional_str(p.as_dict(header).get("bubbleId"))
        stored = bubbles.get(bubble_id or "")
        if bubble_id is None or stored is None:
            # Cursor prunes old bubble bodies but keeps their headers.
            continue
        line, bubble = stored
        if bubble.get("type") == 1:
            turn = p.turn_id(session.id, bubble_id)
        _add_bubble(
            builder,
            session,
            line=line,
            key=f"bubbleId:{composer_id}:{bubble_id}",
            part=index,
            turn=turn or p.turn_id(session.id, bubble_id),
            bubble=bubble,
        )

    return session


def _add_bubble(
    builder: EventBuilder,
    session: Session,
    *,
    line: int,
    key: str,
    part: int,
    turn: str | None,
    bubble: dict[str, Any],
) -> None:
    is_user = bubble.get("type") == 1
    timestamp = iso_from_epoch_ms(bubble.get("createdAt")) or session.started_at
    session.model = session.model or p.as_optional_str(bubble.get("modelType"))

    text = p.content_text(bubble.get("text"))
    if text is not None:
        builder.add_message(
            line=line,
            native_type=f"cursor/bubble/{'user' if is_user else 'assistant'}",
            native_id=key,
            part=part,
            timestamp=timestamp,
            turn_id=turn,
            parent_native_id=None,
            role="user" if is_user else "assistant",
            text=text,
            searchable=True,
        )

    thinking = p.content_text(bubble.get("thinking"))
    if thinking:
        builder.add_reasoning(
            line=line,
            native_type="cursor/bubble/thinking",
            native_id=f"{key}:thinking",
            part=part,
            timestamp=timestamp,
            turn_id=turn,
            parent_native_id=None,
            data=p.content_stats(thinking),
        )

    tool = p.as_dict(bubble.get("toolFormerData"))
    if tool:
        builder.add_tool_call(
            line=line,
            native_type="cursor/bubble/tool",
            native_id=f"{key}:tool",
            timestamp=timestamp,
            turn_id=turn,
            parent_native_id=None,
            tool_call_id=p.as_optional_str(tool.get("toolCallId")),
            tool_name=p.as_optional_str(tool.get("name")) or p.as_optional_str(tool.get("tool")),
            data=p.compact_data(tool.get("params") or tool.get("rawArgs")),
        )


def _model(composer: dict[str, Any]) -> str | None:
    model = composer.get("model")
    if isinstance(model, dict):
        return p.as_optional_str(model.get("modelName"))
    return p.as_optional_str(model)


def _workspace_cwds(global_db: Path) -> dict[str, str]:
    """Map composerId -> workspace folder, from each workspace's own database."""
    workspace_root = global_db.parent.parent / "workspaceStorage"
    if not workspace_root.is_dir():
        return {}
    cwds: dict[str, str] = {}
    for workspace_db in sorted(workspace_root.glob("*/state.vscdb")):
        folder = _workspace_folder(workspace_db.parent / "workspace.json")
        if folder is None:
            continue
        for composer_id in _workspace_composer_ids(workspace_db):
            cwds.setdefault(composer_id, folder)
    return cwds


def _workspace_folder(workspace_json: Path) -> str | None:
    try:
        data = p.as_dict(json.loads(workspace_json.read_text(encoding="utf-8", errors="replace")))
    except (OSError, json.JSONDecodeError):
        return None
    uri = p.as_optional_str(data.get("folder"))
    if uri is None:
        return None
    path = unquote(urlparse(uri).path)
    # file:///c%3A/repo -> /c:/repo; strip the leading slash off a drive letter.
    return path.lstrip("/") if path[2:3] == ":" else path


def _workspace_composer_ids(workspace_db: Path) -> list[str]:
    try:
        with closing(_connect(workspace_db)) as connection:
            row = connection.execute(
                "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
            ).fetchone()
        if row is None:
            return []
        data = p.as_dict(json.loads(row["value"]))
    except (sqlite3.DatabaseError, OSError, json.JSONDecodeError, TypeError):
        return []
    composers = data.get("allComposers")
    return [
        composer_id
        for entry in (composers if isinstance(composers, list) else [])
        if (composer_id := p.as_optional_str(p.as_dict(entry).get("composerId")))
    ]
