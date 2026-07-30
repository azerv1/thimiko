"""Dialect sniffing, redaction, and small formatting helpers shared project-wide.

`detect_source` / `sniff_file_source` are the sniffing primitives `ChatSource`
adapters use for `matches()`. `classify` / `redact` / `markdown_table` exist for
the reporting scripts in `scripts/` (schema summaries, regex search reports).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Field-name substrings whose values are redacted / never echoed verbatim.
SENSITIVE_NAME_PARTS = (
    "secret",
    "token",
    "password",
    "key",
    "auth",
    "encrypted",
    "message",
    "content",
    "stdout",
    "stderr",
    "text",
    "base_instructions",
    "replacement_history",
    # Claude-specific
    "thinking",
    "snapshot",
    "input",
    "arguments",
    "signature",
)

# Claude envelope keys that are structural noise rather than schema signal.
CLAUDE_ENVELOPE_KEYS = frozenset(
    {
        "cwd",
        "entrypoint",
        "gitBranch",
        "isSidechain",
        "parentUuid",
        "sessionId",
        "timestamp",
        "type",
        "userType",
        "uuid",
        "version",
    }
)


@dataclass
class Classification:
    """Dialect-agnostic view of one JSONL record, for schema reporting."""

    source: str
    top_type: str
    sub_type: str
    sub_keys: tuple[str, ...]
    content_block_types: tuple[str, ...] = ()


def discover_jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.jsonl") if p.is_file())
    raise FileNotFoundError(f"Path does not exist: {path}")


def iso_from_epoch_ms(value: Any) -> str | None:
    """Render an epoch-milliseconds timestamp as a UTC `…Z` string.

    Millisecond precision matches the stored ISO timestamps from the JSONL
    providers, so values compare lexicographically as a time order.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    moment = datetime.fromtimestamp(value / 1000, tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return f"<{type(value).__name__}>"


def sorted_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, dict):
        return ()
    return tuple(sorted(str(k) for k in value.keys()))


def is_sensitive_field(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in SENSITIVE_NAME_PARTS)


def preview_string(value: str, limit: int = 80) -> str:
    clean = value.replace("\r", "\\r").replace("\n", "\\n")
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def redact(value: Any, *, field_name: str = "", depth: int = 0, max_depth: int = 2) -> Any:
    if is_sensitive_field(field_name):
        if isinstance(value, str):
            return f"<redacted str len={len(value)} preview={preview_string(value, 40)!r}>"
        if isinstance(value, list):
            return f"<redacted array len={len(value)}>"
        if isinstance(value, dict):
            return f"<redacted object keys={len(value)}>"
        return f"<redacted {type(value).__name__}>"

    if isinstance(value, str):
        return f"<str len={len(value)} preview={preview_string(value)!r}>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return f"<array len={len(value)}>"
    if isinstance(value, dict):
        if depth >= max_depth:
            return f"<object keys={len(value)}>"
        return {
            str(k): redact(v, field_name=str(k), depth=depth + 1, max_depth=max_depth)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return f"<{type(value).__name__}>"


def escape_md(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(str(cell)) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def format_keys(keys: tuple[str, ...]) -> str:
    if not keys:
        return "<none>"
    return ",".join(keys)


def detect_source(record: Any) -> str:
    """Sniff the dialect of a single record.

    Codex records carry a ``payload`` object; Claude records carry ``message``
    or a ``uuid``. Anything else is ``unknown``.
    """
    if not isinstance(record, dict):
        return "unknown"
    if isinstance(record.get("payload"), dict):
        return "codex"
    if "message" in record or "uuid" in record:
        return "claude"
    return "unknown"


def sniff_file_source(file_path: Path, max_lines: int = 200) -> str:
    """Determine a file's dialect from its first classifiable records.

    A session file is written entirely by one CLI, but a few record types
    (Claude's ``file-history-snapshot``, ``ai-title``, ``mode`` ...) lack the
    per-record sniff markers. Reading a handful of lines yields a stable source
    for the whole file, so those auxiliary records are attributed correctly.
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if line_number > max_lines:
                    break
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                source = detect_source(record)
                if source != "unknown":
                    return source
    except OSError:
        return "unknown"
    return "unknown"


def _claude_content_block_types(message: dict[str, Any]) -> tuple[str, ...]:
    content = message.get("content")
    if not isinstance(content, list):
        return ()
    seen: set[str] = set()
    for item in content:
        if isinstance(item, dict):
            seen.add(as_string(item.get("type")) or "<untyped>")
        elif isinstance(item, str):
            seen.add("<str>")
    return tuple(sorted(seen))


def classify(record: Any, source: str | None = None) -> Classification:
    """Flatten a record into a dialect-agnostic :class:`Classification`.

    ``source`` forces a dialect (skipping :func:`detect_source`); when ``None``
    the dialect is sniffed per record.
    """
    resolved = source or detect_source(record)

    if not isinstance(record, dict):
        return Classification(
            source=resolved,
            top_type=f"<{type(record).__name__}>",
            sub_type="",
            sub_keys=(f"<{type(record).__name__}>",),
        )

    if resolved == "codex":
        payload = record.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        return Classification(
            source="codex",
            top_type=as_string(record.get("type")),
            sub_type=as_string(payload.get("type")),
            sub_keys=sorted_keys(payload),
        )

    if resolved == "claude":
        top_type = as_string(record.get("type"))
        message = record.get("message")
        if isinstance(message, dict):
            return Classification(
                source="claude",
                top_type=top_type,
                sub_type=as_string(message.get("role")),
                sub_keys=sorted_keys(message),
                content_block_types=_claude_content_block_types(message),
            )
        # Non-message record types (attachment, file-history-snapshot, ...).
        own_keys = tuple(sorted(str(k) for k in record.keys() if k not in CLAUDE_ENVELOPE_KEYS))
        return Classification(
            source="claude",
            top_type=top_type,
            sub_type="",
            sub_keys=own_keys,
        )

    # Unknown dialect: best-effort top-level view.
    return Classification(
        source="unknown",
        top_type=as_string(record.get("type")),
        sub_type="",
        sub_keys=sorted_keys(record),
    )
