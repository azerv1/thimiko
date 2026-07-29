"""JSONL/content-shape helpers shared by the Codex and Claude source adapters."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def as_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def content_text(value: Any) -> str | None:
    """Extract human-readable text without stringifying binary/image payloads."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        parts = [content_text(item) for item in value]
        joined = "\n".join(part for part in parts if part)
        return joined or None
    if isinstance(value, dict):
        value_type = value.get("type")
        if value_type in {"image", "image_url"}:
            return None
        for key in ("text", "content", "output"):
            text = content_text(value.get(key))
            if text:
                return text
    return None


def content_stats(value: Any) -> dict[str, Any] | None:
    text = content_text(value)
    if text is None:
        return None
    return {"characters": len(text)}


def native_type(record: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    top = str(record.get("type") or "unknown")
    sub = str((payload or {}).get("type") or "")
    return f"{top}/{sub}" if sub else top


def scalar_metadata(record: dict[str, Any], excluded: set[str]) -> dict[str, Any] | None:
    result = {
        str(key): value
        for key, value in record.items()
        if key not in excluded and (value is None or isinstance(value, (str, int, float, bool)))
    }
    return result or None


def compact_data(value: Any, *, depth: int = 0) -> Any:
    """Keep useful tool metadata without copying arbitrarily large raw payloads."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if len(value) <= 2000:
            return value
        return {"preview": value[:2000], "characters": len(value), "truncated": True}
    if depth >= 3:
        return {"value_type": type(value).__name__, "truncated": True}
    if isinstance(value, list):
        result: list[Any] = [compact_data(item, depth=depth + 1) for item in value[:20]]
        if len(value) > 20:
            result.append({"omitted_items": len(value) - 20})
        return result
    if isinstance(value, dict):
        result_dict: dict[str, Any] = {
            str(key): compact_data(item, depth=depth + 1) for key, item in list(value.items())[:30]
        }
        if len(value) > 30:
            result_dict["_omitted_keys"] = len(value) - 30
        return result_dict
    return str(value)


def read_records(file_path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line, raw_line in enumerate(handle, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append((line, record))
    return records


def timestamp_bounds(
    records: Iterable[tuple[int, dict[str, Any]]],
) -> tuple[str | None, str | None]:
    timestamps = [
        timestamp
        for _, record in records
        if (timestamp := as_optional_str(record.get("timestamp"))) is not None
    ]
    return (min(timestamps), max(timestamps)) if timestamps else (None, None)


def fallback_session_id(file_path: Path) -> str:
    stem = file_path.stem
    if stem.startswith("rollout-") and len(stem) >= 36:
        return stem[-36:]
    return stem


def turn_id(canonical_session_id: str, native_turn_id: str | None) -> str | None:
    if not native_turn_id:
        return None
    return f"{canonical_session_id}:turn:{native_turn_id}"
