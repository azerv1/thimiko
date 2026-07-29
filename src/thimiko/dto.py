"""Storage-agnostic transfer types shared across the indexing/search layers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

_ROLE_LABEL = re.compile(r"\[(?:USER|ASSISTANT|TOOL|UNKNOWN)\]")
_WHITESPACE = re.compile(r"\s+")


@dataclass(kw_only=True)
class SearchDocument:
    """One turn-level chunk ready for indexing."""

    id: str
    session_id: str
    turn_id: str
    chunk_index: int
    source: str
    title: str | None
    cwd: str | None
    git_branch: str | None
    started_at: str | None
    ended_at: str | None
    text: str
    provenance: list[dict[str, Any]]


@dataclass(kw_only=True)
class SearchResult:
    """One ranked search hit."""

    id: str
    session_id: str
    turn_id: str
    source: str
    title: str | None
    cwd: str | None
    git_branch: str | None
    started_at: str | None
    ended_at: str | None
    snippet: str
    score: float
    model: str | None
    provenance: list[dict[str, Any]]


def iso_days_ago(days: int) -> str:
    """UTC cutoff `days` before now as an ISO-8601 'Z' string.

    Formatted so it compares lexicographically against the stored `…Z`
    timestamps as an inclusive lower bound.
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _ago(count: int, unit: str) -> str:
    return f"{count} {unit}{'s' if count != 1 else ''} ago"


def relative_time(iso: str | None) -> str:
    """Human 'N units ago' rendering of an ISO-8601 timestamp."""
    if not iso:
        return "no date"
    try:
        moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    seconds = int((datetime.now(UTC) - moment).total_seconds())
    minute, hour, day, week, month, year = 60, 3600, 86400, 604800, 2592000, 31536000
    if seconds < minute:
        return "just now"
    if seconds < hour:
        return _ago(seconds // minute, "minute")
    if seconds < day:
        return _ago(seconds // hour, "hour")
    if seconds < week:
        return _ago(seconds // day, "day")
    if seconds < month:
        return _ago(seconds // week, "week")
    if seconds < year:
        return _ago(seconds // month, "month")
    return _ago(seconds // year, "year")


def clean_snippet(snippet: str) -> str:
    """Drop role labels and collapse whitespace; keep `[term]` match highlights."""
    return _WHITESPACE.sub(" ", _ROLE_LABEL.sub(" ", snippet)).strip()


def answer_dict(result: SearchResult) -> dict[str, Any]:
    """Reader-friendly, model-consumable shape for one hit."""
    first = result.provenance[0] if result.provenance else {}
    return {
        "title": result.title or result.session_id,
        "source": result.source,
        "model": result.model,
        "when": relative_time(result.started_at),
        "timestamp": result.started_at,
        "cwd": result.cwd,
        "path": first.get("path"),
        "line": first.get("line"),
        "snippet": clean_snippet(result.snippet),
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "score": result.score,
    }
