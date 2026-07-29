"""Pointer back to the exact raw source line an event was parsed from."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where one event came from in the original provider JSONL file."""

    path: str
    line: int
    native_type: str
    native_id: str | None = None
    part: int | None = None
