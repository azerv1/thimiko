#!/usr/bin/env python3
"""Export Codex and Claude histories as canonical chat-search JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from thimiko.models import Session
from thimiko.sources import detect, iter_session_files, resolve_input_paths

DEFAULT_OUT = Path("reports") / "canonical.jsonl"


def iter_records(session: Session) -> Iterable[dict[str, Any]]:
    """Yield a session header followed by its canonical events."""
    yield session.header()
    for event in session.events:
        yield event.to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Codex and Claude session JSONL for retrieval and indexing."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSONL files or directories. Defaults to both history roots.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "codex", "claude"),
        default="auto",
        help="Force a source dialect instead of detecting each file.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Canonical JSONL output. Defaults to {DEFAULT_OUT}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    forced_source = None if args.source == "auto" else args.source
    try:
        files = iter_session_files(resolve_input_paths(args.paths))

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        session_count = 0
        event_count = 0
        searchable_count = 0
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for file_path in files:
                source = detect(file_path, forced_source)
                if source is None:
                    continue
                session = source.parse(file_path)
                for record in iter_records(session):
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                    handle.write("\n")
                session_count += 1
                event_count += len(session.events)
                searchable_count += sum(event.searchable for event in session.events)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote canonical JSONL: {args.out}")
    print(
        f"Sessions: {session_count}; events: {event_count}; "
        f"searchable events: {searchable_count}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
