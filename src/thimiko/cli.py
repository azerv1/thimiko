"""Command-line entry point: build | update | search | list | mcp.

Bare `thimiko` prints help. The MCP server only starts through `thimiko mcp`.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

from thimiko.config import DEFAULT_DB
from thimiko.dto import SearchResult, answer_dict, clean_snippet, iso_days_ago, relative_time
from thimiko.indexing import Indexer
from thimiko.search import KeywordRetriever
from thimiko.sources import all_sources, session_files_by_source
from thimiko.storage.base import Store


def _print_json(query: str, days: int | None, results: list[SearchResult]) -> None:
    payload = {
        "query": query,
        "days": days,
        "count": len(results),
        "results": [answer_dict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_text(query: str, days: int | None, results: list[SearchResult]) -> None:
    scope = f" (last {days} days)" if days else ""
    if not results:
        print(f'No results for "{query}"{scope}.')
        return
    print(f'{len(results)} result(s) for "{query}"{scope}\n')
    for index, result in enumerate(results, start=1):
        first = result.provenance[0] if result.provenance else {}
        title = result.title or result.session_id
        parts = [result.source]
        if result.model:
            parts.append(result.model)
        parts.append(relative_time(result.started_at))
        if result.cwd:
            parts.append(result.cwd)
        meta = "   " + " | ".join(parts)
        print(f"{index}. {title}")
        print(meta)
        for line in textwrap.wrap(clean_snippet(result.snippet), width=96) or [""]:
            print(f"   {line}")
        print(f"   {first.get('path', '')}:{first.get('line', '')}\n")


def _source_rows(*, counts: bool, store: Store | None) -> list[dict[str, Any]]:
    """One row per registered source; chat counts added when `counts` is set.

    `store` is None when no index exists yet — on-disk counts still work, and
    every source reports zero indexed chats.
    """
    on_disk = session_files_by_source() if counts else {}
    indexed = store.session_counts() if store is not None else {}
    rows: list[dict[str, Any]] = []
    for source in all_sources():
        roots = source.default_roots()
        row: dict[str, Any] = {
            "name": source.name,
            "roots": [str(root) for root in roots],
            "detected": any(root.exists() for root in roots),
        }
        if counts:
            row["indexed"] = indexed.get(source.name, 0)
            row["on_disk"] = len(on_disk.get(source.name, []))
        rows.append(row)
    return rows


def _print_list_json(rows: list[dict[str, Any]], db_path: Path, *, has_index: bool) -> None:
    payload = {"db": str(db_path), "indexed": has_index, "sources": rows}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_list_text(rows: list[dict[str, Any]], db_path: Path, *, has_index: bool) -> None:
    verbose = bool(rows) and "indexed" in rows[0]
    width = max(len(row["name"]) for row in rows)
    if verbose:
        print(f"{'SOURCE'.ljust(width)}  {'INDEXED':>9}  {'ON DISK':>9}")
    for row in rows:
        name = str(row["name"]).ljust(width)
        if verbose:
            indexed = "-" if not has_index else f"{row['indexed']:,}"
            print(f"{name}  {indexed:>9}  {row['on_disk']:>9,}")
        else:
            print(f"{name}  {'detected' if row['detected'] else 'not found'}")
        for root in row["roots"]:
            print(f"    {root}")
    if verbose and not has_index:
        print(f"\n(index not built at {db_path} — run `thimiko build`)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build, update, and search local chat history.")
    parser.add_argument(
        "--db", default=str(DEFAULT_DB), help=f"SQLite index path; default {DEFAULT_DB}"
    )
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Rebuild the index from chat histories")
    build_parser.add_argument(
        "paths", nargs="*", help="Files/directories; defaults to all registered sources' roots"
    )
    build_parser.add_argument(
        "--source",
        choices=("auto", "codex", "claude", "copilot", "gemini", "cursor"),
        default="auto",
    )

    update_parser = subparsers.add_parser("update", help="Incrementally update an existing index")
    update_parser.add_argument(
        "paths", nargs="*", help="Files/directories; defaults to all registered sources' roots"
    )
    update_parser.add_argument(
        "--source",
        choices=("auto", "codex", "claude", "copilot", "gemini", "cursor"),
        default="auto",
    )
    update_parser.add_argument(
        "--prune", action="store_true", help="Remove sessions whose source file no longer exists"
    )

    search_parser = subparsers.add_parser("search", help="Search indexed turn documents")
    search_parser.add_argument("query")
    search_parser.add_argument(
        "--source", choices=("codex", "claude", "copilot", "gemini", "cursor")
    )
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument(
        "--days", type=int, default=None, help="Only turns from the last N days"
    )
    search_parser.add_argument(
        "--raw-fts", action="store_true", help="Use SQLite FTS5 query syntax"
    )
    search_parser.add_argument(
        "--text", action="store_true", help="Human-readable output instead of JSON"
    )

    list_parser = subparsers.add_parser("list", help="List supported chat sources")
    list_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include indexed and on-disk chat counts per source",
    )
    list_parser.add_argument(
        "--text", action="store_true", help="Human-readable output instead of JSON"
    )

    subparsers.add_parser("mcp", help="Launch the MCP server over stdio")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command is None:
        return 0

    db_path = Path(args.db)

    if args.command == "mcp":
        from thimiko.mcp import run_stdio

        run_stdio(db_path)
        return 0

    from thimiko.storage import SqliteStore

    if args.command == "list":
        # Only open (and thereby create) the DB when counts were actually asked for.
        has_index = db_path.exists()
        store = SqliteStore(db_path) if args.verbose and has_index else None
        try:
            rows = _source_rows(counts=args.verbose, store=store)
        finally:
            if store is not None:
                store.close()
        if args.text:
            _print_list_text(rows, db_path, has_index=has_index)
        else:
            _print_list_json(rows, db_path, has_index=has_index)
        return 0

    try:
        if args.command == "build":
            forced_source = None if args.source == "auto" else args.source
            store = SqliteStore(db_path)
            try:
                build_result = Indexer(store).build(
                    [Path(path) for path in args.paths], forced_source
                )
            finally:
                store.close()
            print(f"Built index: {db_path}")
            print(f"Sessions: {build_result.sessions}; turn chunks: {build_result.documents}.")
            return 0

        if args.command == "update":
            forced_source = None if args.source == "auto" else args.source
            store = SqliteStore(db_path)
            try:
                update_result = Indexer(store).update(
                    [Path(path) for path in args.paths], forced_source, prune=args.prune
                )
            finally:
                store.close()
            print(f"Updated index: {db_path}")
            print(
                f"Added: {update_result.added}; updated: {update_result.updated}; "
                f"skipped: {update_result.skipped}; pruned: {update_result.pruned}."
            )
            return 0

        since = iso_days_ago(args.days) if args.days else None
        store = SqliteStore(db_path)
        try:
            results = KeywordRetriever(store).search(
                args.query, source=args.source, limit=args.limit, raw_fts=args.raw_fts, since=since
            )
        finally:
            store.close()
        if args.text:
            _print_text(args.query, args.days, results)
        else:
            _print_json(args.query, args.days, results)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
