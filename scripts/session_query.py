#!/usr/bin/env python3
"""Search Codex and Claude session JSONL chats and report knowledge-gap candidates.

Text extraction is dialect-aware (built on `thimiko.utils.classify`): Codex text
comes from ``payload`` messages, Claude text from ``message.content[]``. Dialect
is auto-detected per record (``--source`` forces one). With no path argument,
both history roots are scanned.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thimiko.sources import resolve_input_paths
from thimiko.utils import detect_source, discover_jsonl_files, markdown_table, sniff_file_source

DEFAULT_OUT = Path("reports") / "query_report.md"

GAP_PATTERNS: dict[str, list[str]] = {
    "uncertainty": [
        r"\bnot sure\b",
        r"\bunsure\b",
        r"\bunclear\b",
        r"\bunknown\b",
        r"\bi don't know\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bmight\b",
        r"\bmay\b",
        r"\bassum",
        r"\binfer",
    ],
    "missing info": [
        r"\bmissing\b",
        r"\bnot provided\b",
        r"\bneed(?:s|ed)?\b",
        r"\brequires?\b",
        r"\bclarif",
        r"\bquestion\b",
        r"\btodo\b",
        r"\bfixme\b",
    ],
    "blocked or failed": [
        r"\bblocked\b",
        r"\bfailed\b",
        r"\bfail(?:ure|ed)?\b",
        r"\berror\b",
        r"\bexception\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bunable\b",
        r"\btimeout\b",
        r"\bdenied\b",
    ],
    "verification needed": [
        r"\bverify\b",
        r"\bconfirm\b",
        r"\bcheck\b",
        r"\bsearch\b",
        r"\bbrowse\b",
        r"\bcurrent\b",
        r"\blatest\b",
        r"\bup[- ]to[- ]date\b",
    ],
}


@dataclass
class TextRecord:
    file: Path
    line: int
    timestamp: str
    role: str
    source: str
    text: str


@dataclass
class GapCandidate:
    label: str
    line: int
    timestamp: str
    role: str
    source: str
    snippet: str


@dataclass
class MatchRecord:
    line: int
    timestamp: str
    role: str
    source: str
    snippet: str


@dataclass
class SessionResult:
    file: Path
    bytes: int
    source: str = "unknown"
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    lines: int = 0
    text_records: int = 0
    matches: list[MatchRecord] = field(default_factory=list)
    gaps: list[GapCandidate] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search session JSONL chats for a term and report knowledge-gap candidates."
    )
    parser.add_argument("query", help="Text or regex to search for.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSONL files or directories to scan. Defaults to both history roots.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "codex", "claude"),
        default="auto",
        help="Force a dialect instead of auto-detecting per record.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Markdown output path. Defaults to {DEFAULT_OUT}.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional machine-readable JSON output path.",
    )
    parser.add_argument(
        "--regex",
        action="store_true",
        help="Treat query as a regular expression.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive matching.",
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Include developer/system/internal messages. Default is user and assistant only.",
    )
    parser.add_argument(
        "--include-tools",
        action="store_true",
        help="Include tool call inputs/outputs in search and gap detection.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4,
        help=(
            "Number of neighboring text records to inspect for gaps around each match. "
            "Defaults to 4."
        ),
    )
    parser.add_argument(
        "--max-matches-per-session",
        type=int,
        default=20,
        help="Maximum match snippets to print per session. Defaults to 20.",
    )
    parser.add_argument(
        "--max-gaps-per-session",
        type=int,
        default=20,
        help="Maximum gap snippets to print per session. Defaults to 20.",
    )
    return parser.parse_args()


def compile_query(query: str, regex: bool, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = query if regex else re.escape(query)
    return re.compile(pattern, flags)


def compile_gap_patterns() -> list[tuple[str, re.Pattern[str]]]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for label, patterns in GAP_PATTERNS.items():
        for pattern in patterns:
            compiled.append((label, re.compile(pattern, re.IGNORECASE)))
    return compiled


def content_to_text(content: Any, include_tools: bool = False) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"tool_use", "tool_result"} and not include_tools:
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif isinstance(item.get("content"), list):
                    parts.append(content_to_text(item["content"], include_tools))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        content_value = content.get("content")
        if isinstance(content_value, str):
            return content_value
    return ""


def extract_codex(
    record: dict[str, Any], include_internal: bool, include_tools: bool
) -> tuple[str, str, str]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return "", "", ""

    top_type = str(record.get("type", ""))
    payload_type = str(payload.get("type", ""))
    source = f"{top_type}/{payload_type}".strip("/")

    if top_type == "event_msg" and payload_type in {"user_message", "agent_message"}:
        role = "user" if payload_type == "user_message" else "assistant"
        return role, source, str(payload.get("message") or "")

    if top_type == "event_msg" and payload_type == "task_complete":
        return "assistant", source, str(payload.get("last_agent_message") or "")

    if payload_type in {"message", "agent_message"}:
        role = str(payload.get("role") or payload.get("author") or "assistant")
        if not include_internal and role not in {"user", "assistant"}:
            return "", "", ""
        return role, source, content_to_text(payload.get("content"))

    if include_tools and payload_type in {
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
    }:
        role = "tool"
        text = "\n".join(
            str(payload.get(key) or "")
            for key in ("name", "arguments", "input", "output")
            if payload.get(key) is not None
        )
        return role, source, text

    return "", "", ""


def extract_claude(
    record: dict[str, Any], include_internal: bool, include_tools: bool
) -> tuple[str, str, str]:
    top_type = str(record.get("type", ""))
    message = record.get("message")

    if isinstance(message, dict):
        role = str(message.get("role") or top_type or "assistant")
        source = f"{top_type}/{role}".strip("/")
        text = content_to_text(message.get("content"), include_tools)
        return role, source, text

    if include_internal and top_type == "system":
        return "system", top_type, str(record.get("content") or "")

    return "", "", ""


def extract_text(
    record: dict[str, Any],
    forced_source: str | None,
    include_internal: bool,
    include_tools: bool,
) -> tuple[str, str, str]:
    source = forced_source or detect_source(record)
    if source == "codex":
        return extract_codex(record, include_internal, include_tools)
    if source == "claude":
        return extract_claude(record, include_internal, include_tools)
    return "", "", ""


def split_sentences(text: str) -> Iterable[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]


def make_snippet(text: str, pattern: re.Pattern[str] | None = None, limit: int = 280) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    start = 0
    if pattern is not None:
        match = pattern.search(compact)
        if match:
            start = max(0, match.start() - limit // 3)
    end = min(len(compact), start + limit)
    if end - start < limit:
        start = max(0, end - limit)
    snippet = compact[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(compact):
        snippet += "..."
    return snippet


def find_gap_candidates(
    record: TextRecord,
    gap_patterns: list[tuple[str, re.Pattern[str]]],
    seen: set[tuple[str, int, str, str]],
) -> list[GapCandidate]:
    candidates: list[GapCandidate] = []
    sentences = split_sentences(record.text)
    if not sentences and "?" in record.text:
        sentences = [record.text]

    for sentence in sentences:
        labels = sorted({label for label, pattern in gap_patterns if pattern.search(sentence)})
        if "?" in sentence:
            labels.append("open question")
        for label in sorted(set(labels)):
            snippet = make_snippet(sentence, None, 260)
            key = (str(record.file), record.line, label, snippet)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                GapCandidate(
                    label=label,
                    line=record.line,
                    timestamp=record.timestamp,
                    role=record.role,
                    source=record.source,
                    snippet=snippet,
                )
            )
    return candidates


def analyze(
    paths: list[Path],
    query_pattern: re.Pattern[str],
    forced_source: str | None,
    include_internal: bool,
    include_tools: bool,
    context: int,
) -> tuple[list[SessionResult], int, int]:
    gap_patterns = compile_gap_patterns()
    session_results: list[SessionResult] = []
    total_lines = 0
    invalid_lines = 0

    files: list[Path] = []
    for path in paths:
        files.extend(discover_jsonl_files(path))

    for file_path in files:
        try:
            file_bytes = file_path.stat().st_size
        except OSError:
            file_bytes = 0
        file_source = forced_source or sniff_file_source(file_path)
        session = SessionResult(file=file_path, bytes=file_bytes, source=file_source)
        text_records: list[TextRecord] = []
        effective_source = file_source if file_source != "unknown" else None

        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                total_lines += 1
                session.lines += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue

                if not isinstance(record, dict):
                    continue

                timestamp = str(record.get("timestamp") or "")
                if timestamp:
                    if session.first_timestamp is None:
                        session.first_timestamp = timestamp
                    session.last_timestamp = timestamp

                role, source, text = extract_text(
                    record, effective_source, include_internal, include_tools
                )
                if not text:
                    continue
                text_record = TextRecord(file_path, line_number, timestamp, role, source, text)
                text_records.append(text_record)
                session.text_records += 1

        matching_indexes: list[int] = []
        for index, text_record in enumerate(text_records):
            if query_pattern.search(text_record.text):
                matching_indexes.append(index)
                session.matches.append(
                    MatchRecord(
                        line=text_record.line,
                        timestamp=text_record.timestamp,
                        role=text_record.role,
                        source=text_record.source,
                        snippet=make_snippet(text_record.text, query_pattern),
                    )
                )

        if matching_indexes:
            seen_gap_keys: set[tuple[str, int, str, str]] = set()
            indexes_to_check: set[int] = set()
            for index in matching_indexes:
                start = max(0, index - context)
                end = min(len(text_records), index + context + 1)
                indexes_to_check.update(range(start, end))
            for index in sorted(indexes_to_check):
                session.gaps.extend(
                    find_gap_candidates(text_records[index], gap_patterns, seen_gap_keys)
                )
            session_results.append(session)

    return session_results, total_lines, invalid_lines


def write_markdown(
    out_path: Path,
    query: str,
    regex: bool,
    results: list[SessionResult],
    total_lines: int,
    invalid_lines: int,
    max_matches: int,
    max_gaps: int,
) -> None:
    total_matches = sum(len(session.matches) for session in results)
    total_gaps = sum(len(session.gaps) for session in results)
    lines: list[str] = []
    lines.append("# Session Query Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Query", query],
                ["Regex mode", regex],
                ["Matching sessions", len(results)],
                ["Matching text records", total_matches],
                ["Knowledge-gap candidates", total_gaps],
                ["Total JSONL lines scanned", total_lines],
                ["Invalid JSON lines", invalid_lines],
            ],
        )
    )

    lines.append("## Matching Sessions")
    lines.append("")
    session_rows = [
        [
            session.source,
            session.file,
            session.first_timestamp or "",
            session.last_timestamp or "",
            len(session.matches),
            len(session.gaps),
        ]
        for session in results
    ]
    lines.append(
        markdown_table(
            ["Source", "File", "First timestamp", "Last timestamp", "Matches", "Gap candidates"],
            session_rows,
        )
    )

    for session in results:
        lines.append(f"## {session.file}")
        lines.append("")
        lines.append(f"- Source: `{session.source}`")
        lines.append(
            f"- Time range: `{session.first_timestamp or ''}` to `{session.last_timestamp or ''}`"
        )
        lines.append(f"- Lines: `{session.lines}`")
        lines.append(f"- Text records searched: `{session.text_records}`")
        lines.append("")

        lines.append("### Matches")
        lines.append("")
        match_rows = [
            [match.line, match.timestamp, match.role, match.source, match.snippet]
            for match in session.matches[:max_matches]
        ]
        lines.append(markdown_table(["Line", "Timestamp", "Role", "Source", "Snippet"], match_rows))
        if len(session.matches) > max_matches:
            lines.append(f"_Hidden matches in this session: {len(session.matches) - max_matches}_")
            lines.append("")

        lines.append("### Knowledge-Gap Candidates")
        lines.append("")
        gap_rows = [
            [gap.label, gap.line, gap.timestamp, gap.role, gap.source, gap.snippet]
            for gap in session.gaps[:max_gaps]
        ]
        lines.append(
            markdown_table(["Label", "Line", "Timestamp", "Role", "Source", "Snippet"], gap_rows)
        )
        if len(session.gaps) > max_gaps:
            lines.append(f"_Hidden gap candidates in this session: {len(session.gaps) - max_gaps}_")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(
    out_path: Path, results: list[SessionResult], total_lines: int, invalid_lines: int
) -> None:
    payload = {
        "total_lines": total_lines,
        "invalid_lines": invalid_lines,
        "sessions": [
            {
                "file": str(session.file),
                "source": session.source,
                "bytes": session.bytes,
                "first_timestamp": session.first_timestamp,
                "last_timestamp": session.last_timestamp,
                "lines": session.lines,
                "text_records": session.text_records,
                "matches": [match.__dict__ for match in session.matches],
                "knowledge_gap_candidates": [gap.__dict__ for gap in session.gaps],
            }
            for session in results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    forced_source = None if args.source == "auto" else args.source
    try:
        query_pattern = compile_query(args.query, args.regex, args.case_sensitive)
        results, total_lines, invalid_lines = analyze(
            resolve_input_paths(args.paths),
            query_pattern,
            forced_source,
            args.include_internal,
            args.include_tools,
            args.context,
        )
        write_markdown(
            Path(args.out),
            args.query,
            args.regex,
            results,
            total_lines,
            invalid_lines,
            args.max_matches_per_session,
            args.max_gaps_per_session,
        )
        if args.json_out:
            write_json(Path(args.json_out), results, total_lines, invalid_lines)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote Markdown report: {args.out}")
    print(
        f"Matching sessions: {len(results)}; "
        f"matches: {sum(len(session.matches) for session in results)}; "
        f"gap candidates: {sum(len(session.gaps) for session in results)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
