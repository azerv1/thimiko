#!/usr/bin/env python3
"""Stream Codex and Claude session JSONL files and summarize observed schemas.

Produces a single combined Markdown report whose every table carries a
``Source`` column, so the two dialects can be compared side by side. Dialect is
auto-detected per record (``--source`` forces one). With no path argument, both
history roots are scanned.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thimiko.sources import resolve_input_paths
from thimiko.utils import (
    Classification,
    as_string,
    classify,
    discover_jsonl_files,
    format_keys,
    markdown_table,
    preview_string,
    redact,
    sniff_file_source,
    sorted_keys,
)


@dataclass
class FileStats:
    path: Path
    bytes: int
    source: str = "unknown"
    lines: int = 0
    valid: int = 0
    invalid: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None


@dataclass
class ShapeStats:
    count: int = 0
    first_file: str = ""
    first_line: int = 0
    last_file: str = ""
    last_line: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Analysis:
    scanned_paths: list[Path] = field(default_factory=list)
    files: list[FileStats] = field(default_factory=list)
    # keyed by (source, top_keys)
    top_level_shapes: Counter[tuple[str, tuple[str, ...]]] = field(default_factory=Counter)
    # keyed by (source, top_type, sub_type)
    event_categories: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    # keyed by (source, block_type)
    content_blocks: Counter[tuple[str, str]] = field(default_factory=Counter)
    # keyed by (source, top_type, sub_type, sub_keys)
    record_shapes: dict[tuple[str, str, str, tuple[str, ...]], ShapeStats] = field(
        default_factory=dict
    )
    records_by_source: Counter[str] = field(default_factory=Counter)
    invalid_examples: list[dict[str, Any]] = field(default_factory=list)
    total_lines: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    total_bytes: int = 0
    truncated: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Codex and Claude JSONL session schemas into one report."
    )
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
        default=str(Path("reports") / "schema_report.md"),
        help="Markdown output path. Defaults to reports/schema_report.md.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum total JSONL lines to read across all files.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=2,
        help="Redacted examples to keep per record shape. Defaults to 2.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth for redacted example summaries. Defaults to 2.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    return parser.parse_args()


def analyze(
    paths: list[Path],
    forced_source: str | None,
    max_lines: int | None,
    examples: int,
    max_depth: int,
) -> Analysis:
    files: list[Path] = []
    for path in paths:
        files.extend(discover_jsonl_files(path))

    result = Analysis(scanned_paths=paths)

    for file_path in files:
        if max_lines is not None and result.total_lines >= max_lines:
            result.truncated = True
            break

        try:
            file_bytes = file_path.stat().st_size
        except OSError:
            file_bytes = 0

        file_source = forced_source or sniff_file_source(file_path)
        file_stats = FileStats(path=file_path, bytes=file_bytes, source=file_source)
        result.total_bytes += file_bytes

        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if max_lines is not None and result.total_lines >= max_lines:
                    result.truncated = True
                    break

                result.total_lines += 1
                file_stats.lines += 1
                line = raw_line.rstrip("\r\n")

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    file_stats.invalid += 1
                    result.invalid_records += 1
                    if len(result.invalid_examples) < 100:
                        result.invalid_examples.append(
                            {
                                "file": str(file_path),
                                "line": line_number,
                                "error": exc.msg,
                                "preview": preview_string(line, 160),
                            }
                        )
                    continue

                file_stats.valid += 1
                result.valid_records += 1

                info: Classification = classify(
                    record, file_source if file_source != "unknown" else None
                )

                if isinstance(record, dict):
                    top_keys = sorted_keys(record)
                    timestamp = as_string(record.get("timestamp"))
                else:
                    top_keys = (f"<{type(record).__name__}>",)
                    timestamp = ""

                if timestamp:
                    if file_stats.first_timestamp is None:
                        file_stats.first_timestamp = timestamp
                    file_stats.last_timestamp = timestamp

                result.records_by_source[info.source] += 1
                result.top_level_shapes[(info.source, top_keys)] += 1
                result.event_categories[(info.source, info.top_type, info.sub_type)] += 1
                for block_type in info.content_block_types:
                    result.content_blocks[(info.source, block_type)] += 1

                shape_key = (info.source, info.top_type, info.sub_type, info.sub_keys)
                shape = result.record_shapes.setdefault(shape_key, ShapeStats())
                shape.count += 1
                shape.last_file = str(file_path)
                shape.last_line = line_number
                if not shape.first_file:
                    shape.first_file = str(file_path)
                    shape.first_line = line_number
                if len(shape.examples) < examples:
                    shape.examples.append(
                        {
                            "file": str(file_path),
                            "line": line_number,
                            "record": redact(record, max_depth=max_depth),
                        }
                    )

        result.files.append(file_stats)

    return result


def write_markdown(result: Analysis, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Session JSONL Schema Report (Codex + Claude)")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    overview_rows: list[list[Any]] = [
        ["Scanned paths", ", ".join(str(p) for p in result.scanned_paths)],
        ["Files", len(result.files)],
        ["Total bytes", result.total_bytes],
        ["Total lines read", result.total_lines],
        ["Valid records", result.valid_records],
        ["Invalid records", result.invalid_records],
        ["Truncated by --max-lines", result.truncated],
    ]
    for source, count in result.records_by_source.most_common():
        overview_rows.append([f"Records ({source})", count])
    lines.append(markdown_table(["Metric", "Value"], overview_rows))

    lines.append("## Files")
    lines.append("")
    file_rows = [
        [
            stats.source,
            stats.path,
            stats.bytes,
            stats.lines,
            stats.valid,
            stats.invalid,
            stats.first_timestamp or "",
            stats.last_timestamp or "",
        ]
        for stats in result.files
    ]
    lines.append(
        markdown_table(
            [
                "Source",
                "Path",
                "Bytes",
                "Lines",
                "Valid",
                "Invalid",
                "First timestamp",
                "Last timestamp",
            ],
            file_rows,
        )
    )

    lines.append("## Top-Level Schemas")
    lines.append("")
    top_rows = [
        [count, source, format_keys(keys)]
        for (source, keys), count in result.top_level_shapes.most_common()
    ]
    lines.append(markdown_table(["Count", "Source", "Keys"], top_rows))

    lines.append("## Event Categories")
    lines.append("")
    category_rows = [
        [count, source, top_type or "<missing>", sub_type or "<missing>"]
        for (source, top_type, sub_type), count in result.event_categories.most_common()
    ]
    lines.append(markdown_table(["Count", "Source", "Top type", "Sub type"], category_rows))

    lines.append("## Content Blocks")
    lines.append("")
    block_rows = [
        [count, source, block_type or "<missing>"]
        for (source, block_type), count in result.content_blocks.most_common()
    ]
    lines.append(markdown_table(["Count", "Source", "Block type"], block_rows))

    lines.append("## Record Shapes")
    lines.append("")
    shape_rows = []
    for (source, top_type, sub_type, sub_keys), stats in sorted(
        result.record_shapes.items(), key=lambda item: item[1].count, reverse=True
    ):
        shape_rows.append(
            [
                stats.count,
                source,
                top_type or "<missing>",
                sub_type or "<missing>",
                format_keys(sub_keys),
                f"{stats.first_file}:{stats.first_line}",
                f"{stats.last_file}:{stats.last_line}",
            ]
        )
    lines.append(
        markdown_table(
            ["Count", "Source", "Top type", "Sub type", "Sub keys", "First seen", "Last seen"],
            shape_rows,
        )
    )

    lines.append("## Examples")
    lines.append("")
    for (source, top_type, sub_type, sub_keys), stats in sorted(
        result.record_shapes.items(), key=lambda item: item[1].count, reverse=True
    ):
        lines.append(
            f"### {stats.count}x [{source}] top={top_type or '<missing>'} "
            f"sub={sub_type or '<missing>'} keys={format_keys(sub_keys)}"
        )
        lines.append("")
        for example in stats.examples:
            lines.append(f"- `{example['file']}:{example['line']}`")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(example["record"], indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

    if result.invalid_examples:
        lines.append("## Invalid Lines")
        lines.append("")
        invalid_rows = [
            [item["file"], item["line"], item["error"], item["preview"]]
            for item in result.invalid_examples
        ]
        lines.append(markdown_table(["File", "Line", "Error", "Preview"], invalid_rows))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def to_jsonable(result: Analysis) -> dict[str, Any]:
    return {
        "scanned_paths": [str(p) for p in result.scanned_paths],
        "records_by_source": dict(result.records_by_source),
        "files": [
            {
                "source": stats.source,
                "path": str(stats.path),
                "bytes": stats.bytes,
                "lines": stats.lines,
                "valid": stats.valid,
                "invalid": stats.invalid,
                "first_timestamp": stats.first_timestamp,
                "last_timestamp": stats.last_timestamp,
            }
            for stats in result.files
        ],
        "top_level_shapes": [
            {"source": source, "keys": list(keys), "count": count}
            for (source, keys), count in result.top_level_shapes.most_common()
        ],
        "event_categories": [
            {"source": source, "top_type": top_type, "sub_type": sub_type, "count": count}
            for (source, top_type, sub_type), count in result.event_categories.most_common()
        ],
        "content_blocks": [
            {"source": source, "block_type": block_type, "count": count}
            for (source, block_type), count in result.content_blocks.most_common()
        ],
        "record_shapes": [
            {
                "source": source,
                "top_type": top_type,
                "sub_type": sub_type,
                "sub_keys": list(sub_keys),
                "count": stats.count,
                "first_file": stats.first_file,
                "first_line": stats.first_line,
                "last_file": stats.last_file,
                "last_line": stats.last_line,
                "examples": stats.examples,
            }
            for (source, top_type, sub_type, sub_keys), stats in sorted(
                result.record_shapes.items(), key=lambda item: item[1].count, reverse=True
            )
        ],
        "invalid_examples": result.invalid_examples,
        "total_lines": result.total_lines,
        "valid_records": result.valid_records,
        "invalid_records": result.invalid_records,
        "total_bytes": result.total_bytes,
        "truncated": result.truncated,
    }


def write_json(result: Analysis, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(to_jsonable(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    paths = resolve_input_paths(args.paths)
    forced_source = None if args.source == "auto" else args.source
    out_path = Path(args.out)

    try:
        result = analyze(paths, forced_source, args.max_lines, args.examples, args.max_depth)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_markdown(result, out_path)
    print(f"Wrote Markdown report: {out_path}")

    if args.json_out:
        json_out_path = Path(args.json_out)
        write_json(result, json_out_path)
        print(f"Wrote JSON report: {json_out_path}")

    by_source = ", ".join(
        f"{source}={count}" for source, count in result.records_by_source.most_common()
    )
    print(
        "Scanned "
        f"{len(result.files)} files, {result.total_lines} lines, "
        f"{result.valid_records} valid records, {result.invalid_records} invalid records "
        f"({by_source})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
