# thimiko — local chat-history memory

*thimiko* (θύμηση, Greek for "memory") normalizes the JSONL session history
written by two different AI CLIs into one canonical model, indexes it for
ranked full-text search, and serves it to a human or an LLM through an MCP
server.

| Source | Location | Envelope |
| --- | --- | --- |
| **Codex** | `~/.codex/sessions/**/*.jsonl` | uniform `{timestamp, type, payload}`; sub-type in `payload.type` |
| **Claude Code** | `~/.claude/projects/**/*.jsonl` | flat records; nested object is `message` (`role` + `content[]` blocks), plus auxiliary types (`file-history-snapshot`, `ai-title`, `mode`, `attachment`, …) |

See `ARCHITECTURE.md` for the layered design (sources -> models -> storage ->
indexing -> search -> CLI/MCP) and how to extend any layer.

## Setup

Install once as a [uv tool](https://docs.astral.sh/uv/guides/tools/); the
`thimiko` executable is then on your `PATH`:

```powershell
uv tool install "thimiko[mcp]"    # installs the `thimiko` exe (with MCP support)
```

To upgrade later: `uv tool upgrade thimiko`. For local development from a
checkout, use an editable install: `uv tool install -e ".[mcp]"`.

## Usage

```powershell
thimiko build                          # full rebuild (default index: %LOCALAPPDATA%\thimiko\thimiko.sqlite)
thimiko update                         # incremental: only new/changed files
thimiko update --prune                  # also drop sessions whose source file is gone
thimiko search "database migration"     # ranked BM25 search (JSON by default)
thimiko search "permission denied" --source codex --limit 5
thimiko search "rate limit" --days 10    # only turns from the last 10 days
thimiko search "..." --text             # human-readable output instead of JSON
thimiko mcp                            # MCP server over stdio (also the bare `thimiko` default)
```

Every subcommand accepts an explicit path list (files or directories) in
place of the default source roots, plus `--source {auto,codex,claude}` to
force a dialect instead of auto-detecting.

### Reports (`scripts/`)

Standalone, stdlib-only reporting tools — not part of the indexed
build/update/search path:

```powershell
uv run python scripts/schema_analyzer.py --out reports\schema_report.md
uv run python scripts/schema_analyzer.py --json-out reports\schema.json
uv run python scripts/session_query.py "error" --out reports\query_report.md
uv run python scripts/session_query.py "rate limit" --regex --include-tools
uv run python scripts/normalize_sessions.py --out reports\canonical.jsonl
```

## MCP server

`thimiko mcp` launches a read-only MCP server over stdio exposing:

- `search_chats(query, source=None, limit=10, days=None, raw_fts=False)` — ranked BM25 search; returns a JSON envelope of enriched hits (title, source, model, relative `when`, cleaned snippet, `path`+`line`). `days` restricts to the last N days.
- `get_session(session_id)` — a session's header plus all of its turns.
- `get_turn(session_id, turn_id, neighbors=1)` — a turn's chunks plus neighboring turns.

Register it in `.mcp.json` pointing at `thimiko mcp`, or use the MCP
inspector (`mcp dev src/thimiko/mcp.py`) during development.

## Layout

```
src/thimiko/
├── models/      OOP domain: Session, Turn, Event (+ Message/ToolCall/ToolResult/Reasoning/Attachment)
├── sources/     ChatSource ABC + CodexSource/ClaudeSource adapters (pluggable)
├── storage/     Store ABC + SqliteStore (FTS5, pluggable)
├── indexing/    turn -> SearchDocument chunking + Indexer (build/update)
├── search/      Retriever ABC + KeywordRetriever (BM25, pluggable)
├── cli.py       build | update | search | mcp
├── mcp.py       MCP server (semble idiom)
├── utils.py     dialect sniffing, redaction, formatting (used by scripts/)
├── config.py    default data dir / DB path
└── types.py     SearchDocument / SearchResult
scripts/         standalone report generators (schema, regex search, canonical export)
tests/           pytest
data/            scratch dir for dev (gitignored); default index actually lives in %LOCALAPPDATA%\thimiko\ (override with THIMIKO_DATA_DIR)
```

## Dialect-adapter design

Two methods abstract over the schemas, in `thimiko.utils`:

- **`detect_source(record)`** sniffs one record: a `payload` object → `codex`;
  `message`/`uuid` → `claude`. `sniff_file_source()` reads the first lines of a
  file to give a stable source for the whole file, so auxiliary record types
  that lack per-record markers are still attributed correctly. Each
  `ChatSource.matches()` is built on this.
- **`classify(record, source)`** flattens either schema into one
  `Classification(source, top_type, sub_type, sub_keys, content_block_types)`,
  used by `scripts/schema_analyzer.py`.

To support a third source, implement `ChatSource` (see `ARCHITECTURE.md`) —
nothing else needs to change.

## Redaction

Example records embedded in the schema report are redacted: string values are
reduced to `<str len=… preview=…>`, and fields whose names match
`SENSITIVE_NAME_PARTS` (`content`, `message`, `thinking`, `token`, `key`,
`arguments`, `snapshot`, …) are further masked. Raw conversation text is never
written into the schema report.

## Canonical model and search

`Session`/`Event` deliberately don't merge the complete provider schemas —
they're append-only event logs with different duplication and grouping
rules. Every `Event` points back to the exact source file and line via
`Provenance`; the provider logs remain the lossless source of truth.

Only ordinary user/assistant messages are `searchable`. Bootstrap prompts,
hidden reasoning, tool arguments/results, and attachments remain available for
context but stay out of the default search corpus. Codex's duplicated display
events are removed when authoritative response messages exist. Claude's
fragmented content blocks remain in source order, and each `tool_result`
links to its `tool_call`.

Search documents are built at the **turn** level, not per event: searchable
events are concatenated with role labels, long turns are chunked, and each
document keeps `session_id`, `source`, `turn_id`, `cwd`, `git_branch`, and a
time range. `KeywordRetriever` ranks with SQLite FTS5/BM25 and can expand a
winning chunk to its neighboring turns (`get_turn`/`expand`).

## Why not Turso?

Turso Database (`pyturso`, formerly Limbo) does not support SQLite FTS5 (its
own compatibility doc says "use Turso FTS instead") and is still experimental
— adopting it would mean rewriting this project's entire FTS5/BM25 search
onto an unrelated engine. **libSQL** (`pip install libsql`) is the viable
"Turso" path if this ever needs cloud sync or native vector search: it's a
SQLite fork that keeps FTS5. `storage.Store` is the swap point — see
`ARCHITECTURE.md`.
