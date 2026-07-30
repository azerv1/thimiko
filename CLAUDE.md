# thimiko — agent guide

Layered, searchable memory over local Codex and Claude Code chat history. See
`ARCHITECTURE.md` for the layer/interface design before changing structure.

## Commands

```powershell
uv sync                          # create .venv + uv.lock
uv run thimiko build              # full rebuild of the index (default: %LOCALAPPDATA%\thimiko\thimiko.sqlite)
uv run thimiko update [--prune]   # incremental: only changed/new files; --prune drops deleted files' sessions
uv run thimiko search "query"     # BM25 ranked search; JSON by default (--text for humans, --days N for recency)
uv run thimiko mcp                # launch the MCP server over stdio (also the bare `thimiko` default)
```

`--db PATH` overrides the index location on any subcommand.

## Invariants (don't break these)

- Never merge the two providers' raw schemas — normalize into the canonical
  `Session`/`Event` model only. See `ARCHITECTURE.md`.
- Every `Event` carries a `Provenance` (source file + line). Never lose that
  link — it's how `get_turn`/`get_session` point back to the raw record.
- Only ordinary user/assistant `Message`s are `searchable`. Hidden reasoning,
  tool calls/results, and attachments stay out of the default search corpus.
- FTS5 (`documents_fts`) is kept in sync via triggers on `documents`, not a
  manual reindex step — if you touch `SqliteStore`, insert/delete through
  `documents`, don't write to `documents_fts` directly.
- `Store`, `Retriever`, and `ChatSource` are ABCs for a reason: code above a
  layer must depend only on the interface (e.g. `cli.py`/`mcp.py` never
  import `sqlite3` directly).

## Quality bar

- Every function fully typed (params + return). `mypy --strict` and
  `ruff` (including `ANN`) are enforced — run before calling anything done:
  ```powershell
  uv run ruff check .
  uv run mypy -p thimiko
  uv run mypy scripts tests --explicit-package-bases
  uv run pytest
  ```
  (`-p thimiko` / `--explicit-package-bases` avoid a mypy false positive where
  the editable install's `src` path collides with the directory argument.)
- KISS/YAGNI: don't add abstractions, config, or fallback handling beyond
  what's asked. Reuse an existing helper (`thimiko.utils`, `sources/_parsing.py`,
  `sources/_builder.py`) before writing a new one.
- This machine runs Windows; use `uv run python ...`, never a bare `python`.

## Where things live

- `src/thimiko/models/` — OOP domain model (`Session`, `Turn`, `Event` + subclasses).
- `src/thimiko/sources/` — pluggable ingestion adapters (`CodexSource`, `ClaudeSource`, `CopilotSource`).
- `src/thimiko/storage/` — pluggable persistence (`SqliteStore`).
- `src/thimiko/indexing/` — chunking + `Indexer` (build/update).
- `src/thimiko/search/` — pluggable retrieval (`KeywordRetriever`).
- `src/thimiko/cli.py`, `src/thimiko/mcp.py` — the only two callers that pick concrete implementations.
- `scripts/` — standalone reports (schema summary, regex/knowledge-gap search, canonical JSONL export); run via `uv run python scripts/<name>.py`.
- `tests/` — pytest; run via `uv run pytest`.
