# Architecture

thimiko normalizes local Codex, Claude Code, GitHub Copilot, and Gemini CLI chat history into one
canonical model, indexes it for full-text search, and serves it to a human or an LLM
(via MCP). The pipeline is layered as four small interfaces; everything above
a layer depends only on that layer's ABC, never on a concrete implementation.
Each layer can be extended or swapped without touching the others.

```
 sources (ingestion)        models (domain)        storage (persistence)
 ChatSource ABC        ->   Session / Event    ->   Store ABC
 CodexSource                Turn                    SqliteStore
 ClaudeSource
 CopilotSource
 GeminiSource
                                  |
                                  v
                          indexing (pipeline)
                          documents_for_session()
                          Indexer.build() / .update()
                                  |
                                  v
                          search (retrieval)
                          Retriever ABC
                          KeywordRetriever (BM25)
                                  |
                                  v
                          interfaces
                          cli.py (build/update/search/mcp)
                          mcp.py (search_chats/get_session/get_turn)
```

## Data flow

1. A `ChatSource` recognizes and parses one provider's session file into a
   `Session` of ordered `Event`s (Phase 1/2).
2. `indexing.chunking.documents_for_session()` groups a session's events into
   `Turn`s, keeps only searchable `Message`s, and chunks long turns into
   `SearchDocument`s (Phase 4).
3. `Indexer` drives a `Store` through `build()` (full rebuild) or `update()`
   (incremental, mtime/size-keyed) using any registered sources (Phase 4).
4. A `Retriever` (currently `KeywordRetriever`, BM25 via FTS5) turns a query
   into ranked `SearchResult`s and expands a hit into neighboring turns
   (Phase 5).
5. `cli.py` and `mcp.py` are the only two things that construct concrete
   `Store`/`Retriever` instances; every other module only sees the ABCs.

## The interfaces

### `ChatSource` (`src/thimiko/sources/base.py`)

```python
class ChatSource(ABC):
    name: str

    def default_roots(self) -> list[Path]: ...
    def discover(self, root: Path) -> list[Path]: ...  # defaults to *.jsonl
    def matches(self, path: Path) -> bool: ...
    def parse(self, path: Path) -> Session: ...
```

**To add a new chat-history provider** ("agy history" or anything else):
implement this ABC in a new module under `sources/`, using `ChatSource` plus
the shared `sources/_parsing.py` helpers and `sources/_builder.py`'s
`EventBuilder` to sequence events. Call `thimiko.sources.register(YourSource())`
once. `Indexer`, the CLI, and the MCP server all pick it up automatically
through `all_sources()` / `detect()` / `iter_session_files()` — nothing else
changes.

Discovery is **per-source**: `iter_session_files()` asks each source to
`discover()` its own files under its roots. The default `discover()` does
recursive `*.jsonl`; a provider stored differently overrides it (e.g.
`CopilotSource` globs VSCode's mixed `.json`/`.jsonl` `chatSessions/`). Concrete
sources today: `CodexSource`, `ClaudeSource`, `CopilotSource` (GitHub Copilot /
VSCode chat), and `GeminiSource` (legacy JSON snapshots plus current append-only
JSONL sessions, including checkpoints and rewinds).

### `Store` (`src/thimiko/storage/base.py`)

Persistence backend for sessions and their search documents: `create_schema`,
`upsert_session`, `delete_session`, `file_state`/`record_file`/`forget_file`/
`known_files` (incremental-update bookkeeping), `search`, `get_session`,
`get_turn`, `close`.

**Current implementation**: `SqliteStore` — stdlib `sqlite3` + FTS5. Schema:
`sessions`, `documents`, `documents_fts` (FTS5 virtual table over `documents`),
`embeddings` (reserved, unused — see below), `indexed_files` (path -> mtime/
size/session_id, drives `update`), `metadata`.

FTS5 is kept in sync with `documents` by three triggers (`documents_ai`,
`documents_ad`, `documents_au`) rather than a bulk reindex step. This means
`build` (bulk insert) and `update` (targeted insert/delete per changed file)
both keep the search index consistent automatically — there is no separate
"reindex" phase.

**Turso/libSQL note**: Turso Database (`pyturso`, ex-Limbo) does **not**
support FTS5 (its own compat doc says "use Turso FTS instead"), so it isn't a
drop-in here. **libSQL** (`pip install libsql`) is a SQLite fork that *does*
keep FTS5 and adds native vector search + optional cloud sync — the plausible
next `Store` implementation if this ever needs to run somewhere other than
one machine, or wants embeddings without a separate service. To add it:
implement `Store` in `storage/libsql_store.py`; nothing in `Indexer`,
`Retriever`, `cli.py`, or `mcp.py` needs to change.

The `embeddings` table exists today only as a reserved slot for a future
embedding-backed retriever; nothing writes to it yet.

### `Retriever` (`src/thimiko/search/base.py`)

```python
class Retriever(ABC):
    def search(self, query, *, source=None, limit=10, raw_fts=False) -> list[SearchResult]: ...
    def expand(self, session_id, turn_id, neighbors=1) -> dict[str, Any] | None: ...
```

**Current implementation**: `KeywordRetriever` — BM25 ranking via
`Store.search`, and `expand()` (a winning chunk plus its neighboring turns)
via `Store.get_turn`. A future `HybridRetriever` or `VectorRetriever` would
implement this same ABC and read from the (currently reserved) `embeddings`
table; `cli.py` and `mcp.py` would need only a one-line swap of which
retriever they construct.

## The domain model (`src/thimiko/models/`)

- `Provenance` — path + line back to the exact raw source record.
- `Event` (ABC) with concrete subclasses `Message`, `ToolCall`, `ToolResult`,
  `Reasoning`, `Attachment` — replaces a single tagged dataclass with one
  class per event category, each carrying only its relevant fields.
- `Session` — provider identity, time range, cwd/branch/model, and the
  ordered `Event` list; `turns()` groups events by `turn_id` (falling back to
  a per-event singleton turn when none is set, matching each source's own
  fallback-turn numbering).
- `Turn` — one user/assistant exchange; `searchable_messages()` is what
  feeds the chunker.

Only ordinary user/assistant `Message`s are ever `searchable`. Hidden
reasoning, tool arguments/results, and attachments stay available (for
`get_session`/`get_turn` context) but are excluded from the default search
corpus. Codex's duplicated `event_msg` display events are dropped whenever
authoritative `response_item` messages exist. Claude's fragmented content
blocks are kept in source order, and each `tool_result` links back to its
`tool_call` via `parent_id`.

## Interfaces (`cli.py`, `mcp.py`)

`cli.py` is a plain `argparse` CLI: `build`, `update`, `search`, `mcp`.
Bare `thimiko` prints help; `thimiko mcp` starts the server. It constructs a `SqliteStore` and either
an `Indexer` or a `KeywordRetriever` — the only two places in the codebase
that reference `SqliteStore` by name.

`mcp.py` mirrors the semble (`MinishLab/semble`) idiom: a single in-package
module using the SDK's high-level server (`MCPServer` — the installed SDK's
current name for what semble imports as `FastMCP` from `mcp.server.fastmcp`;
same API), async `@server.tool()` functions returning JSON strings, and a
`create_server()` / `serve()` split. It exposes three read-only tools:
`search_chats`, `get_session`, `get_turn`. Build/update are intentionally
CLI-only — the MCP server never mutates the index.

## Adding a new backend, source, or retriever

1. **New chat-history provider**: implement `ChatSource`, register it. See
   `sources/codex.py` / `sources/claude.py` as templates.
2. **New storage backend**: implement `Store` in `storage/`. `Indexer`,
   `KeywordRetriever`, `cli.py`, and `mcp.py` need one constructor call
   changed, nothing else.
3. **New retrieval strategy**: implement `Retriever` in `search/`. Same
   one-line swap in `cli.py`/`mcp.py`.
