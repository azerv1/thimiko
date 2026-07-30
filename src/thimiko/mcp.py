"""MCP server exposing search + context expansion over the local chat-history index.

Mirrors the semble (MinishLab/semble) idiom: a single in-package `mcp.py` using
the SDK's high-level server, async `@server.tool()` functions returning JSON
strings, and a `create_server()` / `serve()` split so the server can be driven
directly or via `thimiko mcp`. Read-only: build/update stay CLI-only operations.

(Semble imports this class as `FastMCP` from `mcp.server.fastmcp`; the installed
SDK has since renamed it to `MCPServer` in `mcp.server.mcpserver` — same API.)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

from thimiko.dto import answer_dict, iso_days_ago
from thimiko.search import KeywordRetriever
from thimiko.storage import SqliteStore

_INSTRUCTIONS = (
    "Search local Codex, Claude Code, and GitHub Copilot chat history. Call search_chats with a "
    "focused query first; it returns ranked snippets with session_id/turn_id and "
    "provenance (source file + line). Use get_turn to expand a promising hit into "
    "its neighboring turns, or get_session for a whole conversation's turns."
)


def create_server(db_path: Path) -> MCPServer:
    store = SqliteStore(db_path)
    retriever = KeywordRetriever(store)
    server = MCPServer("thimiko", instructions=_INSTRUCTIONS)

    @server.tool()
    async def search_chats(
        query: Annotated[str, Field(description="Text to search for across chat history.")],
        source: Annotated[
            str | None,
            Field(description="Restrict to 'codex', 'claude', or 'copilot'; omit for all."),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100, description="Maximum results to return.")] = 10,
        days: Annotated[
            int | None, Field(description="Only turns asked in the last N days; omit for all time.")
        ] = None,
        raw_fts: Annotated[
            bool, Field(description="Treat query as raw SQLite FTS5 MATCH syntax.")
        ] = False,
    ) -> str:
        """Ranked full-text search over indexed chat turns.

        Each result carries title, source, model, a relative `when`, cleaned
        snippet, and provenance (source file + line).
        """
        since = iso_days_ago(days) if days else None
        results = retriever.search(query, source=source, limit=limit, raw_fts=raw_fts, since=since)
        payload = {
            "query": query,
            "days": days,
            "count": len(results),
            "results": [answer_dict(result) for result in results],
        }
        return json.dumps(payload, ensure_ascii=False)

    @server.tool()
    async def get_session(
        session_id: Annotated[
            str, Field(description="Session id returned by search_chats, e.g. 'codex:abc123'.")
        ],
    ) -> str:
        """Session header plus all of its turns, in order."""
        session = store.get_session(session_id)
        if session is None:
            return json.dumps({"error": f"unknown session_id: {session_id}"})
        return json.dumps(session, ensure_ascii=False)

    @server.tool()
    async def get_turn(
        session_id: Annotated[str, Field(description="Session id containing the turn.")],
        turn_id: Annotated[str, Field(description="Turn id returned by search_chats.")],
        neighbors: Annotated[
            int, Field(ge=0, le=10, description="Turns to include before and after.")
        ] = 1,
    ) -> str:
        """A turn's chunks plus neighboring turns, each with provenance."""
        turn = retriever.expand(session_id, turn_id, neighbors)
        if turn is None:
            return json.dumps({"error": f"unknown turn_id: {turn_id} in session {session_id}"})
        return json.dumps(turn, ensure_ascii=False)

    return server


async def serve(db_path: Path) -> None:
    server = create_server(db_path)
    await server.run_stdio_async()


def run_stdio(db_path: Path) -> None:
    asyncio.run(serve(db_path))
