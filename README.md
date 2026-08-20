# thimiko

**thimiko** is **θυμικό**.

It makes your local AI coding chats searchable. It reads Codex, Claude Code,
GitHub Copilot, Gemini CLI, Cursor, and OpenCode history, builds a local SQLite
index, and lets you search it from the terminal or through MCP.

Your original chat files are left alone.

## Why

Useful things get buried in old chats: fixes, decisions, failed approaches,
commands, and the one detail you need six weeks later.

With thimiko, you can ask things like:

- "Have I leaked a secret that starts with `0xSecret` in our chats?"
- "Did we talk about X last week?"
- "Can you collect everything from our chats about Y and make me a study plan?"
- "Why did we decide not to use Redis for the job queue?"
- "Find the chat where we fixed the flaky migration test."
- "What command did we use to recover the broken production index?"
- "Have I pasted any customer names, internal URLs, or production hostnames?"
- "Show me previous attempts at this error, including the surrounding messages."

It is especially handy for secret audits, incident reconstruction, architecture
decision archaeology, compliance checks, recurring bugs, and picking up an old
project without starting from zero.

## Install

```powershell
uv tool install "thimiko[mcp]"
```

For local development:

```powershell
uv tool install -e ".[mcp]"
```

## Use it

Run `thimiko` by itself to see the available commands.

Build the index once:

```powershell
thimiko build
```

Then keep it fresh and search it:

```powershell
thimiko update
thimiko search "database migration"
thimiko search "permission denied" --source codex --limit 5
thimiko search "rate limit" --days 10 --text
```

See which providers this installation supports, where it looks for their history, and
— with `-v` — how many chats are indexed versus present on disk:

```powershell
thimiko list
thimiko list -v --text
```

By default, thimiko reads:

- Codex: `~/.codex/sessions/**/*.jsonl`
- Claude Code: `~/.claude/projects/**/*.jsonl`
- GitHub Copilot: VS Code's `workspaceStorage/*/chatSessions/`
- Gemini CLI: `~/.gemini/tmp/*/chats/`
- Cursor: `%APPDATA%\Cursor\User\globalStorage\state.vscdb` (SQLite — one file, many chats)
- OpenCode: `%USERPROFILE%\.local\share\opencode\opencode.db` on Windows;
  `$XDG_DATA_HOME/opencode/opencode.db` when `XDG_DATA_HOME` is set; otherwise
  `~/.local/share/opencode/opencode.db` (SQLite — one file, many chats)

You can also pass a file or directory directly to `build` or `update`.
Set `THIMIKO_OPENCODE_ROOT` to override OpenCode discovery with either its data
directory or the database file.

`opencode.db` is always an input source and is opened read-only. Do not pass it
as Thimiko's `--db`: that option names Thimiko's separate derived search index.

## MCP

Run the read-only MCP server with:

```powershell
thimiko mcp
```

It exposes three tools: `search_chats`, `get_turn`, and `get_session`. Point your
MCP client at `thimiko mcp`, then your assistant can search old chats and open
the relevant surrounding conversation.

The separate Thimiko index stays local. Its default location is
`%LOCALAPPDATA%\thimiko\thimiko.sqlite` on Windows.

For implementation details and instructions for adding another chat source,
see [ARCHITECTURE.md](ARCHITECTURE.md).
