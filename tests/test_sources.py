from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from thimiko.models import Message, Reasoning, ToolCall
from thimiko.sources import detect
from thimiko.sources.claude import ClaudeSource
from thimiko.sources.codex import CodexSource
from thimiko.sources.copilot import CopilotSource
from thimiko.sources.cursor import CursorSource
from thimiko.sources.gemini import GeminiSource


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def write_vscdb(path: Path, rows: dict[str, Any], table: str = "cursorDiskKV") -> None:
    """Write a Cursor-shaped key/value SQLite database (values stored as JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(f"CREATE TABLE {table} (key TEXT PRIMARY KEY, value TEXT)")
        connection.executemany(
            f"INSERT INTO {table} VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in rows.items()],
        )
        connection.commit()
    finally:
        connection.close()


def _cursor_rows() -> dict[str, Any]:
    return {
        "composerData:c1": {
            "name": "Refactor the parser",
            "createdAt": 1767225600000,
            "lastUpdatedAt": 1767225660000,
            "fullConversationHeadersOnly": [
                {"bubbleId": "b1", "type": 1},
                {"bubbleId": "b2", "type": 2},
                {"bubbleId": "gone", "type": 2},
            ],
        },
        "bubbleId:c1:b1": {"type": 1, "text": "where is the tokenizer"},
        "bubbleId:c1:b2": {
            "type": 2,
            "text": "in lexer.py",
            "thinking": "checking the imports",
            "toolFormerData": {"name": "read_file", "params": {"path": "lexer.py"}},
            "modelType": "claude-opus-5",
        },
    }


def test_codex_deduplicates_display_events_and_links_tool_result(tmp_path: Path) -> None:
    records = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"session_id": "c1", "cwd": "C:/repo"},
        },
        {
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "turn_context",
            "payload": {"turn_id": "t1", "model": "gpt-test"},
        },
        {
            "timestamp": "2026-01-01T00:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "find the parser"}],
            },
        },
        {
            "timestamp": "2026-01-01T00:00:02Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "find the parser"},
        },
        {
            "timestamp": "2026-01-01T00:00:03Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": "fc1",
                "call_id": "call1",
                "name": "search",
                "arguments": '{"q":"parser"}',
            },
        },
        {
            "timestamp": "2026-01-01T00:00:04Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call1",
                "output": "found canonical.py",
            },
        },
        {
            "timestamp": "2026-01-01T00:00:05Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "It is in canonical.py"}],
            },
        },
        {
            "timestamp": "2026-01-01T00:00:05Z",
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "It is in canonical.py"},
        },
    ]
    path = tmp_path / "codex.jsonl"
    write_jsonl(path, records)
    session = CodexSource().parse(path)

    assert session.id == "codex:c1"
    assert session.source_stream_id == "c1"
    assert session.cwd == "C:/repo"
    assert session.model == "gpt-test"
    assert [event.kind for event in session.events] == [
        "message",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert sum(event.searchable for event in session.events) == 2
    assert session.events[2].parent_id == session.events[1].id
    assert all(event.turn_id == "codex:c1:turn:t1" for event in session.events)


def test_codex_uses_thread_id_for_subagent_stream(tmp_path: Path) -> None:
    records = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {
                "session_id": "parent1",
                "id": "thread2",
                "parent_thread_id": "parent1",
                "thread_source": "subagent",
                "agent_path": "/root/reader",
            },
        },
        {
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "result"}],
            },
        },
    ]
    path = tmp_path / "codex.jsonl"
    write_jsonl(path, records)
    session = CodexSource().parse(path)

    assert session.id == "codex:thread2"
    assert session.source_session_id == "parent1"
    assert session.source_stream_id == "thread2"
    assert session.parent_session_id == "codex:parent1"
    assert session.agent_id == "/root/reader"


def test_claude_preserves_fragment_order_and_tool_linkage(tmp_path: Path) -> None:
    records: list[dict[str, Any]] = [
        {"type": "ai-title", "aiTitle": "Parser work", "sessionId": "a1"},
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "sessionId": "a1",
            "promptId": "p1",
            "timestamp": "2026-01-01T00:00:00Z",
            "cwd": "C:/repo",
            "gitBranch": "main",
            "message": {"role": "user", "content": "find the parser"},
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "u1",
            "sessionId": "a1",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {
                "id": "msg1",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "I will inspect it."}],
            },
        },
        {
            "type": "assistant",
            "uuid": "a3",
            "parentUuid": "a2",
            "sessionId": "a1",
            "timestamp": "2026-01-01T00:00:02Z",
            "message": {
                "id": "msg1",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool1",
                        "name": "Search",
                        "input": {"query": "parser"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a3",
            "sessionId": "a1",
            "timestamp": "2026-01-01T00:00:03Z",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool1", "content": "canonical.py"}
                ],
            },
        },
        {
            "type": "user",
            "uuid": "meta1",
            "parentUuid": "u2",
            "sessionId": "a1",
            "timestamp": "2026-01-01T00:00:04Z",
            "isMeta": True,
            "message": {"role": "user", "content": "hidden bootstrap"},
        },
    ]
    path = tmp_path / "claude.jsonl"
    write_jsonl(path, records)
    session = ClaudeSource().parse(path)

    assert session.title == "Parser work"
    assert session.cwd == "C:/repo"
    assert session.git_branch == "main"
    assert session.model == "claude-test"
    assert [event.kind for event in session.events] == [
        "message",
        "message",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert session.events[3].parent_id == session.events[2].id
    assert session.events[-1].searchable is False
    assert sum(event.searchable for event in session.events) == 2
    assert all(event.turn_id == "claude:a1:turn:p1" for event in session.events[:4])


def test_claude_namespaces_subagent_stream_below_parent_session(tmp_path: Path) -> None:
    records = [
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "parent1",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "subagent answer"}],
            },
        }
    ]
    subagents = tmp_path / "subagents"
    subagents.mkdir()
    path = subagents / "agent-abc.jsonl"
    write_jsonl(path, records)
    session = ClaudeSource().parse(path)

    assert session.id == "claude:parent1:agent:agent-abc"
    assert session.parent_session_id == "claude:parent1"
    assert session.agent_id == "agent-abc"
    assert session.source_stream_id == "agent-abc"
    turn_id = session.events[0].turn_id
    assert turn_id is not None
    assert turn_id.startswith(session.id)


def _copilot_request(
    request_id: str, user_text: str, response: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "requestId": request_id,
        "timestamp": 1781883475448,
        "modelId": "copilot/auto",
        "message": {"text": user_text},
        "response": response,
    }


def test_copilot_parses_json_snapshot_and_orders_events(tmp_path: Path) -> None:
    snapshot = {
        "version": 3,
        "sessionId": "cs1",
        "responderUsername": "GitHub Copilot",
        "creationDate": 1781883470000,
        "lastMessageDate": 1781883480000,
        "requests": [
            _copilot_request(
                "req1",
                "how do I fix the dns import error?",
                [
                    {"kind": "thinking", "value": "the dns module is missing"},
                    {"value": "Run pip install dnspython to fix it."},
                    {
                        "kind": "toolInvocationSerialized",
                        "toolName": "run_in_terminal",
                        "invocationMessage": "Running pip install",
                    },
                ],
            )
        ],
    }
    path = tmp_path / "cs1.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    session = CopilotSource().parse(path)

    assert session.id == "copilot:cs1"
    assert session.source == "copilot"
    assert session.model == "copilot/auto"
    assert session.started_at is not None and session.started_at.endswith("Z")
    assert [event.kind for event in session.events] == [
        "message",
        "reasoning",
        "tool_call",
        "message",
    ]
    assert sum(event.searchable for event in session.events) == 2
    assistant = session.events[-1]
    assert isinstance(assistant, Message)
    assert assistant.text == "Run pip install dnspython to fix it."
    assert all(event.turn_id == "copilot:cs1:turn:req1" for event in session.events)


def test_copilot_reconstructs_jsonl_base_plus_patches(tmp_path: Path) -> None:
    base = {
        "version": 3,
        "sessionId": "cs2",
        "responderUsername": "GitHub Copilot",
        "creationDate": 1781883470000,
        "requests": [
            _copilot_request(
                "req1",
                "explain the flag enum",
                [{"kind": "thinking", "value": "flags use powers of two"}],
            )
        ],
    }
    records: list[dict[str, Any]] = [
        {"kind": 0, "v": base},
        {"kind": 1, "k": ["requests", 0, "response", 1], "v": {"value": "Each member is a bit."}},
        {"kind": 1, "k": ["customTitle"], "v": "Flag enum"},
        {"kind": 2, "v": None},
    ]
    path = tmp_path / "cs2.jsonl"
    write_jsonl(path, records)
    session = CopilotSource().parse(path)

    assert session.id == "copilot:cs2"
    assert session.title == "Flag enum"
    assistant = session.events[-1]
    assert isinstance(assistant, Message)
    assert assistant.text == "Each member is a bit."
    assert session.searchable_messages()[-1].text == "Each member is a bit."


def test_copilot_matches_own_files_and_rejects_claude(tmp_path: Path) -> None:
    copilot = tmp_path / "cs.json"
    copilot.write_text(
        json.dumps({"sessionId": "cs", "requests": [], "responderUsername": "GitHub Copilot"}),
        encoding="utf-8",
    )
    claude = tmp_path / "claude.jsonl"
    write_jsonl(claude, [{"type": "user", "uuid": "u1", "message": {"role": "user"}}])

    source = CopilotSource()
    assert source.matches(copilot) is True
    assert source.matches(claude) is False


def test_copilot_discovers_chat_sessions_tree(tmp_path: Path) -> None:
    sessions = tmp_path / "workspaceStorage" / "abc123" / "chatSessions"
    sessions.mkdir(parents=True)
    (sessions / "a.json").write_text("{}", encoding="utf-8")
    (sessions / "b.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "stray.json").write_text("{}", encoding="utf-8")

    found = CopilotSource().discover(tmp_path)
    assert found == sorted([sessions / "a.json", sessions / "b.jsonl"])


def test_gemini_parses_legacy_json_with_thoughts_and_tools(tmp_path: Path) -> None:
    project = tmp_path / "project-hash"
    chats = project / "chats"
    chats.mkdir(parents=True)
    (project / ".project_root").write_text("C:/repo\n", encoding="utf-8")
    snapshot = {
        "sessionId": "gs1",
        "projectHash": "project-hash",
        "startTime": "2026-07-01T10:00:00Z",
        "lastUpdated": "2026-07-01T10:00:03Z",
        "summary": "DNS repair",
        "messages": [
            {
                "id": "user-1",
                "timestamp": "2026-07-01T10:00:01Z",
                "type": "user",
                "content": [{"text": "why does the dns import fail?"}],
            },
            {
                "id": "gemini-1",
                "timestamp": "2026-07-01T10:00:02Z",
                "type": "gemini",
                "model": "gemini-test",
                "content": [{"text": "Install dnspython."}],
                "thoughts": [
                    {
                        "subject": "Dependency",
                        "description": "The module is not installed.",
                        "timestamp": "2026-07-01T10:00:01Z",
                    }
                ],
                "toolCalls": [
                    {
                        "id": "tool-1",
                        "name": "run_shell_command",
                        "args": {"command": "pip install dnspython"},
                        "result": [{"text": "installed"}],
                        "status": "success",
                        "timestamp": "2026-07-01T10:00:02Z",
                    }
                ],
            },
            {
                "id": "info-1",
                "timestamp": "2026-07-01T10:00:03Z",
                "type": "info",
                "content": "session restored",
            },
        ],
    }
    path = chats / "session-gs1.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    session = GeminiSource().parse(path)

    assert session.id == "gemini:gs1"
    assert session.source == "gemini"
    assert session.model == "gemini-test"
    assert session.title == "DNS repair"
    assert session.cwd == "C:/repo"
    assert [event.kind for event in session.events] == [
        "message",
        "reasoning",
        "tool_call",
        "tool_result",
        "message",
        "message",
    ]
    assert session.events[3].parent_id == session.events[2].id
    assert sum(event.searchable for event in session.events) == 2
    assert all(event.turn_id == "gemini:gs1:turn:user-1" for event in session.events)


def test_gemini_reconstructs_jsonl_checkpoint_and_rewind(tmp_path: Path) -> None:
    user_1 = {
        "id": "user-1",
        "timestamp": "2026-07-02T10:00:01Z",
        "type": "user",
        "content": "explain the migration",
    }
    gemini_1 = {
        "id": "gemini-1",
        "timestamp": "2026-07-02T10:00:02Z",
        "type": "gemini",
        "content": "The migration copies rows.",
    }
    user_2 = {
        "id": "user-2",
        "timestamp": "2026-07-02T10:00:03Z",
        "type": "user",
        "content": "delete this branch",
    }
    gemini_2 = {
        "id": "gemini-2",
        "timestamp": "2026-07-02T10:00:04Z",
        "type": "gemini",
        "content": "This answer is rewound.",
    }
    records: list[dict[str, Any]] = [
        {
            "sessionId": "gs2",
            "projectHash": "hash-2",
            "startTime": "2026-07-02T10:00:00Z",
            "lastUpdated": "2026-07-02T10:00:00Z",
        },
        user_1,
        gemini_1,
        {
            "$set": {
                "messages": [user_1, gemini_1, user_2, gemini_2],
                "summary": "Migration notes",
                "lastUpdated": "2026-07-02T10:00:05Z",
            }
        },
        {"$rewindTo": "user-2"},
    ]
    path = tmp_path / "session-gs2.jsonl"
    write_jsonl(path, records)

    session = GeminiSource().parse(path)

    assert session.id == "gemini:gs2"
    assert session.title == "Migration notes"
    assert session.ended_at == "2026-07-02T10:00:05Z"
    assert [message.text for message in session.searchable_messages()] == [
        "explain the migration",
        "The migration copies rows.",
    ]
    assert all(event.provenance.line == 4 for event in session.events)


def test_gemini_matches_discovers_and_namespaces_subagents(tmp_path: Path) -> None:
    chats = tmp_path / "hash" / "chats"
    subagents = chats / "parent-session"
    subagents.mkdir(parents=True)
    main_path = chats / "session-main.json"
    main_path.write_text(
        json.dumps(
            {
                "sessionId": "main",
                "projectHash": "hash",
                "startTime": "2026-07-03T10:00:00Z",
                "messages": [],
            }
        ),
        encoding="utf-8",
    )
    subagent_path = subagents / "agent-1.jsonl"
    write_jsonl(
        subagent_path,
        [
            {
                "sessionId": "agent-1",
                "projectHash": "hash",
                "startTime": "2026-07-03T10:00:00Z",
                "kind": "subagent",
            },
            {
                "id": "gemini-1",
                "timestamp": "2026-07-03T10:00:01Z",
                "type": "gemini",
                "content": "subagent result",
            },
        ],
    )
    (tmp_path / "stray.json").write_text("{}", encoding="utf-8")

    source = GeminiSource()
    assert source.discover(tmp_path) == sorted([main_path, subagent_path])
    assert source.matches(main_path) is True
    assert source.matches(subagent_path) is True
    detected = detect(main_path)
    assert detected is not None
    assert detected.name == "gemini"

    session = source.parse(subagent_path)
    assert session.id == "gemini:parent-session:agent:agent-1"
    assert session.parent_session_id == "gemini:parent-session"
    assert session.agent_id == "agent-1"
    assert session.events[0].turn_id == f"{session.id}:turn:fallback-1"


def test_cursor_parses_composer_bubbles_in_header_order(tmp_path: Path) -> None:
    db_path = tmp_path / "globalStorage" / "state.vscdb"
    write_vscdb(db_path, _cursor_rows())

    session = CursorSource().parse(db_path)

    assert session.id == "cursor:c1"
    assert session.title == "Refactor the parser"
    assert session.started_at == "2026-01-01T00:00:00.000Z"
    assert session.ended_at == "2026-01-01T00:01:00.000Z"
    assert session.model == "claude-opus-5"

    messages = [event for event in session.events if isinstance(event, Message)]
    assert [(m.role, m.text) for m in messages] == [
        ("user", "where is the tokenizer"),
        ("assistant", "in lexer.py"),
    ]
    # The assistant bubble joins the preceding user bubble's turn.
    assert messages[0].turn_id == messages[1].turn_id == "cursor:c1:turn:b1"
    assert messages[1].provenance.native_id == "bubbleId:c1:b2"
    # A header whose bubble body was pruned is skipped, not faked.
    assert len(messages) == 2


def test_cursor_keeps_reasoning_and_tools_out_of_the_search_corpus(tmp_path: Path) -> None:
    db_path = tmp_path / "globalStorage" / "state.vscdb"
    write_vscdb(db_path, _cursor_rows())

    session = CursorSource().parse(db_path)

    assert any(isinstance(event, Reasoning) for event in session.events)
    tool_calls = [event for event in session.events if isinstance(event, ToolCall)]
    assert [call.tool_name for call in tool_calls] == ["read_file"]
    assert all(message.searchable for message in session.searchable_messages())
    assert len(session.searchable_messages()) == 2


def test_cursor_yields_one_session_per_composer(tmp_path: Path) -> None:
    rows = _cursor_rows()
    rows["composerData:c2"] = {
        "name": "Second chat",
        "createdAt": 1767312000000,
        "fullConversationHeadersOnly": [{"bubbleId": "b9", "type": 1}],
    }
    rows["bubbleId:c2:b9"] = {"type": 1, "text": "unrelated question"}
    db_path = tmp_path / "globalStorage" / "state.vscdb"
    write_vscdb(db_path, rows)

    sessions = CursorSource().parse_all(db_path)

    assert [session.id for session in sessions] == ["cursor:c1", "cursor:c2"]
    assert sessions[1].title == "Second chat"


def test_cursor_discovers_and_matches_only_its_own_database(tmp_path: Path) -> None:
    db_path = tmp_path / "globalStorage" / "state.vscdb"
    write_vscdb(db_path, _cursor_rows())
    not_sqlite = tmp_path / "globalStorage" / "other.vscdb"
    not_sqlite.write_text("not a database", encoding="utf-8")
    jsonl_path = tmp_path / "codex.jsonl"
    write_jsonl(jsonl_path, [{"type": "session_meta", "payload": {"session_id": "x"}}])

    source = CursorSource()
    assert source.discover(tmp_path) == [db_path]
    assert source.matches(db_path) is True
    assert source.matches(not_sqlite) is False
    assert source.matches(jsonl_path) is False
    detected = detect(db_path)
    assert detected is not None
    assert detected.name == "cursor"


def test_cursor_maps_workspace_folder_to_cwd(tmp_path: Path) -> None:
    db_path = tmp_path / "globalStorage" / "state.vscdb"
    write_vscdb(db_path, _cursor_rows())
    workspace = tmp_path / "workspaceStorage" / "abc123"
    write_vscdb(
        workspace / "state.vscdb",
        {"composer.composerData": {"allComposers": [{"composerId": "c1"}]}},
        table="ItemTable",
    )
    (workspace / "workspace.json").write_text(
        json.dumps({"folder": "file:///c%3A/Users/dev/repo"}), encoding="utf-8"
    )

    session = CursorSource().parse(db_path)

    assert session.cwd == "c:/Users/dev/repo"
