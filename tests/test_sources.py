from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thimiko.models import Message
from thimiko.sources.claude import ClaudeSource
from thimiko.sources.codex import CodexSource
from thimiko.sources.copilot import CopilotSource


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


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
