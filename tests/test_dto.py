from __future__ import annotations

from thimiko.dto import SearchResult, answer_dict, clean_snippet, iso_days_ago


def test_clean_snippet_strips_role_labels_keeps_highlights() -> None:
    raw = "[USER]\nwhat is [fts5]?\n\n[ASSISTANT]\n**[FTS5]** is full-text search"
    out = clean_snippet(raw)
    assert "[USER]" not in out
    assert "[ASSISTANT]" not in out
    assert "[fts5]" in out  # match highlight preserved
    assert "  " not in out  # whitespace collapsed
    assert "\n" not in out


def test_iso_days_ago_is_a_z_timestamp_in_the_past() -> None:
    cutoff = iso_days_ago(10)
    assert cutoff.endswith("Z")
    assert cutoff < iso_days_ago(0)  # 10 days ago is earlier than now


def test_answer_dict_shape_includes_model_and_flat_provenance() -> None:
    result = SearchResult(
        id="d1",
        session_id="claude:s1",
        turn_id="claude:s1:turn:t1",
        source="claude",
        title="Some title",
        cwd="C:/repo",
        git_branch="main",
        started_at="2026-07-29T10:00:00.000Z",
        ended_at="2026-07-29T10:01:00.000Z",
        snippet="[USER] hello [world]",
        score=-1.5,
        model="claude-sonnet-5",
        provenance=[{"path": "C:/x.jsonl", "line": 42, "role": "user"}],
    )
    out = answer_dict(result)
    assert out["title"] == "Some title"
    assert out["model"] == "claude-sonnet-5"
    assert out["path"] == "C:/x.jsonl"
    assert out["line"] == 42
    assert out["snippet"] == "hello [world]"
    assert "when" in out and out["timestamp"] == "2026-07-29T10:00:00.000Z"
