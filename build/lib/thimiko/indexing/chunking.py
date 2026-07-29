"""Turn -> `SearchDocument` chunking for the search index."""

from __future__ import annotations

from thimiko.dto import SearchDocument
from thimiko.models import Session

MAX_CHUNK_CHARS = 8000
CHUNK_OVERLAP_CHARS = 800


def _chunk_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind("\n", start + max_chars // 2, end)
            if boundary < 0:
                boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def documents_for_session(session: Session) -> list[SearchDocument]:
    """Concatenate each turn's searchable messages into chunked retrieval documents."""
    documents: list[SearchDocument] = []
    for turn in session.turns():
        messages = turn.searchable_messages()
        if not messages:
            continue
        blocks = [
            f"[{(message.role or 'unknown').upper()}]\n{message.text}" for message in messages
        ]
        text = "\n\n".join(blocks)
        timestamps = [message.timestamp for message in messages if message.timestamp]
        provenance = [
            {
                "event_id": message.id,
                "path": message.provenance.path,
                "line": message.provenance.line,
                "role": message.role,
            }
            for message in messages
        ]
        for chunk_index, chunk in enumerate(_chunk_text(text)):
            documents.append(
                SearchDocument(
                    id=f"{turn.id}:chunk:{chunk_index:03d}",
                    session_id=session.id,
                    turn_id=turn.id,
                    chunk_index=chunk_index,
                    source=session.source,
                    title=session.title,
                    cwd=session.cwd,
                    git_branch=session.git_branch,
                    started_at=min(timestamps) if timestamps else None,
                    ended_at=max(timestamps) if timestamps else None,
                    text=chunk,
                    provenance=provenance,
                )
            )
    return documents
