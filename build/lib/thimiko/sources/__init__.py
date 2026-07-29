"""Registry of pluggable `ChatSource` adapters.

To support a new chat-history provider, implement `ChatSource` in a new module
and call `register()` with an instance — the indexer, CLI, and MCP server pick
it up automatically through `all_sources()` / `detect()` / `iter_session_files()`.
"""

from __future__ import annotations

from pathlib import Path

from thimiko.utils import discover_jsonl_files

from .base import ChatSource
from .claude import ClaudeSource
from .codex import CodexSource

_REGISTRY: list[ChatSource] = [CodexSource(), ClaudeSource()]


def register(source: ChatSource) -> None:
    """Add a new source so build/update/detect pick it up."""
    _REGISTRY.append(source)


def all_sources() -> list[ChatSource]:
    return list(_REGISTRY)


def detect(path: Path, forced_name: str | None = None) -> ChatSource | None:
    """Find the registered source that recognizes `path`.

    Pass `forced_name` to skip sniffing and select a source by name instead.
    """
    if forced_name:
        return next((source for source in _REGISTRY if source.name == forced_name), None)
    return next((source for source in _REGISTRY if source.matches(path)), None)


def default_roots() -> list[Path]:
    """Union of every registered source's default scan roots."""
    roots: list[Path] = []
    for source in _REGISTRY:
        roots.extend(source.default_roots())
    return roots


def iter_session_files(paths: list[Path] | None = None) -> list[Path]:
    """Discover JSONL session files under `paths`, or all default roots when omitted."""
    search_paths = paths if paths else default_roots()
    files: list[Path] = []
    for path in search_paths:
        if not path.exists():
            continue
        files.extend(discover_jsonl_files(path))
    return files


def resolve_input_paths(paths: list[str]) -> list[Path]:
    """Return scan paths, defaulting to every registered source's roots when none given."""
    if paths:
        return [Path(path) for path in paths]
    return default_roots()


__all__ = [
    "ChatSource",
    "ClaudeSource",
    "CodexSource",
    "all_sources",
    "default_roots",
    "detect",
    "iter_session_files",
    "register",
    "resolve_input_paths",
]
