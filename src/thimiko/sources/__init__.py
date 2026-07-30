"""Registry of pluggable `ChatSource` adapters.

To support a new chat-history provider, implement `ChatSource` in a new module
and call `register()` with an instance — the indexer, CLI, and MCP server pick
it up automatically through `all_sources()` / `detect()` / `iter_session_files()`.
"""

from __future__ import annotations

from pathlib import Path

from .base import ChatSource
from .claude import ClaudeSource
from .codex import CodexSource
from .copilot import CopilotSource

_REGISTRY: list[ChatSource] = [CodexSource(), ClaudeSource(), CopilotSource()]


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
    """Discover session files under `paths`, or every source's default roots.

    Each source discovers its own files (see `ChatSource.discover`), so providers
    that don't store plain `*.jsonl` are found too. Per-file routing still goes
    through `detect()`, so overlapping globs are deduped harmlessly.
    """
    if paths:
        roots = [(source, path) for source in _REGISTRY for path in paths]
    else:
        roots = [(source, root) for source in _REGISTRY for root in source.default_roots()]
    files: set[Path] = set()
    for source, root in roots:
        if root.exists():
            files.update(source.discover(root))
    return sorted(files)


def resolve_input_paths(paths: list[str]) -> list[Path]:
    """Return scan paths, defaulting to every registered source's roots when none given."""
    if paths:
        return [Path(path) for path in paths]
    return default_roots()


__all__ = [
    "ChatSource",
    "ClaudeSource",
    "CodexSource",
    "CopilotSource",
    "all_sources",
    "default_roots",
    "detect",
    "iter_session_files",
    "register",
    "resolve_input_paths",
]
