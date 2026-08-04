"""Indexer: build a fresh index, or incrementally update an existing one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thimiko.sources import ChatSource, iter_session_files
from thimiko.sources import detect as detect_source
from thimiko.storage import Store

from .chunking import documents_for_session


@dataclass(kw_only=True)
class BuildResult:
    sessions: int
    documents: int


@dataclass(kw_only=True)
class UpdateResult:
    added: int
    updated: int
    skipped: int
    pruned: int


class Indexer:
    """Drives one `Store` through the ingest pipeline using registered `ChatSource`s."""

    def __init__(self, store: Store, sources: list[ChatSource] | None = None) -> None:
        self.store = store
        self._sources = sources

    def _detect(self, path: Path, forced_source: str | None) -> ChatSource | None:
        if self._sources is not None:
            if forced_source:
                return next((s for s in self._sources if s.name == forced_source), None)
            return next((s for s in self._sources if s.matches(path)), None)
        return detect_source(path, forced_source)

    def build(self, paths: list[Path], forced_source: str | None = None) -> BuildResult:
        files = iter_session_files(paths or None)
        if not files:
            joined = ", ".join(str(path) for path in paths) if paths else "default roots"
            raise FileNotFoundError(f"No chat history files found under: {joined}")

        self.store.create_schema(reset=True)
        session_count = 0
        document_count = 0
        for file_path in files:
            source = self._detect(file_path, forced_source)
            if source is None:
                continue
            sessions = source.parse_all(file_path)
            for session in sessions:
                documents = documents_for_session(session)
                self.store.upsert_session(session, documents)
                document_count += len(documents)
            stat = file_path.stat()
            self.store.record_file(
                str(file_path), stat.st_mtime, stat.st_size, [s.id for s in sessions]
            )
            session_count += len(sessions)
        return BuildResult(sessions=session_count, documents=document_count)

    def update(
        self,
        paths: list[Path],
        forced_source: str | None = None,
        *,
        prune: bool = False,
    ) -> UpdateResult:
        self.store.create_schema(reset=False)
        files = iter_session_files(paths or None)
        known = self.store.known_files()
        seen_paths: set[str] = set()
        added = 0
        updated = 0
        skipped = 0

        for file_path in files:
            path_key = str(file_path)
            seen_paths.add(path_key)
            stat = file_path.stat()
            previous = self.store.file_state(path_key)
            if previous is not None and previous == (stat.st_mtime, stat.st_size):
                skipped += 1
                continue

            source = self._detect(file_path, forced_source)
            if source is None:
                continue
            sessions = source.parse_all(file_path)

            prior_session_ids = known.get(path_key)
            if prior_session_ids:
                for prior_session_id in prior_session_ids:
                    self.store.delete_session(prior_session_id)
                updated += 1
            else:
                added += 1

            for session in sessions:
                self.store.upsert_session(session, documents_for_session(session))
            self.store.record_file(path_key, stat.st_mtime, stat.st_size, [s.id for s in sessions])

        pruned = 0
        if prune:
            for path_key, session_ids in known.items():
                if path_key not in seen_paths and not Path(path_key).exists():
                    for session_id in session_ids:
                        self.store.delete_session(session_id)
                    self.store.forget_file(path_key)
                    pruned += 1

        return UpdateResult(added=added, updated=updated, skipped=skipped, pruned=pruned)
