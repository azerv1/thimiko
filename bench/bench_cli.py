"""Repeatable benchmark runner for thimiko's build/update/search workloads.

Usage:

    uv run python bench/bench_cli.py --size small
    uv run python bench/bench_cli.py --sessions 500 --turns 6 --queries 50 --out run1.json

Fixtures are synthetic Codex-format sessions, generated fresh into a temp
directory each run (deterministic and seeded) — nothing here touches real
chat history.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from thimiko.indexing import Indexer
from thimiko.search import KeywordRetriever
from thimiko.storage import SqliteStore

_SIZES: dict[str, tuple[int, int]] = {
    "small": (50, 4),
    "medium": (500, 6),
}

_WORDS = (
    "the quick brown fox jumps over lazy dog while refactoring sqlite index "
    "and searching turn documents for relevant context across many sessions "
    "profile build update search commit transaction batch parser latency"
).split()


def _lorem(rng: random.Random, word_count: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(word_count))


def _session_records(rng: random.Random, session_id: str, turns: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"session_id": session_id, "cwd": "C:/repo"},
        }
    ]
    for turn in range(turns):
        records.append(
            {
                "timestamp": f"2026-01-01T{turn // 60:02d}:{turn % 60:02d}:00Z",
                "type": "turn_context",
                "payload": {"turn_id": f"turn{turn}"},
            }
        )
        records.append(
            {
                "timestamp": f"2026-01-01T{turn // 60:02d}:{turn % 60:02d}:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": _lorem(rng, 15)}],
                },
            }
        )
        records.append(
            {
                "timestamp": f"2026-01-01T{turn // 60:02d}:{turn % 60:02d}:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": _lorem(rng, 60)}],
                },
            }
        )
    return records


def generate_fixtures(
    directory: Path, *, session_count: int, turns_per_session: int, seed: int
) -> list[Path]:
    """Write `session_count` synthetic Codex-format JSONL files under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: list[Path] = []
    for index in range(session_count):
        session_id = f"bench-{index:05d}"
        path = directory / f"rollout-{session_id}.jsonl"
        records = _session_records(rng, session_id, turns_per_session)
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        paths.append(path)
    return paths


@dataclass(kw_only=True)
class BenchResult:
    sessions: int
    turns_per_session: int
    build_seconds: float
    build_sessions_per_second: float
    noop_update_seconds: float
    changed_file_update_seconds: float
    search_p50_ms: float
    search_p95_ms: float
    db_bytes: int


def run(*, session_count: int, turns_per_session: int, queries: int, seed: int) -> BenchResult:
    with tempfile.TemporaryDirectory(prefix="thimiko-bench-") as tmp:
        root = Path(tmp)
        fixtures_dir = root / "fixtures"
        db_path = root / "thimiko.sqlite"

        paths = generate_fixtures(
            fixtures_dir,
            session_count=session_count,
            turns_per_session=turns_per_session,
            seed=seed,
        )

        store = SqliteStore(db_path)
        try:
            indexer = Indexer(store)

            build_start = time.perf_counter()
            indexer.build([fixtures_dir], forced_source="codex")
            build_seconds = time.perf_counter() - build_start

            noop_start = time.perf_counter()
            indexer.update([fixtures_dir], forced_source="codex")
            noop_seconds = time.perf_counter() - noop_start

            # Touch one file so exactly one session is reparsed and re-upserted.
            changed = paths[0]
            changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            changed_start = time.perf_counter()
            indexer.update([fixtures_dir], forced_source="codex")
            changed_seconds = time.perf_counter() - changed_start

            retriever = KeywordRetriever(store)
            rng = random.Random(seed)
            latencies_ms: list[float] = []
            for _ in range(queries):
                query = " ".join(rng.sample(_WORDS, 2))
                query_start = time.perf_counter()
                retriever.search(query, limit=10)
                latencies_ms.append((time.perf_counter() - query_start) * 1000)
        finally:
            store.close()

        latencies_ms.sort()
        return BenchResult(
            sessions=session_count,
            turns_per_session=turns_per_session,
            build_seconds=build_seconds,
            build_sessions_per_second=session_count / build_seconds if build_seconds else 0.0,
            noop_update_seconds=noop_seconds,
            changed_file_update_seconds=changed_seconds,
            search_p50_ms=statistics.median(latencies_ms) if latencies_ms else 0.0,
            search_p95_ms=(
                latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0
            ),
            db_bytes=db_path.stat().st_size,
        )


def _print_result(result: BenchResult) -> None:
    print(f"sessions={result.sessions} turns/session={result.turns_per_session}")
    print(
        f"  build:              {result.build_seconds:.3f}s "
        f"({result.build_sessions_per_second:.1f} sessions/s)"
    )
    print(f"  update (no-op):     {result.noop_update_seconds:.3f}s")
    print(f"  update (1 changed): {result.changed_file_update_seconds:.3f}s")
    print(f"  search p50:         {result.search_p50_ms:.2f}ms")
    print(f"  search p95:         {result.search_p95_ms:.2f}ms")
    print(f"  db size:            {result.db_bytes:,} bytes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run thimiko benchmarks against synthetic data")
    parser.add_argument("--size", choices=sorted(_SIZES), help="Preset session/turn count")
    parser.add_argument("--sessions", type=int, help="Override session count")
    parser.add_argument("--turns", type=int, help="Override turns per session")
    parser.add_argument("--queries", type=int, default=30, help="Search queries to time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="Write JSON result to bench/results/<name>")
    args = parser.parse_args(argv)

    preset = _SIZES[args.size or "small"]
    session_count = args.sessions if args.sessions is not None else preset[0]
    turns_per_session = args.turns if args.turns is not None else preset[1]

    result = run(
        session_count=session_count,
        turns_per_session=turns_per_session,
        queries=args.queries,
        seed=args.seed,
    )
    _print_result(result)

    if args.out:
        out_path = Path(__file__).parent / "results" / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
