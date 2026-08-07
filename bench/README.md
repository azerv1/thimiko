# bench

Repeatable performance measurements for `thimiko build`/`update`/`search`,
kept separate from `tests/` — these measure speed, not correctness.

## Run

```powershell
uv run python bench/bench_cli.py --size small
uv run python bench/bench_cli.py --size medium
uv run python bench/bench_cli.py --sessions 500 --turns 6 --queries 50 --out run1.json
```

`--size small` (50 sessions) and `--size medium` (500 sessions) are presets;
`--sessions`/`--turns` override them. `--out <name>` writes the result as JSON
to `bench/results/<name>` (gitignored — these are local run artifacts, not
checked in).

## What it measures

Each run generates synthetic Codex-format session files (seeded and
deterministic — no real chat history involved), then times:

- `build`: full ingest from scratch
- `update` (no-op): incremental update with nothing changed
- `update` (1 changed file): incremental update after touching one session file
- `search`: p50/p95 latency over randomized keyword queries

## Profiling

For a CPU profile of a specific workload:

```powershell
uv run python -m cProfile -o reports/build.prof bench/bench_cli.py --size medium
uv run python -m pstats reports/build.prof
```

`py-spy`/`scalene` (install via `uv sync --extra profile`) are better for
flame graphs on real runs:

```powershell
py-spy record -o reports/build.svg -- uv run python bench/bench_cli.py --size medium
```

`reports/` is gitignored; profiler output doesn't need to be checked in.
