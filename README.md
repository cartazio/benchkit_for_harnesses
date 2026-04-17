# BenchKit for Harnesses

Unified benchmarking toolkit for CLI coding-agent harnesses. Runs standard
long-context benchmarks (BABILong, InfiniteBench, LongBench-v2) and custom
experiments (IFEval+, bundled-bench) against any compatible CLI.

## Supported harnesses

| Harness | Invocation shape | Auth / config |
|---|---|---|
| `ohp`, `punkin`, `opencode`, `monopi`, `omp` | `BIN -p --no-session --model X @promptfile` | as per harness |
| `claude` (Claude Code) | `claude -p --model X --no-session-persistence` with stdin | `claude login` or `ANTHROPIC_API_KEY` |
| `codex` (OpenAI Codex) | `codex exec -m X -` with stdin, extracts `--output-last-message` | `codex login` |

Adding a new harness = one adapter function in `src/benchkit_for_harnesses/harnesses/dispatch.py`. See [`docs/harnesses.md`](docs/harnesses.md).

## Install

Using `uv` (recommended):

```bash
cd ~/local_dev/dynamic_science/benchkit_for_harnesses
uv sync
```

Or pip:

```bash
pip install -e ".[dev]"
```

Either installs the `benchkit` console script. If you prefer not to activate the venv (or not to install at all), every command below can be prefixed with `uv run` and executed from the repo root:

```bash
uv run benchkit probe --harness claude --model sonnet
uv run python -m benchkit_for_harnesses.ifeval.experiment run -c heavy -m sonnet -n 50
```

## Quick start

```bash
# Sanity-check a harness+model pair (one trivial prompt)
benchkit probe --harness claude --model sonnet

# Enumerate what's available
benchkit list benchmarks
benchkit list harnesses

# BABILong qa1 at 0k context, 10 items, against punkin
benchkit run --benchmark babilong --task qa1 --length 0k --limit 10 \
    --harness punkin --model anthropic/claude-opus-4-6 \
    --output ./results

# InfiniteBench passkey, 10 items, against Claude Code
benchkit run --benchmark infinitebench --task passkey --limit 10 \
    --harness claude --model sonnet \
    --output ./results

# LongBench-v2 multi-choice, 10 items, against Codex
benchkit run --benchmark longbenchv2 --limit 10 \
    --harness codex --model gpt-5.3-codex-spark \
    --output ./results

# Sub-experiments — remaining args forward to the sub-experiment CLI
benchkit ifeval run --condition heavy -m sonnet -n 50 -o ./results
benchkit ifeval compare baseline.jsonl heavy.jsonl
benchkit bundled --preset core --bundle-sizes 1,3,5 --dry-run
```

Validation is upfront — omitting `--task`/`--length` on benchmarks that
require them produces an actionable error listing available values.

## Benchmarks

| Name | Tasks | Lengths | Eval | Notes |
|---|---|---|---|---|
| `babilong` | qa1–qa10 | 0k–128k | loose bracket | Synthetic short-answer QA |
| `infinitebench` | passkey, kv_retrieval, number_string, code_run, code_debug, math_find, longdialogue_qa_eng, longbook_qa_eng | n/a | loose bracket | Streamed, some pipe-target multi-answer |
| `longbenchv2` | (single-split) | n/a | **strict** bracket | Multiple-choice A/B/C/D — strict to kill single-letter substring false positives |

All benchmarks use the **answer-bracket protocol**: the model wraps its
final answer in `{[{[ … ]}]}` markers. See [`docs/benchmarks.md`](docs/benchmarks.md)
for details on the protocol, draft markers, and strict vs loose eval.

## Python API

```python
from benchkit_for_harnesses import run_benchmark_batch

results, output_path = run_benchmark_batch(
    benchmark_name="babilong",
    harness="claude",
    model="sonnet",
    output_dir="./results",
    task="qa1",
    length="0k",
    limit=100,
)

for r in results:
    print(f"idx={r.idx} correct={r.correct} latency={r.latency_ms}ms")
print(f"Archive: {output_path}")
```

## Persistence model

Two layers, each with a different granularity:

| Layer | Granularity | Location | Contents |
|---|---|---|---|
| **Archive** | one file per `benchkit run` invocation | `$BENCHKIT_HOME/runs/{desc}_v{n}_{ts}NYC_{hash}.jsonl` | Per-item records (prompt, target, response, correct, latency). Content-hashed filename — self-certifying. |
| **Ledger** | one line per `benchkit` invocation (any subcommand) | `$BENCHKIT_HOME/ledger.jsonl` | Invocation metadata: timestamp, argv, exit code, command-specific summary (accuracy, output path, error message). |

`$BENCHKIT_HOME` defaults to `~/.benchkit/`; override via env var.
`benchkit run` auto-persists archives to the runs dir when `--output` is omitted — no accidental data loss.
`benchkit log -n 20` tails the ledger for a quick history view.

See [`docs/output.md`](docs/output.md) for archive filename semantics, full JSONL schemas per subcommand, and the `ArchiveWriter` context-manager API for custom pipelines.

## Extra tools

Two sub-experiments live in the same package:

- **IFEval+** (`benchkit ifeval …`) — measures instruction-following
  degradation under heavy system prompts.
- **bundled-bench** (`benchkit bundled …`) — measures alignment-tax on
  multi-question prompts (base vs instruct).

Both stream to the same archive format.

## Development

```bash
.venv/bin/pytest tests/           # run tests
.venv/bin/pyright                 # strict type check (src + tests)
```

Conventions: pyright strict mode (see `pyproject.toml`), pytest.

## Further reading

- [`docs/architecture.md`](docs/architecture.md) — module map, data flow
- [`docs/harnesses.md`](docs/harnesses.md) — adapter pattern, adding a harness
- [`docs/benchmarks.md`](docs/benchmarks.md) — bracket protocol, eval modes
- [`docs/output.md`](docs/output.md) — archive format, JSONL schema

## Author

Carter Schonwald

## License

MIT
