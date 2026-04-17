# Architecture

## Module map

```
src/benchkit_for_harnesses/
├── __init__.py          package surface
├── cli.py               argparse entry point (the `benchkit` console script)
│
├── core.py              UNIFIED LOOP SUBSTRATE — run_items / run_items_async
├── responders.py        Responder factories: harness / api_chat / api_model
├── archive.py           Carter archive format — ArchiveWriter + primitives
├── brackets.py          Answer-bracket protocol — extraction, strict/loose eval
├── results.py           JSONL I/O helpers
├── api_runner.py        HTTP dispatch (OpenAI-compat / Anthropic / base completion)
│
├── benchmarks/
│   └── config.py        benchmark registry (BABILong, InfiniteBench, LongBench-v2)
│
├── harnesses/
│   └── dispatch.py      CLI harness dispatcher + per-harness adapters
│
├── runner.py            standard benchmark runner (uses core.run_items)
│
├── ifeval/              IFEval+ sub-experiment (uses core.run_items)
│   ├── experiment.py    runner + compare + dry-run
│   ├── checkers.py      per-instruction-type verification
│   ├── prompts.py       baseline / heavy / heavy_plus conditions
│   ├── mock.py          deterministic mock for pipeline testing
│   └── types.py         ResultRecord
│
└── bundled_bench/       Bundled-question sub-experiment (uses core.run_items_async)
    ├── harness.py       dataclasses + bundle construction + parse_responses
    ├── runner.py        async orchestrator (failure-record discipline)
    ├── models.py        ModelSpec registry (base + instruct pairs)
    ├── questions.py     MMLU loaders + synthetic fallback
    └── experiment.py    CLI entry point (matrix runner)
```

## The substrate

Every runner in the kit reduces to the same loop:

> iterate inputs → format each into a prompt → dispatch to a responder →
> build a record from `(idx, item, prompt, meta, response, latency)` →
> stream the record to an archive.

That loop lives in [`core.py`](../src/benchkit_for_harnesses/core.py) as two
entry points:

| Function | Transport | Concurrency | Failure discipline |
|---|---|---|---|
| `run_items(items, responder, format_fn, build_record, …)` | sync (CLI subprocess) | 1 | exceptions propagate |
| `run_items_async(items, responder, format_fn, build_record, *, build_failure_record, max_concurrent, …)` | async (HTTP) | bounded | exceptions → failure records (no silent drops) |

Both funnel through `ArchiveWriter` and both use the same record-building
contract so callers pick the variant their transport needs.

### The responder contract

```python
SyncResponder  = Callable[[str, str | None], tuple[str, int]]
AsyncResponder = Callable[[str, str | None], Awaitable[tuple[str, int]]]
```

Input: `(prompt, system_prompt or None)`.
Output: `(response_text, latency_ms)`.

Three factories in [`responders.py`](../src/benchkit_for_harnesses/responders.py):

| Factory | Transport | Used by |
|---|---|---|
| `harness_responder(name, model, timeout_sec)` | CLI subprocess | `runner.py`, `ifeval/experiment.py` |
| `api_chat_responder(model_spec)` | HTTP chat | (available for custom experiments) |
| `api_model_responder(model_spec)` | HTTP chat OR completion (dispatches on `model_spec.is_base`) | `bundled_bench/runner.py` |

Adding a new transport = one factory in `responders.py`. The core loop
doesn't change.

### The record-building contract

```python
FormatFn    = Callable[[T], tuple[str, U]]
BuildRecord = Callable[[int, T, str, U, str, int], R]
```

- `format_fn(item) → (prompt, meta)` — `meta` is whatever the record
  builder needs (the ground-truth target, the full item, a bundle
  structure, …). It's a typed passthrough — core doesn't inspect it.
- `build_record(idx, item, prompt, meta, response, latency_ms) → R` —
  returns the typed record shape for this runner (dataclass or TypedDict).

The substrate serializes records via a default that handles both
dataclasses (via `asdict`) and `Mapping`s. Callers with custom record
types can pass `serialize=...`.

### Failure records (async path only)

For the async path, each `build_failure_record(idx, item, prompt, meta, exc)
→ R` converts a responder exception into a well-formed record — the
experiment sees *every* bundle attempted, not just the ones that succeeded.
Cell n-counts stay honest; failure details go into the archive.

If `build_failure_record` is `None`, exceptions re-raise (strict mode).

## Data flow — benchkit main runner

```
cli.py
  └─> runner.run_benchmark_batch(benchmark_name, harness, model, task, length, …)
        │
        ├─> _validate_args                 fail-fast on missing task/length
        ├─> load_dataset(hf_path, …)       HuggingFace datasets
        ├─> responder = harness_responder(harness, model)
        └─> core.run_items(
              items=dataset,
              responder=responder,
              format_fn=config.format_fn,      ← returns (prompt, target:str)
              build_record=_build,             ← returns BenchmarkResult
              description=…, output_dir=…,
            )
               │
               └─> for idx, item in dataset:
                     prompt, target = format_fn(item)
                     response, ms   = responder(prompt, system_prompt)
                     record         = build_record(idx, item, prompt, target, response, ms)
                     writer.write_record(record)    ← streamed + flushed

        ← (list[BenchmarkResult], final_archive_path)
```

## Data flow — bundled-bench trial

```
bundled_bench/experiment.py
  └─> for trial_config in matrix:
        bundles = make_bundles(questions, trial_config.bundle_size, n_bundles)
        │
        └─> bundled_bench.runner.run_experiment(bundles, trial_config, max_concurrent)
              │
              ├─> responder = api_model_responder(trial_config.model)
              │    (dispatches chat vs completion by is_base)
              └─> core.run_items_async(
                    items=bundles,
                    responder=responder,
                    format_fn=format_bundle_prompt + bundle passthrough,
                    build_record=_build_success,   ← parse_responses → BundleResult
                    build_failure_record=_build_failure,  ← zero-accuracy record
                    max_concurrent=max_concurrent,
                  )
```

## Key invariants

- **Streaming archive.** Every record is flushed per line. A crash at
  item 99 preserves items 0–98 and finalizes the archive via
  `ArchiveWriter.__exit__`. Async path gathers concurrently but writes
  records in input order so archives are deterministic.
- **Content-hashed filenames.** `{desc}_v{n}_{ts}NYC_{hash}.jsonl` with
  `hash = SHA3-256(file_bytes)[:12]`. Draft files use a placeholder
  `000000000000`; finalization composes the final name from parsed parts
  (never `str.replace`), so descriptions containing twelve zeros survive
  intact.
- **Error discipline at the responder boundary.** CLI failures become
  `[ERROR …]`-prefixed strings; HTTP failures raise exceptions that the
  async path converts into explicit failure records. Callers never have
  to peek at return codes or exception types.
- **Honest n-counts.** No benchmark iteration drops items silently. Every
  attempt produces exactly one record — success, failure, or explicit
  `[ERROR …]` string.

## Answer-bracket protocol (short version)

- `{[{[ answer ]}]}` — the final answer (scored)
- `(_( draft )_)` — intermediate reasoning (not scored)
- `extract_bracketed_answer` returns the **last** match — robust to
  leaked visible-CoT from reasoning models.

See [benchmarks.md](benchmarks.md) for strict vs loose eval and the full
protocol semantics.

## Third-party type stubs

`typings/` carries minimal stubs for `datasets` and `langdetect` (neither
ships `py.typed`). `httpx` types come from upstream. Pyright is strict on
`src/` and `tests/`.
