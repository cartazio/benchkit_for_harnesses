# Output format

## Archive filename

```
{desc}_v{n}_{YYYYMMDDTHHMMSS}NYC_{hash}.{ext}
```

| Field | Meaning |
|---|---|
| `desc` | Human-readable slug (spaces → underscores). For benchkit runs: `{benchmark}_{harness}_{model}` |
| `n` | Version number (starts at 1) |
| `YYYYMMDDTHHMMSS` | Timestamp, America/New_York, second resolution, fixed-width |
| `hash` | First 12 hex chars of SHA3-256 over the finalized file contents |
| `ext` | `jsonl` for benchmark runs |

### Content-hashed names as self-certification

The filename is a self-certifying integrity check: anyone can recompute
SHA3-256 over the file contents and compare against the hash embedded in
the filename. Silent corruption is impossible to miss.

```bash
f=./results/babilong_punkin_sonnet_v1_20260417T163732NYC_5032f5270e12.jsonl
echo "claimed:  $(basename "$f" | sed 's/.*_\([0-9a-f]\{12\}\)\.jsonl/\1/')"
echo "computed: $(python3 -c "import hashlib; print(hashlib.sha3_256(open('$f','rb').read()).hexdigest()[:12])")"
```

### Draft → finalized

The writer creates the file with a placeholder hash `000000000000` and
atomically renames it on close. The renaming composes the final name
from **parsed parts** — never a blind `str.replace("0"*12, hash)` — so
descriptions containing twelve zeros survive intact (regression-tested
in `tests/test_archive.py`).

## JSONL schema — `run_benchmark_batch`

Record shape is the `BenchmarkResult` dataclass from `runner.py`:

```json
{
  "idx": 0,
  "benchmark": "babilong",
  "task": "qa1",
  "length": "0k",
  "model": "claude-sonnet-4",
  "harness": "claude",
  "prompt_chars": 2048,
  "target": "bathroom",
  "response": "<model text, or '[ERROR ...]' on failure>",
  "correct": true,
  "latency_ms": 1234,
  "system_prompt": null
}
```

| Field | Type | Notes |
|---|---|---|
| `idx` | int | Position in the dataset iterator |
| `benchmark` | str | Registry name |
| `task` | str | Task name, or `"default"` for single-split benchmarks |
| `length` | str \| null | Context-length band (BABILong only) |
| `model` | str | Exact `--model` string passed to the harness |
| `harness` | str | Harness name |
| `prompt_chars` | int | Character count of the fully-formatted prompt |
| `target` | str | Ground-truth answer; pipe-separated for multi-answer |
| `response` | str | Harness stdout. Starts with `[ERROR …]` on failure |
| `correct` | bool | Result of `config.eval_fn(response, target)` |
| `latency_ms` | int | Wall-clock time of the harness invocation |
| `system_prompt` | str \| null | System prompt override (if passed) |

## JSONL schema — IFEval+

Record shape is the `ResultRecord` TypedDict from `ifeval/types.py`:

```json
{
  "idx": 0,
  "key": 12345,
  "condition": "heavy",
  "model": "claude-sonnet-4",
  "prompt": "<truncated to 500 chars>",
  "instruction_ids": ["punctuation:no_comma", "length_constraints:number_words"],
  "response": "<truncated to 2000 chars>",
  "follow_all": false,
  "follow_list": [true, false],
  "latency_ms": 1234
}
```

`follow_all = all(follow_list)`. `instruction_ids[i]` corresponds to
`follow_list[i]`.

## JSONL schema — bundled-bench

Record shape is the saved projection of `BundleResult` from
`bundled_bench/harness.py`:

```json
{
  "bundle_id": "bundle_abc123def456",
  "model": "llama-3.1-8b-instruct",
  "is_base": false,
  "condition": "minimal",
  "bundle_size": 3,
  "seed": 42,
  "accuracy": 0.667,
  "completion_rate": 1.0,
  "latency_ms": 3421.5,
  "timestamp": "2026-04-17T15:34:12.123456+00:00",
  "questions": [
    {
      "question_id": "mmlu:anatomy:0",
      "position": 0,
      "expected": "B",
      "raw_response": "B",
      "correct": true,
      "attempted": true
    }
  ]
}
```

**Failed bundles** land in the archive with `accuracy = 0.0`,
`completion_rate = 0.0`, and all `questions[].attempted = false`. This
keeps cell n-counts honest — dropped bundles would phantom-inflate
per-cell accuracy.

## Programmatic writing — `ArchiveWriter`

```python
from benchkit_for_harnesses.archive import ArchiveWriter

with ArchiveWriter("./results", "my_experiment") as w:
    for record in results:
        w.write_record(record)          # JSONL, per-line flush
        # or w.write_line("raw text")

print(w.final_path)                     # content-hashed Path
```

Guarantees:

- **Per-line flush.** Data is durable the moment `write_record` returns.
- **Finalize on any exit.** Including exceptions. A crash at item 99
  preserves items 0–98 with the correct content hash.
- **Empty archive still valid.** Finalizes to an empty file with the
  hash of the empty byte string.

## Running custom pipelines through the substrate

```python
from benchkit_for_harnesses.core import run_items
from benchkit_for_harnesses.responders import harness_responder

responder = harness_responder("claude", "sonnet")

def format_fn(item):
    return item["question"], item["answer"]

def build_record(idx, item, prompt, target, response, latency_ms):
    return {
        "idx": idx,
        "target": target,
        "response": response,
        "correct": target.lower() in response.lower(),
        "latency_ms": latency_ms,
    }

records, path = run_items(
    items=my_items,
    responder=responder,
    format_fn=format_fn,
    build_record=build_record,
    description="my_custom_bench",
    output_dir="./results",
    limit=50,
)
```

For async transports, use `run_items_async` with a responder from
`api_chat_responder(...)` or `api_model_responder(...)`, plus an optional
`build_failure_record` for explicit failure semantics.


## The run ledger

Separate from the per-benchmark archive: `$BENCHKIT_HOME/ledger.jsonl`
appends **one line per `benchkit` invocation** (regardless of subcommand).
Archives are per-run data; the ledger is the cross-run history.

```
$BENCHKIT_HOME/
  ├── ledger.jsonl        (one line per invocation)
  └── runs/
      ├── babilong_claude_sonnet_v1_...NYC_{hash}.jsonl
      ├── infinitebench_codex_gpt-5.3-codex-spark_v1_...NYC_{hash}.jsonl
      └── ...
```

`$BENCHKIT_HOME` defaults to `~/.benchkit/`; override with the env var.

### Ledger entry schema

Every subcommand emits this envelope, plus command-specific fields:

```json
{
  "timestamp": "2026-04-17T17:23:27-04:00",
  "cmd": "run",
  "argv": ["run", "--benchmark", "babilong", "--task", "qa1", "--length", "0k", "--limit", "2", "--harness", "claude", "--model", "sonnet"],
  "duration_ms": 6972,
  "exit_code": 0,
  "benchmark": "babilong",
  "harness": "claude",
  "model": "sonnet",
  "task": "qa1",
  "length": "0k",
  "n_items": 2,
  "correct": 2,
  "accuracy": 1.0,
  "avg_latency_ms": 2934.0,
  "output_path": "/Users/carter/.benchkit/runs/babilong_claude_sonnet_v1_20260417T172321NYC_f4dd20673faf.jsonl"
}
```

For `probe`: `harness`, `model`, `latency_ms`, `success`, `response_preview`.
For `ifeval` / `bundled`: `forwarded_argv` (the raw tail handed to the sub-experiment).
For exceptions: an `exception` field with `type_name: message`.

### Reading the ledger

```bash
benchkit log -n 20            # last 20 invocations, pretty-rendered
```

Programmatic:

```python
from benchkit_for_harnesses.ledger import read_entries, tail_entries

all_runs = read_entries()                        # list[dict]
recent = tail_entries(50)                        # last 50
successes = [e for e in all_runs if e["exit_code"] == 0]
```

### Why a ledger on top of the archive

- Archives answer "what happened inside this run?" — per-item records.
- Ledger answers "what runs exist, when, with what settings?" — history across runs.
- Failed runs (parser errors, harness-not-found, Anthropic 429, ...) never produce an archive, but the ledger still records them with an `exception` field. That's the only trace of failed attempts.
- Corrupt ledger lines (from a partial write) are skipped silently by `read_entries`; the file is append-only in normal operation.