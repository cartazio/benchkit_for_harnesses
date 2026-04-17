# Benchmarks

## The answer-bracket protocol

All built-in benchmarks use the same contract:

- The model wraps its **final answer** in `{[{[` / `]}]}` markers.
- If the model shows reasoning or drafts, it wraps those in `(_(` / `)_)`
  so they are not confused with the scored answer.
- If the model emits multiple `{[{[ … ]}]}` pairs, the **last one wins**.

API in `brackets.py`:

```python
extract_bracketed_answer("a {[{[draft]}]} b {[{[final]}]}")  # → "final"
extract_bracketed_answers("same input")                       # → ["draft", "final"]
eval_bracketed(response, target)                              # → bool (loose)
eval_bracketed(response, target, strict=True)                 # → bool (strict)
```

The shared `ANSWER_INSTRUCTION` constant is stitched into every format
function so models always see the same contract regardless of benchmark.

### Why LAST, not FIRST

Reasoning-model responses can contain leaked visible chain-of-thought:
Anthropic extended thinking, system-prompt-induced `<squiggle>` blocks,
etc. The model may emit an interim guess and then a corrected final —
we want to score the final. Taking the last bracket pair gives that for
free, and makes the protocol robust to CoT-leaky transports without
needing to "turn off" reasoning.

## Evaluation modes

### Loose — free-form targets (BABILong, InfiniteBench)

```
eval_bracketed(response, target)  # strict=False
```

1. Extract last `{[{[ … ]}]}` pair.
2. Accept if candidate equals target (case-insensitive after strip).
3. Accept if target appears as a substring of the candidate.
4. If no brackets were found, fall back to containment against the full
   response.

Multi-answer targets are pipe-separated (`"Paris|paris|PARIS"`) — any
alternative matches.

### Strict — discrete targets (LongBench-v2)

```
eval_bracketed(response, target, strict=True)
```

1. Extract last `{[{[ … ]}]}` pair. If none → incorrect.
2. Accept only on exact case-insensitive equality with any target
   alternative.

Strict is required for MC letters: loose substring matching would
accept `"a"` appearing inside `"answer"`, inflating accuracy.

## Benchmark registry

Each entry in `benchmarks/config.py` declares a `BenchmarkConfig`:

```python
BenchmarkConfig(
    name, hf_path,
    format_fn,    # Mapping[str, object] → (prompt, target)
    eval_fn,      # (response, target) → bool
    tasks=[...],
    lengths=[...],
    split_is_task=False,
    default_split="test",
)
```

### BABILong

- `hf_path`: `RMT-team/babilong`
- Shape: `(config=length, split=task)`
- Tasks: `qa1`–`qa10`
- Lengths: `0k`, `1k`, `2k`, `4k`, `8k`, `16k`, `32k`, `64k`, `128k`
- Eval: loose

### InfiniteBench

- `hf_path`: `xinrongzhang2022/InfiniteBench`
- Shape: `(split=task)`, streaming
- Tasks: `passkey`, `kv_retrieval`, `number_string`, `code_run`,
  `code_debug`, `math_find`, `longdialogue_qa_eng`, `longbook_qa_eng`
- Eval: loose (multi-answer items join with `|`)

### LongBench-v2

- `hf_path`: `zai-org/LongBench-v2`
- Shape: single split (`train`)
- Eval: **strict** (MC letter targets A/B/C/D)

## Sub-experiments

### IFEval+

`python -m benchkit_for_harnesses.ifeval.experiment`

Measures instruction-following degradation under heavy system prompts.
Three conditions:

- `baseline` — minimal helpful-assistant prompt
- `heavy` — representative CarterKit/punkin-pi style prompt
- `heavy_plus` — heavy + additional context-pressure markers

```bash
python -m benchkit_for_harnesses.ifeval.experiment run -c baseline -m sonnet -n 50 -o results/
python -m benchkit_for_harnesses.ifeval.experiment run -c heavy    -m sonnet -n 50 -o results/
python -m benchkit_for_harnesses.ifeval.experiment compare \
    results/ifeval_plus_baseline_*.jsonl results/ifeval_plus_heavy_*.jsonl
```

Checkers (`ifeval/checkers.py`) are simplified implementations of the
core google-research/IFEval instruction types. Unknown instruction types
score as not-followed — accuracy is never silently inflated.

### bundled-bench

`benchkit bundled …` (or `python -m benchkit_for_harnesses.bundled_bench.experiment`)

**Shape**, not hypothesis. bundled-bench is a generic loop for:

> *given a list of independent questions, pack N of them into one prompt,
> ask a model to answer all N, score each answer independently, and record
> per-position + per-bundle + per-cell metrics.*

Nothing about the shape requires a specific dataset, transport, or
hypothesis. The current registry ships:

- `questions.py` loaders for MMLU and synthetic questions; any `list[Question]` works, including your own.
- `models.py` ModelSpecs for Llama / Qwen / Mistral (base + instruct) plus closed-source instruct-only (Claude, GPT-4o-mini).
- HTTP responders (`api_model_responder` dispatching chat vs completion by `is_base`). Base models *need* this because CLI harnesses are chat-only. For instruct-only experiments you could equally wire a harness responder.
- Failure-record discipline: a bundle that raises becomes an explicit zero-accuracy record, so cell n-counts stay stable across partial-failure runs.

**Originally motivated** by a base-vs-instruct alignment-tax question: does
instruction-tuning's default policy degrade multi-question accuracy, and
does a heavy system prompt recover it? That's one experiment you can run
with this shape.

**Other things the same shape answers well**:

- Long-context / high-semantic-density degradation (bundle size as load axis)
- CLI-harness stress testing (does punkin handle N=10 correctly?)
- System-prompt ablations (which clauses of a heavy prompt cost multi-question accuracy?)
- Position-bias studies in reasoning models
- Cross-domain interference (math + geography + code in one prompt)
- Instruction-following-at-scale (bundled IFEval items)

The matrix (`preset` x `bundle-sizes` x `conditions` x `n-bundles`) in
`experiment.py` is the scaffold for one kind of study; swap the responder,
the question loader, or the conditions list and the same scaffold runs a
different study.


## Adding a benchmark

1. Write `format_fn(item) → (prompt, target)` that stitches
   `ANSWER_INSTRUCTION` into the prompt.
2. Pick an `eval_fn` — typically `eval_bracketed` (loose) or a strict
   wrapper for discrete targets (see `_eval_bracketed_strict` in
   `config.py`).
3. Add an entry to `BENCHMARKS`.
4. Verify end-to-end with `benchkit --benchmark NEW --limit 2` against a
   fast harness.

No changes to the core loop, the CLI, or any runner needed.
