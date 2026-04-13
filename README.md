# BenchKit for Harnesses

Unified benchmarking toolkit for **ohp** and **punkin** CLI harnesses.

## Features

- **Multi-benchmark support**: BABILong, InfiniteBench, LongBench-v2
- **Flexible harness selection**: Run against ohp or punkin CLI
- **Long-context evaluation**: Support for up to 128k context windows
- **Archive format output**: Versioned, timestamped, content-hashed JSONL results
- **Modern Python packaging**: pyproject.toml, type-safe, pytest-ready

## Installation

```bash
cd ~/local_dev/dynamic_science/benchkit_for_harnesses
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Run BABILong benchmark with ohp harness
benchkit --benchmark babilong --harness ohp --model claude-sonnet-4

# Run specific task with limit
benchkit --benchmark infinitebench --task passkey --limit 10 --harness punkin

# Save results to directory
benchkit --benchmark longbenchv2 --output ./results/ --harness ohp

# With custom system prompt
benchkit --benchmark babilong --harness ohp --model sonnet --system-prompt "Be concise."
```

### Python API

```python
from benchkit_for_harnesses import run_benchmark_batch

results, output_path = run_benchmark_batch(
    benchmark_name="babilong",
    harness="ohp",
    model="claude-sonnet-4",
    output_dir="./results",
    limit=100,
)

for result in results:
    print(f"Item {result.idx}: {result.correct} in {result.latency_ms}ms")
```

## Benchmarks

### BABILong
- **Tasks**: qa1-qa10 (10 synthetic QA tasks)
- **Context**: 0k-128k tokens
- **Evaluation**: Contains-based matching

### InfiniteBench
- **Tasks**: passkey, kv_retrieval, number_string, code_run, code_debug, math_find, longdialogue_qa_eng, longbook_qa_eng
- **Evaluation**: Contains-based matching

### LongBench-v2
- **Format**: Multi-choice QA
- **Evaluation**: Letter-based matching (A, B, C, D)

## Output Format

Results are saved as **Carter archive format** JSONL:

```
{desc}_v{n}_{YYYYMMDDTHHMMSS}NYC_{hash}.jsonl
```

Each line is a JSON object:

```json
{
  "idx": 0,
  "benchmark": "babilong",
  "task": "qa1",
  "length": "4k",
  "model": "claude-sonnet-4",
  "harness": "ohp",
  "prompt_chars": 2048,
  "target": "answer",
  "response": "model response",
  "correct": true,
  "latency_ms": 1234,
  "system_prompt": null
}
```

## Development

### Type Checking

```bash
pyright src/
```

### Testing

```bash
pytest tests/
```

### Code Quality

```bash
black src/ tests/
ruff check src/ tests/
```

## Author

Carter Schonwald

## License

MIT
