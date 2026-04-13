"""Benchmark orchestration and batch execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from datasets import DatasetLike, load_dataset

from .benchmarks.config import BENCHMARKS
from .harnesses.runner import run_harness, HarnessType
from .archive import make_archive_path, finalize_archive_path

__all__ = ["run_benchmark_batch", "BenchmarkResult"]


@dataclass
class BenchmarkResult:
    """Single benchmark result."""

    idx: int
    benchmark: str
    task: str
    length: str | None
    model: str
    harness: str
    prompt_chars: int
    target: str
    response: str
    correct: bool
    latency_ms: int
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def run_benchmark_batch(
    benchmark_name: str,
    harness: HarnessType,
    model: str,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    task: str | None = None,
    length: str | None = None,
    system_prompt: str | None = None,
) -> tuple[list[BenchmarkResult], Path | None]:
    """
    Run benchmark batch against a harness.

    Args:
        benchmark_name: Benchmark to run (from BENCHMARKS registry)
        harness: Harness to use (ohp, punkin)
        model: Model identifier
        output_dir: Output directory for JSONL results (if None, no output)
        limit: Max items to run (None = all)
        task: Specific task to run (for multi-task benchmarks)
        length: Specific length to run (for long-context benchmarks)
        system_prompt: System prompt override

    Returns:
        Tuple of (results list, output_file path or None)

    Raises:
        ValueError: If benchmark_name not in registry
    """
    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {', '.join(BENCHMARKS.keys())}"
        )

    config = BENCHMARKS[benchmark_name]
    results: list[BenchmarkResult] = []

    # Load dataset
    if config.split_is_task and task:
        dataset: DatasetLike = load_dataset(config.hf_path, split=task)
    else:
        dataset = load_dataset(config.hf_path, split=config.default_split)

    # Filter by limit
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Run benchmark
    for idx, item in enumerate(dataset):
        prompt, target = config.format_fn(item)
        response, latency_ms = run_harness(
            harness=harness,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        correct = config.eval_fn(response, target)

        result = BenchmarkResult(
            idx=idx,
            benchmark=benchmark_name,
            task=task or "default",
            length=length,
            model=model,
            harness=harness,
            prompt_chars=len(prompt),
            target=target,
            response=response,
            correct=correct,
            latency_ms=latency_ms,
            system_prompt=system_prompt,
        )
        results.append(result)

    # Write output if requested
    output_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_path = make_archive_path(
            output_dir,
            f"{benchmark_name}_{harness}_{model}",
            extension="jsonl",
        )

        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + "\n")

        # Finalize with actual content hash
        content_bytes = output_path.read_bytes()
        output_path = finalize_archive_path(content_bytes, output_path)

    return results, output_path
