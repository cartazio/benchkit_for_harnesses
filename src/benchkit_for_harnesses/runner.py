"""Standard benchmark batch runner (BABILong, InfiniteBench, LongBench-v2).

Thin orchestrator on top of :func:`benchkit_for_harnesses.core.run_items`:
validate args, load the dataset, delegate the loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset  # type: ignore[import-untyped]

from .benchmarks.config import BENCHMARKS
from .core import run_items
from .harnesses.dispatch import HarnessType
from .responders import harness_responder

__all__ = ["BenchmarkResult", "run_benchmark_batch"]


@dataclass
class BenchmarkResult:
    """Single benchmark result record."""

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
        return asdict(self)


def _validate_args(
    benchmark_name: str, task: str | None, length: str | None
) -> None:
    """Fail fast with actionable messages before any dataset I/O."""
    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {', '.join(BENCHMARKS.keys())}"
        )
    config = BENCHMARKS[benchmark_name]

    if config.split_is_task and task is None:
        available = ", ".join(config.tasks) if config.tasks else "(none registered)"
        raise ValueError(
            f"Benchmark '{benchmark_name}' requires --task (splits are tasks). "
            f"Available tasks: {available}"
        )

    if task is not None and config.tasks and task not in config.tasks:
        raise ValueError(
            f"Unknown task for '{benchmark_name}': {task}. "
            f"Available: {', '.join(config.tasks)}"
        )

    if config.lengths and length is None:
        raise ValueError(
            f"Benchmark '{benchmark_name}' requires --length. "
            f"Available lengths: {', '.join(config.lengths)}"
        )

    if length is not None and config.lengths and length not in config.lengths:
        raise ValueError(
            f"Unknown length for '{benchmark_name}': {length}. "
            f"Available: {', '.join(config.lengths)}"
        )


def _load_dataset_for(config: Any, task: str | None, length: str | None) -> Any:
    """Load the HF dataset with the right (config, split) permutation."""
    if config.split_is_task and task:
        if config.lengths and length:
            # BABILong: config=length, split=task
            return load_dataset(config.hf_path, length, split=task)
        # InfiniteBench: split=task, streaming avoids broken splits
        return load_dataset(config.hf_path, split=task, streaming=True)
    return load_dataset(config.hf_path, split=config.default_split)


def _apply_hf_limit(dataset: Any, limit: int | None) -> Any:
    """Apply HF .select() for non-streaming datasets; streaming is handled
    by the core loop's own limit."""
    is_streaming = hasattr(dataset, "__iter__") and not hasattr(dataset, "select")
    if not is_streaming and limit:
        return dataset.select(range(min(limit, len(dataset))))
    return dataset


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
    """Run a benchmark batch against a harness.

    Returns (results list, output archive path or None).
    """
    _validate_args(benchmark_name, task, length)
    config = BENCHMARKS[benchmark_name]
    dataset = _apply_hf_limit(_load_dataset_for(config, task, length), limit)

    responder = harness_responder(harness, model)
    description = f"{benchmark_name}_{harness}_{model.replace('/', '_')}"

    def _format(item: Mapping[str, object]) -> tuple[str, str]:
        return config.format_fn(item)

    def _build(
        idx: int,
        _item: Mapping[str, object],
        prompt: str,
        target: str,
        response: str,
        latency_ms: int,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            idx=idx,
            benchmark=benchmark_name,
            task=task or "default",
            length=length,
            model=model,
            harness=harness,
            prompt_chars=len(prompt),
            target=target,
            response=response,
            correct=config.eval_fn(response, target),
            latency_ms=latency_ms,
            system_prompt=system_prompt,
        )

    return run_items(
        items=dataset,
        responder=responder,
        format_fn=_format,
        build_record=_build,
        description=description,
        output_dir=output_dir,
        system_prompt=system_prompt,
        limit=limit,
    )
