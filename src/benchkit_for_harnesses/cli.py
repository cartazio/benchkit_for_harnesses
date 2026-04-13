"""Command-line interface for benchkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmarks import BENCHMARKS
from .runner import run_benchmark_batch

__all__ = ["main"]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BenchKit: unified benchmarking for ohp/punkin harnesses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  benchkit --benchmark babilong --harness ohp --model claude-sonnet-4
  benchkit --benchmark infinitebench --task passkey --limit 10 --harness punkin
  benchkit --benchmark longbenchv2 --output results/ --harness ohp
""",
    )

    parser.add_argument(
        "--benchmark",
        "-B",
        required=True,
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to run",
    )
    parser.add_argument(
        "--harness",
        "-H",
        default="ohp",
        choices=["ohp", "punkin"],
        help="Harness to use (default: ohp)",
    )
    parser.add_argument(
        "--model",
        "-M",
        required=True,
        help="Model identifier (e.g., claude-sonnet-4)",
    )
    parser.add_argument(
        "--task",
        "-T",
        default=None,
        help="Specific task within benchmark (if multi-task)",
    )
    parser.add_argument(
        "--length",
        "-L",
        default=None,
        help="Specific context length (if applicable)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to run (None = all)",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=Path,
        default=None,
        help="Output directory for JSONL results",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt override",
    )

    args = parser.parse_args()

    try:
        results, output_path = run_benchmark_batch(
            benchmark_name=args.benchmark,
            harness=args.harness,
            model=args.model,
            output_dir=args.output,
            limit=args.limit,
            task=args.task,
            length=args.length,
            system_prompt=args.system_prompt,
        )

        # Print summary
        correct = sum(1 for r in results if r.correct)
        accuracy = correct / len(results) if results else 0.0
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        print(f"\n{'='*60}")
        print(f"Benchmark: {args.benchmark}")
        print(f"Harness: {args.harness} | Model: {args.model}")
        print(f"{'='*60}")
        print(f"Completed: {len(results)} items")
        print(f"Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
        print(f"Avg Latency: {avg_latency:.0f}ms")

        if output_path:
            print(f"Output: {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
