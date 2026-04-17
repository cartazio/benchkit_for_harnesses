"""IFEval+ experiment runner and CLI.

Measures instruction-following degradation under heavy system prompts by
running the google/IFEval dataset against a harness under different
system-prompt conditions, then comparing.

Usage:
    # Dry run — see what would be tested
    python -m benchkit_for_harnesses.ifeval.experiment --dry-run -n 10

    # Run baseline (minimal system prompt)
    python -m benchkit_for_harnesses.ifeval.experiment run --condition baseline \\
        -m claude-opus-4 -n 50

    # Run treatment (heavy system prompt)
    python -m benchkit_for_harnesses.ifeval.experiment run --condition heavy \\
        -m claude-opus-4 -n 50

    # Compare results
    python -m benchkit_for_harnesses.ifeval.experiment compare \\
        baseline_results.jsonl heavy_results.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

from benchkit_for_harnesses.core import run_items
from benchkit_for_harnesses.responders import harness_responder
from benchkit_for_harnesses.results import load_jsonl

from .checkers import evaluate_response
from .mock import mock_response
from .prompts import SYSTEM_PROMPTS
from .types import ResultRecord


def run_benchmark(
    condition: str,
    model: str,
    harness: str,
    limit: int,
    output_dir: Path,
    mock: bool = False,
    verbose: bool = False,
) -> Path:
    """Run IFEval with a given system-prompt condition."""

    system_prompt = SYSTEM_PROMPTS[condition]
    if verbose:
        print("Loading IFEval dataset...", file=sys.stderr)
    ds = load_dataset("google/IFEval", split="train")
    items = [dict(item) for item in list(ds)[:limit]]
    if verbose:
        print(f"Loaded {len(items)} items", file=sys.stderr)
        print(
            f"Running {condition} condition with {harness} --model {model}...",
            file=sys.stderr,
        )

    # Responder: mock or real harness
    if mock:
        def _responder(prompt: str, system: str | None) -> tuple[str, int]:
            seed = _responder._seed  # type: ignore[attr-defined]
            _responder._seed += 1    # type: ignore[attr-defined]
            return mock_response(prompt, system or "", seed=seed), 100
        _responder._seed = 0  # type: ignore[attr-defined]
        responder = _responder
    else:
        responder = harness_responder(harness, model, timeout_sec=120)

    desc = f"ifeval_plus_{condition}_{model.replace('/', '_')}"

    def _format(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # Pass the whole item as "meta" so build_record has access to
        # key / instruction_ids / kwargs_list without re-looking-up.
        return item["prompt"], item

    def _build(
        idx: int,
        _item: dict[str, Any],
        prompt: str,
        meta: dict[str, Any],
        response: str,
        latency_ms: int,
    ) -> ResultRecord:
        follow_all, follow_list = evaluate_response(
            response=response,
            instruction_ids=meta["instruction_id_list"],
            kwargs_list=meta["kwargs"],
            prompt=prompt,
        )
        return ResultRecord(
            idx=idx,
            key=meta["key"],
            condition=condition,
            model=model,
            prompt=prompt[:500],
            instruction_ids=list(meta["instruction_id_list"]),
            response=response[:2000],
            follow_all=follow_all,
            follow_list=follow_list,
            latency_ms=latency_ms,
        )

    def _progress(idx: int, record: ResultRecord) -> None:
        if not verbose:
            return
        followed = sum(record["follow_list"])
        total = len(record["follow_list"])
        status = "\u2713" if record["follow_all"] else "\u2717"
        print(
            f"[{idx + 1}/{len(items)}] {status} {followed}/{total} "
            f"latency={record['latency_ms']}ms",
            file=sys.stderr,
        )

    records, final_path = run_items(
        items=items,
        responder=responder,
        format_fn=_format,
        build_record=_build,
        description=desc,
        output_dir=output_dir,
        system_prompt=system_prompt,
        on_progress=_progress,
        serialize=lambda r: dict(r),
    )
    assert final_path is not None

    # Summary
    total_instructions = sum(len(r["follow_list"]) for r in records)
    followed_instructions = sum(sum(r["follow_list"]) for r in records)
    prompts_all_followed = sum(1 for r in records if r["follow_all"])
    prompt_acc = prompts_all_followed / len(records) if records else 0.0
    instr_acc = (
        followed_instructions / total_instructions if total_instructions else 0.0
    )

    print(f"\n=== {condition.upper()} Results ===", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(
        f"Prompt-level accuracy: {prompts_all_followed}/{len(records)} "
        f"({prompt_acc:.1%})",
        file=sys.stderr,
    )
    print(
        f"Instruction-level accuracy: {followed_instructions}/{total_instructions} "
        f"({instr_acc:.1%})",
        file=sys.stderr,
    )
    print(f"Output: {final_path}", file=sys.stderr)
    return final_path


def compare_results(baseline_path: Path, treatment_path: Path) -> None:
    """Compare baseline vs treatment result files."""

    baseline = load_jsonl(baseline_path)
    treatment = load_jsonl(treatment_path)

    def compute_stats(results: list[dict[str, Any]]) -> tuple[float, float]:
        total_prompts = len(results)
        prompts_all = sum(1 for r in results if r["follow_all"])
        total_instr = sum(len(r["follow_list"]) for r in results)
        instr_followed = sum(sum(r["follow_list"]) for r in results)
        return prompts_all / total_prompts, instr_followed / total_instr

    b_prompt, b_instr = compute_stats(baseline)
    t_prompt, t_instr = compute_stats(treatment)

    print("=" * 60)
    print("IFEval+ Comparison: System Prompt Overhead Analysis")
    print("=" * 60)
    print()
    print(f"{'Metric':<30} {'Baseline':>12} {'Treatment':>12} {'Delta':>12}")
    print("-" * 60)
    print(
        f"{'Prompt-level accuracy':<30} {b_prompt:>11.1%} {t_prompt:>11.1%} "
        f"{t_prompt - b_prompt:>+11.1%}"
    )
    print(
        f"{'Instruction-level accuracy':<30} {b_instr:>11.1%} {t_instr:>11.1%} "
        f"{t_instr - b_instr:>+11.1%}"
    )
    print()

    # Breakdown by instruction type
    print("Per-instruction-type breakdown:")
    print("-" * 60)

    baseline_by_type: dict[str, list[bool]] = {}
    treatment_by_type: dict[str, list[bool]] = {}

    for r in baseline:
        for instr_id, followed in zip(r["instruction_ids"], r["follow_list"]):
            baseline_by_type.setdefault(instr_id, []).append(followed)

    for r in treatment:
        for instr_id, followed in zip(r["instruction_ids"], r["follow_list"]):
            treatment_by_type.setdefault(instr_id, []).append(followed)

    all_types = sorted(set(baseline_by_type.keys()) | set(treatment_by_type.keys()))
    for instr_type in all_types:
        b_vals = baseline_by_type.get(instr_type, [])
        t_vals = treatment_by_type.get(instr_type, [])
        b_acc = sum(b_vals) / len(b_vals) if b_vals else 0
        t_acc = sum(t_vals) / len(t_vals) if t_vals else 0
        delta = t_acc - b_acc
        if abs(delta) > 0.05:
            print(
                f"  {instr_type:<40} {b_acc:>6.0%} \u2192 {t_acc:>6.0%} ({delta:>+5.0%})"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="IFEval+ harness for measuring system prompt overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument(
        "--condition", "-c",
        choices=list(SYSTEM_PROMPTS.keys()),
        default="baseline",
        help="System prompt condition (default: baseline)",
    )
    run_parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4",
        help="Model identifier (default: claude-sonnet-4)",
    )
    run_parser.add_argument(
        "--harness", "-H",
        default="ohp",
        help="CLI harness to use (default: ohp)",
    )
    run_parser.add_argument(
        "--limit", "-n", type=int, default=50,
        help="Max items to evaluate (default: 50)",
    )
    run_parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("."),
        help="Output directory (default: current)",
    )
    run_parser.add_argument(
        "--mock", action="store_true",
        help="Use mock responses (for testing pipeline)",
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two result files")
    compare_parser.add_argument("baseline", type=Path, help="Baseline results JSONL")
    compare_parser.add_argument("treatment", type=Path, help="Treatment results JSONL")

    # Dry-run (default if no subcommand)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be tested without running",
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=10,
        help="Items to show in dry-run (default: 10)",
    )

    args = parser.parse_args(argv)

    if args.command == "compare":
        compare_results(args.baseline, args.treatment)
        return 0

    if args.command == "run":
        run_benchmark(
            condition=args.condition,
            model=args.model,
            harness=args.harness,
            limit=args.limit,
            output_dir=args.output_dir,
            mock=args.mock,
            verbose=args.verbose,
        )
        return 0

    # Default: dry-run
    if args.dry_run or args.command is None:
        ds = load_dataset("google/IFEval", split="train")
        items_dry = [dict(item) for item in list(ds)[:args.limit]]

        print("=== IFEval+ Dry Run ===\n")
        print(f"Total dataset size: {len(ds)}")
        print(f"Showing first {len(items_dry)} items:\n")

        for i, item in enumerate(items_dry):
            print(f"--- Item {i} (key={item['key']}) ---")
            print(f"Instructions: {item['instruction_id_list']}")
            prompt_text = str(item["prompt"])
            print(f"Prompt: {prompt_text[:200]}...")
            print()

        print("\nSystem prompt conditions available:")
        for name, prompt in SYSTEM_PROMPTS.items():
            print(f"  {name}: {len(prompt)} chars")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
