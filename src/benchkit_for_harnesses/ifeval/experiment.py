"""IFEval+ experiment runner and CLI.

Measures instruction-following degradation under heavy system prompts.

Usage:
    # Dry run — see what would be tested
    python -m benchkit_for_harnesses.ifeval.experiment --dry-run -n 10

    # Run baseline (minimal system prompt)
    python -m benchkit_for_harnesses.ifeval.experiment run --condition baseline -m claude-opus-4 -n 50

    # Run treatment (heavy system prompt)
    python -m benchkit_for_harnesses.ifeval.experiment run --condition heavy -m claude-opus-4 -n 50

    # Compare results
    python -m benchkit_for_harnesses.ifeval.experiment compare baseline_results.jsonl heavy_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

from benchkit_for_harnesses.archive import make_archive_path, finalize_archive_path
from benchkit_for_harnesses.harnesses.runner import run_harness
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
    """Run IFEval with specified condition."""
    
    system_prompt = SYSTEM_PROMPTS[condition]
    
    # Load dataset
    if verbose:
        print(f"Loading IFEval dataset...", file=sys.stderr)
    ds = load_dataset("google/IFEval", split="train")
    
    items = [dict(item) for item in list(ds)[:limit]]
    if verbose:
        print(f"Loaded {len(items)} items", file=sys.stderr)
    
    # Setup output — B's make_archive_path takes (base_dir, description)
    desc = f"ifeval_plus_{condition}_{model.replace('/', '_')}"
    output_path = make_archive_path(output_dir, desc)
    
    results: list[ResultRecord] = []
    total_instructions = 0
    followed_instructions = 0
    prompts_all_followed = 0
    
    if verbose:
        print(f"Running {condition} condition with {harness} --model {model}...", file=sys.stderr)
    
    with open(output_path, "w") as f:
        for i, item in enumerate(items):
            prompt = item["prompt"]
            instruction_ids = item["instruction_id_list"]
            kwargs_list = item["kwargs"]
            
            # Get response
            if mock:
                response = mock_response(prompt, system_prompt, seed=i)
                latency_ms = 100
            else:
                response, latency_ms = run_harness(
                    harness=harness,
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    timeout_sec=120,
                )
            
            # Evaluate
            follow_all, follow_list = evaluate_response(
                response=response,
                instruction_ids=instruction_ids,
                kwargs_list=kwargs_list,
                prompt=prompt,
            )
            
            # Accumulate stats
            total_instructions += len(follow_list)
            followed_instructions += sum(follow_list)
            if follow_all:
                prompts_all_followed += 1
            
            record: ResultRecord = {
                "idx": i,
                "key": item["key"],
                "condition": condition,
                "model": model,
                "prompt": prompt[:500],  # Truncate for sanity
                "instruction_ids": instruction_ids,
                "response": response[:2000],
                "follow_all": follow_all,
                "follow_list": follow_list,
                "latency_ms": latency_ms,
            }
            results.append(record)
            
            f.write(json.dumps(record) + "\n")
            f.flush()
            
            if verbose:
                status = "\u2713" if follow_all else "\u2717"
                followed = sum(follow_list)
                total = len(follow_list)
                print(f"[{i+1}/{len(items)}] {status} {followed}/{total} latency={latency_ms}ms", 
                      file=sys.stderr)
    
    # Finalize — B's finalize_archive_path takes (content_bytes, draft_path)
    content_bytes = output_path.read_bytes()
    final_path = finalize_archive_path(content_bytes, output_path)
    
    # Summary
    prompt_acc = prompts_all_followed / len(items) if items else 0
    instr_acc = followed_instructions / total_instructions if total_instructions else 0
    
    print(f"\n=== {condition.upper()} Results ===", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(f"Prompt-level accuracy: {prompts_all_followed}/{len(items)} ({prompt_acc:.1%})", file=sys.stderr)
    print(f"Instruction-level accuracy: {followed_instructions}/{total_instructions} ({instr_acc:.1%})", file=sys.stderr)
    print(f"Output: {final_path}", file=sys.stderr)
    
    return final_path


def compare_results(baseline_path: Path, treatment_path: Path) -> None:
    """Compare baseline vs treatment results."""
    
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
    print(f"{'Prompt-level accuracy':<30} {b_prompt:>11.1%} {t_prompt:>11.1%} {t_prompt - b_prompt:>+11.1%}")
    print(f"{'Instruction-level accuracy':<30} {b_instr:>11.1%} {t_instr:>11.1%} {t_instr - b_instr:>+11.1%}")
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
        if abs(delta) > 0.05:  # Only show significant deltas
            print(f"  {instr_type:<40} {b_acc:>6.0%} \u2192 {t_acc:>6.0%} ({delta:>+5.0%})")


def main() -> int:
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
        "--limit", "-n",
        type=int,
        default=50,
        help="Max items to evaluate (default: 50)",
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("."),
        help="Output directory (default: current)",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock responses (for testing pipeline)",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress",
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two result files")
    compare_parser.add_argument("baseline", type=Path, help="Baseline results JSONL")
    compare_parser.add_argument("treatment", type=Path, help="Treatment results JSONL")
    
    # Dry-run (default if no subcommand)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without running",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=10,
        help="Items to show in dry-run (default: 10)",
    )
    
    args = parser.parse_args()
    
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
        items = [dict(item) for item in list(ds)[:args.limit]]
        
        print("=== IFEval+ Dry Run ===\n")
        print(f"Total dataset size: {len(ds)}")
        print(f"Showing first {len(items)} items:\n")
        
        for i, item in enumerate(items):
            print(f"--- Item {i} (key={item['key']}) ---")
            print(f"Instructions: {item['instruction_id_list']}")
            prompt_text = str(item['prompt'])
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
