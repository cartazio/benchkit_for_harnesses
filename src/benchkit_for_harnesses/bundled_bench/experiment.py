#!/usr/bin/env python3
"""bundled-bench experiment runner: matrix of (model, bundle_size, condition) trials.

Primary entry point is the unified CLI: ``benchkit bundled …``.
Direct invocation ``python -m benchkit_for_harnesses.bundled_bench.experiment``
also works for developers running out of the source tree.

Usage (via benchkit):
  benchkit bundled --dry-run                                  # preview matrix
  benchkit bundled --preset core --bundle-sizes 1,3,5,7       # core experiment
  benchkit bundled --model llama-3.1-8b-instruct --bundle-sizes 1,5
  benchkit bundled --preset core --base-only --bundle-sizes 1,3   # base coherence
  benchkit bundled --preset full --bundle-sizes 1,3,5,7,10    # full matrix
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from benchkit_for_harnesses.results import clear_results

from .harness import (
    BundleResult,
    ModelSpec,
    PromptCondition,
    TrialConfig,
    make_bundles,
    save_results,
    load_results,
    summarize_results,
)
from .models import ALL_MODELS, CORE_MODELS, EXTENDED_MODELS, FULL_MODELS
from .questions import load_synthetic, load_mmlu_from_hf
from .runner import run_experiment


def build_configs(
    models: list[ModelSpec],
    bundle_sizes: list[int],
    conditions: list[PromptCondition],
    base_only: bool = False,
    seed: int = 42,
) -> list[TrialConfig]:
    """Build the full matrix of trial configurations."""
    configs: list[TrialConfig] = []
    for model in models:
        if base_only and not model.is_base:
            continue

        for bsize in bundle_sizes:
            for cond in conditions:
                # Base models only run with NONE condition
                # (they don't follow system prompts)
                if model.is_base and cond != PromptCondition.NONE:
                    continue
                # Instruct models run all conditions
                configs.append(TrialConfig(
                    model=model,
                    condition=cond,
                    bundle_size=bsize,
                    seed=seed,
                ))
    return configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bundled-bench experiment runner")
    parser.add_argument(
        "--preset",
        choices=["core", "extended", "full"],
        default="core",
        help="Model preset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run a specific model by name (overrides --preset)",
    )
    parser.add_argument(
        "--bundle-sizes",
        type=str,
        default="1,3,5,7",
        help="Comma-separated bundle sizes (default: 1,3,5,7)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="none,minimal,full",
        help="Comma-separated conditions (default: none,minimal,full)",
    )
    parser.add_argument(
        "--n-bundles",
        type=int,
        default=20,
        help="Number of bundles per (model, condition, size) cell",
    )
    parser.add_argument(
        "--question-source",
        choices=["synthetic", "mmlu"],
        default="synthetic",
        help="Question source (default: synthetic for testing)",
    )
    parser.add_argument(
        "--mmlu-subjects",
        type=str,
        default=None,
        help="Comma-separated MMLU subjects (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only run base models (for coherence validation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment plan without running",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bundle construction",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API calls",
    )

    args = parser.parse_args(argv)

    # Parse bundle sizes and conditions
    bundle_sizes = [int(x) for x in args.bundle_sizes.split(",")]
    conditions = [PromptCondition(x.strip()) for x in args.conditions.split(",")]

    # Select models
    if args.model:
        if args.model not in ALL_MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(ALL_MODELS.keys())}")
            sys.exit(1)
        models = [ALL_MODELS[args.model]]
    else:
        presets = {
            "core": CORE_MODELS,
            "extended": EXTENDED_MODELS,
            "full": FULL_MODELS,
        }
        models = presets[args.preset]

    # Build trial configs
    configs = build_configs(
        models=models,
        bundle_sizes=bundle_sizes,
        conditions=conditions,
        base_only=args.base_only,
        seed=args.seed,
    )

    # Load questions
    if args.question_source == "synthetic":
        all_questions = load_synthetic(500)
        print(f"Loaded {len(all_questions)} synthetic questions")
    else:
        subjects = None
        if args.mmlu_subjects:
            subjects = [s.strip() for s in args.mmlu_subjects.split(",")]
        all_questions = load_mmlu_from_hf(subjects=subjects)
        print(f"Loaded {len(all_questions)} MMLU questions")

    # Experiment plan
    print(f"\n{'='*60}")
    print(f"EXPERIMENT PLAN")
    print(f"{'='*60}")
    print(f"Models:       {len(set(c.model.name for c in configs))}")
    print(f"Bundle sizes: {bundle_sizes}")
    print(f"Conditions:   {[c.value for c in conditions]}")
    print(f"Bundles/cell: {args.n_bundles}")
    print(f"Total cells:  {len(configs)}")
    print(f"Total API calls: {len(configs) * args.n_bundles}")
    print(f"Question pool: {len(all_questions)}")
    print(f"Output: {args.output}")
    print()

    for c in configs:
        base_tag = "[BASE]" if c.model.is_base else "[INST]"
        print(f"  {base_tag} {c.model.name:30s} N={c.bundle_size:2d} cond={c.condition.value}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute {len(configs) * args.n_bundles} API calls.")
        return 0

    # Run experiment
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clear_results(output_path)

    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT")
    print(f"{'='*60}")

    async def _run_all() -> list[BundleResult]:
        all_results: list[BundleResult] = []
        for i, config in enumerate(configs):
            base_tag = "[BASE]" if config.model.is_base else "[INST]"
            print(
                f"\n[{i+1}/{len(configs)}] {base_tag} {config.model.name} "
                f"N={config.bundle_size} cond={config.condition.value}"
            )

            bundles = make_bundles(
                all_questions,
                bundle_size=config.bundle_size,
                n_bundles=args.n_bundles,
                seed=config.seed,
            )
            print(f"  Created {len(bundles)} bundles")

            t0 = time.monotonic()
            results = await run_experiment(
                bundles, config, max_concurrent=args.max_concurrent,
            )
            elapsed = time.monotonic() - t0

            save_results(results, output_path)

            accs = [r.accuracy for r in results]
            comps = [r.completion_rate for r in results]
            mean_acc = sum(accs) / len(accs) if accs else 0
            mean_comp = sum(comps) / len(comps) if comps else 0

            print(
                f"  {len(results)} bundles in {elapsed:.1f}s | "
                f"acc={mean_acc:.3f} comp={mean_comp:.3f}"
            )
            all_results.extend(results)

        return all_results

    asyncio.run(_run_all())

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    results_data = load_results(output_path)
    summary = summarize_results(results_data)

    for _key, stats in sorted(summary.items()):
        base_tag = "[BASE]" if stats["is_base"] else "[INST]"
        print(
            f"  {base_tag} {stats['model']:30s} "
            f"N={stats['bundle_size']:2d} "
            f"cond={stats['condition']:8s} "
            f"acc={stats['mean_accuracy']:.3f} "
            f"comp={stats['mean_completion']:.3f} "
            f"(n={stats['n_bundles']})"
        )

    print(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
