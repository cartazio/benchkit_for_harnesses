"""
Model runner for bundled-bench: dispatches to benchkit_for_harnesses.api_runner for HTTP,
wires bundled-bench-specific types for orchestration.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from benchkit_for_harnesses.api_runner import run_chat, run_completion

from .harness import (
    Bundle,
    BundleResult,
    SYSTEM_PROMPTS,
    TrialConfig,
    format_bundle_prompt,
    parse_responses,
)


# ---------------------------------------------------------------------------
# Runner protocol
# ---------------------------------------------------------------------------

async def run_bundle(
    bundle: Bundle,
    config: TrialConfig,
) -> BundleResult:
    """
    Run a single bundle against a model under a specific condition.
    Dispatches to the appropriate provider via benchkit_for_harnesses.api_runner.
    """
    prompt = format_bundle_prompt(bundle)
    system = SYSTEM_PROMPTS.get(config.condition, "")

    t0 = time.monotonic()

    if config.model.is_base:
        raw_output = await run_completion(config.model, prompt, system)
    else:
        raw_output = await run_chat(config.model, prompt, system)

    latency_ms = (time.monotonic() - t0) * 1000

    question_results = parse_responses(raw_output, bundle)

    return BundleResult(
        bundle_id=bundle.id,
        trial_config=config,
        question_results=question_results,
        raw_model_output=raw_output[:5000],
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Batch runner with concurrency control
# ---------------------------------------------------------------------------

async def run_experiment(
    bundles: list[Bundle],
    config: TrialConfig,
    max_concurrent: int = 5,
) -> list[BundleResult]:
    """
    Run all bundles for a given config with concurrency limiting.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(bundle: Bundle) -> BundleResult:
        async with semaphore:
            return await run_bundle(bundle, config)

    tasks = [_run_one(b) for b in bundles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions, log them
    good: list[BundleResult] = []
    for r in results:
        if isinstance(r, BaseException):
            print(f"ERROR: {r}")
        else:
            good.append(r)

    return good
