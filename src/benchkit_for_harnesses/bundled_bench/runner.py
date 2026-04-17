"""Async orchestration for bundled-bench trials.

Thin adapter over :func:`benchkit_for_harnesses.core.run_items_async`:
per-bundle responders pick chat vs base-completion by ModelSpec, and a
failure-record builder keeps cell n-counts honest when an individual
bundle's API call throws.
"""

from __future__ import annotations

from datetime import datetime, timezone

from benchkit_for_harnesses.core import run_items_async
from benchkit_for_harnesses.responders import api_model_responder

from .harness import (
    Bundle,
    BundleResult,
    QuestionResult,
    SYSTEM_PROMPTS,
    TrialConfig,
    format_bundle_prompt,
    parse_responses,
)

__all__ = ["run_experiment"]


def _build_success(
    _idx: int,
    bundle: Bundle,
    _prompt: str,
    _meta: Bundle,
    response: str,
    latency_ms: int,
    config: TrialConfig,
) -> BundleResult:
    question_results = parse_responses(response, bundle)
    return BundleResult(
        bundle_id=bundle.id,
        trial_config=config,
        question_results=question_results,
        raw_model_output=response[:5000],
        latency_ms=float(latency_ms),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _build_failure(
    _idx: int,
    bundle: Bundle,
    _prompt: str,
    _meta: Bundle,
    exc: BaseException,
    config: TrialConfig,
) -> BundleResult:
    """Represent a failed call as a zero-accuracy result \u2014 no silent drops."""
    qrs = [
        QuestionResult(
            question_id=q.id,
            position=j,
            expected=q.answer,
            raw_response="",
            correct=False,
            attempted=False,
        )
        for j, q in enumerate(bundle.questions)
    ]
    msg = f"[ERROR] {type(exc).__name__}: {exc}"
    return BundleResult(
        bundle_id=bundle.id,
        trial_config=config,
        question_results=qrs,
        raw_model_output=msg[:5000],
        latency_ms=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def run_experiment(
    bundles: list[Bundle],
    config: TrialConfig,
    max_concurrent: int = 5,
) -> list[BundleResult]:
    """Run all bundles for a given trial config with bounded concurrency.

    Returned list matches the input order. Failures are explicit failure
    records so cell sizes are stable across partial-failure runs.
    """
    responder = api_model_responder(config.model)
    system = SYSTEM_PROMPTS.get(config.condition, "")

    def _format(bundle: Bundle) -> tuple[str, Bundle]:
        return format_bundle_prompt(bundle), bundle

    results, _ = await run_items_async(
        items=bundles,
        responder=responder,
        format_fn=_format,
        build_record=lambda i, b, p, m, r, ms: _build_success(i, b, p, m, r, ms, config),
        build_failure_record=lambda i, b, p, m, e: _build_failure(i, b, p, m, e, config),
        description="bundled_bench_trial",  # caller writes via save_results, not via core
        output_dir=None,  # trial-level archiving lives in run.py
        system_prompt=system,
        max_concurrent=max_concurrent,
    )
    return results
