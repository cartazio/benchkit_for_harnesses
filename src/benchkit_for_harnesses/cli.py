"""Command-line interface for benchkit.

Subcommands:

    benchkit run      — run a standard benchmark (BABILong / InfiniteBench / LongBench-v2)
    benchkit ifeval   — IFEval+ system-prompt overhead experiment (forwarded)
    benchkit bundled  — bundled-bench alignment-tax experiment (forwarded)
    benchkit list     — enumerate benchmarks or harnesses
    benchkit probe    — one-shot sanity-check a harness+model pair
    benchkit log      — tail the persistent run ledger

Every invocation appends one entry to the run ledger at
``$BENCHKIT_HOME/ledger.jsonl`` (default ``~/.benchkit/ledger.jsonl``),
capturing timestamp, argv, exit code, and command-specific metrics.
`benchkit run` auto-defaults ``--output`` to ``$BENCHKIT_HOME/runs/`` so
archives are always persisted even if the user forgets the flag.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from .benchmarks import BENCHMARKS
from .harnesses.dispatch import KNOWN_HARNESSES, run_harness
from .ledger import (
    append_entry,
    format_entries,
    ledger_path,
    runs_dir,
    tail_entries,
)
from .runner import run_benchmark_batch

__all__ = ["main"]

# Each handler returns (exit_code, ledger_extra). The wrapping main()
# merges those extras into a standard envelope and appends to the ledger.
HandlerResult = tuple[int, dict[str, Any]]


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> HandlerResult:
    """Run a standard benchmark via runner.run_benchmark_batch."""
    # Auto-persist: default --output to the benchkit runs dir if not set.
    output_dir = args.output if args.output is not None else runs_dir()

    try:
        results, output_path = run_benchmark_batch(
            benchmark_name=args.benchmark,
            harness=args.harness,
            model=args.model,
            output_dir=output_dir,
            limit=args.limit,
            task=args.task,
            length=args.length,
            system_prompt=args.system_prompt,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2, {
            "benchmark": args.benchmark,
            "harness": args.harness,
            "model": args.model,
            "error": str(e),
        }

    correct = sum(1 for r in results if r.correct)
    n = len(results)
    accuracy = correct / n if n else 0.0
    avg_latency = sum(r.latency_ms for r in results) / n if n else 0.0

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Harness: {args.harness} | Model: {args.model}")
    print(sep)
    print(f"Completed: {n} items")
    print(f"Accuracy: {correct}/{n} ({accuracy:.1%})")
    print(f"Avg Latency: {avg_latency:.0f}ms")
    if output_path is not None:
        print(f"Output: {output_path}")

    return 0, {
        "benchmark": args.benchmark,
        "harness": args.harness,
        "model": args.model,
        "task": args.task,
        "length": args.length,
        "n_items": n,
        "correct": correct,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "output_path": str(output_path) if output_path else None,
    }


def cmd_list(args: argparse.Namespace) -> HandlerResult:
    """Enumerate benchmarks or harnesses."""
    if args.target == "benchmarks":
        for name, cfg in BENCHMARKS.items():
            tasks = ", ".join(cfg.tasks) if cfg.tasks else "(single split)"
            lengths = ", ".join(cfg.lengths) if cfg.lengths else "—"
            # Best-effort eval-mode inference: the strict wrapper name
            # starts with '_eval_bracketed_strict'; anything else is loose.
            eval_mode = (
                "strict bracket"
                if getattr(cfg.eval_fn, "__name__", "").endswith("strict")
                else "loose bracket"
            )
            print(f"{name}")
            print(f"  hf_path: {cfg.hf_path}")
            print(f"  tasks:   {tasks}")
            print(f"  lengths: {lengths}")
            print(f"  eval:    {eval_mode}")
            print()
    elif args.target == "harnesses":
        for h in KNOWN_HARNESSES:
            print(h)
        print()
        print("(unknown names fall through to the default ohp-style adapter)")
    return 0, {"target": args.target}


def cmd_probe(args: argparse.Namespace) -> HandlerResult:
    """Run one trivial prompt through a harness to sanity-check it."""
    prompt = args.prompt or (
        "Say PONG wrapped in {[{[ and ]}]} markers. Nothing else."
    )
    response, latency_ms = run_harness(
        harness=args.harness,
        model=args.model,
        prompt=prompt,
        system_prompt=args.system_prompt,
        timeout_sec=args.timeout,
    )
    print(f"[{latency_ms}ms] {response}")
    is_error = response.startswith("[ERROR")
    return (1 if is_error else 0), {
        "harness": args.harness,
        "model": args.model,
        "latency_ms": latency_ms,
        "success": not is_error,
        "response_preview": response[:200],
    }


def cmd_log(args: argparse.Namespace) -> HandlerResult:
    """Tail the persistent run ledger."""
    entries = tail_entries(args.n)
    if not entries:
        print(f"(empty ledger at {ledger_path()})", file=sys.stderr)
        return 0, {"count": 0}
    print(format_entries(entries))
    return 0, {"count": len(entries)}


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchkit",
        description="Unified benchmarking toolkit for coding-agent CLI harnesses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Examples:
  benchkit run --benchmark babilong --task qa1 --length 0k --harness punkin --model sonnet
  benchkit run --benchmark longbenchv2 --harness claude --model sonnet --limit 20
  benchkit ifeval run --condition heavy -m sonnet -n 50
  benchkit ifeval compare baseline.jsonl heavy.jsonl
  benchkit bundled --preset core --bundle-sizes 1,3,5 --dry-run
  benchkit list benchmarks
  benchkit probe --harness codex --model gpt-5.3-codex-spark
  benchkit log -n 20

Persistent state:
  Ledger:   {ledger_path()}
  Runs:     {runs_dir()}
  Override: export BENCHKIT_HOME=/some/path
""",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # ---- run ----
    run_p = sub.add_parser("run", help="Run a standard benchmark against a harness")
    run_p.add_argument(
        "--benchmark", "-B", required=True,
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to run",
    )
    run_p.add_argument(
        "--harness", "-H", default="ohp",
        help="Harness CLI: ohp, punkin, opencode, monopi, omp, claude, codex, or any compatible CLI in PATH",
    )
    run_p.add_argument(
        "--model", "-M", required=True,
        help="Model identifier passed to the harness",
    )
    run_p.add_argument(
        "--task", "-T", default=None,
        help="Task within benchmark (required for multi-task; see 'benchkit list benchmarks')",
    )
    run_p.add_argument(
        "--length", "-L", default=None,
        help="Context-length band (BABILong only; see 'benchkit list benchmarks')",
    )
    run_p.add_argument(
        "--limit", type=int, default=None,
        help="Max items to run (default: all)",
    )
    run_p.add_argument(
        "--output", "-O", type=Path, default=None,
        help=f"Output directory (default: {runs_dir()})",
    )
    run_p.add_argument(
        "--system-prompt", default=None,
        help=(
            "System prompt override (behavior differs per harness: "
            "ohp/punkin pass --system-prompt FILE; claude uses "
            "--append-system-prompt-file; codex has no system flag on "
            "exec so the string is prepended to the user prompt)"
        ),
    )
    run_p.set_defaults(handler=cmd_run)

    # ---- ifeval / bundled forwarders ----
    # Added as placeholder subparsers so 'benchkit --help' lists them, but
    # actual dispatch is short-circuited in main() before argparse — see
    # comment in main().
    sub.add_parser(
        "ifeval",
        help="IFEval+ system-prompt overhead experiment (try 'benchkit ifeval --help')",
        add_help=False,
    )
    sub.add_parser(
        "bundled",
        help="bundled-bench alignment-tax experiment (try 'benchkit bundled --help')",
        add_help=False,
    )

    # ---- list ----
    list_p = sub.add_parser("list", help="Enumerate available benchmarks or harnesses")
    list_p.add_argument(
        "target", choices=["benchmarks", "harnesses"],
        help="What to list",
    )
    list_p.set_defaults(handler=cmd_list)

    # ---- probe ----
    probe_p = sub.add_parser(
        "probe", help="One-shot sanity check: run a trivial prompt through a harness"
    )
    probe_p.add_argument("--harness", "-H", required=True, help="Harness to probe")
    probe_p.add_argument("--model", "-M", required=True, help="Model identifier")
    probe_p.add_argument(
        "--prompt", default=None,
        help="Prompt text (default: a PONG bracket test)",
    )
    probe_p.add_argument(
        "--system-prompt", default=None, help="System prompt override"
    )
    probe_p.add_argument(
        "--timeout", type=int, default=60, help="Timeout seconds (default: 60)"
    )
    probe_p.set_defaults(handler=cmd_probe)

    # ---- log ----
    log_p = sub.add_parser("log", help="Tail the persistent run ledger")
    log_p.add_argument(
        "-n", type=int, default=20,
        help="How many recent entries to show (default: 20)",
    )
    log_p.set_defaults(handler=cmd_log)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _dispatch_forwarded(subcmd: str, rest: list[str]) -> HandlerResult:
    """Run a forwarded subcommand (ifeval / bundled) and capture its exit code.

    We can't introspect the sub-experiment's outcome in detail, but we can
    record the command line and exit status so the ledger sees every run.
    """
    if subcmd == "ifeval":
        from .ifeval.experiment import main as ifeval_main
        code = ifeval_main(rest)
    elif subcmd == "bundled":
        from .bundled_bench.experiment import main as bundled_main
        code = bundled_main(rest)
    else:
        raise ValueError(f"unknown forwarded subcommand: {subcmd}")
    return code, {"forwarded_argv": rest}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Always appends to the ledger on exit."""
    if argv is None:
        argv = sys.argv[1:]

    start_ms = time.monotonic()
    cmd = argv[0] if argv else "(none)"
    exit_code = 0
    extras: dict[str, Any] = {}
    exc_repr: str | None = None

    try:
        # Forwarding subcommands bypass argparse entirely — argparse.REMAINDER
        # drops args starting with '--', which breaks the sub-experiments'
        # own flag parsing. Short-circuit cleanly here.
        if argv and argv[0] in ("ifeval", "bundled"):
            exit_code, extras = _dispatch_forwarded(argv[0], argv[1:])
        else:
            parser = _build_parser()
            args = parser.parse_args(argv)
            handler = args.handler  # set_defaults assigned this per subcommand
            exit_code, extras = handler(args)
    except SystemExit as e:
        # argparse uses SystemExit for --help and parse errors; propagate
        # but still record.
        exit_code = int(e.code) if isinstance(e.code, int) else 2
        raise
    except BaseException as e:
        exc_repr = f"{type(e).__name__}: {e}"
        exit_code = 1
        raise
    finally:
        duration_ms = int((time.monotonic() - start_ms) * 1000)
        entry: dict[str, Any] = {
            "cmd": cmd,
            "argv": list(argv),
            "duration_ms": duration_ms,
            "exit_code": exit_code,
            **extras,
        }
        if exc_repr is not None:
            entry["exception"] = exc_repr
        try:
            append_entry(entry)
        except OSError:
            # Ledger write must never take down a real run.
            pass

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
