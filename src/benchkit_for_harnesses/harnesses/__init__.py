"""CLI harness dispatch — adapter pattern for ohp/punkin/claude/codex/..."""

from .dispatch import HarnessType, run_harness

__all__ = ["HarnessType", "run_harness"]
