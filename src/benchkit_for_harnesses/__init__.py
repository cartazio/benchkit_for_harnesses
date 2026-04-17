"""BenchKit for Harnesses — unified benchmarking toolkit for coding-agent CLIs.

Primary entry point is the ``benchkit`` console script (see
:mod:`benchkit_for_harnesses.cli`). This module re-exports a minimal
Python-API surface for programmatic use.

Architecture:

* :mod:`.core`       — unified loop substrate (``run_items`` / ``run_items_async``)
* :mod:`.responders` — transport adapters (harness / api_chat / api_model)
* :mod:`.archive`    — Carter archive format (``ArchiveWriter``)
* :mod:`.ledger`     — persistent run ledger
* :mod:`.brackets`   — answer-bracket protocol (extraction + strict/loose eval)
* :mod:`.runner`     — standard benchmark batch (BABILong / InfiniteBench / LongBench-v2)
* :mod:`.harnesses`  — CLI dispatch (ohp / punkin / claude / codex / ...)
* :mod:`.ifeval`     — IFEval+ system-prompt overhead experiment
* :mod:`.bundled_bench` — bundled-question alignment-tax experiment
"""

from .archive import ArchiveWriter, make_archive_path
from .harnesses.dispatch import run_harness
from .runner import run_benchmark_batch

__version__ = "0.1.0"
__author__ = "Carter Schonwald"

__all__ = [
    "ArchiveWriter",
    "make_archive_path",
    "run_benchmark_batch",
    "run_harness",
]
