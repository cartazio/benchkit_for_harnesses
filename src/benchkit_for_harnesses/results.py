"""JSONL results I/O: save, load, clear, stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def clear_results(path: Path) -> None:
    """Clear results file for fresh experiment run."""
    if path.exists():
        path.unlink()


def save_jsonl(records: list[dict[str, Any]], path: Path, *, append: bool = True) -> None:
    """Save records as JSON lines."""
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from JSON lines."""
    results: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results



