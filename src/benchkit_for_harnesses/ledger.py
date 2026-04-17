"""Persistent run ledger.

Every `benchkit` invocation appends one JSONL entry capturing:

  * when it ran (America/New_York, ISO-8601 with offset)
  * how long it took
  * what was invoked (subcommand + argv)
  * the outcome (exit code + command-specific metrics)
  * where the run's archive landed (if any)

Location: ``$BENCHKIT_HOME/ledger.jsonl`` (default ``~/.benchkit/ledger.jsonl``).
Override via the ``BENCHKIT_HOME`` env var.

The ledger is crash-safe: each entry is appended with a trailing newline
and flushed before the function returns. Concurrent writers are serialized
by the OS append guarantee on POSIX filesystems.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

__all__ = [
    "LEDGER_FILENAME",
    "benchkit_home",
    "ledger_path",
    "runs_dir",
    "append_entry",
    "read_entries",
    "tail_entries",
]

LEDGER_FILENAME = "ledger.jsonl"
_NYC = ZoneInfo("America/New_York")


def benchkit_home() -> Path:
    """Resolve the benchkit home directory.

    Defaults to ``~/.benchkit``; overridden by the ``BENCHKIT_HOME`` env
    var. Always returns an absolute path; creates the directory lazily
    when called from write paths.
    """
    raw = os.environ.get("BENCHKIT_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".benchkit"


def ledger_path() -> Path:
    """Absolute path to the ledger JSONL."""
    return benchkit_home() / LEDGER_FILENAME


def runs_dir() -> Path:
    """Default archive directory used when `benchkit run` omits --output."""
    return benchkit_home() / "runs"


def now_nyc_iso() -> str:
    """Current time as NYC-local ISO-8601 with offset."""
    return datetime.now(_NYC).isoformat(timespec="seconds")


def append_entry(entry: dict[str, Any]) -> Path:
    """Append one entry to the ledger, flushing before return.

    The entry gets ``timestamp`` auto-set if missing. Any caller-supplied
    timestamp is preserved (used by tests to pin clock).
    """
    entry.setdefault("timestamp", now_nyc_iso())
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
    return path


def read_entries(path: Path | None = None) -> list[dict[str, Any]]:
    """Read all entries from the ledger. Silent empty list if no ledger yet."""
    p = path or ledger_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial write (very rare) should not poison reads.
                continue
    return out


def tail_entries(n: int, path: Path | None = None) -> list[dict[str, Any]]:
    """Return the last ``n`` entries in chronological order."""
    entries = read_entries(path)
    if n <= 0 or not entries:
        return []
    return entries[-n:]


def format_entry_line(entry: dict[str, Any]) -> str:
    """Render one ledger entry as a single compact terminal line."""
    ts = entry.get("timestamp", "?")
    cmd = entry.get("cmd", "?")
    rc = entry.get("exit_code", "?")
    status = "OK" if rc == 0 else f"rc={rc}"
    extra: list[str] = []
    if cmd == "run":
        extra.append(
            f"{entry.get('benchmark', '?')} {entry.get('harness', '?')}/{entry.get('model', '?')}"
        )
        acc = entry.get("accuracy")
        n = entry.get("n_items")
        if acc is not None and n is not None:
            extra.append(f"acc={acc:.1%} n={n}")
        if entry.get("output_path"):
            extra.append(f"→ {entry['output_path']}")
    elif cmd == "probe":
        extra.append(f"{entry.get('harness', '?')}/{entry.get('model', '?')}")
        if "latency_ms" in entry:
            extra.append(f"{entry['latency_ms']}ms")
    elif cmd in ("ifeval", "bundled"):
        argv: list[str] = entry.get("argv") or []
        if argv:
            extra.append(" ".join(argv[1:6]))  # skip the subcommand itself
    dur = entry.get("duration_ms")
    dur_s = f"{dur}ms" if dur is not None else ""
    return f"{ts}  {cmd:<7} {status:<6} {dur_s:<8} " + "  ".join(extra)


def format_entries(entries: Iterable[dict[str, Any]]) -> str:
    """Render a sequence of ledger entries as newline-joined lines."""
    return "\n".join(format_entry_line(e) for e in entries)
