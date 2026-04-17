"""Carter archive format utilities (versioned, timestamped, hashed).

Filename: {desc}_v{n}_{YYYYMMDDTHHMMSS}NYC_{hash}.{ext}
  - hash: SHA3-256(file_contents), first 12 hex chars
  - ts: America/New_York, seconds resolution

The canonical entry point is `ArchiveWriter` — a context manager that writes
to a draft path (with a placeholder hash), streams records with per-line
flush, and atomically renames to the final (content-hashed) path on close.
Partial data is preserved even if the body raises.

`make_archive_path` / `finalize_archive_path` remain available for callers
that need manual two-phase control, but new code should prefer the writer.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

__all__ = [
    "ArchiveWriter",
    "compute_hash",
    "finalize_archive_path",
    "make_archive_path",
]

NYC_ZONE = ZoneInfo("America/New_York")
_PLACEHOLDER_HASH = "0" * 12


def compute_hash(content: bytes, length: int = 12) -> str:
    """Compute SHA3-256 hash of content, return first `length` hex chars."""
    return hashlib.sha3_256(content).hexdigest()[:length]


@dataclass(frozen=True)
class _Filename:
    """Structured archive filename — compose, don't string-replace."""

    description: str
    version: int
    timestamp: str  # YYYYMMDDTHHMMSS
    hash_hex: str
    extension: str

    def render(self) -> str:
        safe_desc = self.description.replace(" ", "_")
        return (
            f"{safe_desc}_v{self.version}_{self.timestamp}NYC_{self.hash_hex}"
            f".{self.extension}"
        )

    def with_hash(self, hash_hex: str) -> "_Filename":
        return _Filename(
            description=self.description,
            version=self.version,
            timestamp=self.timestamp,
            hash_hex=hash_hex,
            extension=self.extension,
        )


def _now_timestamp() -> str:
    return datetime.now(NYC_ZONE).strftime("%Y%m%dT%H%M%S")


def make_archive_path(
    base_dir: str | Path,
    description: str,
    version: int = 1,
    extension: str = "jsonl",
) -> Path:
    """Generate a draft archive path with a placeholder hash.

    The file is NOT created here. Caller is responsible for writing and then
    calling `finalize_archive_path` — or, preferably, using `ArchiveWriter`.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    name = _Filename(
        description=description,
        version=version,
        timestamp=_now_timestamp(),
        hash_hex=_PLACEHOLDER_HASH,
        extension=extension,
    )
    return base / name.render()


def _parse_filename(name: str) -> _Filename | None:
    """Parse a rendered archive filename back into its parts.

    Returns None if the name doesn't match the expected shape. We parse from
    the right so description may contain underscores, digits, or the literal
    placeholder hash value without confusion.
    """
    if "." not in name:
        return None
    stem, _, ext = name.rpartition(".")
    # stem = {desc}_v{n}_{ts}NYC_{hash}
    if not stem.endswith(tuple("0123456789abcdef")):
        return None
    try:
        prefix, hash_hex = stem.rsplit("_", 1)
        if len(hash_hex) != 12 or any(c not in "0123456789abcdef" for c in hash_hex):
            return None
        prefix2, ts_nyc = prefix.rsplit("_", 1)
        if not ts_nyc.endswith("NYC") or len(ts_nyc) != len("YYYYMMDDTHHMMSS") + 3:
            return None
        timestamp = ts_nyc[:-3]
        desc, ver_token = prefix2.rsplit("_v", 1)
        version = int(ver_token)
    except (ValueError, IndexError):
        return None
    return _Filename(
        description=desc,
        version=version,
        timestamp=timestamp,
        hash_hex=hash_hex,
        extension=ext,
    )


def finalize_archive_path(content_bytes: bytes, draft_path: Path) -> Path:
    """Rename a draft archive file to its content-hashed final path.

    Fails loudly if the draft filename doesn't parse — better to raise than
    produce a corrupt archive name silently.
    """
    parsed = _parse_filename(draft_path.name)
    if parsed is None:
        raise ValueError(
            f"Cannot finalize: {draft_path.name!r} is not a valid archive filename"
        )
    if parsed.hash_hex != _PLACEHOLDER_HASH:
        # Already finalized — idempotent no-op if content matches.
        return draft_path

    final = parsed.with_hash(compute_hash(content_bytes))
    final_path = draft_path.parent / final.render()
    if draft_path.exists() and draft_path != final_path:
        draft_path.rename(final_path)
    return final_path


class ArchiveWriter:
    """Context-manager writer for archive-format files.

    Usage:
        with ArchiveWriter(out_dir, "babilong_ohp_claude") as w:
            for record in results:
                w.write_record(record)
        print(w.final_path)  # content-hashed final path

    Guarantees:
      - every `write_line` / `write_record` is flushed (crash-safe progress)
      - on exit (success OR exception), the draft file is renamed to the
        content-hashed final path; partial data is preserved
      - if the body writes nothing, an empty file is still finalized
    """

    def __init__(
        self,
        base_dir: str | Path,
        description: str,
        version: int = 1,
        extension: str = "jsonl",
    ) -> None:
        self._base_dir = Path(base_dir)
        self._description = description
        self._version = version
        self._extension = extension
        self.draft_path: Path | None = None
        self.final_path: Path | None = None
        self._fh: io.TextIOWrapper | None = None

    def __enter__(self) -> "ArchiveWriter":
        self.draft_path = make_archive_path(
            self._base_dir, self._description, self._version, self._extension
        )
        self._fh = open(self.draft_path, "w", encoding="utf-8")
        return self

    def write_line(self, line: str) -> None:
        """Write a line of text (newline appended if missing) and flush."""
        if self._fh is None:
            raise RuntimeError("ArchiveWriter used outside its context")
        if not line.endswith("\n"):
            line = line + "\n"
        self._fh.write(line)
        self._fh.flush()

    def write_record(self, record: Mapping[str, Any]) -> None:
        """JSON-serialize a record and write as a JSONL line."""
        self.write_line(json.dumps(dict(record)))

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.draft_path is not None and self.draft_path.exists():
            self.final_path = finalize_archive_path(
                self.draft_path.read_bytes(), self.draft_path
            )
