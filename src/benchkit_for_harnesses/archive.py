"""Carter archive format utilities (versioned, timestamped, hashed)."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

__all__ = ["make_archive_path", "compute_hash"]

NYC_ZONE = ZoneInfo("America/New_York")


def compute_hash(content: bytes, length: int = 12) -> str:
    """
    Compute SHA3-256 hash of content.

    Args:
        content: Bytes to hash
        length: Number of hex characters to return (default: 12)

    Returns:
        First `length` hex characters of SHA3-256 hash
    """
    return hashlib.sha3_256(content).hexdigest()[:length]


def make_archive_path(
    base_dir: str | Path,
    description: str,
    version: int = 1,
    extension: str = "jsonl",
) -> Path:
    """
    Generate Carter archive filename.

    Format: {desc}_v{n}_{YYYYMMDDTHHMMSS}NYC_{hash}

    Args:
        base_dir: Directory to place file in
        description: Human-readable description (spaces → underscores)
        version: Version number (default: 1)
        extension: File extension (default: jsonl)

    Returns:
        Path to generated filename (file not created, just path)
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(NYC_ZONE)
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    safe_desc = description.replace(" ", "_")

    # For path generation, we use a placeholder hash that will be updated
    # when actual content is written
    placeholder_hash = "0" * 12

    filename = f"{safe_desc}_v{version}_{timestamp}NYC_{placeholder_hash}.{extension}"
    return base_dir / filename


def finalize_archive_path(
    content_bytes: bytes,
    draft_path: Path,
) -> Path:
    """
    Rename draft file to final archive path with actual content hash.

    Args:
        content_bytes: Final content to hash
        draft_path: Current path (with placeholder hash)

    Returns:
        Final path (file already renamed)
    """
    actual_hash = compute_hash(content_bytes)
    final_name = draft_path.name.replace("0" * 12, actual_hash)
    final_path = draft_path.parent / final_name

    if draft_path.exists() and draft_path != final_path:
        draft_path.rename(final_path)

    return final_path
