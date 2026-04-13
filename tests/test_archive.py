"""Tests for archive module."""

from pathlib import Path

from benchkit_for_harnesses.archive import make_archive_path, compute_hash


def test_compute_hash() -> None:
    """Test hash computation."""
    content = b"test content"
    hash_val = compute_hash(content)
    assert len(hash_val) == 12
    assert all(c in "0123456789abcdef" for c in hash_val)


def test_make_archive_path(tmp_path: Path) -> None:
    """Test archive path generation."""
    path = make_archive_path(
        tmp_path,
        "test benchmark",
        version=1,
        extension="jsonl",
    )

    # Check structure
    assert path.parent == tmp_path
    assert "test_benchmark_v1_" in path.name
    assert "NYC_" in path.name
    assert path.name.endswith(".jsonl")
