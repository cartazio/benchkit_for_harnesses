"""Tests for archive module."""

from __future__ import annotations

import re
from pathlib import Path

from benchkit_for_harnesses.archive import (
    ArchiveWriter,
    compute_hash,
    finalize_archive_path,
    make_archive_path,
)


_NAME_RE = re.compile(
    r"^(?P<desc>.+)_v(?P<ver>\d+)_(?P<ts>\d{8}T\d{6})NYC_(?P<hash>[0-9a-f]{12})\.(?P<ext>\w+)$"
)


def test_compute_hash() -> None:
    content = b"test content"
    hash_val = compute_hash(content)
    assert len(hash_val) == 12
    assert all(c in "0123456789abcdef" for c in hash_val)


def test_compute_hash_deterministic() -> None:
    assert compute_hash(b"abc") == compute_hash(b"abc")
    assert compute_hash(b"abc") != compute_hash(b"abd")


def test_make_archive_path_structure(tmp_path: Path) -> None:
    path = make_archive_path(tmp_path, "test benchmark", version=1, extension="jsonl")
    assert path.parent == tmp_path
    m = _NAME_RE.match(path.name)
    assert m is not None, f"filename shape wrong: {path.name}"
    assert m["desc"] == "test_benchmark"
    assert m["ver"] == "1"
    assert m["ext"] == "jsonl"
    assert m["hash"] == "000000000000"  # placeholder


def test_finalize_rewrites_only_hash_field(tmp_path: Path) -> None:
    # Regression: a description containing twelve consecutive zeros must not
    # confuse the finalizer. The old implementation used str.replace and
    # corrupted the filename.
    draft = make_archive_path(tmp_path, "weird_000000000000_desc", version=1)
    draft.write_bytes(b"payload")
    final = finalize_archive_path(draft.read_bytes(), draft)
    m = _NAME_RE.match(final.name)
    assert m is not None, f"finalized name malformed: {final.name}"
    assert m["desc"] == "weird_000000000000_desc", "description zeros must survive"
    assert m["hash"] != "000000000000"
    assert m["hash"] == compute_hash(b"payload")
    assert final.exists()


def test_archive_writer_context_manager(tmp_path: Path) -> None:
    with ArchiveWriter(tmp_path, "demo") as w:
        w.write_line('{"a":1}')
        w.write_line('{"a":2}')
    final = w.final_path
    assert final is not None
    assert final.exists()
    m = _NAME_RE.match(final.name)
    assert m is not None
    assert m["hash"] == compute_hash(final.read_bytes())
    assert final.read_text().splitlines() == ['{"a":1}', '{"a":2}']


def test_archive_writer_write_record_jsonl(tmp_path: Path) -> None:
    with ArchiveWriter(tmp_path, "records") as w:
        w.write_record({"k": 1})
        w.write_record({"k": 2})
    assert w.final_path is not None
    lines = w.final_path.read_text().splitlines()
    assert lines == ['{"k": 1}', '{"k": 2}']


def test_archive_writer_empty_still_finalizes(tmp_path: Path) -> None:
    with ArchiveWriter(tmp_path, "empty") as w:
        pass
    assert w.final_path is not None
    assert w.final_path.exists()
    assert w.final_path.read_bytes() == b""


def test_archive_writer_finalizes_on_exception(tmp_path: Path) -> None:
    # Partial results should still be written/finalized on error — we don't
    # want to lose hours of benchmark runs because of an exception at item 99.
    w = ArchiveWriter(tmp_path, "crashy")
    try:
        with w:
            w.write_line('{"idx":0}')
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert w.final_path is not None
    assert w.final_path.exists()
    assert w.final_path.read_text() == '{"idx":0}\n'
