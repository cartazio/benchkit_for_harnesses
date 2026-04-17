"""Tests for the persistent run ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchkit_for_harnesses.ledger import (
    LEDGER_FILENAME,
    append_entry,
    benchkit_home,
    format_entries,
    ledger_path,
    read_entries,
    runs_dir,
    tail_entries,
)


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate BENCHKIT_HOME to a tmp dir for each test."""
    monkeypatch.setenv("BENCHKIT_HOME", str(tmp_path))
    return tmp_path


def test_benchkit_home_from_env(tmp_home: Path) -> None:
    assert benchkit_home() == tmp_home


def test_ledger_path_under_home(tmp_home: Path) -> None:
    assert ledger_path() == tmp_home / LEDGER_FILENAME


def test_runs_dir_under_home(tmp_home: Path) -> None:
    assert runs_dir() == tmp_home / "runs"


def test_append_creates_ledger_and_parent(tmp_home: Path) -> None:
    # parent may not exist yet — append_entry must create it.
    entry = {"cmd": "probe", "argv": ["probe"], "exit_code": 0}
    path = append_entry(entry)
    assert path == ledger_path()
    assert path.exists()
    assert path.parent == tmp_home
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["cmd"] == "probe"
    assert "timestamp" in parsed  # auto-set


def test_append_preserves_supplied_timestamp(tmp_home: Path) -> None:
    append_entry(
        {"cmd": "probe", "timestamp": "2026-01-01T00:00:00-05:00", "exit_code": 0}
    )
    entries = read_entries()
    assert entries[0]["timestamp"] == "2026-01-01T00:00:00-05:00"


def test_multiple_appends_ordered(tmp_home: Path) -> None:
    for i in range(5):
        append_entry({"cmd": "run", "exit_code": 0, "idx": i})
    entries = read_entries()
    assert [e["idx"] for e in entries] == [0, 1, 2, 3, 4]


def test_read_empty_when_no_ledger(tmp_home: Path) -> None:
    assert read_entries() == []


def test_tail_entries(tmp_home: Path) -> None:
    for i in range(5):
        append_entry({"cmd": "run", "exit_code": 0, "idx": i})
    assert [e["idx"] for e in tail_entries(2)] == [3, 4]
    assert tail_entries(0) == []
    assert [e["idx"] for e in tail_entries(100)] == [0, 1, 2, 3, 4]


def test_read_skips_corrupt_lines(tmp_home: Path) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"cmd": "run", "exit_code": 0}\n'
        '{not valid json\n'
        '{"cmd": "probe", "exit_code": 0}\n'
    )
    entries = read_entries()
    assert len(entries) == 2
    assert [e["cmd"] for e in entries] == ["run", "probe"]


def test_format_entries_render(tmp_home: Path) -> None:
    entries = [
        {
            "timestamp": "2026-04-17T16:53:07-04:00",
            "cmd": "run",
            "argv": ["run", "--benchmark", "babilong"],
            "exit_code": 0,
            "duration_ms": 1234,
            "benchmark": "babilong",
            "harness": "claude",
            "model": "sonnet",
            "accuracy": 1.0,
            "n_items": 2,
            "output_path": "/tmp/x.jsonl",
        },
        {
            "timestamp": "2026-04-17T16:54:12-04:00",
            "cmd": "probe",
            "exit_code": 1,
            "duration_ms": 500,
            "harness": "codex",
            "model": "gpt",
            "latency_ms": 450,
        },
    ]
    rendered = format_entries(entries)
    assert "run" in rendered
    assert "OK" in rendered
    assert "babilong" in rendered
    assert "100.0%" in rendered
    assert "rc=1" in rendered  # probe failed
    assert "codex" in rendered


def test_isolation_between_tests(tmp_home: Path) -> None:
    # Fresh ledger per test (proves the fixture actually isolates).
    assert read_entries() == []
