"""Integration tests for the unified `benchkit` CLI (argv-driven, no subprocess)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchkit_for_harnesses.cli import main

@pytest.fixture(autouse=True)
def _isolate_ledger(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Every CLI test writes to a throwaway BENCHKIT_HOME."""
    monkeypatch.setenv("BENCHKIT_HOME", str(tmp_path))
    yield



def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "benchkit" in out
    assert "run" in out
    assert "ifeval" in out
    assert "bundled" in out
    assert "list" in out
    assert "probe" in out


def test_list_benchmarks(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["list", "benchmarks"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "babilong" in out
    assert "infinitebench" in out
    assert "longbenchv2" in out


def test_list_harnesses(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["list", "harnesses"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "claude" in out
    assert "codex" in out
    assert "punkin" in out


def test_run_missing_task_fails_fast(capsys: pytest.CaptureFixture[str]) -> None:
    # BABILong needs --task AND --length; omitting both must fail without
    # ever reaching the dataset loader.
    rc = main([
        "run",
        "--benchmark", "babilong",
        "--harness", "nonexistent-cli",
        "--model", "whatever",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "requires --task" in err


def test_run_bad_task_fails_fast(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([
        "run",
        "--benchmark", "babilong",
        "--task", "qa_nonexistent",
        "--length", "0k",
        "--harness", "nonexistent-cli",
        "--model", "whatever",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Unknown task" in err


def test_ifeval_dry_run_forwarded(capsys: pytest.CaptureFixture[str]) -> None:
    # Forwarded subcommand must pass --dry-run through to ifeval's parser
    # despite arg starting with '--'. This regression-tests the short-circuit.
    rc = main(["ifeval", "--dry-run", "-n", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "IFEval+ Dry Run" in out


def test_unknown_subcommand_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["nosuchcmd"])
    assert exc.value.code != 0


def test_run_benchmark_choice_validation(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([
            "run",
            "--benchmark", "nosuchbenchmark",
            "--harness", "punkin",
            "--model", "whatever",
        ])
