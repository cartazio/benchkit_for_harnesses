"""Tests for per-harness invocation construction (no subprocess execution)."""

from __future__ import annotations

import os
from pathlib import Path

from benchkit_for_harnesses.harnesses.dispatch import build_invocation


def _cleanup(tempfiles: list[str]) -> None:
    for p in tempfiles:
        try:
            os.unlink(p)
        except OSError:
            pass


def test_default_adapter_punkin_shape() -> None:
    inv = build_invocation("punkin", "claude-sonnet-4", "hello world", None)
    try:
        assert inv.cmd[0] == "punkin"
        assert "-p" in inv.cmd
        assert "--no-session" in inv.cmd
        assert inv.cmd[inv.cmd.index("--model") + 1] == "claude-sonnet-4"
        # Last arg is the @promptfile reference
        assert inv.cmd[-1].startswith("@")
        assert inv.stdin is None
        # Prompt was written to a temp file
        assert len(inv.tempfiles) == 1
        assert Path(inv.tempfiles[0]).read_text() == "hello world"
    finally:
        _cleanup(inv.tempfiles)


def test_default_adapter_with_system_prompt_adds_flag() -> None:
    inv = build_invocation("ohp", "opus", "p", "be brief")
    try:
        assert "--system-prompt" in inv.cmd
        sp_path = inv.cmd[inv.cmd.index("--system-prompt") + 1]
        assert Path(sp_path).read_text() == "be brief"
        assert len(inv.tempfiles) == 2  # prompt + system
    finally:
        _cleanup(inv.tempfiles)


def test_default_adapter_unknown_name_falls_through() -> None:
    # Any unknown name becomes the executable with the default shape.
    inv = build_invocation("my-custom-cli", "m", "p", None)
    try:
        assert inv.cmd[0] == "my-custom-cli"
        assert "--no-session" in inv.cmd
    finally:
        _cleanup(inv.tempfiles)


def test_claude_adapter_stdin_and_model() -> None:
    inv = build_invocation("claude", "sonnet", "PING", None)
    try:
        assert inv.cmd[0] == "claude"
        assert "-p" in inv.cmd
        assert inv.cmd[inv.cmd.index("--model") + 1] == "sonnet"
        assert "--no-session-persistence" in inv.cmd
        assert inv.stdin == "PING"
        # No system prompt → no temp files
        assert inv.tempfiles == []
    finally:
        _cleanup(inv.tempfiles)


def test_claude_adapter_system_prompt_via_file() -> None:
    inv = build_invocation("claude", "sonnet", "PING", "be terse")
    try:
        assert "--append-system-prompt-file" in inv.cmd
        sp_path = inv.cmd[inv.cmd.index("--append-system-prompt-file") + 1]
        assert Path(sp_path).read_text() == "be terse"
        assert len(inv.tempfiles) == 1
    finally:
        _cleanup(inv.tempfiles)


def test_codex_adapter_stdin_and_last_message_flag() -> None:
    inv = build_invocation("codex", "gpt-5", "PING", None)
    try:
        assert inv.cmd[0:2] == ["codex", "exec"]
        assert "--ephemeral" in inv.cmd
        assert "--skip-git-repo-check" in inv.cmd
        assert "-o" in inv.cmd
        last_path = inv.cmd[inv.cmd.index("-o") + 1]
        assert last_path.endswith(".lastmsg")
        assert inv.cmd[inv.cmd.index("-m") + 1] == "gpt-5"
        assert inv.cmd[-1] == "-"  # read prompt from stdin
        assert inv.stdin == "PING"
    finally:
        _cleanup(inv.tempfiles)


def test_codex_system_prompt_prepended() -> None:
    # Codex has no system flag → system is prepended to the user prompt.
    inv = build_invocation("codex", "gpt-5", "user message", "system directive")
    try:
        assert inv.stdin is not None
        assert "system directive" in inv.stdin
        assert "user message" in inv.stdin
        # System must appear before user content
        assert inv.stdin.index("system directive") < inv.stdin.index("user message")
    finally:
        _cleanup(inv.tempfiles)
