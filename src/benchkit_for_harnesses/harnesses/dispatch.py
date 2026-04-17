"""CLI harness runner.

Supports multiple CLI shapes via per-harness adapters:

  * **default** — `ohp`, `punkin`, `opencode`, `monopi`, `omp` (and any CLI
    that follows the shape ``-p --no-session --model X @promptfile``).
    User-message is passed as an ``@file`` reference so long prompts bypass
    the OS ARG_MAX limit.

  * **claude** — Anthropic Claude Code CLI. Prompt via stdin in ``-p`` mode.
    System prompt (if any) goes through ``--append-system-prompt-file``.

  * **codex** — OpenAI Codex CLI. Invoked as ``codex exec`` with the prompt
    on stdin and ``--output-last-message`` pointing to a temp file that we
    read back (codex's stdout is streamed chrome; only the last-message file
    is the clean agent response). No dedicated system-prompt flag, so the
    system prompt is prepended to the user prompt.

Error discipline: every non-success exit path returns a response starting
with ``[ERROR ...]``. Callers need not inspect return codes.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

HarnessType = str

__all__ = ["HarnessType", "KNOWN_HARNESSES", "build_invocation", "run_harness"]

# CLIs with a native shape adapter. Unknown names fall back to the default
# (ohp-style) adapter with the name used as the executable.
KNOWN_HARNESSES = ("ohp", "punkin", "opencode", "monopi", "omp", "claude", "codex")


@dataclass
class HarnessInvocation:
    """One CLI invocation plan.

    Keep the tempfile list so the runner can clean them up after the process
    exits. ``extract`` pulls the final response out of stdout + stderr +
    any side-channel files the adapter created (e.g. codex's ``-o`` file).
    """

    cmd: list[str]
    stdin: str | None = None
    tempfiles: list[str] = field(default_factory=lambda: [])
    extract: Callable[[subprocess.CompletedProcess[str]], str] = lambda p: p.stdout.strip()


def _write_temp(content: str, suffix: str = ".txt") -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


def _build_default(
    harness: str, model: str, prompt: str, system_prompt: str | None
) -> HarnessInvocation:
    """ohp/punkin/opencode/monopi/omp — prompt via @file, system via file flag."""
    prompt_path = _write_temp(prompt)
    cmd = [harness, "-p", "--no-session", "--model", model, f"@{prompt_path}"]
    tempfiles = [prompt_path]

    if system_prompt:
        sp_path = _write_temp(system_prompt)
        cmd.extend(["--system-prompt", sp_path])
        tempfiles.append(sp_path)

    return HarnessInvocation(cmd=cmd, tempfiles=tempfiles)


def _build_claude(
    model: str, prompt: str, system_prompt: str | None
) -> HarnessInvocation:
    """Claude Code — prompt via stdin, system via --append-system-prompt-file."""
    cmd = [
        "claude",
        "-p",
        "--model", model,
        "--no-session-persistence",
        "--permission-mode", "bypassPermissions",
    ]
    tempfiles: list[str] = []
    if system_prompt:
        # Always file-mode — dodges ARG_MAX on long system prompts.
        sp_path = _write_temp(system_prompt)
        cmd.extend(["--append-system-prompt-file", sp_path])
        tempfiles.append(sp_path)
    return HarnessInvocation(cmd=cmd, stdin=prompt, tempfiles=tempfiles)


def _build_codex(
    model: str, prompt: str, system_prompt: str | None
) -> HarnessInvocation:
    """Codex — prompt via stdin, final message extracted from -o file.

    Codex has no dedicated system-prompt flag on ``exec``. System content is
    prepended to the user prompt, delimited so the model can still tell the
    sections apart.
    """
    last_path = _write_temp("", suffix=".lastmsg")
    cmd = [
        "codex", "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color", "never",
        "-o", last_path,
        "-m", model,
        "-",  # read prompt from stdin
    ]
    tempfiles = [last_path]

    effective_prompt = prompt
    if system_prompt:
        effective_prompt = (
            f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n\n"
            f"[USER REQUEST]\n{prompt}"
        )

    def _extract(p: subprocess.CompletedProcess[str]) -> str:
        try:
            text = Path(last_path).read_text(encoding="utf-8").strip()
        except OSError as e:
            return f"[ERROR codex] could not read last-message file: {e}"
        if not text:
            # Codex prints "Warning: no last agent message" to stdout when the
            # model didn't produce a message (auth failure, refusal, etc.).
            return f"[ERROR codex] empty last-message; stderr={p.stderr.strip()[:500]}"
        return text

    return HarnessInvocation(
        cmd=cmd,
        stdin=effective_prompt,
        tempfiles=tempfiles,
        extract=_extract,
    )


def build_invocation(
    harness: str, model: str, prompt: str, system_prompt: str | None
) -> HarnessInvocation:
    """Build the invocation plan for a given harness name.

    Exposed for tests — runtime callers should use :func:`run_harness`.
    """
    if harness == "claude":
        return _build_claude(model, prompt, system_prompt)
    if harness == "codex":
        return _build_codex(model, prompt, system_prompt)
    # default shape for ohp/punkin/opencode/monopi/omp and any unknown CLI
    return _build_default(harness, model, prompt, system_prompt)


def run_harness(
    harness: str,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    timeout_sec: int = 300,
) -> tuple[str, int]:
    """Run prompt through a CLI harness, returning ``(response, latency_ms)``.

    The response always starts with ``[ERROR ...]`` when the run failed, so
    callers do not need to inspect a return code.
    """
    start = datetime.now(timezone.utc)
    tempfiles: list[str] = []

    try:
        invocation = build_invocation(harness, model, prompt, system_prompt)
        tempfiles = invocation.tempfiles

        result = subprocess.run(
            invocation.cmd,
            input=invocation.stdin,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        if result.returncode != 0:
            response = (
                f"[ERROR rc={result.returncode}] {result.stderr.strip()}\n"
                f"---STDOUT---\n{result.stdout.strip()}"
            )
        else:
            response = invocation.extract(result)
    except subprocess.TimeoutExpired:
        response = f"[ERROR timeout={timeout_sec}s] harness did not return within timeout"
    except FileNotFoundError:
        response = f"[ERROR] Harness '{harness}' not found in PATH"
    finally:
        for path in tempfiles:
            try:
                os.unlink(path)
            except OSError:
                pass

    elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
    return response, elapsed_ms
