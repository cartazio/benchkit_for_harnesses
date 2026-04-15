"""CLI harness runner for ohp/punkin.

Uses temp files for prompt and system prompt to avoid ARG_MAX limits
on long-context benchmarks. ohp's --system-prompt flag auto-resolves
file paths, and @file syntax passes file content as the user message.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime, timezone

# Known harness CLIs. Any CLI binary accepting -p --no-session --model is valid.
KNOWN_HARNESSES = ("ohp", "punkin", "opencode", "monopi", "omp")
HarnessType = str


def run_harness(
    harness: str,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    timeout_sec: int = 300,
) -> tuple[str, int]:
    """
    Run prompt through CLI harness, return (response, latency_ms).

    Writes prompt and system prompt to temp files to avoid OS ARG_MAX
    limits (macOS ~256KB) on long-context benchmarks where prompts can
    exceed 100K characters.

    Args:
        harness: CLI command (ohp or punkin)
        model: Model identifier
        prompt: User prompt
        system_prompt: System prompt override
        timeout_sec: Timeout in seconds

    Returns:
        Tuple of (response_text, latency_milliseconds)
    """
    sp_path: str | None = None
    p_path: str | None = None

    start = datetime.now(timezone.utc)

    try:
        # Write prompt to temp file — avoids ARG_MAX on long prompts
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as p_f:
            p_f.write(prompt)
            p_path = p_f.name

        cmd = [
            harness,
            "-p",
            "--no-session",
            "--model", model,
            f"@{p_path}",
        ]

        # Write system prompt to temp file — ohp resolves file paths on --system-prompt
        if system_prompt:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as sp_f:
                sp_f.write(system_prompt)
                sp_path = sp_f.name
            cmd.extend(["--system-prompt", sp_path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        response = result.stdout.strip()
        if result.returncode != 0 and not response:
            response = f"[ERROR] {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        response = "[TIMEOUT]"
    except FileNotFoundError:
        response = f"[ERROR] Harness '{harness}' not found in PATH"
    finally:
        for path in (p_path, sp_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
    return response, elapsed_ms
