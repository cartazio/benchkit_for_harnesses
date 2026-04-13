"""CLI harness runner for ohp/punkin."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Literal


HarnessType = Literal["ohp", "punkin"]


def run_harness(
    harness: HarnessType,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    timeout_sec: int = 300,
) -> tuple[str, int]:
    """
    Run prompt through CLI harness, return (response, latency_ms).

    Args:
        harness: CLI command (ohp or punkin)
        model: Model identifier
        prompt: User prompt
        system_prompt: System prompt override (if provided, used with --system-prompt flag)
        timeout_sec: Timeout in seconds

    Returns:
        Tuple of (response_text, latency_milliseconds)

    Raises:
        FileNotFoundError: If harness binary not found in PATH
        subprocess.TimeoutExpired: If execution exceeds timeout_sec
    """
    cmd = [
        harness,
        "-p",
        "--no-session",
        "--model",
        model,
    ]

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    cmd.append(prompt)

    start = datetime.now(timezone.utc)
    try:
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

    elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
    return response, elapsed_ms
