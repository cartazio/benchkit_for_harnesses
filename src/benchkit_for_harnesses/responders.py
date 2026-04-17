"""Responder factories for :mod:`benchkit_for_harnesses.core`.

A responder is a callable ``(prompt, system) -> (response, latency_ms)``.
This module converts the kit's transports (CLI harnesses, HTTP chat/
completion) into that uniform shape so the core loop doesn't care how
bytes move.
"""

from __future__ import annotations

import time
from typing import Protocol

from .api_runner import ModelLike, run_chat, run_completion
from .core import AsyncResponder, SyncResponder
from .harnesses.dispatch import run_harness

__all__ = [
    "ModelDispatch",
    "api_chat_responder",
    "api_model_responder",
    "harness_responder",
]


def harness_responder(
    harness: str, model: str, *, timeout_sec: int = 300
) -> SyncResponder:
    """Bind a CLI harness into the uniform responder shape.

    Propagates :func:`~benchkit_for_harnesses.harnesses.dispatch.run_harness`'s
    error discipline — responses starting with ``[ERROR …]`` on any
    failure path.
    """
    def _call(prompt: str, system: str | None) -> tuple[str, int]:
        return run_harness(
            harness=harness,
            model=model,
            prompt=prompt,
            system_prompt=system,
            timeout_sec=timeout_sec,
        )

    return _call


def api_chat_responder(model: ModelLike) -> AsyncResponder:
    """Chat-completions HTTP responder (OpenAI / Anthropic / compat)."""

    async def _call(prompt: str, system: str | None) -> tuple[str, int]:
        t0 = time.monotonic()
        text = await run_chat(model, prompt, system or "")
        ms = int((time.monotonic() - t0) * 1000)
        return text, ms

    return _call


class ModelDispatch(Protocol):
    """A model spec that knows whether to use chat vs base completion."""

    @property
    def is_base(self) -> bool: ...
    @property
    def provider(self) -> str: ...
    @property
    def api_base(self) -> str | None: ...
    @property
    def model_id(self) -> str: ...
    @property
    def max_tokens(self) -> int: ...


def api_model_responder(model: ModelDispatch) -> AsyncResponder:
    """HTTP responder that dispatches base→completion vs instruct→chat.

    Used by bundled-bench where the experiment compares base and instruct
    variants of the same family and needs to talk to each through the
    right endpoint.
    """

    async def _call(prompt: str, system: str | None) -> tuple[str, int]:
        t0 = time.monotonic()
        if model.is_base:
            text = await run_completion(model, prompt, system or "")
        else:
            text = await run_chat(model, prompt, system or "")
        ms = int((time.monotonic() - t0) * 1000)
        return text, ms

    return _call
