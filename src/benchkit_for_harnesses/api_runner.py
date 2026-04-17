"""
API runner: direct HTTP calls to LLM providers (no CLI harness).

Supports:
  - OpenAI-compatible chat: OpenAI, Together, vLLM local
  - Anthropic chat (including extended-thinking content blocks)
  - Base-model completion: Together, local, OpenAI (legacy)

Environment variables for base URL overrides:
  OPENAI_BASE_URL      - Override OpenAI API endpoint
  TOGETHER_BASE_URL    - Override Together AI endpoint
  ANTHROPIC_BASE_URL   - Override Anthropic API endpoint
  LOCAL_BASE_URL       - Override local vLLM/Ollama endpoint (default: http://localhost:8000/v1)

Priority: env var > ModelSpec.api_base > hardcoded default.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx


class ModelLike(Protocol):
    @property
    def provider(self) -> str: ...
    @property
    def api_base(self) -> str | None: ...
    @property
    def model_id(self) -> str: ...
    @property
    def max_tokens(self) -> int: ...


def _dget(obj: Any, key: str, default: Any = None) -> Any:
    """Dict.get that returns Any regardless of pyright narrowing."""
    if isinstance(obj, dict):
        return obj.get(key, default)  # type: ignore[return-value]
    return default



# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def run_chat(model: ModelLike, prompt: str, system: str) -> str:
    """Dispatch chat completion to the appropriate provider."""
    if model.provider == "anthropic":
        return await _run_anthropic_chat(model, prompt, system)
    return await _run_openai_chat(model, prompt, system)


def _openai_compat_endpoint(model: ModelLike) -> tuple[str, str, str]:
    """Return (base_url, api_key, provider_label) for OpenAI-compatible endpoints."""
    if model.provider == "together":
        base = (
            os.environ.get("TOGETHER_BASE_URL")
            or model.api_base
            or "https://api.together.xyz/v1"
        )
        return base, os.environ.get("TOGETHER_API_KEY", ""), "together"
    if model.provider == "local":
        base = (
            os.environ.get("LOCAL_BASE_URL")
            or model.api_base
            or "http://localhost:8000/v1"
        )
        return base, "not-needed", "local"
    # Default to OpenAI for "openai" and any unknown openai-compatible provider.
    base = (
        os.environ.get("OPENAI_BASE_URL")
        or model.api_base
        or "https://api.openai.com/v1"
    )
    return base, os.environ.get("OPENAI_API_KEY", ""), model.provider or "openai"


async def _run_openai_chat(model: ModelLike, prompt: str, system: str) -> str:
    base_url, api_key, label = _openai_compat_endpoint(model)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model.model_id,
        "messages": messages,
        "max_tokens": model.max_tokens,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"{label} chat {resp.status_code}: {resp.text[:500]}")
    data: dict[str, Any] = resp.json()
    choices: list[Any] = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"{label} chat returned no choices: {str(data)[:500]}")
    first = choices[0]
    message = _dget(first, "message")
    content = _dget(message, "content")
    if not isinstance(content, str):
        raise RuntimeError(f"{label} chat returned non-string content: {str(message)[:500]}")
    return content


async def _run_anthropic_chat(model: ModelLike, prompt: str, system: str) -> str:
    """Anthropic messages API, resilient to extended-thinking content blocks.

    Anthropic's content array may contain `{type: "thinking", thinking: "..."}`
    blocks before the `{type: "text", text: "..."}` block(s). We must NOT
    score the thinking prose as the answer — iterate and concatenate only
    the text blocks.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = (
        os.environ.get("ANTHROPIC_BASE_URL")
        or model.api_base
        or "https://api.anthropic.com/v1/messages"
    )

    payload: dict[str, Any] = {
        "model": model.model_id,
        "max_tokens": model.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"anthropic {resp.status_code}: {resp.text[:500]}")
    data: dict[str, Any] = resp.json()

    blocks: list[Any] = data.get("content") or []
    text_parts: list[str] = []
    block_types: list[str] = []
    for block in blocks:
        btype = str(_dget(block, "type", ""))
        block_types.append(btype)
        if btype == "text":
            t = _dget(block, "text")
            if isinstance(t, str):
                text_parts.append(t)

    if not text_parts:
        stop_reason = data.get("stop_reason")
        raise RuntimeError(
            "anthropic response has no text blocks "
            f"(stop_reason={stop_reason!r}, block_types={block_types})"
        )
    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# Completion (base models)
# ---------------------------------------------------------------------------


def _completion_endpoint(model: ModelLike) -> tuple[str, str, str]:
    """Return (base_url, api_key, provider_label) for the /completions endpoint."""
    if model.provider == "local":
        base = (
            os.environ.get("LOCAL_BASE_URL")
            or model.api_base
            or "http://localhost:8000/v1"
        )
        return base, "not-needed", "local"
    if model.provider == "together":
        base = (
            os.environ.get("TOGETHER_BASE_URL")
            or model.api_base
            or "https://api.together.xyz/v1"
        )
        return base, os.environ.get("TOGETHER_API_KEY", ""), "together"
    if model.provider == "openai":
        base = (
            os.environ.get("OPENAI_BASE_URL")
            or model.api_base
            or "https://api.openai.com/v1"
        )
        return base, os.environ.get("OPENAI_API_KEY", ""), "openai"
    raise ValueError(
        f"run_completion: provider {model.provider!r} has no text-completion endpoint"
    )


async def run_completion(model: ModelLike, prompt: str, system: str) -> str:
    """Text completion for base (non-instruct) models.

    Base models respond to format cues, not instructions. A minimal few-shot
    preamble is prepended demonstrating question/answer format.
    """
    base_url, api_key, label = _completion_endpoint(model)

    few_shot = (
        "The following are questions with answers.\n\n"
        "Question 1: What is the capital of France?\n"
        "Answer: Paris\n\n"
        "Question 2: What is 7 * 8?\n"
        "Answer: 56\n\n"
        "---\n\n"
    )

    full_prompt = ""
    if system:
        full_prompt += system + "\n\n"
    full_prompt += few_shot + prompt + "\n\n"

    payload: dict[str, Any] = {
        "model": model.model_id,
        "prompt": full_prompt,
        "max_tokens": model.max_tokens,
        "temperature": 0.0,
        "stop": ["\n\n---"],
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"{label} completion {resp.status_code}: {resp.text[:500]}")
    data: dict[str, Any] = resp.json()
    choices: list[Any] = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"{label} completion returned no choices: {str(data)[:500]}")
    first = choices[0]
    text = _dget(first, "text")
    if not isinstance(text, str):
        raise RuntimeError(f"{label} completion returned non-string text: {str(first)[:500]}")
    return text
