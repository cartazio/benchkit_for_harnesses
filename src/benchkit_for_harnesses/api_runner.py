"""
API runner: direct HTTP calls to LLM providers (no CLI harness).

Supports:
  - OpenAI-compatible (OpenAI, Together, vLLM local)
  - Anthropic
  - Base models via completion endpoint (not chat)

Environment variables for base URL overrides:
  OPENAI_BASE_URL      - Override OpenAI API endpoint
  TOGETHER_BASE_URL    - Override Together AI endpoint
  ANTHROPIC_BASE_URL   - Override Anthropic API endpoint
  LOCAL_BASE_URL       - Override local vLLM/Ollama endpoint (default: http://localhost:8000/v1)

Priority: env var > ModelSpec.api_base > hardcoded default
"""

from __future__ import annotations

import os


async def run_chat(model: object, prompt: str, system: str) -> str:
    """Dispatch chat completion to the appropriate provider."""
    if model.provider == "anthropic":  # type: ignore[union-attr]
        return await _run_anthropic_chat(model, prompt, system)
    return await _run_openai_chat(model, prompt, system)


async def _run_openai_chat(model: object, prompt: str, system: str) -> str:
    import httpx

    base_url = os.environ.get("OPENAI_BASE_URL") or model.api_base or "https://api.openai.com/v1"  # type: ignore[union-attr]
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if model.provider == "together":  # type: ignore[union-attr]
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        base_url = os.environ.get("TOGETHER_BASE_URL") or model.api_base or "https://api.together.xyz/v1"  # type: ignore[union-attr]
    elif model.provider == "local":  # type: ignore[union-attr]
        api_key = "not-needed"
        base_url = os.environ.get("LOCAL_BASE_URL") or model.api_base or "http://localhost:8000/v1"  # type: ignore[union-attr]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model.model_id,  # type: ignore[union-attr]
        "messages": messages,
        "max_tokens": model.max_tokens,  # type: ignore[union-attr]
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
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _run_anthropic_chat(model: object, prompt: str, system: str) -> str:
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or model.api_base or "https://api.anthropic.com/v1/messages"  # type: ignore[union-attr]

    payload = {
        "model": model.model_id,  # type: ignore[union-attr]
        "max_tokens": model.max_tokens,  # type: ignore[union-attr]
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
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


async def run_completion(model: object, prompt: str, system: str) -> str:
    """
    Text completion for base (non-instruct) models.

    Base models respond to format cues, not instructions.
    Constructs a few-shot prompt demonstrating expected Q&A format.
    """
    import httpx

    base_url = os.environ.get("TOGETHER_BASE_URL") or model.api_base or "https://api.together.xyz/v1"  # type: ignore[union-attr]
    api_key = os.environ.get("TOGETHER_API_KEY", "")

    if model.provider == "local":  # type: ignore[union-attr]
        api_key = "not-needed"
        base_url = os.environ.get("LOCAL_BASE_URL") or model.api_base or "http://localhost:8000/v1"  # type: ignore[union-attr]

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

    payload = {
        "model": model.model_id,  # type: ignore[union-attr]
        "prompt": full_prompt,
        "max_tokens": model.max_tokens,  # type: ignore[union-attr]
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
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]
