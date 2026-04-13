"""
Model registry: predefined ModelSpec instances for the experiment.

Organized by family to enable base-vs-instruct comparison.
Each family has a base and instruct variant at the same scale.

Provider options:
  - "together": Together AI API (hosts both base and instruct)
  - "anthropic": Anthropic API (instruct only, no base available)
  - "openai": OpenAI API (instruct only, no base available)
  - "local": local vLLM/Ollama endpoint
"""

from .harness import ModelSpec


# ---------------------------------------------------------------------------
# Llama 3.1 family
# ---------------------------------------------------------------------------

LLAMA_31_8B_BASE = ModelSpec(
    name="llama-3.1-8b-base",
    provider="together",
    model_id="meta-llama/Meta-Llama-3.1-8B",
    is_base=True,
)

LLAMA_31_8B_INSTRUCT = ModelSpec(
    name="llama-3.1-8b-instruct",
    provider="together",
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    is_base=False,
)

LLAMA_31_70B_BASE = ModelSpec(
    name="llama-3.1-70b-base",
    provider="together",
    model_id="meta-llama/Meta-Llama-3.1-70B",
    is_base=True,
)

LLAMA_31_70B_INSTRUCT = ModelSpec(
    name="llama-3.1-70b-instruct",
    provider="together",
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    is_base=False,
)


# ---------------------------------------------------------------------------
# Qwen 2.5 family
# ---------------------------------------------------------------------------

QWEN_25_7B_BASE = ModelSpec(
    name="qwen-2.5-7b-base",
    provider="together",
    model_id="Qwen/Qwen2.5-7B",
    is_base=True,
)

QWEN_25_7B_INSTRUCT = ModelSpec(
    name="qwen-2.5-7b-instruct",
    provider="together",
    model_id="Qwen/Qwen2.5-7B-Instruct",
    is_base=False,
)

QWEN_25_72B_BASE = ModelSpec(
    name="qwen-2.5-72b-base",
    provider="together",
    model_id="Qwen/Qwen2.5-72B",
    is_base=True,
)

QWEN_25_72B_INSTRUCT = ModelSpec(
    name="qwen-2.5-72b-instruct",
    provider="together",
    model_id="Qwen/Qwen2.5-72B-Instruct",
    is_base=False,
)


# ---------------------------------------------------------------------------
# Mistral family
# ---------------------------------------------------------------------------

MISTRAL_7B_BASE = ModelSpec(
    name="mistral-7b-base",
    provider="together",
    model_id="mistralai/Mistral-7B-v0.3",
    is_base=True,
)

MISTRAL_7B_INSTRUCT = ModelSpec(
    name="mistral-7b-instruct",
    provider="together",
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    is_base=False,
)


# ---------------------------------------------------------------------------
# Closed-source instruct-only (no base available)
# ---------------------------------------------------------------------------

CLAUDE_SONNET = ModelSpec(
    name="claude-sonnet-4",
    provider="anthropic",
    model_id="claude-sonnet-4-20250514",
    is_base=False,
)

CLAUDE_HAIKU = ModelSpec(
    name="claude-haiku-3.5",
    provider="anthropic",
    model_id="claude-3-5-haiku-20241022",
    is_base=False,
)

GPT4O_MINI = ModelSpec(
    name="gpt-4o-mini",
    provider="openai",
    model_id="gpt-4o-mini",
    is_base=False,
)


# ---------------------------------------------------------------------------
# Experiment presets
# ---------------------------------------------------------------------------

# Core experiment: base-vs-instruct pairs at 7-8B scale
CORE_MODELS = [
    LLAMA_31_8B_BASE,
    LLAMA_31_8B_INSTRUCT,
    QWEN_25_7B_BASE,
    QWEN_25_7B_INSTRUCT,
    MISTRAL_7B_BASE,
    MISTRAL_7B_INSTRUCT,
]

# Extended: add 70B+ pairs for scale analysis
EXTENDED_MODELS = CORE_MODELS + [
    LLAMA_31_70B_BASE,
    LLAMA_31_70B_INSTRUCT,
    QWEN_25_72B_BASE,
    QWEN_25_72B_INSTRUCT,
]

# Full: add closed-source instruct-only for comparison
FULL_MODELS = EXTENDED_MODELS + [
    CLAUDE_SONNET,
    CLAUDE_HAIKU,
    GPT4O_MINI,
]

ALL_MODELS = {m.name: m for m in FULL_MODELS}
