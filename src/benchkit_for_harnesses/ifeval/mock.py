"""IFEval+ mock response generator for testing evaluation pipeline."""

from __future__ import annotations

import random


def mock_response(prompt: str, system_prompt: str, seed: int = 42) -> str:
    """
    Generate mock response for testing evaluation pipeline.
    
    This simulates a model that follows instructions ~70% of the time,
    with degradation under heavier system prompts.
    """
    rng = random.Random(seed)
    
    # Heavier prompts → worse performance (simulated)
    base_success_rate = 0.85
    if "punkin-pi" in system_prompt:
        base_success_rate -= 0.15
    if "pressure" in system_prompt.lower():
        base_success_rate -= 0.10
    
    # Extract what the prompt is asking for (very crude)
    response_parts: list[str] = []
    
    if "no comma" in prompt.lower():
        if rng.random() < base_success_rate:
            response_parts.append("Here is my response without any comma characters.")
        else:
            response_parts.append("Here is my response, which might have commas.")
    
    if "json" in prompt.lower():
        if rng.random() < base_success_rate:
            response_parts.append('{"response": "valid json"}')
        else:
            response_parts.append("This is not valid JSON")
    
    if "300" in prompt and "word" in prompt.lower():
        if rng.random() < base_success_rate:
            response_parts.append(" ".join(["word"] * 350))
        else:
            response_parts.append(" ".join(["word"] * 50))
    
    if not response_parts:
        response_parts.append("This is a mock response for testing the evaluation pipeline.")
    
    return "\n\n".join(response_parts)
