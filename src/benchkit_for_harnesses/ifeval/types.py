"""IFEval+ types: result records and evaluation summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


IFEvalItem = dict[str, Any]


class ResultRecord(TypedDict):
    """Single IFEval result."""
    idx: int
    key: int
    condition: str
    model: str
    prompt: str
    instruction_ids: list[str]
    response: str
    follow_all: bool
    follow_list: list[bool]
    latency_ms: int


@dataclass
class EvalSummary:
    """Aggregate evaluation summary (loose evaluation mode)."""
    condition: str
    model: str
    total: int
    prompt_acc: float  # % prompts where ALL verifiable instructions followed
    instruction_acc: float  # % of individual verifiable instructions followed
