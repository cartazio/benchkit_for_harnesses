"""Benchmark dataset configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark dataset."""

    name: str
    """Benchmark identifier (e.g., 'babilong')"""

    hf_path: str
    """HuggingFace dataset path"""

    format_fn: Callable[[Mapping[str, Any]], tuple[str, str]]
    """Function to format dataset items to (prompt, target)"""

    eval_fn: Callable[[str, str], bool]
    """Function to evaluate (response, target) -> correct: bool"""

    tasks: list[str] = field(default_factory=lambda: [])
    """Available tasks within the benchmark"""

    lengths: list[str] = field(default_factory=lambda: [])
    """Context lengths (for long-context benchmarks)"""

    split_is_task: bool = False
    """If True, use split= parameter for task selection"""

    default_split: str = "test"
    """Default split to load"""


# Format functions
def format_babilong(item: Mapping[str, Any]) -> tuple[str, str]:
    """Format BABILong item to (prompt, target)."""
    input_text = item.get("input", "")
    question = item.get("question", "")
    target_raw = item.get("target", "")
    prompt = f"{input_text}\n\nQuestion: {question}\nAnswer:"
    target = str(target_raw).strip()
    return prompt, target


def format_infinitebench(item: Mapping[str, Any]) -> tuple[str, str]:
    """Format InfiniteBench item to (prompt, target)."""
    context = item.get("context", "")
    input_text = item.get("input", "")
    prompt = f"{context}\n\n{input_text}"
    answer_raw: Any = item.get("answer", "")
    if isinstance(answer_raw, list) and answer_raw:
        target = str(answer_raw[0])
    else:
        target = str(answer_raw)
    return prompt, target.strip()


def format_longbenchv2(item: Mapping[str, Any]) -> tuple[str, str]:
    """Format LongBench-v2 item to (prompt, target)."""
    prompt = (
        f"{item.get('context', '')}\n\n"
        f"Question: {item.get('question', '')}\n"
        f"A) {item.get('choice_A', '')}\n"
        f"B) {item.get('choice_B', '')}\n"
        f"C) {item.get('choice_C', '')}\n"
        f"D) {item.get('choice_D', '')}\n"
        f"Answer with just the letter (A, B, C, or D):"
    )
    target = str(item.get("answer", "")).strip().upper()
    return prompt, target


# Evaluation functions
def eval_contains(response: str, target: str) -> bool:
    """Check if target appears in response (case-insensitive)."""
    return target.strip().lower() in response.lower()


def eval_letter_match(response: str, target: str) -> bool:
    """Check if response contains the target letter answer."""
    resp_clean = response.strip().upper()
    target_clean = target.strip().upper()
    if resp_clean.startswith(target_clean):
        return True
    if target_clean in resp_clean.split():
        return True
    return target_clean in resp_clean


# Registry
BENCHMARKS: dict[str, BenchmarkConfig] = {
    "babilong": BenchmarkConfig(
        name="babilong",
        hf_path="RMT-team/babilong",
        format_fn=format_babilong,
        eval_fn=eval_contains,
        tasks=["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"],
        lengths=["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"],
        split_is_task=True,
    ),
    "infinitebench": BenchmarkConfig(
        name="infinitebench",
        hf_path="xinrongzhang2022/InfiniteBench",
        format_fn=format_infinitebench,
        eval_fn=eval_contains,
        tasks=[
            "passkey",
            "kv_retrieval",
            "number_string",
            "code_run",
            "code_debug",
            "math_find",
            "longdialogue_qa_eng",
            "longbook_qa_eng",
        ],
        split_is_task=True,
    ),
    "longbenchv2": BenchmarkConfig(
        name="longbenchv2",
        hf_path="zai-org/LongBench-v2",
        format_fn=format_longbenchv2,
        eval_fn=eval_letter_match,
        default_split="train",
    ),
}
