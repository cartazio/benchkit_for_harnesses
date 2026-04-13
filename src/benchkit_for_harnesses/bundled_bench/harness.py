"""
bundled-bench: Measuring instruction-tuning policy tax on multi-question prompts.

Hypothesis: instruction tuning introduces a default policy that degrades
multi-question accuracy. Targeted system prompt instructions recover capability
toward (or beyond) base model performance.

Experimental matrix:
  A: base model, no system prompt
  B: instruct model, no system prompt
  C: instruct model, minimal system prompt
  D: instruct model, full harness system prompt

Independent variables:
  - model (base vs instruct variants)
  - bundle_size N (1, 3, 5, 7, 10)
  - system_prompt_condition (none, minimal, full)

Dependent variable:
  - per-question accuracy within bundle
  - positional accuracy (does position in bundle affect correctness?)
  - completion rate (did the model attempt all N questions?)
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from benchkit_for_harnesses.brackets import extract_bracketed_answers
from benchkit_for_harnesses.results import load_jsonl, save_jsonl


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class PromptCondition(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    FULL = "full"


@dataclass(frozen=True)
class Question:
    """Single question with ground truth."""
    id: str
    text: str
    answer: str  # canonical answer string
    source: str  # e.g. "mmlu:abstract_algebra:42"
    choices: Optional[list[str]] = None  # for multiple choice


@dataclass
class Bundle:
    """N questions grouped into a single prompt."""
    id: str
    questions: list[Question]
    bundle_size: int = field(init=False)

    def __post_init__(self):
        self.bundle_size = len(self.questions)


@dataclass
class ModelSpec:
    """Describes a model endpoint."""
    name: str            # human-readable
    provider: str        # "openai", "anthropic", "together", "local"
    model_id: str        # API model string
    is_base: bool        # True = base/pretrained, False = instruct-tuned
    api_base: Optional[str] = None  # for local/custom endpoints
    max_tokens: int = 2048


@dataclass
class TrialConfig:
    """Full specification of one experimental trial."""
    model: ModelSpec
    condition: PromptCondition
    bundle_size: int
    seed: int = 42


@dataclass
class QuestionResult:
    """Result for a single question within a bundle."""
    question_id: str
    position: int        # 0-indexed position in bundle
    expected: str
    raw_response: str    # the model's answer for this question
    correct: bool
    attempted: bool      # did the model produce an answer at all?


@dataclass
class BundleResult:
    """Result for one bundle evaluation."""
    bundle_id: str
    trial_config: TrialConfig
    question_results: list[QuestionResult]
    raw_model_output: str
    latency_ms: float
    timestamp: str

    @property
    def accuracy(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(1 for q in self.question_results if q.correct) / len(self.question_results)

    @property
    def completion_rate(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(1 for q in self.question_results if q.attempted) / len(self.question_results)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[PromptCondition, str] = {
    PromptCondition.NONE: "",

    PromptCondition.MINIMAL: (
        "You will be given multiple questions. "
        "Answer each question separately and completely. "
        "Label each answer with its question number. "
        "Wrap each final answer in {[{[ and ]}]} markers."
    ),

    PromptCondition.FULL: (
        "You will be given multiple questions in a single message. "
        "You must answer ALL questions. Do not skip, merge, or summarize.\n\n"
        "Requirements:\n"
        "- Address each question independently with its own labeled answer\n"
        "- Maintain the same quality for each answer regardless of position\n"
        "- If a question is ambiguous, state your interpretation and answer it\n"
        "- Do not sacrifice later answers for earlier ones\n"
        "- Do not add disclaimers, hedges, or meta-commentary\n"
        "- Show your reasoning for each question before giving the answer\n"
        "- Format: 'Question N: [reasoning] Answer: {[{[ answer ]}]}'\n"
    ),
}


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------

def make_bundles(
    questions: list[Question],
    bundle_size: int,
    n_bundles: int,
    seed: int = 42,
) -> list[Bundle]:
    """
    Partition questions into bundles of size `bundle_size`.
    Returns up to `n_bundles` bundles.
    Shuffles questions first (deterministically via seed).
    """
    rng = random.Random(seed)
    qs = list(questions)
    rng.shuffle(qs)

    bundles: list[Bundle] = []
    for i in range(0, len(qs) - bundle_size + 1, bundle_size):
        if len(bundles) >= n_bundles:
            break
        chunk = qs[i : i + bundle_size]
        bid = hashlib.sha256(
            "|".join(q.id for q in chunk).encode()
        ).hexdigest()[:12]
        bundles.append(Bundle(id=f"bundle_{bid}", questions=chunk))

    return bundles


def format_bundle_prompt(bundle: Bundle) -> str:
    """
    Format a bundle into a user-facing prompt string.
    For multiple choice, includes choices. For free-form, just the question.
    """
    parts: list[str] = []
    for i, q in enumerate(bundle.questions, 1):
        part = f"Question {i}: {q.text}"
        if q.choices:
            for j, choice in enumerate(q.choices):
                label = chr(ord('A') + j)
                part += f"\n  {label}) {choice}"
        parts.append(part)

    header = f"Answer all {bundle.bundle_size} questions below.\nWrap each answer in {{[{{[ and ]}}]}} markers.\n\n"
    return header + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_responses(
    raw_output: str,
    bundle: Bundle,
) -> list[QuestionResult]:
    """
    Parse model output into per-question results using bracket extraction.
    Looks for {[{[ ... ]}]} markers to identify answers.
    """
    answers = extract_bracketed_answers(raw_output)
    results: list[QuestionResult] = []

    for i, q in enumerate(bundle.questions):
        if i < len(answers):
            raw_answer = answers[i]
            attempted = True
            correct = normalize_answer(raw_answer, q.answer)
        else:
            raw_answer = ""
            attempted = False
            correct = False

        results.append(QuestionResult(
            question_id=q.id,
            position=i,
            expected=q.answer,
            raw_response=raw_answer[:500],
            correct=correct,
            attempted=attempted,
        ))

    return results


def normalize_answer(predicted: str, expected: str) -> bool:
    """
    Check if predicted answer matches expected.
    Handles: multiple choice letters, numeric answers, short text.
    """
    pred = predicted.strip().lower().rstrip('.')
    exp = expected.strip().lower().rstrip('.')

    # Direct match
    if pred == exp:
        return True

    # Multiple choice: extract single letter
    pred_letter = re.match(r'^([a-d])$', pred) or re.search(r'(?:answer\s+is\s+)([a-d])', pred, re.IGNORECASE)
    exp_letter = re.search(r'\b([a-d])\b', exp)
    if pred_letter and exp_letter:
        return pred_letter.group(1) == exp_letter.group(1)

    # Check if expected appears in predicted (for short answers)
    if len(exp) > 2 and exp in pred:
        return True

    return False


# ---------------------------------------------------------------------------
# Results I/O (delegates to benchkit_for_harnesses.results)
# ---------------------------------------------------------------------------

def save_results(results: list[BundleResult], path: Path) -> None:
    """Save results as JSON lines via shared results module."""
    records = [{
        "bundle_id": r.bundle_id,
        "model": r.trial_config.model.name,
        "is_base": r.trial_config.model.is_base,
        "condition": r.trial_config.condition.value,
        "bundle_size": r.trial_config.bundle_size,
        "seed": r.trial_config.seed,
        "accuracy": r.accuracy,
        "completion_rate": r.completion_rate,
        "latency_ms": r.latency_ms,
        "timestamp": r.timestamp,
        "questions": [asdict(qr) for qr in r.question_results],
    } for r in results]
    save_jsonl(records, path)


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load results from JSON lines via shared results module."""
    return load_jsonl(path)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute summary statistics across results.
    Groups by (model, is_base, condition, bundle_size).
    """
    import statistics

    groups: dict[tuple[str, bool, str, int], list[float]] = defaultdict(list)
    completion_groups: dict[tuple[str, bool, str, int], list[float]] = defaultdict(list)
    positional: dict[tuple[str, bool, str, int, int], list[bool]] = defaultdict(list)

    for r in results:
        key: tuple[str, bool, str, int] = (r["model"], r["is_base"], r["condition"], r["bundle_size"])
        groups[key].append(r["accuracy"])
        completion_groups[key].append(r["completion_rate"])

        for q in r["questions"]:
            pos_key: tuple[str, bool, str, int, int] = (*key, q["position"])
            positional[pos_key].append(q["correct"])

    summary: dict[str, Any] = {}
    for key, accs in groups.items():
        model, is_base, condition, bsize = key
        comps = completion_groups[key]
        summary[str(key)] = {
            "model": model,
            "is_base": is_base,
            "condition": condition,
            "bundle_size": bsize,
            "n_bundles": len(accs),
            "mean_accuracy": statistics.mean(accs),
            "std_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0,
            "mean_completion": statistics.mean(comps),
            "positional_accuracy": {
                pos: sum(corrects) / len(corrects)
                for pos_key, corrects in positional.items()
                if pos_key[:4] == key
                for pos in [pos_key[4]]
            },
        }

    return summary
