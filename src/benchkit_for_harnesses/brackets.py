"""Answer bracket protocol: {[{[ answer ]}]} extraction for benchmark evaluation."""

from __future__ import annotations

import re

ANSWER_BRACKET_RE = re.compile(r'\{\[\{\[\s*(.*?)\s*\]\}\]\}', re.DOTALL)

# Draft markers: for intermediate reasoning that should NOT be scored.
# Visually distinct from answer brackets so extraction is unambiguous.
DRAFT_OPEN = "(_("
DRAFT_CLOSE = ")_)"

ANSWER_INSTRUCTION = (
    "Wrap your FINAL answer in {[{[ and ]}]} markers — exactly one pair, "
    "at the end of your response. "
    "If you show intermediate drafts or reasoning steps, wrap those in "
    f"{DRAFT_OPEN} and {DRAFT_CLOSE} so they are not confused with the final answer."
)


def extract_bracketed_answer(text: str) -> str | None:
    """Extract the LAST {[{[ ... ]}]} bracketed answer from text.

    Prompt says "wrap your final answer" — if the model emits multiple bracket
    pairs (e.g. mid-reasoning drafts before the final), take the last one.
    Also handles leaked visible reasoning without --print-thoughts: intermediate
    answers appear earlier in the stream, final answer comes last.
    """
    matches = ANSWER_BRACKET_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_bracketed_answers(text: str) -> list[str]:
    """Extract all {[{[ ... ]}]} bracketed answers from text, in order."""
    return [m.strip() for m in ANSWER_BRACKET_RE.findall(text)]


def eval_bracketed(response: str, target: str) -> bool:
    """
    Extract bracketed answer and compare to target(s).

    Targets may be pipe-separated for multi-answer (e.g. "Paris|paris").
    Falls back to direct containment if no brackets found.
    """
    extracted = extract_bracketed_answer(response)
    targets = [t.strip().lower() for t in target.split("|")]

    if extracted is None:
        # Fallback: direct containment
        resp_lower = response.strip().lower()
        return any(t in resp_lower for t in targets)

    extracted_lower = extracted.lower().strip()
    return any(extracted_lower == t or t in extracted_lower for t in targets)
