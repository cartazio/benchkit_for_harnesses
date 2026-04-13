"""Answer bracket protocol: {[{[ answer ]}]} extraction for benchmark evaluation."""

from __future__ import annotations

import re

ANSWER_BRACKET_RE = re.compile(r'\{\[\{\[\s*(.*?)\s*\]\}\]\}', re.DOTALL)


def extract_bracketed_answer(text: str) -> str | None:
    """Extract first {[{[ ... ]}]} bracketed answer from text."""
    m = ANSWER_BRACKET_RE.search(text)
    return m.group(1).strip() if m else None


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
