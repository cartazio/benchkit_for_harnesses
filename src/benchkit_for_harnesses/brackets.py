"""Answer bracket protocol: {[{[ answer ]}]} extraction for benchmark evaluation.

Design notes
------------
Models with visible reasoning (Anthropic extended thinking, system-prompt
induced <squiggle>, etc.) may emit multiple bracket pairs — drafts during
reasoning, then a final answer. We extract the LAST pair, which matches the
prompt contract ("wrap your FINAL answer"). Callers who want all drafts can
use :func:`extract_bracketed_answers`.

Draft markers ``(_( ... )_)`` are provided so the model can mark intermediate
reasoning without polluting the answer-bracket stream; they are not scored.
"""

from __future__ import annotations

import re

__all__ = [
    "ANSWER_BRACKET_RE",
    "ANSWER_INSTRUCTION",
    "DRAFT_CLOSE",
    "DRAFT_OPEN",
    "eval_bracketed",
    "extract_bracketed_answer",
    "extract_bracketed_answers",
]

ANSWER_BRACKET_RE = re.compile(r"\{\[\{\[\s*(.*?)\s*\]\}\]\}", re.DOTALL)

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
    """Extract the LAST ``{[{[ ... ]}]}`` bracketed answer from text.

    The prompt contract says "wrap your FINAL answer" — if multiple pairs are
    present (draft reasoning followed by the answer, or leaked visible-CoT
    from a reasoning model), the final one is authoritative.
    """
    matches = ANSWER_BRACKET_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_bracketed_answers(text: str) -> list[str]:
    """Extract all ``{[{[ ... ]}]}`` bracketed answers from text, in order."""
    return [m.strip() for m in ANSWER_BRACKET_RE.findall(text)]


def eval_bracketed(response: str, target: str, *, strict: bool = False) -> bool:
    """Evaluate a model response against a target using the bracket protocol.

    Targets may be pipe-separated for multi-answer (e.g. ``"Paris|paris"``).
    Comparison is case-insensitive after stripping.

    Modes
    -----
    ``strict=False`` (default, loose): for free-form benchmarks like BABILong
    / InfiniteBench. Accepts case-insensitive equality or target-as-substring
    against the extracted answer. Falls back to containment against the full
    response when no brackets are present.

    ``strict=True``: for benchmarks with discrete targets (e.g. LongBench-v2's
    A/B/C/D letters). Requires brackets AND exact case-insensitive equality —
    one-letter substring matches ("a" ⊂ "answer") are rejected.
    """
    targets = [t.strip().lower() for t in target.split("|") if t.strip()]
    if not targets:
        return False

    extracted = extract_bracketed_answer(response)

    if strict:
        if extracted is None:
            return False
        extracted_lower = extracted.lower().strip()
        return extracted_lower in targets

    if extracted is None:
        # Loose: fall back to containment against the full response.
        resp_lower = response.strip().lower()
        return any(t in resp_lower for t in targets)

    extracted_lower = extracted.lower().strip()
    return any(extracted_lower == t or t in extracted_lower for t in targets)
