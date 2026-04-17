"""Tests for answer-bracket extraction and evaluation."""

from __future__ import annotations

from benchkit_for_harnesses.brackets import (
    eval_bracketed,
    extract_bracketed_answer,
    extract_bracketed_answers,
)


def test_extract_returns_last_pair() -> None:
    # LAST match wins — matches prompt contract ("wrap your FINAL answer").
    assert extract_bracketed_answer("first {[{[ draft ]}]} then {[{[ final ]}]}") == "final"



def test_extract_single() -> None:
    assert extract_bracketed_answer("noise {[{[ Paris ]}]} more noise") == "Paris"


def test_extract_single_none_when_absent() -> None:
    assert extract_bracketed_answer("no brackets here") is None


def test_extract_multi_ordered() -> None:
    text = "a {[{[ one ]}]} b {[{[ two ]}]} c {[{[ three ]}]}"
    assert extract_bracketed_answers(text) == ["one", "two", "three"]


def test_eval_exact_match() -> None:
    assert eval_bracketed("{[{[ Paris ]}]}", "Paris") is True


def test_eval_case_insensitive() -> None:
    assert eval_bracketed("{[{[ paris ]}]}", "Paris") is True


def test_eval_multi_target_pipe() -> None:
    assert eval_bracketed("{[{[ paris ]}]}", "Paris|London") is True
    assert eval_bracketed("{[{[ tokyo ]}]}", "Paris|London") is False


def test_eval_loose_containment_fallback_when_no_brackets() -> None:
    # Loose mode (default): when no brackets, fall back to containment
    assert eval_bracketed("The answer is Paris.", "Paris") is True


def test_eval_strict_requires_brackets() -> None:
    # Strict mode: no brackets → not correct regardless of content
    assert eval_bracketed("The answer is Paris.", "Paris", strict=True) is False


def test_eval_strict_mc_letter_exact() -> None:
    # LongBench-v2: target is a single letter. Strict mode must require exact equality.
    assert eval_bracketed("{[{[ B ]}]}", "B", strict=True) is True
    assert eval_bracketed("{[{[ B ]}]}", "A", strict=True) is False


def test_eval_strict_rejects_substring_letter_match() -> None:
    # In loose mode, "a" ⊂ "answer" silently passes — the exact bug.
    # Strict must reject.
    assert eval_bracketed("{[{[ answer ]}]}", "a", strict=True) is False


def test_eval_loose_accepts_substring() -> None:
    # Document current loose behavior (used by BABILong/InfiniteBench).
    assert eval_bracketed("{[{[ the answer is yes ]}]}", "yes") is True
