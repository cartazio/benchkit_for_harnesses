"""Smoke tests for IFEval instruction checkers."""

from __future__ import annotations

from benchkit_for_harnesses.ifeval.checkers import (
    check_instruction_following,
    evaluate_response,
)


def _chk(instr: str, kwargs: dict[str, object], resp: str, prompt: str = "") -> bool:
    return check_instruction_following(resp, instr, kwargs, prompt)


def test_no_comma_pass() -> None:
    assert _chk("punctuation:no_comma", {}, "no commas here") is True


def test_no_comma_fail() -> None:
    assert _chk("punctuation:no_comma", {}, "has, a comma") is False


def test_json_format_pass() -> None:
    assert _chk("detectable_format:json_format", {}, '{"a": 1}') is True


def test_json_format_pass_fenced() -> None:
    assert _chk("detectable_format:json_format", {}, "```json\n{\"a\": 1}\n```") is True


def test_json_format_fail() -> None:
    assert _chk("detectable_format:json_format", {}, "not json") is False


def test_number_words_at_least_pass() -> None:
    assert _chk(
        "length_constraints:number_words",
        {"num_words": 3, "relation": "at least"},
        "one two three four",
    ) is True


def test_number_words_at_least_fail() -> None:
    assert _chk(
        "length_constraints:number_words",
        {"num_words": 10, "relation": "at least"},
        "too short",
    ) is False


def test_keywords_existence_pass() -> None:
    assert _chk(
        "keywords:existence",
        {"keywords": ["Paris", "Tokyo"]},
        "Paris and tokyo are capitals",
    ) is True


def test_keywords_forbidden_fail() -> None:
    assert _chk(
        "keywords:forbidden_words",
        {"forbidden_words": ["foo"]},
        "avoid foo in text",
    ) is False


def test_english_lowercase_pass() -> None:
    assert _chk("change_case:english_lowercase", {}, "all lowercase 123") is True


def test_english_lowercase_fail() -> None:
    assert _chk("change_case:english_lowercase", {}, "Has Capitals") is False


def test_quotation_pass() -> None:
    assert _chk("startend:quotation", {}, '"wrapped in quotes"') is True


def test_quotation_fail() -> None:
    assert _chk("startend:quotation", {}, "unquoted") is False


def test_evaluate_response_all_and_list() -> None:
    follow_all, follow_list = evaluate_response(
        response="hello world",
        instruction_ids=["punctuation:no_comma", "change_case:english_lowercase"],
        kwargs_list=[{}, {}],
        prompt="",
    )
    assert follow_list == [True, True]
    assert follow_all is True


def test_evaluate_response_mixed() -> None:
    follow_all, follow_list = evaluate_response(
        response="Hello, world",
        instruction_ids=["punctuation:no_comma", "change_case:english_lowercase"],
        kwargs_list=[{}, {}],
        prompt="",
    )
    assert follow_list == [False, False]
    assert follow_all is False


def test_unknown_instruction_scored_not_followed() -> None:
    # Unknown checker must NOT silently inflate accuracy.
    assert _chk("nonexistent:checker", {}, "anything") is False
