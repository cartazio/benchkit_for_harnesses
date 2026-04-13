"""IFEval+ instruction-following checkers.

Adapted from google-research/instruction_following_eval — simplified
implementation of key instruction types.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

import langdetect


def check_instruction_following(
    response: str,
    instruction_id: str,
    kwargs: dict[str, Any],
    prompt: str,
) -> bool:
    """
    Check if response follows a single instruction.
    
    Simplified implementation of key instruction types.
    For full fidelity, use the google-research evaluation library.
    """
    response = response.strip()
    if not response:
        return False
    
    # Implement key checkers
    if instruction_id == "punctuation:no_comma":
        return "," not in response
    
    elif instruction_id == "detectable_format:json_format":
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if json_match:
                try:
                    json.loads(json_match.group(1))
                    return True
                except json.JSONDecodeError:
                    pass
            return False
    
    elif instruction_id == "detectable_format:title":
        # Check for title wrapped in <<>>
        return "<<" in response and ">>" in response
    
    elif instruction_id == "length_constraints:number_words":
        num_words = kwargs.get("num_words", 0)
        relation = kwargs.get("relation", "at least")
        word_count = len(response.split())
        if relation == "at least":
            return word_count >= num_words
        else:  # "less than"
            return word_count < num_words
    
    elif instruction_id == "length_constraints:number_paragraphs":
        num_paragraphs = kwargs.get("num_paragraphs", 1)
        # Count paragraphs (sequences separated by blank lines)
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        return len(paragraphs) >= num_paragraphs
    
    elif instruction_id == "length_constraints:number_sentences":
        num_sentences = kwargs.get("num_sentences", 1)
        relation = kwargs.get("relation", "at least")
        # Simple sentence count (periods, !, ?)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        count = len(sentences)
        if relation == "at least":
            return count >= num_sentences
        else:
            return count < num_sentences
    
    elif instruction_id == "keywords:existence":
        keywords = kwargs.get("keywords", [])
        response_lower = response.lower()
        return all(kw.lower() in response_lower for kw in keywords)
    
    elif instruction_id == "keywords:forbidden_words":
        forbidden = kwargs.get("forbidden_words", [])
        response_lower = response.lower()
        return not any(fw.lower() in response_lower for fw in forbidden)
    
    elif instruction_id == "keywords:frequency":
        keyword = kwargs.get("keyword", "")
        frequency = kwargs.get("frequency", 1)
        relation = kwargs.get("relation", "at least")
        count = response.lower().count(keyword.lower())
        if relation == "at least":
            return count >= frequency
        else:
            return count < frequency
    
    elif instruction_id == "startend:end_checker":
        end_phrase = kwargs.get("end_phrase", "")
        return response.rstrip().endswith(end_phrase)
    
    elif instruction_id == "startend:quotation":
        # Response should be wrapped in quotes
        stripped = response.strip()
        return (stripped.startswith('"') and stripped.endswith('"')) or \
               (stripped.startswith("'") and stripped.endswith("'"))
    
    elif instruction_id == "detectable_content:postscript":
        marker = kwargs.get("postscript_marker", "P.S.")
        return marker in response
    
    elif instruction_id == "detectable_format:number_bullet_lists":
        num_bullets = kwargs.get("num_bullets", 1)
        # Count distinct bullet *lists* (contiguous blocks of bullet lines)
        bullet_re = re.compile(r'^\s*(?:[\*\-\u2022]|\d+[.\)])\s+', re.MULTILINE)
        lines = response.split('\n')
        in_list = False
        list_count = 0
        for line in lines:
            is_bullet = bool(bullet_re.match(line))
            if is_bullet and not in_list:
                list_count += 1
                in_list = True
            elif not is_bullet and line.strip():
                in_list = False
        return list_count >= num_bullets

    elif instruction_id == "detectable_format:number_highlighted_sections":
        num_highlights = kwargs.get("num_highlights", 1)
        # Count markdown bold or *text*
        highlights = re.findall(r'\*\*[^*]+\*\*|\*[^*]+\*', response)
        return len(highlights) >= num_highlights
    
    elif instruction_id == "combination:two_responses":
        # Require explicit paired markers indicating two distinct responses
        resp_lower = response.lower()
        has_pair = (
            ("response 1" in resp_lower and "response 2" in resp_lower) or
            ("response a" in resp_lower and "response b" in resp_lower) or
            re.search(r'\b1st response\b', resp_lower) is not None and
            re.search(r'\b2nd response\b', resp_lower) is not None
        )
        return has_pair
    
    elif instruction_id == "change_case:english_lowercase":
        # All letters should be lowercase
        return response == response.lower()
    
    elif instruction_id == "change_case:english_capital":
        # All letters should be uppercase
        return response == response.upper()
    
    elif instruction_id == "detectable_format:constrained_response":
        # Response must be one of specific options
        options = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
        return any(opt in response for opt in options)
    
    elif instruction_id == "language:response_language":
        language = kwargs.get("language", "en")
        detected = langdetect.detect(response)
        return detected == language
    
    # Default: can't verify this instruction type — count as not followed
    # to avoid inflating accuracy. Log for visibility.
    print(f"WARNING: unhandled instruction type '{instruction_id}', scoring as not-followed", file=sys.stderr)
    return False


def evaluate_response(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict[str, Any]],
    prompt: str,
) -> tuple[bool, list[bool]]:
    """
    Evaluate if response follows all instructions.
    
    Returns:
        (follow_all, follow_list) where follow_list[i] indicates
        whether instruction_ids[i] was followed.
    """
    follow_list = []
    for instr_id, kw in zip(instruction_ids, kwargs_list):
        follows = check_instruction_following(response, instr_id, kw, prompt)
        follow_list.append(follows)
    
    return all(follow_list), follow_list
