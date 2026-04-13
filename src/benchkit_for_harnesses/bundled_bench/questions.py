"""
Question bank: loads questions from standard benchmarks.

Currently supports:
  - MMLU (via HuggingFace datasets or local CSV)
  - Synthetic (for testing the harness)

MMLU is ideal because:
  1. Multiple-choice with known answers (automatable scoring)
  2. Originally benchmarked few-shot on base models (established coherence)
  3. Wide domain coverage (can control for domain effects)
  4. Well-understood baseline numbers across models
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .harness import Question


# ---------------------------------------------------------------------------
# MMLU loader
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_mmlu_from_hf(
    subjects: Optional[list[str]] = None,
    split: str = "test",
    max_per_subject: int = 50,
) -> list[Question]:
    """
    Load MMLU questions from HuggingFace datasets.

    Requires: pip install datasets
    """
    from datasets import load_dataset

    if subjects is None:
        subjects = MMLU_SUBJECTS

    questions = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split=split)
        except Exception as e:
            print(f"  SKIP {subj}: {e}")
            continue

        for i, row in enumerate(ds):
            if i >= max_per_subject:
                break

            qid = f"mmlu:{subj}:{i}"
            answer_idx = row["answer"]
            answer_letter = ANSWER_MAP.get(answer_idx, str(answer_idx))

            questions.append(Question(
                id=qid,
                text=row["question"],
                answer=answer_letter,
                source=f"mmlu:{subj}",
                choices=row["choices"],
            ))

    return questions


def load_mmlu_from_csv(
    data_dir: Path,
    subjects: Optional[list[str]] = None,
    split: str = "test",
    max_per_subject: int = 50,
) -> list[Question]:
    """
    Load MMLU from local CSV files.
    Expected structure: data_dir/{split}/{subject}_{split}.csv
    CSV columns: question, A, B, C, D, answer
    """
    if subjects is None:
        subjects = MMLU_SUBJECTS

    questions = []
    for subj in subjects:
        csv_path = data_dir / split / f"{subj}_{split}.csv"
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= max_per_subject:
                    break
                if len(row) < 6:
                    continue

                question_text = row[0]
                choices = [row[1], row[2], row[3], row[4]]
                answer = row[5].strip().upper()

                qid = f"mmlu:{subj}:{i}"
                questions.append(Question(
                    id=qid,
                    text=question_text,
                    answer=answer,
                    source=f"mmlu:{subj}",
                    choices=choices,
                ))

    return questions


# ---------------------------------------------------------------------------
# Synthetic questions (for harness testing)
# ---------------------------------------------------------------------------

def load_synthetic(n: int = 100) -> list[Question]:
    """
    Generate simple synthetic questions for harness validation.
    These have unambiguous answers for testing scoring logic.
    """
    questions = []

    # Arithmetic
    rng = random.Random(42)
    for i in range(min(n, 50)):
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b

        questions.append(Question(
            id=f"synth:arith:{i}",
            text=f"What is {a} {op} {b}?",
            answer=str(ans),
            source="synthetic:arithmetic",
        ))

    # Capitals
    capitals = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("Brazil", "Brasilia"), ("Australia", "Canberra"),
        ("Canada", "Ottawa"), ("Italy", "Rome"), ("Spain", "Madrid"),
        ("Egypt", "Cairo"), ("Mexico", "Mexico City"),
        ("India", "New Delhi"), ("China", "Beijing"),
        ("Russia", "Moscow"), ("Argentina", "Buenos Aires"),
        ("South Korea", "Seoul"), ("Thailand", "Bangkok"),
        ("Turkey", "Ankara"), ("Poland", "Warsaw"),
        ("Sweden", "Stockholm"), ("Norway", "Oslo"),
    ]
    for i, (country, capital) in enumerate(capitals):
        if len(questions) >= n:
            break
        questions.append(Question(
            id=f"synth:capital:{i}",
            text=f"What is the capital of {country}?",
            answer=capital,
            source="synthetic:capitals",
        ))

    # Multiple choice
    mc_questions = [
        ("Which planet is largest in our solar system?",
         ["Mars", "Jupiter", "Saturn", "Venus"], "B"),
        ("What is the boiling point of water in Celsius?",
         ["50", "100", "150", "212"], "B"),
        ("Which element has atomic number 1?",
         ["Helium", "Hydrogen", "Lithium", "Carbon"], "B"),
        ("How many continents are there?",
         ["5", "6", "7", "8"], "C"),
        ("What is the speed of light approximately in km/s?",
         ["100,000", "200,000", "300,000", "400,000"], "C"),
    ]
    for i, (text, choices, answer) in enumerate(mc_questions):
        if len(questions) >= n:
            break
        questions.append(Question(
            id=f"synth:mc:{i}",
            text=text,
            answer=answer,
            source="synthetic:mc",
            choices=choices,
        ))

    return questions[:n]


# ---------------------------------------------------------------------------
# Domain-balanced sampling
# ---------------------------------------------------------------------------

def sample_balanced(
    questions: list[Question],
    n: int,
    seed: int = 42,
) -> list[Question]:
    """
    Sample n questions, balanced across source domains.
    Ensures domain diversity in bundles.
    """
    rng = random.Random(seed)

    by_domain: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_domain[q.source].append(q)

    # Shuffle within each domain
    for qs in by_domain.values():
        rng.shuffle(qs)

    # Round-robin across domains
    domains = sorted(by_domain.keys())
    result = []
    idx = {d: 0 for d in domains}

    while len(result) < n:
        added_any = False
        for d in domains:
            if len(result) >= n:
                break
            if idx[d] < len(by_domain[d]):
                result.append(by_domain[d][idx[d]])
                idx[d] += 1
                added_any = True
        if not added_any:
            break

    if len(result) < n:
        raise ValueError(
            f"sample_balanced: requested {n} questions but only {len(result)} "
            f"available across {len(domains)} domains (pool size: {len(questions)})"
        )

    return result
