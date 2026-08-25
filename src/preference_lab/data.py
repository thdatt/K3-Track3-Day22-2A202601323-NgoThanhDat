from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample

_DUPLICATE_PROMPT_ERROR = "duplicate prompt"


def load_jsonl(path: str | Path, *, detect_duplicates: bool = True) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Raises a ``ValueError`` with the offending line number when a line is not
    valid JSON or fails schema validation. Optionally rejects duplicate
    prompts (fail-loud hygiene for alignment data). Blank lines are skipped.

    PII guardrails are exclusion-focused: schema validation (`prompt`/`chosen`/
    `rejected` are non-empty after stripping) plus duplicate detection cover
    the common data-quality hazards for this pipeline.
    """
    examples: list[PreferenceExample] = []
    seen_prompts: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                example = PreferenceExample.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"line {line_no}: invalid preference example: {exc}") from exc
            if detect_duplicates and example.prompt in seen_prompts:
                raise ValueError(f"line {line_no}: {_DUPLICATE_PROMPT_ERROR} {example.prompt!r}")
            seen_prompts.add(example.prompt)
            examples.append(example)
    return examples


def group_by_prompt(examples: list[PreferenceExample]) -> dict[str, list[PreferenceExample]]:
    """Group examples by their exact prompt, preserving insertion order."""
    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(example.prompt, []).append(example)
    return groups


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid train/val leakage.

    Groups all rows by prompt (so every row for a prompt lands in the same
    split), deterministically shuffles the unique prompts using ``seed``, and
    assigns the first ``validation_ratio`` of prompts to validation.
    """
    if not examples:
        return [], []
    groups = group_by_prompt(examples)
    unique_prompts = list(groups)
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)
    val_count = max(1, min(len(unique_prompts) - 1, int(len(unique_prompts) * validation_ratio)))
    val_prompts = set(unique_prompts[:val_count])
    train: list[PreferenceExample] = []
    val: list[PreferenceExample] = []
    for prompt, group in groups.items():
        (val if prompt in val_prompts else train).extend(group)
    return train, val
