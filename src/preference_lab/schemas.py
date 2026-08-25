from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def _is_near_duplicate(a: str, b: str) -> bool:
    """True when two normalized strings differ only by trailing punctuation."""
    left = _normalize(a).rstrip(".!?;,")
    right = _normalize(b).rstrip(".!?;,")
    return bool(left) and left == right


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty or whitespace only")
        return stripped

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if chosen is None:
            return rejected
        # Robust to whitespace, case, and trivial punctuation differences
        # that still make chosen/rejected effectively the same response.
        if _normalize(chosen) == _normalize(rejected) or _is_near_duplicate(chosen, rejected):
            raise ValueError("chosen and rejected must differ")
        return rejected
