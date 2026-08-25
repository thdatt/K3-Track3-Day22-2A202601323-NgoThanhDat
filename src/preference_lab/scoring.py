from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WORD_RE = re.compile(r"\b[\w'-]+\b")
_SENTENCE_RE = re.compile(r"[.!?]+")
_CONNECTIVES = {
    "because",
    "while",
    "which",
    "where",
    "when",
    "therefore",
    "however",
    "by",
    "through",
    "during",
    "using",
    "allowing",
    "helping",
}

FEATURE_NAMES = (
    "log_chars",
    "log_words",
    "log_sentences",
    "lexical_diversity",
    "connective_rate",
)


def _token_stats(text: str) -> tuple[str, list[str], int]:
    stripped = text.strip()
    if not stripped:
        return "", [], 0
    words = _WORD_RE.findall(stripped.lower())
    sentences = max(1, len(_SENTENCE_RE.findall(stripped)) + (0 if stripped[-1] in ".!?" else 1))
    return stripped, words, sentences


def response_features(text: str) -> np.ndarray:
    """Return generic, deterministic response-quality features.

    The features intentionally avoid domain-specific answer keys. They make the
    CPU training path fully reproducible while exposing its main limitation:
    quality can be confounded with verbosity/structure on tiny datasets.
    """
    stripped, words, sentences = _token_stats(text)
    if not words:
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    unique_ratio = len(set(words)) / len(words)
    connective_rate = sum(word in _CONNECTIVES for word in words) / len(words)
    return np.asarray(
        [
            math.log1p(len(stripped)),
            math.log1p(len(words)),
            math.log1p(sentences),
            unique_ratio,
            connective_rate,
        ],
        dtype=float,
    )


def score_text(text: str) -> float:
    """Deterministic CPU-safe reference score used as a smoke-test baseline."""
    stripped, words, sentences = _token_stats(text)
    if not words:
        return 0.0
    return float(
        math.log1p(len(stripped))
        + 0.5 * math.log1p(len(words))
        + 0.3 * math.log1p(sentences)
    )


@dataclass(frozen=True)
class LinearPreferenceScorer:
    """Small learned scorer representing DPO's implicit policy/reference reward."""

    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def score(self, text: str) -> float:
        x = (response_features(text) - self.feature_mean) / self.feature_scale
        return float(x @ self.weights)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LinearPreferenceScorer":
        return cls(
            weights=np.asarray(data["weights"], dtype=float),
            feature_mean=np.asarray(data["feature_mean"], dtype=float),
            feature_scale=np.asarray(data["feature_scale"], dtype=float),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearPreferenceScorer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data.get("model")
        if not isinstance(model, dict):
            raise ValueError(f"checkpoint {path} does not contain a trained model")
        return cls.from_dict(model)
