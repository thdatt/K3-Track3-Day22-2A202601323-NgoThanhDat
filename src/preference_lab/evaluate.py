from __future__ import annotations

import json
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return fraction where chosen score is greater than rejected score.

    Explicitly validates that score lists align 1:1 with examples and that
    both lists have equal length. Ties are counted as non-wins (they do not
    contribute to accuracy).
    """
    n = len(examples)
    if n == 0:
        return 0.0
    if len(chosen_scores) != n or len(rejected_scores) != n:
        raise ValueError(
            f"score length mismatch: expected {n} scores per pair, "
            f"got chosen={len(chosen_scores)} rejected={len(rejected_scores)}"
        )
    wins = sum(c > r for c, r in zip(chosen_scores, rejected_scores, strict=True))
    return wins / n


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
