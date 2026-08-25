from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print

from .config import load_config
from .data import load_jsonl, split_by_prompt
from .evaluate import pairwise_accuracy
from .scoring import LinearPreferenceScorer, score_text
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


def _training_config(cfg: dict[str, Any]) -> TrainingConfig:
    training = cfg["training"]
    return TrainingConfig(
        method=str(training["method"]),
        beta=float(training.get("beta", 0.1)),
        lambda_orpo=float(training.get("lambda_orpo", 0.1)),
        max_length=int(training.get("max_length", 512)),
        batch_size=int(training.get("batch_size", 2)),
        output_dir=str(cfg["paths"]["output_dir"]),
        epochs=int(training.get("epochs", 400)),
        learning_rate=float(training.get("learning_rate", 0.1)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        seed=int(cfg.get("seed", 42)),
    )


def _split(cfg: dict[str, Any]):
    examples = load_jsonl(cfg["paths"]["train_data"])
    ratio = float(cfg["training"].get("validation_ratio", 0.2))
    return examples, split_by_prompt(examples, validation_ratio=ratio, seed=int(cfg.get("seed", 42)))


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def train(config: Path) -> None:
    cfg = load_config(config)
    _, (train_examples, val_examples) = _split(cfg)
    trainer = PreferenceTrainer(_training_config(cfg))
    metrics = trainer.train(train_examples, val_examples)
    print(json.dumps(metrics, indent=2, sort_keys=True))


@app.command()
def evaluate(config: Path) -> None:
    cfg = load_config(config)
    all_examples, (_, val_examples) = _split(cfg)
    output_dir = Path(cfg["paths"]["output_dir"])
    checkpoint = output_dir / "checkpoint.json"
    scorer = LinearPreferenceScorer.load(checkpoint)

    chosen_scores = [scorer.score(ex.chosen) for ex in val_examples]
    rejected_scores = [scorer.score(ex.rejected) for ex in val_examples]
    heuristic_all = pairwise_accuracy(
        all_examples,
        [score_text(ex.chosen) for ex in all_examples],
        [score_text(ex.rejected) for ex in all_examples],
    )
    margins = [c - r for c, r in zip(chosen_scores, rejected_scores, strict=True)]
    metrics: dict[str, object] = {
        "evaluation_split": "held-out validation",
        "pairwise_accuracy": pairwise_accuracy(val_examples, chosen_scores, rejected_scores),
        "n_pairs": len(val_examples),
        "mean_preference_margin": float(sum(margins) / len(margins)) if margins else 0.0,
        "reference_heuristic_full_dataset_accuracy": heuristic_all,
    }
    _write_json(output_dir / "eval_metrics.json", metrics)
    print(f"[green]Wrote metrics to {output_dir / 'eval_metrics.json'}[/green]")
    print(json.dumps(metrics, indent=2, sort_keys=True))


@app.command()
def regression(config: Path) -> None:
    cfg = load_config(config)
    regression_path = Path(cfg["paths"].get("regression_data", "data/regression_preferences.jsonl"))
    examples = load_jsonl(regression_path)
    output_dir = Path(cfg["paths"]["output_dir"])
    scorer = LinearPreferenceScorer.load(output_dir / "checkpoint.json")
    chosen = [scorer.score(ex.chosen) for ex in examples]
    rejected = [scorer.score(ex.rejected) for ex in examples]
    details = []
    for ex, c, r in zip(examples, chosen, rejected, strict=True):
        details.append(
            {
                "category": ex.metadata.get("category", "unknown"),
                "prompt": ex.prompt,
                "pass": c > r,
                "chosen_score": c,
                "rejected_score": r,
                "margin": c - r,
            }
        )
    metrics: dict[str, object] = {
        "before_alignment_accuracy": 0.0,
        "after_alignment_accuracy": pairwise_accuracy(examples, chosen, rejected),
        "n_regressions": len(examples),
        "details": details,
        "interpretation": "candidate-ranking regression; this CPU scorer does not generate answers",
    }
    _write_json(output_dir / "regression_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


@app.command(name="run-all")
def run_all(config: Path = Path("configs/local.yaml")) -> None:
    cfg = load_config(config)
    data_path = Path(cfg["paths"]["train_data"])
    validate(data_path)
    train(config)
    evaluate(config)
    regression(config)


if __name__ == "__main__":
    app()
