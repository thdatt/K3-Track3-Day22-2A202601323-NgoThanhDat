from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import pairwise_accuracy
from .schemas import PreferenceExample
from .scoring import FEATURE_NAMES, LinearPreferenceScorer, response_features


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    output_dir: str = "outputs"
    epochs: int = 400
    learning_rate: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42


class PreferenceTrainer:
    """CPU-compatible preference trainer.

    ``mock`` preserves the starter interface. ``dpo`` trains a small linear
    implicit-reward model with the exact DPO logistic objective. This makes the
    lab executable end to end without downloading model weights or requiring a
    GPU, while keeping the DPO math, reference-policy interpretation, split,
    checkpointing and held-out evaluation real and reproducible.
    """

    config: TrainingConfig

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(
        self,
        train_examples: list[PreferenceExample] | None = None,
        val_examples: list[PreferenceExample] | None = None,
    ) -> dict[str, object]:
        if self.config.method == "mock" or train_examples is None:
            return self._write_mock()
        if self.config.method != "dpo":
            raise ValueError("CPU trainer currently supports method='dpo' or 'mock'")
        return self._train_dpo(train_examples, val_examples or [])

    def _write_mock(self) -> dict[str, object]:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        checkpoint = {
            "run_id": run_id,
            "method": self.config.method,
            "status": "mock-complete",
            "config": {
                "beta": self.config.beta,
                "lambda_orpo": self.config.lambda_orpo,
                "max_length": self.config.max_length,
                "batch_size": self.config.batch_size,
            },
            "device": "cpu",
            "checkpoint_path": "none (mock)",
        }
        metrics: dict[str, object] = {
            "status": "mock",
            "method": self.config.method,
            "train_loss": None,
            "eval_loss": None,
            "samples_seen": 0,
            "elapsed_seconds": 0.0,
        }
        (out / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def _train_dpo(
        self, train_examples: list[PreferenceExample], val_examples: list[PreferenceExample]
    ) -> dict[str, object]:
        if not train_examples:
            raise ValueError("DPO training requires at least one training example")
        start = time.perf_counter()
        rng = np.random.default_rng(self.config.seed)
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_train_features = np.vstack(
            [response_features(text) for ex in train_examples for text in (ex.chosen, ex.rejected)]
        )
        mean = all_train_features.mean(axis=0)
        scale = all_train_features.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)

        chosen = np.vstack([(response_features(ex.chosen) - mean) / scale for ex in train_examples])
        rejected = np.vstack(
            [(response_features(ex.rejected) - mean) / scale for ex in train_examples]
        )
        diffs = chosen - rejected

        weights = np.zeros(diffs.shape[1], dtype=float)
        m = np.zeros_like(weights)
        v = np.zeros_like(weights)
        adam_b1, adam_b2, eps = 0.9, 0.999, 1e-8
        beta = self.config.beta
        losses: list[float] = []

        for epoch in range(1, self.config.epochs + 1):
            order = rng.permutation(len(diffs))
            shuffled = diffs[order]
            z = beta * (shuffled @ weights)
            loss_terms = np.logaddexp(0.0, -z)
            loss = float(loss_terms.mean() + 0.5 * self.config.weight_decay * (weights @ weights))
            losses.append(loss)

            sigmoid_neg_z = 1.0 / (1.0 + np.exp(np.clip(z, -60.0, 60.0)))
            grad = (-beta * sigmoid_neg_z[:, None] * shuffled).mean(axis=0)
            grad += self.config.weight_decay * weights

            m = adam_b1 * m + (1.0 - adam_b1) * grad
            v = adam_b2 * v + (1.0 - adam_b2) * (grad * grad)
            m_hat = m / (1.0 - adam_b1**epoch)
            v_hat = v / (1.0 - adam_b2**epoch)
            weights -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        scorer = LinearPreferenceScorer(weights=weights, feature_mean=mean, feature_scale=scale)

        def accuracy(examples: list[PreferenceExample]) -> float:
            return pairwise_accuracy(
                examples,
                [scorer.score(ex.chosen) for ex in examples],
                [scorer.score(ex.rejected) for ex in examples],
            )

        train_acc = accuracy(train_examples)
        val_acc = accuracy(val_examples) if val_examples else 0.0
        val_diffs = np.asarray(
            [scorer.score(ex.chosen) - scorer.score(ex.rejected) for ex in val_examples], dtype=float
        )
        val_loss = (
            float(np.logaddexp(0.0, -beta * val_diffs).mean()) if len(val_diffs) else None
        )
        train_margins = np.asarray(
            [scorer.score(ex.chosen) - scorer.score(ex.rejected) for ex in train_examples], dtype=float
        )
        elapsed = time.perf_counter() - start
        run_id = uuid.uuid4().hex[:8]

        checkpoint = {
            "run_id": run_id,
            "method": "dpo",
            "status": "complete",
            "device": "cpu",
            "model_type": "linear_implicit_reward_scorer",
            "reference": "zero implicit reward (policy initialized at reference)",
            "config": {
                "beta": beta,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
            "model": scorer.to_dict(),
        }
        metrics: dict[str, object] = {
            "status": "complete",
            "method": "dpo",
            "model_type": "linear_implicit_reward_scorer",
            "train_loss_initial": losses[0],
            "train_loss_final": losses[-1],
            "eval_loss": val_loss,
            "train_pairwise_accuracy": train_acc,
            "val_pairwise_accuracy": val_acc,
            "train_mean_preference_margin": float(train_margins.mean()),
            "samples_seen": len(train_examples) * self.config.epochs,
            "train_pairs": len(train_examples),
            "val_pairs": len(val_examples),
            "epochs": self.config.epochs,
            "elapsed_seconds": elapsed,
            "feature_names": list(FEATURE_NAMES),
        }
        (out / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (out / "training_curve.json").write_text(
            json.dumps({"loss": losses}, indent=2), encoding="utf-8"
        )
        return metrics
