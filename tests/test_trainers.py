import json

from preference_lab.trainers import PreferenceTrainer, TrainingConfig


def _config(tmp_path, method: str = "mock") -> TrainingConfig:
    return TrainingConfig(method=method, output_dir=str(tmp_path))


def test_mock_trainer_writes_checkpoint(tmp_path) -> None:
    trainer = PreferenceTrainer(_config(tmp_path))
    trainer.train()
    ckpt_path = tmp_path / "checkpoint.json"
    assert ckpt_path.exists()
    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert data["method"] == "mock"


def test_mock_trainer_writes_metrics(tmp_path) -> None:
    trainer = PreferenceTrainer(_config(tmp_path))
    trainer.train()
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "status" in metrics


def test_mock_trainer_creates_output_dir(tmp_path) -> None:
    nested = tmp_path / "nested" / "dir"
    trainer = PreferenceTrainer(_config(nested))
    trainer.train()
    assert (nested / "checkpoint.json").exists()


def test_dpo_trainer_reduces_loss_and_saves_learned_model(tmp_path) -> None:
    from preference_lab.data import load_jsonl, split_by_prompt

    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.2, seed=42)
    cfg = TrainingConfig(
        method="dpo",
        output_dir=str(tmp_path),
        epochs=50,
        learning_rate=0.1,
        weight_decay=0.01,
        seed=42,
    )
    metrics = PreferenceTrainer(cfg).train(train, val)
    assert float(metrics["train_loss_final"]) < float(metrics["train_loss_initial"])
    assert 0.0 <= float(metrics["val_pairwise_accuracy"]) <= 1.0
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "complete"
    assert checkpoint["model_type"] == "linear_implicit_reward_scorer"
    assert len(checkpoint["model"]["weights"]) == 5
