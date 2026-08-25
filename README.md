# Preference Alignment Lab: DPO & ORPO

A small production-style preference-alignment lab covering data hygiene, DPO/ORPO losses, prompt-grouped splitting, held-out evaluation, regression tests, and experiment reporting.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
make run-all
```

If you are running directly from a source checkout without installing the package:

```bash
export PYTHONPATH=src:.
python -m pytest -q
python -m preference_lab.cli run-all --config configs/local.yaml
```

## End-to-end outputs

`run-all` validates the 24 preference pairs, creates an 80/20 prompt-grouped split, trains the CPU DPO implicit-reward scorer, evaluates the four held-out pairs, and runs four safety/instruction regression pairs. Results are written to `outputs/`.

See `docs/REPORT.md` for the completed experiment report and `docs/data_card.md` for dataset documentation.

## Training modes

- `dpo`: reproducible CPU DPO optimization of a lightweight linear implicit-reward scorer. This requires only NumPy and does not download model weights.
- `mock`: preserves the original starter mock interface for tests/examples.
- `orpo_loss`: the ORPO loss utility is implemented and unit-tested, but the end-to-end CPU trainer in this submission is intentionally configured for DPO.

The CPU scorer is an educational alignment model, **not** a full LLM fine-tune. A full TRL/Transformers training path can replace it when model weights and training dependencies are available.

## Commands

```bash
make test
make validate
make train
make run-eval
make regression
make run-all
bash scripts/smoke_test.sh
```

Optional synthetic preference generation uses `scripts/generate_data.py` and requires `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.

## Repository layout

```text
src/preference_lab/     package and DPO/ORPO logic
data/                   training and regression preference pairs
configs/                experiment configuration
docs/                   report, data card, lab notes
scripts/                synthetic-data and smoke-test entrypoints
tests/                  automated tests
outputs/                 generated experiment artifacts (gitignored)
```
