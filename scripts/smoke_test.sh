#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src:."
python -m preference_lab.cli run-all --config configs/local.yaml
cat outputs/train_metrics.json
cat outputs/eval_metrics.json
cat outputs/regression_metrics.json
