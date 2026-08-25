.PHONY: setup test lint typecheck format validate train run-eval regression run-all clean
PYTHON ?= python
ENV = PYTHONPATH=src:.
setup:
	$(PYTHON) -m pip install -e '.[dev]'
test:
	$(ENV) $(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src tests scripts
typecheck:
	$(PYTHON) -m mypy src
format:
	$(PYTHON) -m ruff format src tests scripts
validate:
	$(ENV) $(PYTHON) -m preference_lab.cli validate data/sample_preferences.jsonl
train:
	$(ENV) $(PYTHON) -m preference_lab.cli train --config configs/local.yaml
run-eval:
	$(ENV) $(PYTHON) -m preference_lab.cli evaluate --config configs/local.yaml
regression:
	$(ENV) $(PYTHON) -m preference_lab.cli regression --config configs/local.yaml
run-all:
	$(ENV) $(PYTHON) -m preference_lab.cli run-all --config configs/local.yaml
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache outputs
