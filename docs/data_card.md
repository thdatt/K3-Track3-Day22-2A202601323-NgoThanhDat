# Data Card — Sample Preference Dataset

## Dataset Summary

- **File:** `data/sample_preferences.jsonl`
- **Size:** 24 preference pairs
- **Format:** JSONL with `prompt`, `chosen`, `rejected`, and `metadata`
- **Domain:** machine-learning education / technical explanation
- **Preference rubric:** primarily factual accuracy and explanatory quality
- **Language:** English
- **Repository provenance:** bundled sample data from the starter repository; no external source or authoring history is documented in the repository.

## Intended Use

This small dataset is intended for a teaching lab on preference alignment, data validation, DPO/ORPO losses, train/validation splitting, evaluation, and regression testing. It is not large or diverse enough to support production model alignment claims.

## Validation and Cleaning in This Run

The loader validates every non-empty JSONL line with the `PreferenceExample` schema, strips surrounding whitespace, rejects empty fields, rejects chosen/rejected near-duplicates, reports line-numbered parsing/schema errors, and rejects duplicate prompts by default. The current 24-row file passed validation without manual row edits during this run.

## Split

The experiment uses a deterministic prompt-grouped 80/20 split with seed 42:

- Train: 20 pairs
- Validation: 4 pairs
- Prompt overlap: 0

Grouping by prompt prevents leakage if future datasets contain multiple preference rows for the same prompt.

## Known Biases and Limitations

All 24 preferred responses are longer than their rejected counterparts. Average preferred response length is about 178.7 characters versus 95.0 characters for rejected responses. This creates a strong verbosity/structure shortcut that a simple preference scorer can exploit. The dataset is also narrow in domain and does not directly cover safety, uncertainty, strict formatting, or troubleshooting behaviors.

## Regression Set

`data/regression_preferences.jsonl` contains four hand-authored evaluation-only candidate pairs for medical safety, exact word-count instruction following, uncertainty, and troubleshooting with missing context. These examples are not included in DPO training.

## Privacy and Safety

The bundled sample data contains general educational questions and no intentionally included personal data. The regression set uses fictional/general prompts. No private dataset or API-generated synthetic data was added during this run.
