import pytest

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_load_jsonl_raises_with_line_number_on_malformed_line(tmp_path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"prompt":"a","chosen":"x","rejected":"y"}\n{"prompt":"b","chosen":"broken json\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(p)


def test_load_jsonl_raises_on_duplicate_prompt(tmp_path) -> None:
    p = tmp_path / "dup.jsonl"
    p.write_text(
        '{"prompt":"same","chosen":"x","rejected":"y"}\n'
        '{"prompt":"same","chosen":"a","rejected":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prompt"):
        load_jsonl(p)


def test_load_jsonl_ignores_blank_lines(tmp_path) -> None:
    p = tmp_path / "blank.jsonl"
    p.write_text(
        '{"prompt":"a","chosen":"x","rejected":"y"}\n\n'
        '{"prompt":"b","chosen":"x","rejected":"y"}\n',
        encoding="utf-8",
    )
    assert len(load_jsonl(p)) == 2


def test_split_keeps_prompt_rows_together() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p1", chosen="a2", rejected="b2"),
        PreferenceExample(prompt="p2", chosen="c", rejected="d"),
        PreferenceExample(prompt="p3", chosen="e", rejected="f"),
    ]
    train, val = split_by_prompt(examples, validation_ratio=0.5, seed=1)
    val_prompts = {e.prompt for e in val}
    # p1 has 2 rows; both rows of p1 must land in the same split
    for e in train:
        assert e.prompt not in val_prompts
    assert len(train) + len(val) == 4


def test_split_is_deterministic_for_same_seed() -> None:
    examples = [PreferenceExample(prompt=f"p{i}", chosen="x", rejected="y") for i in range(10)]
    t1, v1 = split_by_prompt(examples, validation_ratio=0.3, seed=7)
    t2, v2 = split_by_prompt(examples, validation_ratio=0.3, seed=7)
    assert [e.prompt for e in t1] == [e.prompt for e in t2]
    assert [e.prompt for e in v1] == [e.prompt for e in v2]


def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)
