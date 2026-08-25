import pytest
from pydantic import ValidationError

from preference_lab.schemas import PreferenceExample


def test_chosen_and_rejected_differ_by_case_rejected() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        PreferenceExample(prompt="p", chosen="Hello World", rejected="hello world")


def test_chosen_and_rejected_differ_by_whitespace_rejected() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        PreferenceExample(prompt="p", chosen="same text", rejected="  same text  ")


def test_near_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(
            prompt="p", chosen="The cat sat on the mat.", rejected="The cat sat on the mat!"
        )


def test_valid_distinct_responses_accepted() -> None:
    ex = PreferenceExample(
        prompt="p", chosen="A detailed correct answer", rejected="A wrong answer"
    )
    assert ex.chosen == "A detailed correct answer"
    assert ex.rejected == "A wrong answer"


def test_empty_after_strip_rejected() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(prompt="p", chosen="   ", rejected="answer")
