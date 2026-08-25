import pytest

from preference_lab.evaluate import pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def _ex(count: int) -> list[PreferenceExample]:
    return [
        PreferenceExample(prompt=f"p{i}", chosen="chosen", rejected="rejected")
        for i in range(count)
    ]


def test_pairwise_accuracy_all_wins() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_loses_when_rejected_higher() -> None:
    examples = _ex(1)
    assert pairwise_accuracy(examples, [1.0], [2.0]) == 0.0


def test_pairwise_accuracy_handles_partial() -> None:
    examples = _ex(3)
    assert pairwise_accuracy(examples, [2.0, 1.5, 3.0], [1.0, 2.0, 0.5]) == pytest.approx(2 / 3)


def test_pairwise_accuracy_ties_not_counted_as_wins() -> None:
    examples = _ex(2)
    # first wins, second ties (not a win)
    acc = pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0])
    assert acc == pytest.approx(1 / 2)


def test_pairwise_accuracy_raises_on_length_mismatch() -> None:
    examples = _ex(2)
    with pytest.raises(ValueError, match="length"):
        pairwise_accuracy(examples, [1.0], [1.0, 2.0])


def test_pairwise_accuracy_empty_returns_zero() -> None:
    assert pairwise_accuracy([], [], []) == 0.0
