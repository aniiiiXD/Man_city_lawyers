"""Pure-Python checks for grouped asynchronous policy updates."""

import pytest

from learning.orchestrator_training import normalized_group_advantages


def test_advantages_are_finite_and_centered() -> None:
    values = normalized_group_advantages([1.0, 2.0, 3.0])

    assert sum(values) == pytest.approx(0.0)
    assert values[0] < values[1] < values[2]
    assert all(value == value for value in values)


def test_advantages_handle_small_or_zero_variance_groups() -> None:
    assert normalized_group_advantages([2.0]) == [0.0]
    assert normalized_group_advantages([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]


def test_advantages_reject_non_finite_returns() -> None:
    with pytest.raises(ValueError, match="finite"):
        normalized_group_advantages([1.0, float("inf")])
