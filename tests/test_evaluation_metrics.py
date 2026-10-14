"""Statistical reporting checks based on task-level paired samples."""

import pytest

from evaluation.metrics import bootstrap_mean_interval, paired_bootstrap_difference


def test_bootstrap_is_reproducible_and_reports_task_count() -> None:
    first = bootstrap_mean_interval([0.0, 1.0, 1.0, 0.0], seed=12, resamples=100)
    second = bootstrap_mean_interval([0.0, 1.0, 1.0, 0.0], seed=12, resamples=100)

    assert first == second
    assert first.estimate == 0.5
    assert first.observations == 4
    assert first.lower <= first.estimate <= first.upper


def test_paired_difference_uses_task_level_matches() -> None:
    interval = paired_bootstrap_difference(
        {"task-a": 0.2, "task-b": 0.5, "task-c": 0.0},
        {"task-a": 0.4, "task-b": 0.7, "task-c": 0.2},
        resamples=50,
    )

    assert interval.estimate == pytest.approx(0.2)
    assert interval.lower == pytest.approx(0.2)
    assert interval.upper == pytest.approx(0.2)
    assert interval.observations == 3


def test_paired_difference_rejects_unmatched_or_invalid_data() -> None:
    with pytest.raises(ValueError, match="same task IDs"):
        paired_bootstrap_difference({"a": 1.0}, {"b": 1.0})
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_interval([float("nan")])
