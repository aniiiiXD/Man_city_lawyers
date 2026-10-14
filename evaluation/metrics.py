"""Task-level paired bootstrap intervals for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random
from statistics import fmean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    observations: int


def _validate(values: Sequence[float], confidence: float, resamples: int) -> None:
    if not values:
        raise ValueError("At least one task-level observation is required")
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("Bootstrap values must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be positive")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Linear-interpolated quantile, avoiding a NumPy dependency."""

    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Estimate a mean and percentile interval by resampling independent tasks."""

    _validate(values, confidence, resamples)
    observations = [float(value) for value in values]
    rng = Random(seed)
    draws = [
        fmean(rng.choices(observations, k=len(observations)))
        for _ in range(resamples)
    ]
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=fmean(observations),
        lower=_quantile(draws, tail),
        upper=_quantile(draws, 1.0 - tail),
        confidence=confidence,
        observations=len(observations),
    )


def paired_bootstrap_difference(
    control: Mapping[str, float],
    treatment: Mapping[str, float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap treatment-minus-control over matched task IDs.

    Task IDs, rather than individual agent events, are the independent units.
    """

    if set(control) != set(treatment):
        raise ValueError("Paired conditions must contain exactly the same task IDs")
    task_ids = sorted(control)
    differences = [float(treatment[key]) - float(control[key]) for key in task_ids]
    return bootstrap_mean_interval(
        differences,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )

