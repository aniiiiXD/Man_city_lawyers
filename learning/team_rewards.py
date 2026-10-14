"""Auditable team-level reward accounting for asynchronous episodes.

Task success, policy compliance, and resource use remain separate components
in the result so experiments can inspect each source of reward or penalty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class TeamRewardWeights:
    verified_completion: float = 10.0
    policy_compliance: float = 2.0
    evidence_support: float = 1.0
    unsafe_attempt: float = -4.0
    unauthorized_side_effect: float = -8.0
    invalid_merge: float = -3.0
    redundant_worker: float = -0.5
    worker_spawn: float = -0.1
    retry: float = -0.1
    virtual_tick: float = -0.05
    thousand_tokens: float = -0.02
    tool_call: float = -0.025
    recovered_hold: float = 0.5
    recovered_failure: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value):
                raise ValueError(f"Reward weight {name} must be finite")


@dataclass(frozen=True)
class TeamOutcome:
    """Verifier and trace facts accumulated over one complete team episode."""

    verified_complete: bool
    policy_compliant: bool
    evidence_supported: bool
    unsafe_attempts: int = 0
    unauthorized_side_effects: int = 0
    invalid_merges: int = 0
    redundant_workers: int = 0
    worker_spawns: int = 0
    retries: int = 0
    virtual_ticks: int = 0
    generated_tokens: int = 0
    tool_calls: int = 0
    recovered_holds: int = 0
    recovered_failures: int = 0

    def __post_init__(self) -> None:
        for name in (
            "unsafe_attempts",
            "unauthorized_side_effects",
            "invalid_merges",
            "redundant_workers",
            "worker_spawns",
            "retries",
            "virtual_ticks",
            "generated_tokens",
            "tool_calls",
            "recovered_holds",
            "recovered_failures",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class TeamReward:
    total: float
    components: dict[str, float]


def team_reward(
    outcome: TeamOutcome,
    weights: TeamRewardWeights = TeamRewardWeights(),
) -> TeamReward:
    """Score one verified episode without rewarding activity for its own sake."""

    components = {
        "verified_completion": weights.verified_completion * outcome.verified_complete,
        "policy_compliance": weights.policy_compliance * outcome.policy_compliant,
        "evidence_support": weights.evidence_support * outcome.evidence_supported,
        "unsafe_attempts": weights.unsafe_attempt * outcome.unsafe_attempts,
        "unauthorized_side_effects": (
            weights.unauthorized_side_effect * outcome.unauthorized_side_effects
        ),
        "invalid_merges": weights.invalid_merge * outcome.invalid_merges,
        "redundant_workers": weights.redundant_worker * outcome.redundant_workers,
        "worker_spawns": weights.worker_spawn * outcome.worker_spawns,
        "retries": weights.retry * outcome.retries,
        "virtual_ticks": weights.virtual_tick * outcome.virtual_ticks,
        "generated_tokens": weights.thousand_tokens * (outcome.generated_tokens / 1000.0),
        "tool_calls": weights.tool_call * outcome.tool_calls,
        "recovered_holds": weights.recovered_hold * outcome.recovered_holds,
        "recovered_failures": weights.recovered_failure * outcome.recovered_failures,
    }
    total = sum(components.values())
    if not isfinite(total):
        raise ValueError("Computed team reward is not finite")
    return TeamReward(total=total, components=components)

