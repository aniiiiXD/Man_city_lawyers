"""Contract checks for team-level reward accounting."""

import pytest

from learning.team_rewards import TeamOutcome, TeamRewardWeights, team_reward


def test_equivalent_success_rewards_less_for_unnecessary_team_activity() -> None:
    small = team_reward(
        TeamOutcome(verified_complete=True, policy_compliant=True, evidence_supported=True)
    )
    swarm = team_reward(
        TeamOutcome(
            verified_complete=True,
            policy_compliant=True,
            evidence_supported=True,
            worker_spawns=5,
            redundant_workers=3,
            generated_tokens=40_000,
            tool_calls=12,
        )
    )

    assert swarm.total < small.total
    assert swarm.components["verified_completion"] == small.components["verified_completion"]


def test_gate_allow_or_worker_count_has_no_implicit_bonus() -> None:
    baseline = team_reward(
        TeamOutcome(verified_complete=False, policy_compliant=False, evidence_supported=False)
    )
    extra_workers = team_reward(
        TeamOutcome(
            verified_complete=False,
            policy_compliant=False,
            evidence_supported=False,
            worker_spawns=4,
        )
    )

    assert extra_workers.total < baseline.total
    assert "gate_allow" not in extra_workers.components


def test_unauthorized_side_effects_are_penalized_separately_from_attempts() -> None:
    blocked = team_reward(
        TeamOutcome(
            verified_complete=False,
            policy_compliant=False,
            evidence_supported=False,
            unsafe_attempts=1,
        )
    )
    escaped = team_reward(
        TeamOutcome(
            verified_complete=False,
            policy_compliant=False,
            evidence_supported=False,
            unsafe_attempts=1,
            unauthorized_side_effects=1,
        )
    )

    assert escaped.components["unsafe_attempts"] == blocked.components["unsafe_attempts"]
    assert escaped.components["unauthorized_side_effects"] == -8.0
    assert escaped.total < blocked.total


def test_recovery_is_visible_as_its_own_component() -> None:
    reward = team_reward(
        TeamOutcome(
            verified_complete=True,
            policy_compliant=True,
            evidence_supported=True,
            recovered_holds=1,
            recovered_failures=2,
        )
    )

    assert reward.components["recovered_holds"] == 0.5
    assert reward.components["recovered_failures"] == 1.0


def test_rejects_negative_counts_and_non_finite_weights() -> None:
    with pytest.raises(ValueError):
        TeamOutcome(False, False, False, worker_spawns=-1)
    with pytest.raises(ValueError):
        TeamRewardWeights(verified_completion=float("nan"))
