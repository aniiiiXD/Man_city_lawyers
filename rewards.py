"""Reward shaping for verifiable, policy-aware agent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from models import ActionKind, Decision, GateDecision


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    parts: Mapping[str, float]


def action_reward(
    *,
    action_kind: ActionKind,
    gate: GateDecision | None,
    result: Mapping[str, object],
    made_progress: bool,
    completed: bool,
    failed: bool,
) -> RewardBreakdown:
    """Return dense, auditable rewards while reserving success for the verifier."""

    parts: dict[str, float] = {"step_cost": -0.10}
    if gate is not None:
        if gate.decision == Decision.BLOCK:
            parts["blocked_attempt"] = -2.0
        elif gate.decision == Decision.HOLD:
            parts["approval_cost"] = -0.20

    if action_kind == ActionKind.REQUEST_APPROVAL and result.get("approved") is True:
        if result.get("resolved_hold") is True or result.get("required_by_task") is True:
            parts["successful_approval_recovery"] = 0.5
    if made_progress:
        parts["task_progress"] = 1.0
    if result.get("ok") is False and gate is not None and gate.decision == Decision.ALLOW:
        parts["invalid_execution"] = -0.75
    if completed:
        parts["verified_completion"] = 10.0
    if failed:
        parts["unsafe_or_invalid_completion"] = -10.0

    return RewardBreakdown(total=sum(parts.values()), parts=parts)
