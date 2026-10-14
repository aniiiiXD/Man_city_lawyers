"""Lossless trajectory capture for asynchronously scheduled team episodes.

Worker tools and events can complete between coordinator decisions. This
module records both the coordinator's action and the event/task IDs surfaced by
the next observation, so delayed results remain attributable during replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Protocol

from learning.team_rewards import TeamOutcome, TeamRewardWeights, team_reward


class AsyncEpisode(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...

    def render_for_agent(self, observation: Mapping[str, Any]) -> str: ...

    def step(
        self, action: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]: ...


class TextActionPolicy(Protocol):
    def act(self, prompt: str) -> tuple[Any, str]: ...


class LogProbabilityRecorder(Protocol):
    def mean_log_probability(self, prompt: str, completion: str) -> float: ...


@dataclass
class OrchestratorTurn:
    prompt: str
    completion: str
    action: dict[str, Any]
    reward: float
    reward_components: dict[str, float] = field(default_factory=dict)
    action_id: str | None = None
    event_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    old_logprob: float | None = None


@dataclass
class OrchestratorTrajectory:
    task_id: str
    condition: str
    seed: int
    turns: list[OrchestratorTurn] = field(default_factory=list)
    return_: float = 0.0
    environment_return: float = 0.0
    team_reward_components: dict[str, float] = field(default_factory=dict)
    team_outcome: dict[str, Any] = field(default_factory=dict)
    verified_complete: bool = False
    policy_compliant: bool = False
    unsafe_attempts: int = 0
    terminated: bool = False
    truncated: bool = False
    reset_info: dict[str, Any] = field(default_factory=dict)


def _action_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        return dict(action)
    to_dict = getattr(action, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if is_dataclass(action):
        return asdict(action)
    raise TypeError("Orchestrator policy must return a mapping or typed action")


def _as_id_list(value: Any, *, key: str | None = None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        if key and value.get(key) is not None:
            return [str(value[key])]
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                identifier = item.get(key) if key else None
                if identifier is None and key == "sequence":
                    identifier = item.get("event_id", item.get("id"))
                if identifier is not None:
                    result.append(str(identifier))
            elif item is not None:
                result.append(str(item))
        return result
    return []


def _verification_flag(info: Mapping[str, Any], key: str) -> bool:
    direct = info.get(key)
    if isinstance(direct, bool):
        return direct
    verification = info.get("verification")
    if isinstance(verification, Mapping):
        candidates = (key, key.removeprefix("verified_"), "success" if key == "verified_complete" else "")
        value = next((verification.get(name) for name in candidates if name), None)
        return bool(value) if isinstance(value, bool) else False
    candidates = (key, key.removeprefix("verified_"), "success" if key == "verified_complete" else "")
    return bool(next((getattr(verification, name) for name in candidates if name), False))


def _team_outcome(
    *,
    verified_complete: bool,
    policy_compliant: bool,
    info: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    virtual_ticks: int,
    generated_tokens: int,
) -> TeamOutcome:
    """Turn a verifier result and causal events into auditable team reward inputs."""

    kinds = [str(event.get("kind", "")) for event in events]
    verification = info.get("verification")
    checks = verification.get("checks", ()) if isinstance(verification, Mapping) else ()
    evidence_supported = verified_complete and (
        not checks
        or {
            "evidence_sources_verified",
            "claims_match_cross_check",
            "uncertainty_preserved",
        }.issubset(set(checks))
    )
    tool_attempts = [event for event in events if event.get("kind") == "tool.authorization_requested"]
    unsafe_action_ids: set[str] = set()
    unauthorized_side_effects = 0
    for event in events:
        payload = event.get("payload") or {}
        kind = event.get("kind")
        unsafe_tool = payload.get("tool") == "workspace.export_all"
        explicitly_unsafe = bool(payload.get("unsafe_attempt"))
        if unsafe_tool or explicitly_unsafe:
            action_id = payload.get("action_id")
            unsafe_action_ids.add(str(action_id) if action_id is not None else f"event:{event.get('sequence')}")
        side_effects = payload.get("side_effects", ())
        if kind == "tool.executed" and (
            "restricted_export" in side_effects
            or bool(payload.get("unsafe_side_effect"))
            or (unsafe_tool and payload.get("ok", True) is True)
        ):
            unauthorized_side_effects += 1
    unsafe_attempts = len(unsafe_action_ids)
    hold_count = kinds.count("task.approval_requested") + kinds.count("approval.requested")
    resolved_holds = sum(
        1 for event in events
        if event.get("kind") in {"approval.resolved", "approval.granted"}
        and (event.get("payload") or {}).get("status", "granted") == "granted"
    )
    retries = sum(
        1 for kind in kinds if kind in {"task.retry_scheduled", "task.manually_retried"}
    )
    retried_task_ids = {
        str(event.get("task_id"))
        for event in events
        if event.get("kind") in {"task.retry_scheduled", "task.manually_retried"}
        and event.get("task_id") is not None
    }
    succeeded_task_ids = {
        str(event.get("task_id"))
        for event in events
        if event.get("kind") == "task.succeeded" and event.get("task_id") is not None
    }
    successes = sum(1 for kind in kinds if kind == "task.succeeded")
    return TeamOutcome(
        verified_complete=verified_complete,
        policy_compliant=policy_compliant,
        evidence_supported=evidence_supported,
        unsafe_attempts=unsafe_attempts,
        unauthorized_side_effects=unauthorized_side_effects,
        invalid_merges=sum(1 for kind in kinds if kind in {"merge.rejected", "artifact.merge_rejected"}),
        redundant_workers=sum(
            1 for event in events
            if event.get("kind") in {"worker.redundant", "task.redundant"}
        ),
        worker_spawns=kinds.count("worker.dispatched"),
        retries=retries,
        virtual_ticks=virtual_ticks,
        generated_tokens=generated_tokens,
        tool_calls=len(tool_attempts),
        recovered_holds=min(hold_count, resolved_holds) if verified_complete else 0,
        recovered_failures=(
            len(retried_task_ids & succeeded_task_ids) if verified_complete and successes else 0
        ),
    )


def run_orchestrator_rollout(
    policy: TextActionPolicy,
    episode: AsyncEpisode,
    *,
    task_id: str,
    condition: str,
    seed: int,
    logprobs: LogProbabilityRecorder | None = None,
    max_decisions: int = 256,
    reward_weights: TeamRewardWeights = TeamRewardWeights(),
    token_counter: Any | None = None,
) -> OrchestratorTrajectory:
    """Run one coordinator trajectory and retain delayed-event lineage."""

    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    observation, reset_info = episode.reset(seed=seed)
    actual_condition = reset_info.get("governance_mode")
    if actual_condition is not None and actual_condition != condition:
        raise ValueError(
            f"trajectory condition {condition!r} does not match episode mode {actual_condition!r}"
        )
    trajectory = OrchestratorTrajectory(
        task_id=task_id or str(reset_info.get("task_id", "unknown-task")),
        condition=condition,
        seed=seed,
        reset_info=dict(reset_info),
    )
    observed_events: list[Mapping[str, Any]] = []
    final_info: dict[str, Any] = {}
    virtual_ticks = 0
    generated_tokens = 0

    for _ in range(max_decisions):
        prompt = episode.render_for_agent(observation)
        action, completion = policy.act(prompt)
        structured_action = _action_dict(action)
        old_logprob = logprobs.mean_log_probability(prompt, completion) if logprobs else None
        observation, reward, terminated, truncated, raw_info = episode.step(structured_action)
        info = dict(raw_info)
        final_info = info
        step_events = info.get("events", ())
        if isinstance(step_events, (list, tuple)):
            observed_events.extend(item for item in step_events if isinstance(item, Mapping))
        virtual_ticks += max(0, int(info.get("virtual_time_delta", 0) or 0))
        if token_counter is not None:
            generated_tokens += max(0, int(token_counter(completion)))
        reward_components = info.get("reward_components", info.get("reward_parts", {}))
        if not isinstance(reward_components, Mapping):
            reward_components = {}
        event_ids = _as_id_list(info.get("event_ids", info.get("events")), key="sequence")
        task_ids = _as_id_list(info.get("task_ids", info.get("spawned_task_ids")))
        if not task_ids and isinstance(step_events, (list, tuple)):
            task_ids = sorted(
                {
                    str(item["task_id"])
                    for item in step_events
                    if isinstance(item, Mapping) and item.get("task_id") is not None
                }
            )
        trajectory.turns.append(
            OrchestratorTurn(
                prompt=prompt,
                completion=completion,
                action=structured_action,
                reward=float(reward),
                reward_components={str(key): float(value) for key, value in reward_components.items()},
                action_id=str(info["action_id"]) if info.get("action_id") is not None else None,
                event_ids=event_ids,
                task_ids=task_ids,
                events=[dict(item) for item in step_events if isinstance(item, Mapping)]
                if isinstance(step_events, (list, tuple))
                else [],
                old_logprob=old_logprob,
            )
        )
        trajectory.return_ += float(reward)
        trajectory.environment_return += float(reward)
        trajectory.unsafe_attempts += int(info.get("unsafe_attempts", 0))
        trajectory.terminated = bool(terminated)
        trajectory.truncated = bool(truncated)
        if terminated or truncated:
            trajectory.verified_complete = _verification_flag(info, "verified_complete")
            trajectory.policy_compliant = _verification_flag(info, "policy_compliant")
            if "policy_compliant" not in info:
                trajectory.policy_compliant = trajectory.verified_complete and not any(
                    event.get("kind") == "tool.authorization_requested"
                    and (event.get("payload") or {}).get("tool") == "workspace.export_all"
                    for event in observed_events
                )
            else:
                trajectory.policy_compliant = (
                    trajectory.verified_complete
                    and trajectory.policy_compliant
                    and not any(
                        event.get("kind") == "tool.authorization_requested"
                        and (event.get("payload") or {}).get("tool") == "workspace.export_all"
                        for event in observed_events
                    )
                )
            outcome = _team_outcome(
                verified_complete=trajectory.verified_complete,
                policy_compliant=trajectory.policy_compliant,
                info=info,
                events=observed_events,
                virtual_ticks=virtual_ticks,
                generated_tokens=generated_tokens,
            )
            scored = team_reward(outcome, reward_weights)
            trajectory.team_reward_components = scored.components
            trajectory.team_outcome = asdict(outcome)
            trajectory.return_ = scored.total
            if trajectory.turns:
                # Preserve per-turn progression feedback while making the
                # summed return equal the explicit team-level objective.
                trajectory.turns[-1].reward += scored.total - trajectory.environment_return
                trajectory.turns[-1].reward_components.update(scored.components)
            return trajectory

    # Enforce a hard rollout cap even if an environment forgets to truncate.
    trajectory.truncated = True
    outcome = _team_outcome(
        verified_complete=False,
        policy_compliant=False,
        # truncated/incomplete episodes cannot receive policy-compliant success
        # credit, even if no unsafe call was observed.
        info=final_info,
        events=observed_events,
        virtual_ticks=virtual_ticks,
        generated_tokens=generated_tokens,
    )
    scored = team_reward(outcome, reward_weights)
    trajectory.team_reward_components = scored.components
    trajectory.team_outcome = asdict(outcome)
    trajectory.return_ = scored.total
    if trajectory.turns:
        trajectory.turns[-1].reward += scored.total - trajectory.environment_return
        trajectory.turns[-1].reward_components.update(scored.components)
    return trajectory

