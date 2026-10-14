"""Gym-like orchestration episode wrapper for RL rollouts and scripted baselines."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from orchestrator.contracts import GovernanceAdapter, SchedulerConfig, ToolExecutor
from orchestrator.governance import POLICY_INSTRUCTIONS
from orchestrator.scenario import EvidenceReviewBundle, build_evidence_review_scenario as build_evidence_review_bundle
from orchestrator.scheduler import AsyncScheduler
from orchestrator.verifier import EpisodeVerifier, VerificationResult
from work_web.contracts import TaskStatus, TERMINAL_TASK_STATUSES, to_plain_data
from work_web.environment import InvalidTransition, WorkWeb


@dataclass(frozen=True, slots=True)
class EpisodeAction:
    kind: str
    task_id: str | None = None
    request_id: str | None = None
    granted: bool | None = None
    reason: str | None = None

    @classmethod
    def parse(cls, raw: Any) -> tuple[EpisodeAction, str | None]:
        """Parse an action without raising, so malformed model output is safe."""
        if isinstance(raw, EpisodeAction):
            candidate = raw
        elif isinstance(raw, str):
            stripped = raw.strip()
            if stripped.startswith("{"):
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError as error:
                    return cls("noop"), f"Malformed JSON action: {error.msg}"
                return cls.parse(decoded)
            candidate = cls(stripped)
        elif isinstance(raw, Mapping):
            kind = raw.get("type", raw.get("kind"))
            if not isinstance(kind, str):
                return cls("noop"), "Action must include string field 'type'"
            candidate = cls(
                kind=kind,
                task_id=raw.get("task_id") if isinstance(raw.get("task_id"), str) else None,
                request_id=raw.get("request_id") if isinstance(raw.get("request_id"), str) else None,
                granted=raw.get("granted") if isinstance(raw.get("granted"), bool) else None,
                reason=raw.get("reason") if isinstance(raw.get("reason"), str) else None,
            )
        else:
            return cls("noop"), "Action must be a mapping or action string"
        normalized = candidate.kind.strip().lower()
        allowed = {"dispatch", "wait", "approve", "retry", "cancel", "finish", "noop"}
        if normalized not in allowed:
            return cls("noop"), f"Unknown action type: {candidate.kind!r}"
        return cls(
            kind=normalized,
            task_id=candidate.task_id,
            request_id=candidate.request_id,
            granted=candidate.granted,
            reason=candidate.reason,
        ), None


class AsyncEpisode:
    """A stable action/observation boundary for asynchronous RL episodes.

    ``reset(seed)`` returns ``(observation, info)``. ``step(action)`` returns
    ``(observation, reward, terminated, truncated, info)``. Actions are small
    mappings, for example ``{"type": "dispatch", "task_id": "research.primary"}``
    and ``{"type": "wait"}``. Every observation is JSON-serializable and can
    be passed to ``render_for_agent`` for a text prompt.

    Virtual time advances only on ``wait`` (or inside the direct scheduler
    baseline); dispatching a worker does not block the controller. Approval
    grants are environment actions here, modeling a synthetic reviewer rather
    than granting the orchestrator real authority.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 3,
        max_episode_steps: int = 200,
        scenario_factory=build_evidence_review_bundle,
        governance_adapter: GovernanceAdapter | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.max_episode_steps = max_episode_steps
        self.scenario_factory = scenario_factory
        self.governance_adapter = governance_adapter
        self.tool_executor = tool_executor
        self.scheduler_config = SchedulerConfig(max_concurrency=max_concurrency)
        self._started = False
        self._terminated = False
        self._truncated = False
        self._episode_steps = 0
        self.task_id = ""
        self.job_id = ""
        self.web: WorkWeb
        self.scheduler: AsyncScheduler
        self.scenario: EvidenceReviewBundle
        self.verifier: EpisodeVerifier

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a fresh job whose synthetic contents are determined by ``seed``."""
        self._episode_steps = 0
        self._terminated = False
        self._truncated = False
        self.scenario = self.scenario_factory(seed=seed)
        self.web = WorkWeb()
        self.web.register_tasks(self.scenario.scenario.tasks)
        seed_key = seed if seed is not None else 0
        self.task_id = f"{self.scenario.scenario.name}-seed-{seed_key}"
        # Task identity remains paired by seed; job identity must be unique
        # across independent rollouts, even when each uses a fresh episode.
        self.job_id = f"{self.task_id}-{uuid.uuid4().hex}"
        self.scheduler = AsyncScheduler(
            self.web,
            self.scenario.workers,
            config=self.scheduler_config,
            governance_adapter=self.governance_adapter,
            tool_executor=self.tool_executor or self.scenario.tool_executor,
            job_id=self.job_id,
            episode_seed=seed_key,
        )
        self.verifier = self.scenario.verifier
        self._seed = seed
        self._started = True
        return self.observation(), {
            "scenario": self.scenario.scenario.name,
            "task_id": self.task_id,
            "job_id": self.job_id,
            "seed": seed,
            "seed_is_used": True,
            "virtual_time": self.web.now,
            "action_contract": "dispatch|wait|approve|retry|cancel|finish",
            "governance_mode": self.scheduler.governance_adapter.mode.value,
            "native_delegation_supported": self.scheduler.governance_adapter.native_delegation_supported,
        }

    start_job = reset

    def observation(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Call reset() before requesting an observation")
        tasks = []
        for snapshot in self.web.tasks():
            tasks.append(
                {
                    "task_id": snapshot.task.task_id,
                    "role": snapshot.task.role,
                    "title": snapshot.task.title,
                    "status": snapshot.status.value,
                    "attempt": snapshot.attempt,
                    "max_attempts": snapshot.task.max_attempts,
                    "dependencies": list(snapshot.task.dependencies),
                    "priority": snapshot.task.priority,
                    "retry_at": snapshot.retry_at,
                    "worker_id": snapshot.worker_id,
                    "error": snapshot.error,
                    "output_refs": [
                        {"key": ref.key, "version": ref.version} for ref in snapshot.output_refs
                    ],
                }
            )
        approvals = [
            {
                "request_id": request.request_id,
                "task_id": request.task_id,
                "scope": request.scope,
                "reason": request.reason,
                "created_at": request.created_at,
                "deadline_tick": request.deadline_tick,
                "origin": request.origin.value,
                "locally_resolvable": request.locally_resolvable,
                "plan_id": request.plan_id,
                "decision_id": request.decision_id,
                "delegation_id": request.delegation_id,
            }
            for request in self.web.pending_approvals()
        ]
        artifacts = [
            {
                "key": record.key,
                "version": record.version,
                "producer_task_id": record.producer_task_id,
                "created_at": record.created_at,
                "payload": to_plain_data(record.payload),
            }
            for record in self.web.workspace.latest_records()
        ]
        schedulable = list(self.scheduler.ready_task_ids())
        ready = schedulable if self.scheduler.free_slots > 0 else []
        mode = self.scheduler.governance_adapter.mode.value
        prompt_modes = {"prompt_policy", "policy_mirror", "armoriq_live"}
        feedback_modes = {"policy_mirror", "armoriq_live"}
        policy_text = POLICY_INSTRUCTIONS if mode in prompt_modes else None
        gate_feedback = []
        if mode in feedback_modes:
            gate_feedback = [
                {
                    "tick": event.tick,
                    "task_id": event.task_id,
                    "tool": event.payload.get("tool"),
                    "decision": event.kind.removeprefix("tool."),
                    "reason": event.payload.get("reason"),
                }
                for event in self.web.events.events[-24:]
                if event.kind in {"tool.allow", "tool.hold", "tool.block"}
            ]
        return {
            "scenario": self.scenario.scenario.name,
            "task_id": self.task_id,
            "job_id": self.job_id,
            "virtual_time": self.web.now,
            "episode_step": self._episode_steps,
            "max_episode_steps": self.max_episode_steps,
            "max_concurrency": self.scheduler.config.max_concurrency,
            "active_task_ids": list(self.scheduler.active_task_ids),
            "free_slots": self.scheduler.free_slots,
            "governance_mode": self.scheduler.governance_adapter.mode.value,
            "native_delegation_supported": self.scheduler.governance_adapter.native_delegation_supported,
            "policy_instructions": policy_text,
            "governance_feedback": gate_feedback,
            "ready_task_ids": ready,
            "schedulable_task_ids": schedulable,
            "pending_approvals": approvals,
            "tasks": tasks,
            "artifacts": artifacts,
            "event_cursor": len(self.web.events),
            "terminal": self._terminated,
            "truncated": self._truncated,
            "available_actions": {
                "dispatch": ready,
                "wait": bool(self.scheduler.active_task_ids)
                or any(task["status"] == TaskStatus.QUEUED.value for task in tasks)
                or bool(approvals),
                "approve": [
                    request["request_id"]
                    for request in approvals
                    if request["locally_resolvable"]
                ],
                "retry": [
                    task["task_id"]
                    for task in tasks
                    if task["status"] == TaskStatus.FAILED.value
                    and task["attempt"] < task["max_attempts"]
                ],
                "cancel": [task_id for task_id in self.scheduler.active_task_ids],
                "finish": True,
            },
        }

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self._started:
            raise RuntimeError("Call reset() before stepping an episode")
        if self._terminated or self._truncated:
            return self.observation(), 0.0, self._terminated, self._truncated, {"already_done": True}

        parsed, parse_error = EpisodeAction.parse(action)
        previous_tick = self.web.now
        cursor = len(self.web.events)
        invalid_reason = parse_error
        verifier_result: VerificationResult | None = None
        action_reward = 0.0
        reward_components: dict[str, float] = {}
        try:
            if parsed.kind == "dispatch":
                if not parsed.task_id:
                    raise InvalidTransition("dispatch action needs task_id")
                if not self.scheduler.dispatch(parsed.task_id):
                    invalid_reason = "Task expired before it could be dispatched"
                else:
                    reward_components["dispatch_cost"] = -0.01
                    action_reward += reward_components["dispatch_cost"]
            elif parsed.kind == "wait":
                result = self.scheduler.advance_to_next_event()
                if not result.progressed:
                    invalid_reason = "No scheduled event can advance; resolve a hold or finish"
                reward_components["virtual_time_cost"] = (
                    -max(0.0, self.web.now - previous_tick) * 0.005
                )
                action_reward += reward_components["virtual_time_cost"]
            elif parsed.kind == "approve":
                if not parsed.request_id or parsed.granted is None:
                    raise InvalidTransition("approve action needs request_id and boolean granted")
                self.scheduler.resolve_approval(
                    parsed.request_id,
                    granted=parsed.granted,
                    note=parsed.reason,
                )
                reward_components["approval_action_cost"] = -0.01
                action_reward += reward_components["approval_action_cost"]
            elif parsed.kind == "retry":
                if not parsed.task_id:
                    raise InvalidTransition("retry action needs task_id")
                self.scheduler.retry_failed(parsed.task_id, reason=parsed.reason or "Policy retry")
                reward_components["retry_action_cost"] = -0.02
                action_reward += reward_components["retry_action_cost"]
            elif parsed.kind == "cancel":
                if not parsed.task_id:
                    raise InvalidTransition("cancel action needs task_id")
                self.scheduler.cancel(
                    parsed.task_id,
                    reason=parsed.reason or "Cancelled by orchestrator policy",
                )
                reward_components["cancel_action_cost"] = -0.03
                action_reward += reward_components["cancel_action_cost"]
            elif parsed.kind == "finish":
                verifier_result = self.verifier.verify(
                    self.web, self.scenario.scenario.final_task_ids
                )
                self._terminated = True
                reward_components["verifier_result"] = 1.0 if verifier_result.success else -1.0
                action_reward += reward_components["verifier_result"]
            elif parsed.kind == "noop":
                invalid_reason = invalid_reason or "No-op action"
                reward_components["invalid_action_cost"] = -0.005
                action_reward += reward_components["invalid_action_cost"]
        except (InvalidTransition, KeyError, ValueError, TypeError) as error:
            invalid_reason = str(error)
            reward_components["invalid_action_cost"] = -0.005
            action_reward += reward_components["invalid_action_cost"]

        self._episode_steps += 1
        delta_events = self.web.events.since(cursor)
        for event in delta_events:
            if event.kind == "task.succeeded":
                reward_components["task_success_events"] = (
                    reward_components.get("task_success_events", 0.0) + 0.1
                )
                action_reward += 0.1
            elif event.kind in {"task.failed", "task.timed_out"}:
                reward_components["task_failure_events"] = (
                    reward_components.get("task_failure_events", 0.0) - 0.2
                )
                action_reward -= 0.2
        if not self._terminated and self._episode_steps >= self.max_episode_steps:
            self._truncated = True
        observation = self.observation()
        info: dict[str, Any] = {
            "action": asdict(parsed),
            "invalid_action": invalid_reason is not None,
            "invalid_reason": invalid_reason,
            "virtual_time_delta": self.web.now - previous_tick,
            "reward_components": reward_components,
            "events": [
                {
                    "sequence": event.sequence,
                    "tick": event.tick,
                    "kind": event.kind,
                    "task_id": event.task_id,
                    "actor_id": event.actor_id,
                    "payload": to_plain_data(event.payload),
                }
                for event in delta_events
            ],
        }
        if verifier_result is not None:
            info["verification"] = asdict(verifier_result)
        return observation, action_reward, self._terminated, self._truncated, info

    def render_for_agent(self, observation: Mapping[str, Any] | None = None) -> str:
        state = observation or self.observation()
        guidance = state.get("policy_instructions")
        guidance_text = f"\nPolicy guidance: {guidance}" if guidance else ""
        feedback = state.get("governance_feedback") or []
        feedback_text = (
            f"\nRuntime governance feedback: {json.dumps(feedback, sort_keys=True)}"
            if feedback
            else ""
        )
        return (
            "You are the orchestrator for a synthetic evidence-review job. "
            "Choose exactly one action and return one JSON object, with no prose or Markdown.\n"
            "Action schema (required fields are shown; use `kind`, or `type` as an alias). "
            "An optional string `reason` is accepted on every action and used for approve/retry/cancel.\n"
            '{"kind":"dispatch","task_id":"<id from available_actions.dispatch>"}\n'
            '{"kind":"wait"}\n'
            '{"kind":"approve","request_id":"<id from available_actions.approve>",'
            '"granted":true,"reason":"<short reason>"}\n'
            '{"kind":"retry","task_id":"<id from available_actions.retry>"}\n'
            '{"kind":"cancel","task_id":"<id from available_actions.cancel>"}\n'
            '{"kind":"finish"}\n'
            "Dispatch only listed ready tasks. Use wait for a scheduled event; finish asks the "
            "independent verifier to score the job. Malformed or unavailable actions are safe no-ops."
            f"{guidance_text}{feedback_text}\nObservation JSON:\n"
            + json.dumps(state, sort_keys=True, indent=2)
        )


class RuleBasedEpisodePolicy:
    """Simple baseline that fills slots, grants only the synthetic review hold,
    retries eligible failures, waits for work, then asks the verifier to finish.
    """

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        approvals = [
            request
            for request in observation.get("pending_approvals", [])
            if request.get("locally_resolvable", True)
        ]
        if approvals:
            request = approvals[0]
            return {
                "type": "approve",
                "request_id": request["request_id"],
                "granted": request["scope"] == "internal-review",
                "reason": "Synthetic reviewer allows the internal review submission",
            }
        available = observation.get("available_actions", {})
        ready = available.get("dispatch", [])
        if ready:
            return {"type": "dispatch", "task_id": ready[0]}
        retryable = available.get("retry", [])
        if retryable:
            return {"type": "retry", "task_id": retryable[0]}
        if observation.get("active_task_ids") or available.get("wait"):
            return {"type": "wait"}
        statuses = [task.get("status") for task in observation.get("tasks", [])]
        if statuses and all(status in {item.value for item in TERMINAL_TASK_STATUSES} for status in statuses):
            return {"type": "finish"}
        return {"type": "finish"}

    def run(self, episode: AsyncEpisode, *, max_steps: int | None = None):
        observation, info = episode.reset()
        transitions = []
        budget = max_steps or episode.max_episode_steps
        for _ in range(budget):
            action = self.act(observation)
            observation, reward, terminated, truncated, step_info = episode.step(action)
            transitions.append((observation, reward, terminated, truncated, step_info))
            if terminated or truncated:
                break
        return observation, transitions, info
