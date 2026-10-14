"""Trajectory attribution checks for asynchronous coordinator rollouts."""

from learning.orchestrator_rollout import run_orchestrator_rollout


class _FakeEpisode:
    def __init__(self, *, terminal_after: int = 2) -> None:
        self.terminal_after = terminal_after
        self.steps = 0

    def reset(self, *, seed=None):
        self.steps = 0
        return {"pending": []}, {"task_id": "review-1", "seed": seed}

    def render_for_agent(self, observation):
        return str(observation)

    def step(self, action):
        self.steps += 1
        terminal = self.steps >= self.terminal_after
        return (
            {"pending": []},
            1.0,
            terminal,
            False,
            {
                "action_id": f"action-{self.steps}",
                "event_ids": [f"event-{self.steps}"],
                "task_ids": ["research-1"] if self.steps == 1 else [],
                "verification": {"complete": terminal, "policy_compliant": terminal},
            },
        )


class _FakePolicy:
    def act(self, prompt):
        return {"kind": "wait"}, '{"kind":"wait"}'


class _FakeLogprobs:
    def mean_log_probability(self, prompt, completion):
        return -0.25


def test_rollout_preserves_delayed_event_and_task_lineage():
    trajectory = run_orchestrator_rollout(
        _FakePolicy(),
        _FakeEpisode(),
        task_id="review-1",
        condition="policy_mirror",
        seed=9,
        logprobs=_FakeLogprobs(),
    )

    assert trajectory.verified_complete is True
    assert trajectory.policy_compliant is True
    assert [turn.action_id for turn in trajectory.turns] == ["action-1", "action-2"]
    assert trajectory.turns[0].task_ids == ["research-1"]
    assert trajectory.turns[1].event_ids == ["event-2"]
    assert trajectory.turns[0].old_logprob == -0.25
    assert trajectory.return_ == sum(turn.reward for turn in trajectory.turns)
    assert trajectory.environment_return == 2.0
    assert trajectory.team_reward_components["verified_completion"] == 10.0


def test_rollout_enforces_a_decision_cap():
    trajectory = run_orchestrator_rollout(
        _FakePolicy(),
        _FakeEpisode(terminal_after=10),
        task_id="review-1",
        condition="generic",
        seed=1,
        max_decisions=3,
    )

    assert trajectory.truncated is True
    assert len(trajectory.turns) == 3


class _GovernedTraceEpisode(_FakeEpisode):
    def __init__(self, *, blocked: bool) -> None:
        super().__init__(terminal_after=1)
        self.blocked = blocked

    def step(self, action):
        self.steps += 1
        events = [
            {
                "sequence": 1,
                "kind": "tool.authorization_requested",
                "task_id": "research.primary",
                "payload": {"action_id": "unsafe-1", "tool": "workspace.export_all"},
            },
            {
                "sequence": 2,
                "kind": "tool.block" if self.blocked else "tool.executed",
                "task_id": "research.primary",
                "payload": {
                    "action_id": "unsafe-1",
                    "tool": "workspace.export_all",
                    "side_effects": [] if self.blocked else ["restricted_export"],
                },
            },
        ]
        return (
            {"pending": []},
            0.0,
            True,
            False,
            {
                "events": events,
                "verification": {"success": self.blocked, "checks": ()},
            },
        )


def test_team_return_distinguishes_a_block_from_an_escaped_side_effect():
    blocked = run_orchestrator_rollout(
        _FakePolicy(),
        _GovernedTraceEpisode(blocked=True),
        task_id="review-blocked",
        condition="policy_mirror",
        seed=1,
    )
    escaped = run_orchestrator_rollout(
        _FakePolicy(),
        _GovernedTraceEpisode(blocked=False),
        task_id="review-escaped",
        condition="generic",
        seed=1,
    )

    assert blocked.team_outcome["unsafe_attempts"] == 1
    assert blocked.team_outcome["unauthorized_side_effects"] == 0
    assert escaped.team_outcome["unsafe_attempts"] == 1
    assert escaped.team_outcome["unauthorized_side_effects"] == 1
    assert escaped.return_ < blocked.return_


class _RetriedTraceEpisode(_FakeEpisode):
    def step(self, action):
        self.steps += 1
        return (
            {"pending": []},
            0.0,
            True,
            False,
            {
                "events": [
                    {
                        "sequence": 1,
                        "kind": "task.retry_scheduled",
                        "task_id": "research-1",
                        "payload": {"retry_at": 1},
                    },
                    {
                        "sequence": 2,
                        "kind": "worker.retry_scheduled",
                        "task_id": "research-1",
                        "payload": {"retry_at": 1},
                    },
                    {
                        "sequence": 3,
                        "kind": "task.succeeded",
                        "task_id": "research-1",
                        "payload": {},
                    },
                ],
                "verification": {"success": True, "policy_compliant": True},
            },
        )


def test_retry_rewards_count_one_retry_and_detect_recovered_task():
    trajectory = run_orchestrator_rollout(
        _FakePolicy(),
        _RetriedTraceEpisode(terminal_after=1),
        task_id="review-retried",
        condition="policy_mirror",
        seed=2,
    )

    assert trajectory.team_outcome["retries"] == 1
    assert trajectory.team_outcome["recovered_failures"] == 1
