"""RL adapter and end-to-end synthetic scenario source checks."""

import pytest

from orchestrator.episode import RuleBasedEpisodePolicy
from orchestrator.contracts import GovernanceMode
from orchestrator.governance import CallbackGovernanceAdapter
from work_web.scenarios import (
    build_armoriq_live_episode,
    build_evidence_review_scenario,
    build_generic_episode,
    build_policy_mirror_episode,
    build_prompt_policy_episode,
)


def _run_policy(episode):
    observation, reset_info = episode.reset(seed=41)
    policy = RuleBasedEpisodePolicy()
    transitions = []
    for _ in range(episode.max_episode_steps):
        observation, reward, terminated, truncated, info = episode.step(policy.act(observation))
        transitions.append((observation, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return observation, reset_info, transitions


def test_episode_reset_is_seeded_and_reports_a_stable_task_id() -> None:
    episode = build_evidence_review_scenario()
    first_observation, first_info = episode.reset(seed=41)
    first_claim = episode.scenario.tool_executor.sources["launch_log"]["claim"]
    second_observation, second_info = episode.reset(seed=41)
    different_observation, different_info = episode.reset(seed=42)
    _, independent_info = build_evidence_review_scenario().reset(seed=41)
    different_claim = episode.scenario.tool_executor.sources["launch_log"]["claim"]

    assert first_info["task_id"] == second_info["task_id"]
    assert first_info["job_id"] != second_info["job_id"]
    assert first_info["job_id"] != independent_info["job_id"]
    assert first_info["task_id"] != different_info["task_id"]
    assert first_info["seed_is_used"] is True
    assert first_claim != different_claim
    assert first_observation["task_id"] == first_info["task_id"]
    assert different_observation["scenario"] == first_observation["scenario"]


def test_malformed_action_is_a_safe_ordinary_transition() -> None:
    episode = build_policy_mirror_episode()
    observation, _ = episode.reset(seed=3)

    observation, reward, terminated, truncated, info = episode.step({"type": "not-a-real-action"})

    assert info["invalid_action"] is True
    assert info["invalid_reason"]
    assert observation["episode_step"] == 1
    assert reward < 0
    assert terminated is False
    assert truncated is False


def test_policy_text_and_gate_feedback_match_experimental_condition() -> None:
    generic = build_generic_episode()
    generic_observation, _ = generic.reset(seed=2)
    prompt_only = build_prompt_policy_episode()
    prompt_observation, _ = prompt_only.reset(seed=2)
    mirror = build_policy_mirror_episode()
    mirror_observation, _ = mirror.reset(seed=2)

    assert generic_observation["policy_instructions"] is None
    assert generic_observation["governance_feedback"] == []
    assert prompt_observation["policy_instructions"]
    assert prompt_observation["policy_instructions"] == mirror_observation["policy_instructions"]
    assert prompt_observation["governance_feedback"] == []
    assert mirror_observation["governance_feedback"] == []

    mirror_observation, _, _, _, _ = mirror.step(
        {"type": "dispatch", "task_id": "research.primary"}
    )
    assert any(
        item["decision"] == "block" and item["tool"] == "workspace.export_all"
        for item in mirror_observation["governance_feedback"]
    )


def test_rendered_prompt_has_parser_compatible_json_actions() -> None:
    episode = build_prompt_policy_episode()
    observation, _ = episode.reset(seed=12)
    prompt = episode.render_for_agent(observation)

    assert '{"kind":"dispatch","task_id":"<id from available_actions.dispatch>"}' in prompt
    assert '{"kind":"wait"}' in prompt
    assert '{"kind":"approve","request_id":"<id from available_actions.approve>"' in prompt
    assert '{"kind":"retry","task_id":"<id from available_actions.retry>"}' in prompt
    assert '{"kind":"cancel","task_id":"<id from available_actions.cancel>"}' in prompt
    assert '{"kind":"finish"}' in prompt
    assert observation["policy_instructions"] in prompt
    assert "Policy guidance:" in prompt


def test_policy_mirror_blocks_unsafe_export_and_episode_verifies() -> None:
    episode = build_policy_mirror_episode()
    observation, reset_info, transitions = _run_policy(episode)
    all_events = [event for *_, info in transitions for event in info["events"]]
    block_events = [
        event
        for event in all_events
        if event["kind"] == "tool.block"
        and event["payload"].get("tool") == "workspace.export_all"
    ]
    unsafe_executions = [
        event
        for event in all_events
        if event["kind"] == "tool.executed"
        and "restricted_export" in event["payload"].get("side_effects", [])
    ]

    assert reset_info["task_id"].endswith("seed-41")
    assert block_events
    assert not unsafe_executions
    assert episode.scenario.tool_executor.side_effects
    assert all(item["kind"] != "restricted_export" for item in episode.scenario.tool_executor.side_effects)
    assert transitions[-1][4]["verification"]["success"] is True
    assert observation["terminal"] is True


def test_unrestricted_conditions_expose_restricted_export_side_effect() -> None:
    for factory in (build_generic_episode, build_prompt_policy_episode):
        episode = factory()
        _, _, transitions = _run_policy(episode)
        all_events = [event for *_, info in transitions for event in info["events"]]
        assert any(
            event["kind"] == "tool.executed"
            and "restricted_export" in event["payload"].get("side_effects", [])
            for event in all_events
        )
        assert transitions[-1][4]["verification"]["success"] is False


def test_approval_hold_does_not_stop_independent_timeline_work() -> None:
    episode = build_policy_mirror_episode(max_concurrency=3)
    observation, _ = episode.reset(seed=8)
    policy = RuleBasedEpisodePolicy()

    for _ in range(100):
        if observation["pending_approvals"]:
            break
        observation, _, terminated, truncated, _ = episode.step(policy.act(observation))
        assert not terminated and not truncated

    assert observation["pending_approvals"]
    assert any(task["task_id"] == "archive.timeline" and task["status"] == "running" for task in observation["tasks"])
    still_held_request = observation["pending_approvals"][0]["request_id"]
    observation, _, _, _, _ = episode.step({"type": "wait"})

    archive = next(task for task in observation["tasks"] if task["task_id"] == "archive.timeline")
    assert archive["status"] == "succeeded"
    assert observation["pending_approvals"][0]["request_id"] == still_held_request


def test_live_callback_failure_blocks_before_tool_execution() -> None:
    def unavailable_live_gate(request):
        raise RuntimeError("SDK session unavailable")

    adapter = CallbackGovernanceAdapter(
        unavailable_live_gate,
        mode=GovernanceMode.ARMIQ_LIVE,
    )
    episode = build_armoriq_live_episode(governance_adapter=adapter)
    _, reset_info = episode.reset(seed=5)
    observation, _, _, _, info = episode.step(
        {"type": "dispatch", "task_id": "research.primary"}
    )

    assert reset_info["governance_mode"] == "armoriq_live"
    assert reset_info["native_delegation_supported"] is False
    assert observation["policy_instructions"]
    assert any(item["decision"] == "block" for item in observation["governance_feedback"])
    assert any(event["kind"] == "tool.block" for event in info["events"])
    assert not any(event["kind"] == "tool.executed" for event in info["events"])


def test_condition_factory_rejects_a_mislabeled_governance_adapter() -> None:
    adapter = CallbackGovernanceAdapter(
        lambda request: None,
        mode=GovernanceMode.ARMIQ_LIVE,
    )

    with pytest.raises(ValueError, match="cannot use an adapter labeled armoriq_live"):
        build_generic_episode(governance_adapter=adapter)
