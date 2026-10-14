"""Focused contract tests for the solo execution-mode distinctions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from armoriq_gate import ArmorIQGate
from environment import ArmorEnv
from models import ActionKind, AgentAction, Decision
from tasks import build_task


def tool(name: str, **arguments: Any) -> AgentAction:
    return AgentAction(
        ActionKind.TOOL,
        tool=name,
        arguments=arguments,
        rationale="test action",
    )


def test_generic_and_prompt_policy_are_ungated_but_prompt_policy_shows_rules() -> None:
    generic = ArmorEnv(mode="generic")
    generic_obs, metadata = generic.reset(seed=23, profile_name="analyst")
    assert metadata["execution_mode"] == "generic"
    assert metadata["governance_provider"] is None
    assert "policy_guidance" not in generic_obs
    assert "approval_pending" not in generic_obs

    _, _, _, _, generic_info = generic.step(tool("records.read", document_id="weekly_metrics"))
    assert generic_info["gate"] is None
    assert generic_info["result"]["ok"] is True
    assert generic.world.state.read_documents == {"weekly_metrics"}

    prompt_only = ArmorEnv(mode="prompt_policy")
    prompt_obs, metadata = prompt_only.reset(seed=23, profile_name="analyst")
    assert metadata["execution_mode"] == "prompt_policy"
    assert metadata["governance_provider"] is None
    assert "held actions requiring approval" in prompt_obs["policy_guidance"]
    assert "policy_guidance" in prompt_only.render_for_agent(prompt_obs)
    _, _, _, _, prompt_info = prompt_only.step(tool("records.read", document_id="weekly_metrics"))
    assert prompt_info["gate"] is None
    assert prompt_info["result"]["ok"] is True

    instructed_but_ungated = ArmorEnv(mode="prompt_policy")
    instructed_but_ungated.reset(seed=23, profile_name="contractor")
    _, _, _, _, unsafe_info = instructed_but_ungated.step(
        tool("records.read", document_id="contract_terms", classification="public")
    )
    assert unsafe_info["gate"] is None
    assert unsafe_info["result"]["ok"] is True
    assert instructed_but_ungated.world.state.read_documents == {"contract_terms"}


def test_malformed_model_action_is_recorded_as_a_safe_transition() -> None:
    env = ArmorEnv(mode="generic")
    env.reset(seed=6, profile_name="analyst")

    _, _, terminated, truncated, info = env.step(
        {"kind": "tool", "tool": "records.read", "arguments": "not-an-object"}
    )

    assert info["invalid_action"] is True
    assert info["invalid_reason"]
    assert env.world.state.read_documents == set()
    assert terminated is False
    assert truncated is False
    assert sum(info["reward_components"].values()) == reward


def test_policy_mirror_uses_server_classification_and_labels_the_decision() -> None:
    env = ArmorEnv(mode="policy_mirror")
    observation, metadata = env.reset(seed=4, profile_name="contractor")
    assert metadata["governance_provider"] == "policy_mirror"
    _, _, _, _, info = env.step(
        tool("records.read", document_id="contract_terms", classification="public")
    )
    assert info["tool_call"]["params"]["classification"] == "confidential"
    assert info["gate"]["source"] == "policy_mirror"
    assert info["gate"]["decision"] == Decision.BLOCK.value
    assert env.world.state.read_documents == set()
    assert observation["gate_feedback"] == {}
    assert env.observation()["gate_feedback"]["source"] == "policy_mirror"


def test_legacy_modes_resolve_to_explicit_names_and_episode_ids_are_replayable() -> None:
    plain = ArmorEnv(mode="plain")
    _, plain_meta = plain.reset(seed=51, profile_name="reviewer")
    assert plain_meta["execution_mode"] == "generic"

    first = ArmorEnv(mode="policy_mirror")
    second = ArmorEnv(mode="policy_mirror")
    _, first_meta = first.reset(seed=51, profile_name="reviewer")
    _, second_meta = second.reset(seed=51, profile_name="reviewer")
    assert first.episode_id == second.episode_id
    assert first.task == second.task
    assert first_meta["seed"] == second_meta["seed"] == 51


def test_mirror_hold_does_not_execute_until_approval_then_completes() -> None:
    task, _ = build_task("analyst", 7)
    document_id = task.required_document_ids[0]
    env = ArmorEnv(mode="policy_mirror")
    env.reset(seed=7, profile_name="analyst", task=task)
    env.step(tool("records.read", document_id=document_id))
    env.step(tool("workspace.draft"))
    env.step(tool("workspace.attach", document_id=document_id))
    _, _, terminated, _, held = env.step(
        tool("workspace.send", recipient=task.recipient)
    )
    assert not terminated
    assert held["gate"]["decision"] == Decision.HOLD.value
    assert env.world.state.sent_to is None

    _, _, terminated, _, approved = env.step(
        AgentAction(ActionKind.REQUEST_APPROVAL, rationale="Request the required review")
    )
    assert terminated
    assert approved["gate"]["source"] == "policy_mirror"
    assert approved["gate"]["decision"] == Decision.ALLOW.value
    assert approved["result"]["approved"] is True
    assert env.world.state.sent_to == task.recipient


@dataclass
class _Plan:
    id: str = "plan-1"


class _FakeArmorIQClient:
    def __init__(self, *, invoke_result: Any = None, invoke_error: Exception | None = None) -> None:
        self.invoke_result = {"ok": True} if invoke_result is None else invoke_result
        self.invoke_error = invoke_error
        self.invocations: list[dict[str, Any]] = []

    def capture_plan(self, **kwargs: Any) -> _Plan:
        return _Plan()

    def get_intent_token(self, plan: _Plan, *, validity_seconds: int) -> str:
        assert plan.id == "plan-1"
        assert validity_seconds > 0
        return "signed-test-token"

    def invoke(self, **kwargs: Any) -> Any:
        self.invocations.append(kwargs)
        if self.invoke_error is not None:
            raise self.invoke_error
        return self.invoke_result


def _live_env(client: _FakeArmorIQClient) -> ArmorEnv:
    return ArmorEnv(
        mode="armoriq_live",
        gate_factory=lambda _profile: ArmorIQGate(
            user_email="analyst@acme.test",
            llm_name="test-model",
            client=client,
        ),
    )


def test_live_gate_calls_sdk_once_and_labels_remote_enforcement() -> None:
    client = _FakeArmorIQClient(invoke_result={"ok": True, "data": {"read": True}})
    env = _live_env(client)
    env.reset(seed=8, profile_name="reviewer")
    _, _, _, _, info = env.step(tool("records.read", document_id="weekly_metrics"))
    assert len(client.invocations) == 1
    assert client.invocations[0]["intent_token"] == "signed-test-token"
    assert info["execution_mode"] == "armoriq_live"
    assert info["gate"]["source"] == "armoriq_live"
    assert info["result"]["remote_result"] == {"ok": True, "data": {"read": True}}
    assert env.world.state.read_documents == {"weekly_metrics"}


def test_live_sdk_error_fails_closed_without_shadow_side_effect() -> None:
    client = _FakeArmorIQClient(invoke_error=RuntimeError("network unavailable"))
    env = _live_env(client)
    env.reset(seed=9, profile_name="reviewer")
    _, _, _, _, info = env.step(tool("records.read", document_id="weekly_metrics"))
    assert info["gate"]["source"] == "armoriq_live"
    assert info["gate"]["decision"] == Decision.BLOCK.value
    assert env.world.state.read_documents == set()


def test_live_hold_cannot_be_locally_approved_or_executed() -> None:
    class PolicyHoldException(Exception):
        delegation_id = "delegation-7"

    client = _FakeArmorIQClient(invoke_error=PolicyHoldException("approval required"))
    env = _live_env(client)
    env.reset(seed=10, profile_name="analyst")
    _, _, _, _, held = env.step(tool("records.read", document_id="weekly_metrics"))
    assert held["gate"]["decision"] == Decision.HOLD.value
    assert env.world.state.read_documents == set()

    _, _, _, _, pending = env.step(AgentAction(ActionKind.REQUEST_APPROVAL))
    assert pending["gate"]["decision"] == Decision.HOLD.value
    assert pending["result"]["pending"] is True
    assert env.world.state.read_documents == set()
    assert len(client.invocations) == 1
