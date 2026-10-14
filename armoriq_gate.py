"""Policy gates for plain, simulated, and live ArmorIQ execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from models import Decision, GateDecision, ToolCall
from policies import PolicyProfile


class ActionGate(Protocol):
    """Authorization boundary used by :class:`environment.ArmorEnv`."""

    def authorize(self, call: ToolCall, *, goal: str, rationale: str) -> GateDecision: ...


class PlainGate:
    """Ablation gate: every well-formed call runs with no policy feedback."""

    def authorize(self, call: ToolCall, *, goal: str, rationale: str) -> GateDecision:
        return GateDecision(
            Decision.ALLOW,
            "Generic mode: action executes without governance",
            source="generic",
        )


@dataclass
class TrainingPolicyGate:
    """Fast deterministic policy surrogate for thousands of training rollouts.

    It intentionally shares the same policy profile used to configure live
    ArmorIQ. It is not presented as a live ArmorIQ decision in evaluation.
    """

    profile: PolicyProfile

    def authorize(self, call: ToolCall, *, goal: str, rationale: str) -> GateDecision:
        return self.profile.decide(call)

    def resolve_hold(
        self,
        call: ToolCall,
        *,
        gate_decision: GateDecision,
        goal: str,
        rationale: str,
    ) -> GateDecision:
        del gate_decision, goal, rationale
        return self.profile.approve(call)


class ArmorIQGate:
    """Live ArmorIQ adapter using capture_plan -> token -> invoke.

    The target MCP must already be registered and reachable through ArmorIQ.
    ``invoke`` executes the configured remote MCP action. Use this adapter for
    live evaluation/demo runs, not the in-memory high-throughput trainer.
    """

    def __init__(
        self,
        *,
        user_email: str,
        llm_name: str,
        validity_seconds: int = 300,
        client: Any | None = None,
        approval_resolver: Callable[..., GateDecision] | None = None,
    ) -> None:
        self.user_email = user_email
        self.llm_name = llm_name
        self.validity_seconds = validity_seconds
        self.approval_resolver = approval_resolver
        if client is None:
            try:
                from armoriq_sdk import ArmorIQClient  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError(
                    "Install armoriq-sdk and authenticate before using ArmorIQGate."
                ) from error
            client = ArmorIQClient()
        self.client = client

    def authorize(self, call: ToolCall, *, goal: str, rationale: str) -> GateDecision:
        plan_payload = {
            "goal": goal,
            "steps": [
                {
                    "action": call.action,
                    "mcp": call.mcp,
                    "params": dict(call.params),
                    "rationale": rationale,
                }
            ],
        }
        plan_id: str | None = None
        try:
            plan = self.client.capture_plan(
                llm=self.llm_name,
                prompt=goal,
                plan=plan_payload,
            )
            plan_id = getattr(plan, "id", None) or getattr(plan, "plan_id", None)
            token = self.client.get_intent_token(plan, validity_seconds=self.validity_seconds)
            result = self.client.invoke(
                mcp=call.mcp,
                action=call.action,
                intent_token=token,
                params=dict(call.params),
                user_email=self.user_email,
            )
        except Exception as error:  # SDK exception types vary by installed SDK version.
            name = type(error).__name__
            if name == "PolicyHoldException":
                return GateDecision(
                    Decision.HOLD,
                    str(error),
                    plan_id=plan_id,
                    delegation_id=getattr(error, "delegation_id", None),
                    source="armoriq_live",
                )
            if name in {"PolicyBlockedException", "IntentMismatchException", "TokenExpiredException"}:
                return GateDecision(Decision.BLOCK, str(error), plan_id=plan_id, source="armoriq_live")
            # SDK/network failures must never degrade into an ungated tool call.
            return GateDecision(
                Decision.BLOCK,
                f"ArmorIQ invocation failed closed ({name})",
                plan_id=plan_id,
                source="armoriq_live",
            )
        remote_result = getattr(result, "data", result)
        return GateDecision(
            Decision.ALLOW,
            "ArmorIQ authorized and invoked the MCP action",
            plan_id=plan_id,
            remote_result=remote_result,
            source="armoriq_live",
        )

    def resolve_hold(
        self,
        call: ToolCall,
        *,
        gate_decision: GateDecision,
        goal: str,
        rationale: str,
    ) -> GateDecision:
        """Resolve a live hold only through an explicitly supplied SDK adapter.

        A local approval button must not execute a live-held action in the
        simulation shadow. The adapter is responsible for following the
        selected ArmorIQ SDK's approval/resume protocol and returning the
        resulting governed decision.
        """
        if self.approval_resolver is None:
            return GateDecision(
                Decision.HOLD,
                "Live approval is still pending; no SDK approval resolver is configured",
                plan_id=gate_decision.plan_id,
                delegation_id=gate_decision.delegation_id,
                source="armoriq_live",
            )
        try:
            decision = self.approval_resolver(
                call=call,
                plan_id=gate_decision.plan_id,
                delegation_id=gate_decision.delegation_id,
                goal=goal,
                rationale=rationale,
            )
        except Exception as error:  # The adapter's SDK exception types vary.
            return GateDecision(
                Decision.BLOCK,
                f"ArmorIQ approval resolution failed closed ({type(error).__name__})",
                plan_id=gate_decision.plan_id,
                delegation_id=gate_decision.delegation_id,
                source="armoriq_live",
            )
        if not isinstance(decision, GateDecision) or decision.source != "armoriq_live":
            return GateDecision(
                Decision.BLOCK,
                "ArmorIQ approval resolver returned an invalid decision",
                plan_id=gate_decision.plan_id,
                delegation_id=gate_decision.delegation_id,
                source="armoriq_live",
            )
        return decision
