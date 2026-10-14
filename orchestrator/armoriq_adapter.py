"""Concrete ArmorIQ SDK adapter for live synthetic Work Web evaluation.

The local episode never executes an ALLOW a second time. ArmorIQ's invoke
call is the execution, and its returned MCP data becomes the worker result.
An ArmorIQ HOLD stays external; this adapter does not translate it into a
local approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from orchestrator.contracts import (
    DecisionStatus,
    GovernanceDecision,
    GovernanceMode,
    ToolRequest,
)


@dataclass(frozen=True, slots=True)
class ArmorIQToolBinding:
    mcp: str
    action: str
    seed_parameter: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mcp, str)
            or not self.mcp.strip()
            or not isinstance(self.action, str)
            or not self.action.strip()
        ):
            raise ValueError("ArmorIQ tool bindings need non-empty MCP and action names")
        if self.seed_parameter is not None and (
            not isinstance(self.seed_parameter, str) or not self.seed_parameter.strip()
        ):
            raise ValueError("seed_parameter cannot be empty")


DEFAULT_TOOL_BINDINGS = {
    "case.read_source": ArmorIQToolBinding(
        "work-web-mcp", "read_source", seed_parameter="episode_seed"
    ),
    "case.audit_sources": ArmorIQToolBinding("work-web-mcp", "audit_sources"),
    "case.compare_sources": ArmorIQToolBinding("work-web-mcp", "compare_sources"),
    "workspace.create_draft": ArmorIQToolBinding("work-web-mcp", "create_draft"),
    "workspace.submit_brief": ArmorIQToolBinding("work-web-mcp", "submit_brief"),
    "workspace.export_all": ArmorIQToolBinding("work-web-mcp", "export_all"),
    "audit.archive_source": ArmorIQToolBinding("work-web-mcp", "archive_source"),
}


class ArmorIQGovernanceAdapter:
    """Capture a one-tool plan, mint its intent token, then invoke the MCP.

    tool_bindings maps the Work Web's qualified synthetic tool names to names
    registered with ArmorIQ. Supply an injected client for tests; otherwise
    the authenticated armoriq-sdk client is loaded lazily.
    """

    mode = GovernanceMode.ARMIQ_LIVE
    native_delegation_supported = False

    def __init__(
        self,
        *,
        user_email: str,
        llm_name: str,
        tool_bindings: Mapping[str, ArmorIQToolBinding] | None = None,
        validity_seconds: int = 300,
        client: Any | None = None,
    ) -> None:
        if not user_email.strip():
            raise ValueError("user_email cannot be empty")
        if not llm_name.strip():
            raise ValueError("llm_name cannot be empty")
        if validity_seconds < 1:
            raise ValueError("validity_seconds must be positive")
        self.user_email = user_email
        self.llm_name = llm_name
        self.validity_seconds = validity_seconds
        self.tool_bindings = dict(
            DEFAULT_TOOL_BINDINGS if tool_bindings is None else tool_bindings
        )
        if any(
            not isinstance(name, str) or not isinstance(binding, ArmorIQToolBinding)
            for name, binding in self.tool_bindings.items()
        ):
            raise TypeError("tool_bindings values must be ArmorIQToolBinding records")
        self.client = client if client is not None else self._create_client()

    @staticmethod
    def _create_client() -> Any:
        try:
            from armoriq_sdk import ArmorIQClient
        except ImportError as error:
            raise RuntimeError(
                "Install the armoriq extra and authenticate with armoriq login."
            ) from error
        return ArmorIQClient()

    def authorize(self, request: ToolRequest) -> GovernanceDecision:
        binding = self.tool_bindings.get(request.tool)
        if binding is None:
            return GovernanceDecision(
                DecisionStatus.BLOCK,
                reason=f"No ArmorIQ MCP binding is configured for {request.tool!r}",
                decision_id=request.action_id,
                locally_resolvable=False,
            )

        plan_id: str | None = None
        params = dict(request.arguments)
        if binding.seed_parameter is not None and request.episode_seed is not None:
            params[binding.seed_parameter] = request.episode_seed
        try:
            goal = request.intent or f"Complete {request.task_id} as {request.role}"
            plan = self.client.capture_plan(
                llm=self.llm_name,
                prompt=(
                    f"{goal}\nRequested tool: {request.tool}. "
                    f"Reason: {request.rationale}"
                ),
                plan={
                    "goal": goal,
                    "steps": [
                        {
                            "action": binding.action,
                            "mcp": binding.mcp,
                            "params": params,
                        }
                    ],
                    "metadata": {
                        "job_id": request.job_id,
                        "task_id": request.task_id,
                        "parent_task_id": request.parent_task_id,
                        "parent_delegation_id": request.delegation_id,
                        "worker_id": request.worker_id,
                        "role": request.role,
                        "action_id": request.action_id,
                        "episode_seed": request.episode_seed,
                    },
                },
            )
            plan_id = self._identifier(plan, "id", "plan_id")
            token = self.client.get_intent_token(
                plan,
                validity_seconds=self.validity_seconds,
            )
            result = self.client.invoke(
                mcp=binding.mcp,
                action=binding.action,
                intent_token=token,
                params=params,
                user_email=self.user_email,
            )
        except Exception as error:
            name = type(error).__name__
            decision_id = self._identifier(error, "decision_id", "request_id")
            if name == "PolicyHoldException":
                return GovernanceDecision(
                    DecisionStatus.HOLD,
                    reason=str(error),
                    plan_id=plan_id,
                    decision_id=decision_id or request.action_id,
                    approval_scope="armoriq-approval",
                    locally_resolvable=False,
                    delegation_id=self._identifier(error, "delegation_id"),
                )
            if name in {
                "PolicyBlockedException",
                "IntentMismatchException",
                "TokenExpiredException",
                "InvalidTokenException",
            }:
                return GovernanceDecision(
                    DecisionStatus.BLOCK,
                    reason=str(error),
                    plan_id=plan_id,
                    decision_id=decision_id or request.action_id,
                    locally_resolvable=False,
                )
            return GovernanceDecision(
                DecisionStatus.BLOCK,
                reason=f"ArmorIQ invocation failed closed ({name})",
                plan_id=plan_id,
                decision_id=request.action_id,
                locally_resolvable=False,
            )

        remote_result = self._result_data(result)
        status_value = getattr(result, "status", None)
        status = str(status_value).lower() if status_value is not None else ""
        if status in {"hold", "pending"}:
            return GovernanceDecision(
                DecisionStatus.HOLD,
                reason="ArmorIQ returned a pending approval result",
                plan_id=plan_id,
                decision_id=self._identifier(result, "invocation_id", "decision_id")
                or request.action_id,
                approval_scope="armoriq-approval",
                remote_result=remote_result,
                locally_resolvable=False,
                delegation_id=self._identifier(result, "delegation_id"),
            )
        if status in {"blocked", "policy_blocked"}:
            return GovernanceDecision(
                DecisionStatus.BLOCK,
                reason="ArmorIQ MCP invocation was blocked by policy",
                plan_id=plan_id,
                decision_id=self._identifier(result, "invocation_id", "decision_id")
                or request.action_id,
                remote_result=remote_result,
                locally_resolvable=False,
            )
        if status_value is not None:
            remote_result.setdefault("ok", status in {"success", "ok"})
        else:
            remote_result.setdefault("ok", False)
        return GovernanceDecision(
            DecisionStatus.ALLOW,
            reason=(
                "ArmorIQ invoked the registered MCP action"
                if remote_result["ok"] is True
                else "ArmorIQ invoked the MCP action and returned an unsuccessful tool result"
            ),
            plan_id=plan_id,
            decision_id=self._identifier(result, "invocation_id", "decision_id")
            or request.action_id,
            remote_result=remote_result,
            locally_resolvable=False,
        )

    def report(
        self,
        request: ToolRequest,
        decision: GovernanceDecision,
        result: Mapping[str, Any],
    ) -> None:
        """ArmorIQ writes its invocation audit; Work Web joins the local event."""
        del request, decision, result

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _identifier(value: Any, *names: str) -> str | None:
        for name in names:
            candidate = getattr(value, name, None)
            if candidate is not None:
                return str(candidate)
        if isinstance(value, Mapping):
            for name in names:
                candidate = value.get(name)
                if candidate is not None:
                    return str(candidate)
        return None

    @classmethod
    def _result_data(cls, result: Any) -> dict[str, Any]:
        data = getattr(result, "data", result)
        if isinstance(data, Mapping):
            return {str(key): cls._plain(value) for key, value in data.items()}
        return {"data": cls._plain(data)}

    @classmethod
    def _plain(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): cls._plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._plain(item) for item in value]
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return cls._plain(value.model_dump())
        if hasattr(value, "__dict__"):
            return cls._plain(vars(value))
        return repr(value)
