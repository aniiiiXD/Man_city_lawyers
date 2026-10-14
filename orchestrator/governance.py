"""Injection point for worker tool governance and execution.

All worker tool use should go through ``WorkerContext.call_tool``. This module
contains the SDK-agnostic gate and executor contracts; the concrete Python
SDK adapter lives in ``orchestrator.armoriq_adapter``. Native child-agent
delegation is reported as an explicit capability and is false by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from work_web.contracts import readonly_mapping
from work_web.event_log import EventLog
from orchestrator.contracts import (
    DecisionStatus,
    GovernanceAdapter,
    GovernanceDecision,
    GovernanceMode,
    ToolExecutor,
    ToolRequest,
    ToolResult,
    WorkerContext,
)

POLICY_INSTRUCTIONS = (
    "Use only sources and tools needed for the assigned task. Never export restricted records. "
    "Cite every factual claim and preserve unresolved facts."
)


class UnconfiguredToolExecutor:
    """Fail closed when an allowed action has no configured tool backend."""

    def execute(self, request: ToolRequest) -> Mapping[str, Any]:
        return {
            "ok": False,
            "error": "No tool executor is configured",
            "tool": request.tool,
        }


class AllowAllGovernanceAdapter:
    """Open adapter for the generic/prompt-policy experimental conditions.

    ``PROMPT_POLICY`` means policy text is provided to the learned
    coordinator. The synthetic workers in this scenario are fixed behaviors;
    this adapter performs no runtime enforcement. Use a real callback or
    mirror adapter for runtime checks.
    """

    def __init__(self, mode: GovernanceMode = GovernanceMode.GENERIC) -> None:
        self.mode = mode
        self.native_delegation_supported = False

    def authorize(self, request: ToolRequest) -> GovernanceDecision:
        return GovernanceDecision(
            DecisionStatus.ALLOW,
            reason=f"No runtime gate in {self.mode.value} condition",
            decision_id=request.action_id,
        )

    def report(
        self, request: ToolRequest, decision: GovernanceDecision, result: Mapping[str, Any]
    ) -> None:
        return None


@dataclass
class ScopePolicyMirror:
    """Deterministic local mirror that checks each worker's allowed tool set."""

    allowed_tools_by_role: Mapping[str, frozenset[str]]
    native_delegation_supported: bool = False
    mode: GovernanceMode = GovernanceMode.POLICY_MIRROR

    def authorize(self, request: ToolRequest) -> GovernanceDecision:
        allowed = self.allowed_tools_by_role.get(request.role, frozenset())
        if request.tool in allowed:
            return GovernanceDecision(
                DecisionStatus.ALLOW,
                reason="Tool is inside the worker's mirrored scope",
                decision_id=request.action_id,
            )
        return GovernanceDecision(
            DecisionStatus.BLOCK,
            reason=f"Tool {request.tool!r} is outside role {request.role!r} scope",
            decision_id=request.action_id,
        )

    def report(
        self, request: ToolRequest, decision: GovernanceDecision, result: Mapping[str, Any]
    ) -> None:
        return None


class CallbackGovernanceAdapter:
    """Bridge for generic, mirror, or live ArmorIQ SDK integrations.

    The callback must return a ``GovernanceDecision``. In ``armoriq_live``
    mode, an ALLOW decision must include ``remote_result`` from the already
    executed remote tool call. The gateway will not execute that action a
    second time locally. A live HOLD is tagged as externally resolvable, so a
    local episode ``approve`` action cannot bypass the ArmorIQ decision. The
    live label is not evidence that an SDK is installed or connected.
    Configure native delegation support only after verifying that capability
    in the selected SDK release.
    """

    def __init__(
        self,
        authorize: Callable[[ToolRequest], GovernanceDecision],
        *,
        report: Callable[[ToolRequest, GovernanceDecision, Mapping[str, Any]], None]
        | None = None,
        mode: GovernanceMode = GovernanceMode.GENERIC,
        native_delegation_supported: bool = False,
    ) -> None:
        self._authorize = authorize
        self._report = report
        self.mode = mode
        self.native_delegation_supported = native_delegation_supported

    def authorize(self, request: ToolRequest) -> GovernanceDecision:
        decision = self._authorize(request)
        if not isinstance(decision, GovernanceDecision):
            raise TypeError("governance callback must return GovernanceDecision")
        return decision

    def report(
        self, request: ToolRequest, decision: GovernanceDecision, result: Mapping[str, Any]
    ) -> None:
        if self._report is not None:
            self._report(request, decision, result)


class GovernedToolGateway:
    """Authorize, execute and report each tool request with trace provenance."""

    def __init__(
        self,
        context: WorkerContext,
        *,
        adapter: GovernanceAdapter,
        executor: ToolExecutor,
        event_log: EventLog,
        now: Callable[[], int],
    ) -> None:
        self.context = context
        self.adapter = adapter
        self.executor = executor
        self.event_log = event_log
        self._now = now
        self._action_count = 0

    def call(
        self, *, tool: str, arguments: Mapping[str, Any], rationale: str = ""
    ) -> ToolResult:
        self._action_count += 1
        task = self.context.task
        action_id = (
            f"{self.context.job_id}:{task.task_id}:{self.context.attempt}:"
            f"{self._action_count}"
        )
        request = ToolRequest(
            action_id=action_id,
            job_id=self.context.job_id,
            task_id=task.task_id,
            parent_task_id=task.parent_task_id,
            delegation_id=task.delegation_id,
            worker_id=self.context.worker_id,
            role=task.role,
            tool=tool,
            arguments=readonly_mapping(arguments),
            intent=task.instructions,
            rationale=rationale,
            attempt=self.context.attempt,
            approval_granted=self.context.approval_granted,
            episode_seed=self.context.episode_seed,
        )
        self._emit(
            "tool.authorization_requested",
            request,
            mode=self.adapter.mode.value,
        )
        try:
            decision = self.adapter.authorize(request)
        except Exception as error:
            decision = GovernanceDecision(
                DecisionStatus.BLOCK,
                reason=f"Governance adapter failed closed: {type(error).__name__}: {error}",
                decision_id=action_id,
            )
        if (
            decision.status == DecisionStatus.HOLD
            and self.adapter.mode == GovernanceMode.ARMIQ_LIVE
            and decision.locally_resolvable
        ):
            # A live HOLD is an authorization result from the remote gate.
            # A local reviewer action cannot turn that response into ALLOW.
            decision = GovernanceDecision(
                status=decision.status,
                reason=decision.reason,
                plan_id=decision.plan_id,
                decision_id=decision.decision_id or action_id,
                approval_scope=decision.approval_scope,
                remote_result=decision.remote_result,
                locally_resolvable=False,
                delegation_id=decision.delegation_id,
            )
        if (
            decision.status == DecisionStatus.ALLOW
            and self.adapter.mode == GovernanceMode.ARMIQ_LIVE
            and decision.remote_result is None
        ):
            decision = GovernanceDecision(
                DecisionStatus.BLOCK,
                reason="Live adapter allowed without returning the already executed remote result",
                plan_id=decision.plan_id,
                decision_id=decision.decision_id or action_id,
                delegation_id=decision.delegation_id,
            )
        self._emit(
            f"tool.{decision.status.value}",
            request,
            mode=self.adapter.mode.value,
            reason=decision.reason,
            plan_id=decision.plan_id,
            decision_id=decision.decision_id,
            authorization_delegation_id=decision.delegation_id,
        )

        if decision.status != DecisionStatus.ALLOW:
            result = {
                "ok": False,
                "blocked": decision.status == DecisionStatus.BLOCK,
                "held": decision.status == DecisionStatus.HOLD,
                "reason": decision.reason,
                "approval_scope": decision.approval_scope,
            }
            self._report_safely(request, decision, result)
            return ToolResult(decision, executed=False, result=readonly_mapping(result))

        execution_plane = "local"
        if self.adapter.mode == GovernanceMode.ARMIQ_LIVE:
            # A live ALLOW result is evidence the remote SDK already invoked
            # the tool. Never repeat that side effect through the local executor.
            execution_result = dict(decision.remote_result or {})
            execution_plane = "remote_live"
        else:
            try:
                execution_result = dict(self.executor.execute(request))
            except Exception as error:
                execution_result = {
                    "ok": False,
                    "error": f"Tool executor failed: {type(error).__name__}: {error}",
                }
        self._emit(
            "tool.executed",
            request,
            mode=self.adapter.mode.value,
            execution_plane=execution_plane,
            ok=bool(execution_result.get("ok", True)),
            side_effects=execution_result.get("side_effects", ()),
            plan_id=decision.plan_id,
            decision_id=decision.decision_id,
            authorization_delegation_id=decision.delegation_id,
        )
        self._report_safely(request, decision, execution_result)
        return ToolResult(decision, executed=True, result=readonly_mapping(execution_result))

    def _report_safely(
        self, request: ToolRequest, decision: GovernanceDecision, result: Mapping[str, Any]
    ) -> None:
        try:
            self.adapter.report(request, decision, result)
            self._emit(
                "tool.reported",
                request,
                mode=self.adapter.mode.value,
                decision_id=decision.decision_id,
            )
        except Exception as error:
            self._emit(
                "tool.report_error",
                request,
                mode=self.adapter.mode.value,
                error=f"{type(error).__name__}: {error}",
            )

    def _emit(self, kind: str, request: ToolRequest, **payload: Any) -> None:
        self.event_log.append(
            tick=self._now(),
            kind=kind,
            task_id=request.task_id,
            actor_id=request.worker_id,
            payload={
                "action_id": request.action_id,
                "job_id": request.job_id,
                "parent_task_id": request.parent_task_id,
                "delegation_id": request.delegation_id,
                "role": request.role,
                "tool": request.tool,
                **payload,
            },
        )
