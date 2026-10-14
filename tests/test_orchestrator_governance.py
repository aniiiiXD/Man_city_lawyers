"""Governance boundary checks, including remote live execution semantics."""

from dataclasses import replace

import pytest

from orchestrator.contracts import (
    DecisionStatus,
    GovernanceDecision,
    GovernanceMode,
    WorkerContext,
)
from orchestrator.governance import CallbackGovernanceAdapter, GovernedToolGateway
from orchestrator.scheduler import AsyncScheduler
from work_web.contracts import ApprovalOrigin, WorkTask
from work_web.event_log import EventLog
from work_web.environment import InvalidTransition, WorkWeb


class CountingExecutor:
    def __init__(self):
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return {"ok": True, "plane": "local"}


def _gateway(adapter, executor, events):
    context = WorkerContext(
        task=WorkTask("task-1", "researcher", "Read", "Read one source"),
        attempt=1,
        now=0,
        worker_id="worker-1",
        job_id="job-1",
    )
    gateway = GovernedToolGateway(
        context,
        adapter=adapter,
        executor=executor,
        event_log=events,
        now=lambda: 0,
    )
    return replace(context, tools=gateway)


def test_live_allow_uses_remote_result_without_local_duplicate_execution() -> None:
    remote_calls = []

    def authorize_and_invoke(request):
        remote_calls.append(request.action_id)
        return GovernanceDecision(
            DecisionStatus.ALLOW,
            decision_id="decision-1",
            remote_result={"ok": True, "side_effects": ["remote_write"]},
        )

    adapter = CallbackGovernanceAdapter(
        authorize_and_invoke,
        mode=GovernanceMode.ARMIQ_LIVE,
    )
    executor = CountingExecutor()
    events = EventLog()
    context = _gateway(adapter, executor, events)

    result = context.call_tool(
        tool="workspace.submit_brief", arguments={"brief_id": "brief-1"}
    )

    assert result.executed is True
    assert result.result["side_effects"] == ("remote_write",)
    assert len(remote_calls) == 1
    assert executor.calls == 0
    executed = [event for event in events.events if event.kind == "tool.executed"]
    assert executed[0].payload["execution_plane"] == "remote_live"


def test_live_allow_without_remote_result_fails_closed() -> None:
    adapter = CallbackGovernanceAdapter(
        lambda request: GovernanceDecision(DecisionStatus.ALLOW),
        mode=GovernanceMode.ARMIQ_LIVE,
    )
    executor = CountingExecutor()
    events = EventLog()
    context = _gateway(adapter, executor, events)

    result = context.call_tool(tool="workspace.submit_brief", arguments={})

    assert result.decision.status == DecisionStatus.BLOCK
    assert result.executed is False
    assert executor.calls == 0
    assert "already executed remote result" in result.decision.reason


def test_local_approval_cannot_clear_live_governance_hold() -> None:
    class LiveHoldingWorker:
        role = "publisher"
        worker_id = "publisher-1"

        def start(self, context):
            context.call_tool(tool="workspace.submit_brief", arguments={})
            raise AssertionError("a held tool call must stop the worker")

    adapter = CallbackGovernanceAdapter(
        lambda request: GovernanceDecision(
            DecisionStatus.HOLD,
            reason="ArmorIQ requires an external approval",
            plan_id="plan-42",
            decision_id="decision-42",
            approval_scope="remote-review",
            delegation_id="delegation-42",
        ),
        mode=GovernanceMode.ARMIQ_LIVE,
    )
    web = WorkWeb()
    web.register_tasks((WorkTask("publish", "publisher", "Publish", "Submit"),))
    scheduler = AsyncScheduler(web, {"publisher": LiveHoldingWorker()}, governance_adapter=adapter)

    scheduler.dispatch("publish")
    scheduler.advance_to_next_event()
    request = web.pending_approvals()[0]

    assert request.origin == ApprovalOrigin.GOVERNANCE_GATE
    assert request.scope == "remote-review"
    assert request.locally_resolvable is False
    assert request.plan_id == "plan-42"
    assert request.decision_id == "decision-42"
    assert request.delegation_id == "delegation-42"
    with pytest.raises(InvalidTransition, match="local approval cannot resolve"):
        scheduler.resolve_approval(request.request_id, granted=True)
    assert web.approval(request.request_id).status.value == "pending"


def test_tool_hold_stops_worker_before_following_tool_calls() -> None:
    local_executor = CountingExecutor()
    requested_tools = []

    class WorkerThatWouldContinue:
        role = "researcher"
        worker_id = "researcher-1"

        def start(self, context):
            context.call_tool(tool="workspace.export_all", arguments={})
            context.call_tool(tool="case.read_source", arguments={})
            raise AssertionError("the held export must stop this worker")

    def hold_export(request):
        requested_tools.append(request.tool)
        return GovernanceDecision(
            DecisionStatus.HOLD,
            reason="Remote review required",
            approval_scope="restricted-export",
        )

    adapter = CallbackGovernanceAdapter(hold_export, mode=GovernanceMode.ARMIQ_LIVE)
    web = WorkWeb()
    web.register_tasks((WorkTask("research", "researcher", "Research", "Read sources"),))
    scheduler = AsyncScheduler(
        web,
        {"researcher": WorkerThatWouldContinue()},
        governance_adapter=adapter,
        tool_executor=local_executor,
    )

    scheduler.dispatch("research")
    scheduler.advance_to_next_event()

    assert requested_tools == ["workspace.export_all"]
    assert local_executor.calls == 0
    assert web.task("research").status.value == "waiting_approval"
    assert web.pending_approvals()[0].locally_resolvable is False
