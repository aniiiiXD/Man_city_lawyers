"""Contract checks for the concrete live ArmorIQ Work Web adapter."""

from dataclasses import dataclass, replace

from orchestrator.armoriq_adapter import ArmorIQGovernanceAdapter
from orchestrator.contracts import (
    DecisionStatus,
    GovernanceMode,
    ToolRequest,
    WorkerContext,
)
from orchestrator.governance import GovernedToolGateway
from work_web.contracts import WorkTask
from work_web.event_log import EventLog


@dataclass
class FakePlan:
    id: str = "plan-17"


@dataclass
class FakeResult:
    status: str = "success"
    data: dict[str, object] | None = None
    invocation_id: str = "invocation-23"

    def __post_init__(self):
        if self.data is None:
            self.data = {"source_id": "launch_log", "claim": "Project launched"}


class FakeArmorIQClient:
    def __init__(self, result=None, error=None):
        self.result = result or FakeResult()
        self.error = error
        self.plan_calls = []
        self.token_calls = []
        self.invoke_calls = []

    def capture_plan(self, **kwargs):
        self.plan_calls.append(kwargs)
        return FakePlan()

    def get_intent_token(self, plan, *, validity_seconds):
        self.token_calls.append((plan, validity_seconds))
        return "signed-token-17"

    def invoke(self, **kwargs):
        self.invoke_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _request(tool="case.read_source", *, episode_seed=None):
    return ToolRequest(
        action_id="action-1",
        job_id="job-1",
        task_id="research.primary",
        parent_task_id=None,
        delegation_id=None,
        worker_id="researcher-1",
        role="researcher",
        tool=tool,
        arguments={"source_id": "launch_log"},
        intent="Read the assigned source record.",
        rationale="Use the source needed for the brief.",
        attempt=1,
        episode_seed=episode_seed,
    )


def test_sdk_adapter_binds_the_call_to_a_plan_and_returns_remote_result() -> None:
    client = FakeArmorIQClient()
    adapter = ArmorIQGovernanceAdapter(
        user_email="agent@example.com",
        llm_name="test-model",
        client=client,
    )

    decision = adapter.authorize(_request())

    assert adapter.mode == GovernanceMode.ARMIQ_LIVE
    assert decision.status == DecisionStatus.ALLOW
    assert decision.plan_id == "plan-17"
    assert decision.decision_id == "invocation-23"
    assert decision.remote_result["ok"] is True
    assert decision.remote_result["source_id"] == "launch_log"
    assert client.plan_calls[0]["plan"]["steps"] == [
        {
            "action": "read_source",
            "mcp": "work-web-mcp",
            "params": {"source_id": "launch_log"},
        }
    ]
    assert client.invoke_calls[0]["intent_token"] == "signed-token-17"
    assert client.invoke_calls[0]["user_email"] == "agent@example.com"
    assert len(client.invoke_calls) == 1


def test_live_gateway_does_not_repeat_the_remote_call_in_the_local_executor() -> None:
    client = FakeArmorIQClient()
    adapter = ArmorIQGovernanceAdapter(
        user_email="agent@example.com",
        llm_name="test-model",
        client=client,
    )

    class CountingExecutor:
        calls = 0

        def execute(self, request):
            self.calls += 1
            return {"ok": True, "source_id": request.arguments["source_id"]}

    executor = CountingExecutor()
    events = EventLog()
    context = WorkerContext(
        task=WorkTask("research", "researcher", "Read source", "Read source."),
        attempt=1,
        now=0,
        worker_id="researcher-1",
        job_id="job-1",
    )
    gateway = GovernedToolGateway(
        context,
        adapter=adapter,
        executor=executor,
        event_log=events,
        now=lambda: 0,
    )
    result = replace(context, tools=gateway).call_tool(
        tool="case.read_source",
        arguments={"source_id": "launch_log"},
    )

    assert result.executed is True
    assert result.result["claim"] == "Project launched"
    assert executor.calls == 0
    executed = [event for event in events.events if event.kind == "tool.executed"]
    assert executed[0].payload["execution_plane"] == "remote_live"
    assert executed[0].payload["plan_id"] == "plan-17"
    assert executed[0].payload["decision_id"] == "invocation-23"


def test_sdk_hold_preserves_external_delegation_and_cannot_be_locally_resolved() -> None:
    class PolicyHoldException(Exception):
        delegation_id = "delegation-9"

    client = FakeArmorIQClient(error=PolicyHoldException("approval required"))
    adapter = ArmorIQGovernanceAdapter(
        user_email="agent@example.com",
        llm_name="test-model",
        client=client,
    )

    decision = adapter.authorize(_request("workspace.submit_brief"))

    assert decision.status == DecisionStatus.HOLD
    assert decision.locally_resolvable is False
    assert decision.delegation_id == "delegation-9"
    assert decision.plan_id == "plan-17"


def test_seeded_read_passes_the_episode_seed_to_the_registered_mcp() -> None:
    client = FakeArmorIQClient()
    adapter = ArmorIQGovernanceAdapter(
        user_email="agent@example.com",
        llm_name="test-model",
        client=client,
    )

    decision = adapter.authorize(_request(episode_seed=41))

    expected_params = {"source_id": "launch_log", "episode_seed": 41}
    assert decision.status == DecisionStatus.ALLOW
    assert client.plan_calls[0]["plan"]["steps"][0]["params"] == expected_params
    assert client.invoke_calls[0]["params"] == expected_params


def test_remote_tool_error_is_not_retried_in_the_local_executor() -> None:
    client = FakeArmorIQClient(
        result=FakeResult(status="error", data={"error": "synthetic MCP failure"})
    )
    adapter = ArmorIQGovernanceAdapter(
        user_email="agent@example.com",
        llm_name="test-model",
        client=client,
    )

    decision = adapter.authorize(_request())

    assert decision.status == DecisionStatus.ALLOW
    assert decision.remote_result["ok"] is False
    assert decision.remote_result["error"] == "synthetic MCP failure"


def test_unmapped_tool_and_unknown_sdk_error_fail_closed() -> None:
    adapter = ArmorIQGovernanceAdapter(
        user_email="agent@example.com",
        llm_name="test-model",
        client=FakeArmorIQClient(error=RuntimeError("network unavailable")),
    )

    unmapped = adapter.authorize(_request("case.unknown"))
    failed = adapter.authorize(_request())

    assert unmapped.status == DecisionStatus.BLOCK
    assert unmapped.remote_result is None
    assert failed.status == DecisionStatus.BLOCK
    assert failed.locally_resolvable is False
