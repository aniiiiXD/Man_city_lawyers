"""Worker and scheduler contracts; kept independent of model/RL libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol

from work_web.contracts import (
    ApprovalOrigin,
    ArtifactDraft,
    ArtifactRecord,
    WorkTask,
    readonly_mapping,
)


@dataclass(frozen=True, slots=True)
class ApprovalHold:
    scope: str
    reason: str
    origin: ApprovalOrigin = ApprovalOrigin.SYNTHETIC_REVIEW
    locally_resolvable: bool = True
    plan_id: str | None = None
    decision_id: str | None = None
    delegation_id: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    reason: str
    retryable: bool = True


@dataclass(frozen=True, slots=True)
class WorkerExecution:
    """A worker's result, delivered after ``delay_ticks`` virtual ticks."""

    delay_ticks: int = 1
    artifacts: tuple[ArtifactDraft, ...] = ()
    hold: ApprovalHold | None = None
    failure: WorkerFailure | None = None

    def __post_init__(self) -> None:
        if self.delay_ticks < 1:
            raise ValueError("worker delay_ticks must be at least one")
        if self.hold is not None and self.failure is not None:
            raise ValueError("a worker result cannot be both a hold and a failure")
        if (self.hold is not None or self.failure is not None) and self.artifacts:
            raise ValueError("held or failed work cannot publish artifacts")


@dataclass(frozen=True, slots=True)
class WorkerContext:
    task: WorkTask
    attempt: int
    now: int
    worker_id: str = "worker"
    inputs: Mapping[str, ArtifactRecord] = field(default_factory=readonly_mapping)
    approval_granted: bool | None = None
    job_id: str = "work-web-job"
    episode_seed: int | None = None
    policy_instructions: str = ""
    tools: ToolGateway | None = None

    def call_tool(
        self,
        *,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        rationale: str = "",
    ) -> ToolResult:
        """Route a worker MCP action through the configured policy boundary."""
        if self.tools is None:
            raise RuntimeError("No governed tool gateway is configured for this worker")
        result = self.tools.call(tool=tool, arguments=arguments or {}, rationale=rationale)
        if result.decision.status == DecisionStatus.HOLD:
            raise ToolAuthorizationHold(result.decision)
        return result


class GovernanceMode(StrEnum):
    GENERIC = "generic"
    PROMPT_POLICY = "prompt_policy"
    POLICY_MIRROR = "policy_mirror"
    ARMIQ_LIVE = "armoriq_live"


class DecisionStatus(StrEnum):
    ALLOW = "allow"
    HOLD = "hold"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class ToolRequest:
    action_id: str
    job_id: str
    task_id: str
    parent_task_id: str | None
    delegation_id: str | None
    worker_id: str
    role: str
    tool: str
    arguments: Mapping[str, Any]
    intent: str
    rationale: str
    attempt: int
    approval_granted: bool | None = None
    episode_seed: int | None = None


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    status: DecisionStatus
    reason: str = ""
    plan_id: str | None = None
    decision_id: str | None = None
    approval_scope: str | None = None
    remote_result: Mapping[str, Any] | None = None
    locally_resolvable: bool = True
    delegation_id: str | None = None

    def __post_init__(self) -> None:
        if self.remote_result is not None:
            object.__setattr__(self, "remote_result", readonly_mapping(self.remote_result))


@dataclass(frozen=True, slots=True)
class ToolResult:
    decision: GovernanceDecision
    executed: bool
    result: Mapping[str, Any] = field(default_factory=readonly_mapping)


class ToolAuthorizationHold(RuntimeError):
    """Stop a worker at the tool boundary until its HOLD is resolved."""

    def __init__(self, decision: GovernanceDecision) -> None:
        super().__init__(decision.reason or "Tool action is awaiting approval")
        self.scope = decision.approval_scope or "unspecified"
        self.reason = decision.reason or "Tool action is awaiting approval"
        self.locally_resolvable = decision.locally_resolvable
        self.plan_id = decision.plan_id
        self.decision_id = decision.decision_id
        self.delegation_id = decision.delegation_id


class ToolGateway(Protocol):
    def call(
        self, *, tool: str, arguments: Mapping[str, Any], rationale: str
    ) -> ToolResult: ...


class ToolExecutor(Protocol):
    def execute(self, request: ToolRequest) -> Mapping[str, Any]: ...


class GovernanceAdapter(Protocol):
    mode: GovernanceMode
    native_delegation_supported: bool

    def authorize(self, request: ToolRequest) -> GovernanceDecision: ...

    def report(
        self, request: ToolRequest, decision: GovernanceDecision, result: Mapping[str, Any]
    ) -> None: ...


class Worker(Protocol):
    worker_id: str
    role: str

    def start(self, context: WorkerContext) -> WorkerExecution: ...


@dataclass(frozen=True, slots=True)
class SchedulerResult:
    now: int
    progressed: bool
    dispatched: tuple[str, ...] = ()
    completed: tuple[str, ...] = ()
    blocked_on_approval: tuple[str, ...] = ()
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_concurrency: int = 3
    retry_backoff_ticks: int = 1

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if self.retry_backoff_ticks < 0:
            raise ValueError("retry_backoff_ticks cannot be negative")
