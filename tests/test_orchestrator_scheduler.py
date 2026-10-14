"""Discrete-event concurrency, hold, and retry behavior."""

import pytest

from orchestrator.contracts import ApprovalHold, WorkerExecution, WorkerFailure
from orchestrator.scheduler import AsyncScheduler
from work_web.contracts import ArtifactDraft, TaskStatus, WorkTask
from work_web.environment import InvalidTransition, WorkWeb


class HoldingWorker:
    role = "held-role"
    worker_id = "held-worker"

    def start(self, context):
        if context.approval_granted is None:
            return WorkerExecution(delay_ticks=1, hold=ApprovalHold("review", "Needs review"))
        return WorkerExecution(
            artifacts=(ArtifactDraft("held.output", {"approved": True}),)
        )


class DelayedWorker:
    role = "other-role"
    worker_id = "other-worker"

    def start(self, context):
        return WorkerExecution(
            delay_ticks=3,
            artifacts=(ArtifactDraft("other.output", {"done": True}),),
        )


class RetryWorker:
    role = "retry-role"
    worker_id = "retry-worker"

    def __init__(self):
        self.calls = 0

    def start(self, context):
        self.calls += 1
        if self.calls == 1:
            return WorkerExecution(failure=WorkerFailure("temporary", retryable=True))
        return WorkerExecution(artifacts=(ArtifactDraft("retry.output", {"ok": True}),))


def test_approval_hold_releases_slot_while_other_work_completes() -> None:
    web = WorkWeb()
    web.register_tasks(
        (
            WorkTask("held", "held-role", "Held work", "Wait for review", output_keys=("held.output",)),
            WorkTask("other", "other-role", "Other work", "Continue independently", output_keys=("other.output",)),
        )
    )
    scheduler = AsyncScheduler(
        web,
        {"held-role": HoldingWorker(), "other-role": DelayedWorker()},
    )
    scheduler.dispatch("held")
    scheduler.dispatch("other")

    scheduler.advance_to_next_event()
    assert web.task("held").status == TaskStatus.WAITING_APPROVAL
    assert scheduler.active_task_ids == ("other",)

    scheduler.advance_to_next_event()
    assert web.task("other").status == TaskStatus.SUCCEEDED
    assert web.task("held").status == TaskStatus.WAITING_APPROVAL
    assert len(web.pending_approvals()) == 1

    request = web.pending_approvals()[0]
    scheduler.resolve_approval(request.request_id, granted=True)
    scheduler.dispatch("held")
    scheduler.advance_to_next_event()
    assert web.task("held").status == TaskStatus.SUCCEEDED


def test_worker_pool_enforces_configured_concurrency() -> None:
    web = WorkWeb()
    web.register_tasks(
        WorkTask(f"task-{index}", "other-role", "Work", "Independent")
        for index in range(3)
    )
    scheduler = AsyncScheduler(web, {"other-role": DelayedWorker()})
    scheduler.dispatch("task-0")
    scheduler.dispatch("task-1")

    assert scheduler.free_slots == 1
    scheduler.dispatch("task-2")
    assert scheduler.free_slots == 0
    with pytest.raises(InvalidTransition):
        scheduler.dispatch("task-0")


def test_retryable_worker_failure_uses_a_later_attempt() -> None:
    web = WorkWeb()
    web.register_tasks(
        (WorkTask("retry", "retry-role", "Retry", "Recover", output_keys=("retry.output",)),)
    )
    worker = RetryWorker()
    scheduler = AsyncScheduler(web, {"retry-role": worker})

    scheduler.step()  # attempt one fails and schedules a retry
    assert web.task("retry").status == TaskStatus.QUEUED
    scheduler.step()  # wait for the deterministic backoff tick
    scheduler.step()  # attempt two succeeds

    assert worker.calls == 2
    assert web.task("retry").status == TaskStatus.SUCCEEDED
    assert web.task("retry").attempt == 2
