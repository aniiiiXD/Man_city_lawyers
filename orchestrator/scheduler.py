"""Bounded event-driven multi-agent scheduler with deterministic virtual time."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from work_web.contracts import (
    ApprovalOrigin,
    ApprovalStatus,
    TaskSnapshot,
    TaskStatus,
    TERMINAL_TASK_STATUSES,
)
from work_web.environment import InvalidTransition, WorkWeb
from orchestrator.contracts import (
    ApprovalHold,
    GovernanceAdapter,
    SchedulerConfig,
    SchedulerResult,
    ToolExecutor,
    Worker,
    WorkerContext,
    WorkerExecution,
    WorkerFailure,
    ToolAuthorizationHold,
)
from orchestrator.controller import DispatchController, RuleBasedController
from orchestrator.governance import (
    AllowAllGovernanceAdapter,
    GovernedToolGateway,
    POLICY_INSTRUCTIONS,
    UnconfiguredToolExecutor,
)


@dataclass(frozen=True, slots=True)
class _ActiveWork:
    task_id: str
    worker_id: str
    execution: WorkerExecution
    ready_at: int
    timeout_at: int


class AsyncScheduler:
    """Discrete-event scheduler; concurrent work consumes virtual time only.

    A worker's ``start`` method returns a delayed result.  The worker occupies
    one bounded slot until that virtual completion time.  An approval hold
    releases its slot while waiting, so unrelated work keeps running.
    """

    def __init__(
        self,
        web: WorkWeb,
        workers: Mapping[str, Worker] | Iterable[Worker],
        *,
        config: SchedulerConfig | None = None,
        controller: DispatchController | None = None,
        governance_adapter: GovernanceAdapter | None = None,
        tool_executor: ToolExecutor | None = None,
        job_id: str = "work-web-job",
        episode_seed: int | None = None,
    ) -> None:
        self.web = web
        self.config = config or SchedulerConfig()
        self.controller = controller or RuleBasedController()
        self.governance_adapter = governance_adapter or AllowAllGovernanceAdapter()
        self.tool_executor = tool_executor or UnconfiguredToolExecutor()
        self.job_id = job_id
        self.episode_seed = episode_seed
        if isinstance(workers, Mapping):
            self._workers = dict(workers)
        else:
            self._workers = {worker.role: worker for worker in workers}
        self._active: dict[str, _ActiveWork] = {}
        self._approval_grants: dict[str, bool] = {}

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    @property
    def free_slots(self) -> int:
        return max(0, self.config.max_concurrency - len(self._active))

    @property
    def all_terminal(self) -> bool:
        return all(task.status in TERMINAL_TASK_STATUSES for task in self.web.tasks())

    def ready_tasks(self) -> tuple[TaskSnapshot, ...]:
        candidates: list[TaskSnapshot] = []
        for task in self.web.tasks():
            if task.status != TaskStatus.QUEUED or task.task.task_id in self._active:
                continue
            if task.retry_at is not None and task.retry_at > self.web.now:
                continue
            dependencies = tuple(self.web.task(dep) for dep in task.task.dependencies)
            if all(dependency.status == TaskStatus.SUCCEEDED for dependency in dependencies):
                candidates.append(task)
        return tuple(candidates)

    def ready_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task.task_id for task in self.ready_tasks())

    def dispatch(self, task_id: str) -> bool:
        """Start one ready task; return false if it expired before dispatch."""
        if self.free_slots <= 0:
            raise InvalidTransition("worker pool is full")
        task_state = self.web.task(task_id)
        if task_id not in self.ready_task_ids():
            raise InvalidTransition(f"task {task_id!r} is not ready to dispatch")
        if task_state.task.deadline_tick is not None and self.web.now >= task_state.task.deadline_tick:
            self.web.timeout_task(task_id, reason="Task deadline reached before dispatch")
            return False
        worker = self._workers.get(task_state.task.role)
        if worker is None:
            self.web.fail_task(task_id, reason=f"No worker registered for role {task_state.task.role!r}")
            return False

        started = self.web.start_task(task_id, worker.worker_id)
        inputs = self._dependency_artifacts(started)
        approval_granted = self._approval_grants.pop(task_id, None)
        context = WorkerContext(
            task=started.task,
            attempt=started.attempt,
            now=self.web.now,
            worker_id=worker.worker_id,
            inputs=inputs,
            approval_granted=approval_granted,
            job_id=self.job_id,
            episode_seed=self.episode_seed,
            policy_instructions=(
                POLICY_INSTRUCTIONS
                if self.governance_adapter.mode.value
                in {"prompt_policy", "policy_mirror", "armoriq_live"}
                else ""
            ),
        )
        gateway = GovernedToolGateway(
            context,
            adapter=self.governance_adapter,
            executor=self.tool_executor,
            event_log=self.web.events,
            now=lambda: self.web.now,
        )
        context = replace(context, tools=gateway)
        try:
            execution = worker.start(context)
            if not isinstance(execution, WorkerExecution):
                raise TypeError("Worker.start must return WorkerExecution")
        except ToolAuthorizationHold as hold:
            execution = WorkerExecution(
                hold=ApprovalHold(
                    scope=hold.scope,
                    reason=hold.reason,
                    origin=ApprovalOrigin.GOVERNANCE_GATE,
                    locally_resolvable=hold.locally_resolvable,
                    plan_id=hold.plan_id,
                    decision_id=hold.decision_id,
                    delegation_id=hold.delegation_id,
                )
            )
        except Exception as error:  # Keep one faulty worker from aborting the episode.
            execution = WorkerExecution(
                failure=WorkerFailure(f"Worker raised {type(error).__name__}: {error}", retryable=False)
            )
        timeout_at = self.web.now + started.task.timeout_ticks
        if started.task.deadline_tick is not None:
            timeout_at = min(timeout_at, started.task.deadline_tick)
        active = _ActiveWork(
            task_id=task_id,
            worker_id=worker.worker_id,
            execution=execution,
            ready_at=self.web.now + execution.delay_ticks,
            timeout_at=timeout_at,
        )
        self._active[task_id] = active
        self.web.events.append(
            tick=self.web.now,
            kind="worker.dispatched",
            task_id=task_id,
            actor_id=worker.worker_id,
            payload={"role": worker.role, "attempt": started.attempt},
        )
        return True

    def dispatch_ready(self, limit: int | None = None) -> tuple[str, ...]:
        slots = self.free_slots if limit is None else min(self.free_slots, max(0, limit))
        if slots == 0:
            return ()
        ready = self.ready_tasks()
        selected = self.controller.select_ready(ready, now=self.web.now, slots=slots)
        dispatched: list[str] = []
        for task_id in selected:
            if len(dispatched) >= slots:
                break
            if self.dispatch(task_id):
                dispatched.append(task_id)
        return tuple(dispatched)

    def resolve_approval(self, request_id: str, *, granted: bool, note: str | None = None) -> None:
        self._expire_at_current_time()
        request = self.web.approval(request_id)
        if not request.locally_resolvable:
            raise InvalidTransition(
                f"approval {request_id} is controlled by {request.origin.value}; "
                "local approval cannot resolve this hold"
            )
        request = self.web.resolve_approval(request_id, granted=granted, note=note)
        if granted:
            self._approval_grants[request.task_id] = True
            self.web.resume_approved_task(request.task_id, request_id=request_id)
        else:
            task = self.web.task(request.task_id)
            if task.status == TaskStatus.WAITING_APPROVAL:
                self.web.fail_task(request.task_id, reason=f"Approval denied: {note or request.scope}")

    def cancel(self, task_id: str, *, reason: str = "Cancelled by controller") -> None:
        self._active.pop(task_id, None)
        self._approval_grants.pop(task_id, None)
        self.web.cancel_task(task_id, reason=reason)

    def retry_failed(self, task_id: str, *, reason: str = "Retry requested by controller") -> None:
        self.web.reopen_failed_task(task_id, reason=reason)

    def advance_to_next_event(self) -> SchedulerResult:
        """Advance to one completion, timeout, retry, or deadline event.

        Waiting approval requests have no automatic wakeup.  A caller may
        resolve a locally resolvable synthetic review hold through
        ``resolve_approval`` while other work is processed normally. Live
        Governance holds from ArmorIQ remain pending until their deadline. This
        scheduler does not yet implement the external ArmorIQ hold resolver.
        """
        self._expire_at_current_time()
        if self.all_terminal:
            return SchedulerResult(self.web.now, False, terminal=True)
        next_tick = self._next_wakeup()
        if next_tick is None:
            blocked = tuple(request.task_id for request in self.web.pending_approvals())
            return SchedulerResult(
                self.web.now,
                False,
                blocked_on_approval=blocked,
                terminal=self.all_terminal,
            )
        previous = self.web.now
        self.web.advance_to(next_tick)
        completed = self._process_due_work()
        self._expire_at_current_time()
        return SchedulerResult(
            now=self.web.now,
            progressed=self.web.now > previous or bool(completed),
            completed=completed,
            blocked_on_approval=tuple(request.task_id for request in self.web.pending_approvals()),
            terminal=self.all_terminal,
        )

    def step(self) -> SchedulerResult:
        """One baseline scheduler turn: dispatch, then advance one event."""
        self._expire_at_current_time()
        if self.all_terminal:
            return SchedulerResult(self.web.now, False, terminal=True)
        dispatched = self.dispatch_ready()
        advanced = self.advance_to_next_event()
        return SchedulerResult(
            now=self.web.now,
            progressed=bool(dispatched) or advanced.progressed,
            dispatched=dispatched,
            completed=advanced.completed,
            blocked_on_approval=advanced.blocked_on_approval,
            terminal=advanced.terminal,
        )

    def run_until_blocked(self, *, max_events: int = 10_000) -> SchedulerResult:
        """Run the rule baseline until terminal, awaiting approval, or budget."""
        if max_events < 1:
            raise ValueError("max_events must be positive")
        last = SchedulerResult(self.web.now, False, terminal=self.all_terminal)
        for _ in range(max_events):
            last = self.step()
            if last.terminal or (not last.progressed and last.blocked_on_approval):
                return last
            if not last.progressed:
                return last
        return last

    def _dependency_artifacts(self, task: TaskSnapshot) -> Mapping[str, object]:
        artifacts = {}
        for dependency_id in task.task.dependencies:
            dependency = self.web.task(dependency_id)
            for ref in dependency.output_refs:
                artifacts[ref.key] = self.web.workspace.resolve(ref)
        return artifacts

    def _process_due_work(self) -> tuple[str, ...]:
        completed: list[str] = []
        # Timeouts win ties with completion deadlines by design.
        self._expire_at_current_time()
        for task_id, active in sorted(tuple(self._active.items())):
            if task_id not in self._active:
                continue
            if active.ready_at > self.web.now:
                continue
            if self.web.now >= active.timeout_at:
                self._active.pop(task_id, None)
                self.web.timeout_task(task_id, reason="Worker exceeded its virtual-time limit")
                completed.append(task_id)
                continue
            self._active.pop(task_id, None)
            execution = active.execution
            if execution.failure is not None:
                self._handle_failure(task_id, active.worker_id, execution.failure)
            elif execution.hold is not None:
                task = self.web.task(task_id)
                scope = execution.hold.scope or task.task.approval_scope or "unspecified"
                self.web.hold_task(
                    task_id,
                    scope=scope,
                    reason=execution.hold.reason,
                    origin=execution.hold.origin,
                    locally_resolvable=execution.hold.locally_resolvable,
                    plan_id=execution.hold.plan_id,
                    decision_id=execution.hold.decision_id,
                    delegation_id=execution.hold.delegation_id,
                )
            else:
                try:
                    self.web.complete_task(task_id, execution.artifacts, worker_id=active.worker_id)
                    completed.append(task_id)
                except Exception as error:
                    self._handle_failure(
                        task_id,
                        active.worker_id,
                        WorkerFailure(f"Invalid worker output: {type(error).__name__}: {error}", False),
                    )
                    completed.append(task_id)
        self._fail_tasks_with_failed_dependencies()
        return tuple(completed)

    def _handle_failure(self, task_id: str, worker_id: str, failure: WorkerFailure) -> None:
        task = self.web.task(task_id)
        reason = failure.reason
        if failure.retryable and task.attempt < task.task.max_attempts:
            retry_at = self.web.now + self.config.retry_backoff_ticks
            self.web.retry_task(task_id, retry_at=retry_at, reason=reason)
            self.web.events.append(
                tick=self.web.now,
                kind="worker.retry_scheduled",
                task_id=task_id,
                actor_id=worker_id,
                payload={"attempt": task.attempt, "retry_at": retry_at, "reason": reason},
            )
        else:
            self.web.fail_task(task_id, reason=reason, worker_id=worker_id)

    def _expire_at_current_time(self) -> None:
        for request in self.web.pending_approvals():
            if request.deadline_tick is not None and self.web.now >= request.deadline_tick:
                self.web.close_approval(
                    request.request_id,
                    status=ApprovalStatus.EXPIRED,
                    note="Approval deadline reached",
                )
                self._active.pop(request.task_id, None)
                snapshot = self.web.task(request.task_id)
                if snapshot.status == TaskStatus.WAITING_APPROVAL:
                    self.web.timeout_task(request.task_id, reason="Approval deadline reached")
        for snapshot in self.web.tasks():
            if snapshot.status in TERMINAL_TASK_STATUSES:
                continue
            deadline = snapshot.task.deadline_tick
            active = self._active.get(snapshot.task.task_id)
            timeout_at = active.timeout_at if active is not None else None
            reached_task_deadline = deadline is not None and self.web.now >= deadline
            reached_attempt_timeout = timeout_at is not None and self.web.now >= timeout_at
            if reached_task_deadline or reached_attempt_timeout:
                self._active.pop(snapshot.task.task_id, None)
                reason = (
                    "Task deadline reached"
                    if reached_task_deadline
                    else "Worker exceeded its virtual-time limit"
                )
                self.web.timeout_task(snapshot.task.task_id, reason=reason)
        self._fail_tasks_with_failed_dependencies()

    def _fail_tasks_with_failed_dependencies(self) -> None:
        for snapshot in self.web.tasks():
            if snapshot.status != TaskStatus.QUEUED:
                continue
            dependencies = tuple(self.web.task(dep) for dep in snapshot.task.dependencies)
            failed = [
                dep.task.task_id
                for dep in dependencies
                if dep.status in TERMINAL_TASK_STATUSES and dep.status != TaskStatus.SUCCEEDED
            ]
            if failed:
                self.web.fail_task(
                    snapshot.task.task_id,
                    reason=f"Dependency did not succeed: {', '.join(failed)}",
                )

    def _next_wakeup(self) -> int | None:
        candidates: list[int] = []
        for active in self._active.values():
            candidates.extend((active.ready_at, active.timeout_at))
        for task in self.web.tasks():
            if task.status in TERMINAL_TASK_STATUSES:
                continue
            if task.task.deadline_tick is not None:
                candidates.append(task.task.deadline_tick)
            if task.status == TaskStatus.QUEUED and task.retry_at is not None:
                candidates.append(task.retry_at)
        for request in self.web.pending_approvals():
            if request.deadline_tick is not None:
                candidates.append(request.deadline_tick)
        future = [tick for tick in candidates if tick > self.web.now]
        return min(future) if future else None
