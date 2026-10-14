"""Run lifecycle for manual/provider-backed coding-agent task graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import uuid

from .contracts import AgentTask, ContractError, TaskGraph, TaskStatus
from .events import Event, EventStore
from .integration import IntegrationError, IntegrationQueue, Proposal
from .paths import PathSafetyError, PathScope, SecureRoot, copy_safe_tree
from .providers import (
    DispatchResult, UnavailableProvider, WorkerProvider, get_provider,
)
from .repo_tools import CheckRunner, RepoToolError, collect_file_changes


class RuntimeErrorWithGuidance(RuntimeError):
    """A user-actionable lifecycle error."""


@dataclass(frozen=True, slots=True)
class RunHandle:
    run_id: str
    graph: TaskGraph
    provider_id: str


class DevWebRuntime:
    """Coordinator for DAG dispatch, manual workspaces, checks, review, and merge."""

    def __init__(self, repository: str | Path, *, check_runner: CheckRunner | None = None) -> None:
        self.repository = SecureRoot(repository)
        self.repo_root = self.repository.root
        self.state_dir = self.repo_root / ".dev_web"
        if self.state_dir.is_symlink():
            raise RuntimeErrorWithGuidance(".dev_web cannot be a symlink")
        self.state_dir.mkdir(exist_ok=True)
        if not self.state_dir.is_dir():
            raise RuntimeErrorWithGuidance(".dev_web must be a directory")
        event_path = self.state_dir / "events.sqlite3"
        if event_path.is_symlink():
            raise RuntimeErrorWithGuidance(".dev_web/events.sqlite3 cannot be a symlink")
        self.events = EventStore(event_path)
        self.check_runner = check_runner or self._load_checks()
        self.integrations = IntegrationQueue(
            self.repo_root, self.state_dir, self.events, self.check_runner
        )

    def _load_checks(self) -> CheckRunner:
        config = self.state_dir / "config.json"
        if not config.exists():
            return CheckRunner()
        try:
            return CheckRunner.from_config(config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeErrorWithGuidance(f"Invalid .dev_web/config.json: {exc}") from exc

    def create_run(
        self, graph: TaskGraph, *, provider_id: str | None, actor: str = "coordinator"
    ) -> RunHandle:
        provider = get_provider(provider_id)
        if isinstance(provider, UnavailableProvider):
            # Fail before creating any run state; this runtime has no native
            # conversation-subagent bridge and must not imply one exists.
            provider.ensure_available()
        run_id = uuid.uuid4().hex
        self.events.append(
            run_id=run_id, event_type="run_created", actor=actor,
            payload={"graph": graph.to_dict(), "provider_id": provider.provider_id},
        )
        self._dispatch_ready(run_id, graph, provider)
        return RunHandle(run_id, graph, provider.provider_id)

    def run_from_json(self, path: str | Path, *, provider_id: str | None,
                      actor: str = "coordinator") -> RunHandle:
        graph_path = Path(path).expanduser().resolve(strict=True)
        try:
            graph = TaskGraph.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ContractError) as exc:
            raise RuntimeErrorWithGuidance(f"Could not load a valid task DAG: {exc}") from exc
        return self.create_run(graph, provider_id=provider_id, actor=actor)

    def _dispatch_ready(self, run_id: str, graph: TaskGraph,
                        provider: WorkerProvider | None = None) -> tuple[DispatchResult, ...]:
        provider = provider or get_provider(self._run_provider(run_id))
        task_events = self.events.events(run_id)
        completed = {
            event.task_id for event in task_events
            if event.event_type == "task_integrated" and event.task_id is not None
        }
        known_queued = {
            event.task_id for event in task_events
            if event.event_type == "task_queued" and event.task_id is not None
        }
        results: list[DispatchResult] = []
        for task_id in graph.ready_task_ids(completed):
            if task_id in known_queued:
                continue
            task = self.task(graph, task_id)
            baseline, workspace = self._task_paths(run_id, task_id)
            self._ensure_state_dir("baselines", run_id)
            self._ensure_state_dir("workspaces", run_id)
            try:
                copy_safe_tree(self.repo_root, baseline)
                copy_safe_tree(self.repo_root, workspace)
            except (OSError, PathSafetyError) as exc:
                raise RuntimeErrorWithGuidance(f"Could not prepare isolated task workspace: {exc}") from exc
            self._ensure_state_dir("queue", run_id, task_id)
            packet_dir = self.state_dir / "queue" / run_id / task_id
            result = provider.dispatch(task, workspace, packet_dir)
            self.events.append(
                run_id=run_id, task_id=task_id, event_type="task_queued", actor=result.actor,
                payload={
                    "provider_id": provider.provider_id, "workspace": str(workspace),
                    "baseline": str(baseline), "packet_path": result.packet_path,
                    "message": result.message,
                },
            )
            results.append(result)
        return tuple(results)

    def resume(self, run_id: str, *, actor: str = "manual-worker") -> tuple[Proposal, ...]:
        graph = self.graph(run_id)
        proposals: list[Proposal] = []
        latest_events = self.events.events(run_id)
        integrated = {
            event.task_id for event in latest_events
            if event.event_type == "task_integrated" and event.task_id is not None
        }
        queued = {
            event.task_id for event in latest_events
            if event.event_type == "task_queued" and event.task_id is not None
        }
        for task_id in graph.topological_order():
            if task_id not in queued or task_id in integrated:
                continue
            task = self.task(graph, task_id)
            baseline, workspace = self._task_paths(run_id, task_id)
            try:
                changes = collect_file_changes(baseline, workspace, PathScope(task.file_scopes))
            except (RepoToolError, PathSafetyError) as exc:
                raise RuntimeErrorWithGuidance(str(exc)) from exc
            if not changes:
                continue
            latest = self.events.latest_by_type(run_id, task_id, ("task_submitted",))
            if latest:
                current_proposal = self.integrations.latest_proposal(run_id, task_id)
                if [change.to_dict() for change in changes] == [
                    change.to_dict() for change in current_proposal.changes
                ]:
                    continue
                attempts = sum(
                    event.event_type == "task_submitted" for event in latest_events
                    if event.task_id == task_id
                )
                if attempts >= task.max_attempts:
                    raise RuntimeErrorWithGuidance(
                        f"Task {task_id!r} exhausted its {task.max_attempts} proposal attempts; "
                        "create a new run with an updated contract to continue."
                    )
            proposals.append(self.integrations.capture(
                run_id=run_id, task=task, baseline=baseline, workspace=workspace, actor=actor
            ))
        provider = get_provider(self._run_provider(run_id))
        self._dispatch_ready(run_id, graph, provider)
        return tuple(proposals)

    def check_task(self, run_id: str, task_id: str):
        task = self.task(self.graph(run_id), task_id)
        _, workspace = self._task_paths(run_id, task_id)
        try:
            return self.integrations.run_checks(run_id=run_id, task=task, workspace=workspace)
        except (IntegrationError, RepoToolError) as exc:
            raise RuntimeErrorWithGuidance(str(exc)) from exc

    def proposal_diff(self, run_id: str, task_id: str) -> str:
        self.task(self.graph(run_id), task_id)
        proposal = self.integrations.latest_proposal(run_id, task_id)
        baseline, _ = self._task_paths(run_id, task_id)
        try:
            return self.integrations.render_diff(proposal, baseline)
        except (IntegrationError, PathSafetyError) as exc:
            raise RuntimeErrorWithGuidance(str(exc)) from exc

    def review_task(self, run_id: str, task_id: str, *, reviewer: str, decision: str, note: str = "") -> Event:
        task = self.task(self.graph(run_id), task_id)
        try:
            return self.integrations.review(
                run_id=run_id, task=task, reviewer=reviewer, decision=decision, note=note
            )
        except IntegrationError as exc:
            raise RuntimeErrorWithGuidance(str(exc)) from exc

    def integrate_task(self, run_id: str, task_id: str) -> tuple[str, ...]:
        task = self.task(self.graph(run_id), task_id)
        _, workspace = self._task_paths(run_id, task_id)
        try:
            paths = self.integrations.integrate(run_id=run_id, task=task, workspace=workspace)
        except IntegrationError as exc:
            raise RuntimeErrorWithGuidance(str(exc)) from exc
        # Dependencies become eligible only after this integration event.
        self._dispatch_ready(run_id, self.graph(run_id))
        return paths

    def graph(self, run_id: str) -> TaskGraph:
        event = next((event for event in self.events.events(run_id)
                      if event.event_type == "run_created"), None)
        if event is None:
            raise RuntimeErrorWithGuidance(f"Unknown run id: {run_id}")
        try:
            return TaskGraph.from_dict(event.payload["graph"])
        except (KeyError, TypeError, ContractError) as exc:
            raise RuntimeErrorWithGuidance(f"Stored task graph is invalid: {exc}") from exc

    @staticmethod
    def task(graph: TaskGraph, task_id: str) -> AgentTask:
        for task in graph.tasks:
            if task.task_id == task_id:
                return task
        raise RuntimeErrorWithGuidance(f"Unknown task id: {task_id}")

    def status(self, run_id: str) -> dict[str, object]:
        graph = self.graph(run_id)
        events = self.events.events(run_id)
        task_rows: list[dict[str, object]] = []
        for task in graph.tasks:
            task_events = [event for event in events if event.task_id == task.task_id]
            queued_event = next(
                (event for event in task_events if event.event_type == "task_queued"), None
            )
            latest_submission = next(
                (event for event in reversed(task_events) if event.event_type == "task_submitted"), None
            )
            if any(event.event_type == "task_integrated" and
                   (latest_submission is None or event.sequence > latest_submission.sequence)
                   for event in task_events):
                status = TaskStatus.INTEGRATED
            elif latest_submission is None:
                status = TaskStatus.QUEUED if any(e.event_type == "task_queued" for e in task_events) else TaskStatus.BLOCKED
            else:
                proposal_id = latest_submission.payload.get("proposal_id")
                check_events = [e for e in task_events if e.event_type == "check_result"
                                and e.payload.get("proposal_id") == proposal_id]
                latest_checks = {str(e.payload.get("check_id")): e for e in check_events}
                checks_ok = bool(task.required_checks) and all(
                    check_id in latest_checks and latest_checks[check_id].payload.get("passed") is True
                    for check_id in task.required_checks
                )
                reviews = [e for e in task_events if e.event_type == "review_recorded"
                           and e.payload.get("proposal_id") == proposal_id]
                review = reviews[-1] if reviews else None
                if review and review.payload.get("decision") == "reject":
                    status = TaskStatus.REVIEW_REJECTED
                elif review and review.payload.get("decision") == "approve" and checks_ok:
                    status = TaskStatus.REVIEW_APPROVED
                elif any(not item.payload.get("passed") for item in latest_checks.values()):
                    count = sum(e.event_type == "task_submitted" for e in task_events)
                    status = TaskStatus.FAILED if count >= task.max_attempts else TaskStatus.CHECKS_FAILED
                elif checks_ok:
                    status = TaskStatus.CHECKS_PASSED
                else:
                    status = TaskStatus.SUBMITTED
            task_rows.append({
                "task_id": task.task_id, "title": task.title, "status": status.value,
                "dependencies": list(task.dependencies), "required_checks": list(task.required_checks),
                "proposal_id": latest_submission.payload.get("proposal_id") if latest_submission else None,
                "workspace": queued_event.payload.get("workspace") if queued_event else None,
                "packet_path": queued_event.payload.get("packet_path") if queued_event else None,
            })
        return {"run_id": run_id, "provider": self._run_provider(run_id), "tasks": task_rows}

    def latest_run_id(self) -> str | None:
        for event in reversed(self.events.events()):
            if event.event_type == "run_created":
                return event.run_id
        return None

    def _run_provider(self, run_id: str) -> str:
        event = next((event for event in self.events.events(run_id)
                      if event.event_type == "run_created"), None)
        if event is None:
            raise RuntimeErrorWithGuidance(f"Unknown run id: {run_id}")
        return str(event.payload.get("provider_id", "unconfigured"))

    def _task_paths(self, run_id: str, task_id: str) -> tuple[Path, Path]:
        # IDs are validated contract identifiers before reaching this helper.
        paths = (
            self.state_dir / "baselines" / run_id / task_id,
            self.state_dir / "workspaces" / run_id / task_id,
        )
        for path in paths:
            current = self.state_dir
            for part in path.relative_to(self.state_dir).parts:
                current = current / part
                if current.is_symlink():
                    raise RuntimeErrorWithGuidance(f"Task state path cannot be a symlink: {current}")
                if current.exists() and not current.is_dir():
                    raise RuntimeErrorWithGuidance(f"Task state path is not a directory: {current}")
        return paths

    def _ensure_state_dir(self, *parts: str) -> Path:
        current = self.state_dir
        for part in parts:
            if part in ("", ".", "..") or "/" in part or "\\" in part:
                raise RuntimeErrorWithGuidance("Unsafe internal state path")
            current = current / part
            try:
                current.mkdir()
            except FileExistsError:
                pass
            if current.is_symlink() or not current.is_dir():
                raise RuntimeErrorWithGuidance(f"Unsafe internal state directory: {current}")
        try:
            current.resolve(strict=True).relative_to(self.state_dir.resolve(strict=True))
        except (OSError, ValueError):
            raise RuntimeErrorWithGuidance("Internal state path escaped .dev_web") from None
        return current

    def close(self) -> None:
        self.events.close()

    def __enter__(self) -> DevWebRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
