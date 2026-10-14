"""Proposal capture and review-gated repository integration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from .contracts import AgentTask
from .events import Event, EventStore
from .paths import PathSafetyError, PathScope, SecureRoot
from .repo_tools import (
    CheckResult, CheckRunner, FileChange, RepoToolError, collect_file_changes,
    render_changes, sha256_bytes,
)


class IntegrationError(RuntimeError):
    """Raised when a proposal is not safe or ready to integrate."""


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_id: str
    run_id: str
    task_id: str
    submitted_by: str
    digest: str
    changes: tuple[FileChange, ...]
    artifact_path: Path


class IntegrationQueue:
    """Owns the proposal -> checks -> independent review -> integration gate."""

    def __init__(self, repository: str | Path, state_dir: str | Path,
                 event_store: EventStore, check_runner: CheckRunner | None = None) -> None:
        self.repository = SecureRoot(repository)
        self.state_dir = Path(state_dir).resolve()
        self.proposals_dir = self.state_dir / "proposals"
        self._ensure_state_dir("proposals")
        self.events = event_store
        self.check_runner = check_runner or CheckRunner()

    def capture(
        self, *, run_id: str, task: AgentTask, baseline: str | Path,
        workspace: str | Path, actor: str,
    ) -> Proposal:
        if not actor.strip():
            raise IntegrationError("A submitting worker identity is required")
        actor = actor.strip()
        try:
            changes = collect_file_changes(baseline, workspace, PathScope(task.file_scopes))
        except (RepoToolError, PathSafetyError) as exc:
            raise IntegrationError(str(exc)) from exc
        if not changes:
            raise IntegrationError(f"Task {task.task_id!r} has no scoped changes to submit")
        binary = [change.path for change in changes if change.is_binary]
        if binary:
            raise IntegrationError(f"Binary changes require a future binary artifact adapter: {binary}")
        proposal_id = uuid.uuid4().hex
        body = {
            "proposal_id": proposal_id, "run_id": run_id, "task_id": task.task_id,
            "submitted_by": actor, "changes": [item.to_dict() for item in changes],
        }
        serialized = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        run_dir = self._ensure_state_dir("proposals", run_id)
        artifact = run_dir / f"{proposal_id}.json"
        with artifact.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
        self.events.append(
            run_id=run_id, task_id=task.task_id, event_type="task_submitted", actor=actor,
            payload={
                "proposal_id": proposal_id, "digest": digest,
                "paths": [item.path for item in changes],
                "artifact": str(artifact.relative_to(self.state_dir)),
            },
        )
        return Proposal(proposal_id, run_id, task.task_id, actor, digest, changes, artifact)

    def latest_proposal(self, run_id: str, task_id: str) -> Proposal:
        event = self.events.latest_by_type(run_id, task_id, ("task_submitted",))
        if event is None:
            raise IntegrationError(f"Task {task_id!r} has no submitted proposal")
        proposal_id = event.payload.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id.isalnum():
            raise IntegrationError("Invalid proposal id in event log")
        run_dir = self._ensure_state_dir("proposals", run_id, create=False)
        artifact = run_dir / f"{proposal_id}.json"
        if artifact.is_symlink() or not artifact.is_file():
            raise IntegrationError("Proposal artifact is missing or is a symlink")
        artifact = artifact.resolve(strict=True)
        try:
            artifact.relative_to(self.proposals_dir.resolve(strict=True))
        except ValueError:
            raise IntegrationError("Proposal artifact escaped its state directory") from None
        body = json.loads(artifact.read_text(encoding="utf-8"))
        changes = tuple(FileChange.from_dict(item) for item in body["changes"])
        serialized = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if digest != event.payload.get("digest"):
            raise IntegrationError("Proposal artifact does not match the append-only event digest")
        return Proposal(
            proposal_id, run_id, task_id, str(body["submitted_by"]), digest, changes, artifact
        )

    def run_checks(self, *, run_id: str, task: AgentTask, workspace: str | Path) -> tuple[CheckResult, ...]:
        proposal = self.latest_proposal(run_id, task.task_id)
        self._assert_workspace_matches(proposal, workspace, task)
        if not task.required_checks:
            raise IntegrationError(
                f"Task {task.task_id!r} has no required checks; integration requires at least one."
            )
        results: list[CheckResult] = []
        for check_id in task.required_checks:
            result = self.check_runner.run(check_id, workspace)
            results.append(result)
            self.events.append(
                run_id=run_id, task_id=task.task_id, event_type="check_result", actor="check-runner",
                payload={"proposal_id": proposal.proposal_id, **result.to_dict()},
            )
            if not result.passed:
                break
        return tuple(results)

    def review(self, *, run_id: str, task: AgentTask, reviewer: str,
               decision: str, note: str = "") -> Event:
        if decision not in ("approve", "reject"):
            raise IntegrationError("Review decision must be approve or reject")
        if not reviewer.strip():
            raise IntegrationError("Reviewer identity is required")
        reviewer = reviewer.strip()
        proposal = self.latest_proposal(run_id, task.task_id)
        if reviewer.casefold() == proposal.submitted_by.casefold():
            raise IntegrationError("The proposal author cannot provide its independent review")
        if decision == "approve" and not self.checks_passed(run_id, task, proposal):
            raise IntegrationError("Required checks must pass before a reviewer can approve")
        return self.events.append(
            run_id=run_id, task_id=task.task_id, event_type="review_recorded", actor=reviewer,
            payload={"proposal_id": proposal.proposal_id, "decision": decision, "note": note[:4000]},
        )

    def checks_passed(self, run_id: str, task: AgentTask, proposal: Proposal | None = None) -> bool:
        proposal = proposal or self.latest_proposal(run_id, task.task_id)
        if not task.required_checks:
            return False
        latest: dict[str, Event] = {}
        for event in self.events.events(run_id, task.task_id):
            if event.event_type == "check_result" and event.payload.get("proposal_id") == proposal.proposal_id:
                latest[str(event.payload.get("check_id"))] = event
        return all(
            check_id in latest and latest[check_id].payload.get("passed") is True
            for check_id in task.required_checks
        )

    def integrate(self, *, run_id: str, task: AgentTask, workspace: str | Path) -> tuple[str, ...]:
        proposal = self.latest_proposal(run_id, task.task_id)
        if not self.checks_passed(run_id, task, proposal):
            raise IntegrationError("All required checks must pass for the current proposal")
        reviews = [
            event for event in self.events.events(run_id, task.task_id)
            if event.event_type == "review_recorded"
            and event.payload.get("proposal_id") == proposal.proposal_id
        ]
        latest_review = reviews[-1] if reviews else None
        if latest_review is None or latest_review.payload.get("decision") != "approve":
            raise IntegrationError("An independent approval for the current proposal is required")
        if latest_review.actor.casefold() == proposal.submitted_by.casefold():
            raise IntegrationError("Proposal author cannot approve their own change")
        self._assert_workspace_matches(proposal, workspace, task)
        self._assert_repository_unchanged(proposal, task)
        paths = self._apply_changes(proposal, task)
        self.events.append(
            run_id=run_id, task_id=task.task_id, event_type="task_integrated", actor="integrator",
            payload={"proposal_id": proposal.proposal_id, "paths": list(paths), "digest": proposal.digest},
        )
        return paths

    def _assert_workspace_matches(self, proposal: Proposal, workspace: str | Path, task: AgentTask) -> None:
        try:
            current = collect_file_changes(
                self._baseline_for(proposal), workspace, PathScope(task.file_scopes)
            )
        except (RepoToolError, PathSafetyError) as exc:
            raise IntegrationError(str(exc)) from exc
        expected = [item.to_dict() for item in proposal.changes]
        actual = [item.to_dict() for item in current]
        if expected != actual:
            raise IntegrationError("Workspace changed after proposal capture; resume and submit a fresh proposal")

    def _baseline_for(self, proposal: Proposal) -> Path:
        # Baseline path is recorded in a controlled sidecar created by Runtime.
        try:
            return self._ensure_state_dir(
                "baselines", proposal.run_id, proposal.task_id, create=False
            )
        except IntegrationError as exc:
            raise IntegrationError("Task baseline snapshot is missing or unsafe") from exc

    def _ensure_state_dir(self, *parts: str, create: bool = True) -> Path:
        current = self.state_dir
        for part in parts:
            if part in ("", ".", "..") or "/" in part or "\\" in part:
                raise IntegrationError("Unsafe internal state path")
            current = current / part
            if create:
                try:
                    current.mkdir()
                except FileExistsError:
                    pass
            if current.is_symlink() or not current.is_dir():
                raise IntegrationError(f"Unsafe internal state directory: {current}")
        try:
            current.resolve(strict=True).relative_to(self.state_dir.resolve(strict=True))
        except (OSError, ValueError):
            raise IntegrationError("Internal proposal path escaped .dev_web") from None
        return current

    def _assert_repository_unchanged(self, proposal: Proposal, task: AgentTask) -> None:
        scope = PathScope(task.file_scopes)
        for change in proposal.changes:
            if not scope.allows(change.path):
                raise IntegrationError(f"Out-of-scope path in proposal: {change.path}")
            try:
                current_path = self.repository.resolve(change.path)
            except PathSafetyError as exc:
                raise IntegrationError(str(exc)) from exc
            if current_path.exists():
                if not current_path.is_file():
                    raise IntegrationError(f"Repository target is not a file: {change.path}")
                current_hash = sha256_bytes(current_path.read_bytes())
            else:
                current_hash = None
            if current_hash != change.before_sha256:
                raise IntegrationError(
                    f"Repository changed since task started at {change.path}; rebase/review the proposal first"
                )

    def _apply_changes(self, proposal: Proposal, task: AgentTask) -> tuple[str, ...]:
        scope = PathScope(task.file_scopes)
        targets: list[tuple[FileChange, Path]] = []
        for change in proposal.changes:
            if change.is_binary or (change.after_sha256 is not None and change.content is None):
                raise IntegrationError(f"Unsupported binary proposal: {change.path}")
            if change.after_sha256 is not None and sha256_bytes(
                (change.content or "").encode("utf-8")
            ) != change.after_sha256:
                raise IntegrationError(f"Proposal content hash mismatch: {change.path}")
            if not scope.allows(change.path):
                raise IntegrationError(f"Out-of-scope path in proposal: {change.path}")
            target = self.repository.resolve(change.path)
            targets.append((change, target))

        originals: dict[str, bytes | None] = {}
        staged: list[tuple[FileChange, Path]] = []
        touched: list[str] = []
        try:
            for change, target in targets:
                originals[change.path] = target.read_bytes() if target.exists() else None
            for change, target in targets:
                if change.after_sha256 is None:
                    continue
                SecureRoot(self.repository.root).ensure_parent(change.path)
                target = self.repository.resolve(change.path)
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{target.name}.devweb-", dir=target.parent, delete=False
                ) as stream:
                    stream.write((change.content or "").encode("utf-8"))
                    staged.append((change, Path(stream.name)))
            for change, target in targets:
                if change.after_sha256 is None:
                    if target.exists():
                        touched.append(change.path)
                        target.unlink()
            staged_by_path = {change.path: temp for change, temp in staged}
            for change, target in targets:
                if change.after_sha256 is not None:
                    touched.append(change.path)
                    os.replace(staged_by_path[change.path], target)
        except Exception as exc:
            # Best-effort rollback keeps a mid-apply filesystem error from
            # leaving a partially integrated multi-file proposal.
            for change, target in reversed(targets):
                if change.path not in touched:
                    continue
                old = originals[change.path]
                try:
                    if old is None:
                        if target.exists():
                            target.unlink()
                    else:
                        SecureRoot(self.repository.root).ensure_parent(change.path)
                        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, delete=False) as stream:
                            stream.write(old)
                            rollback = Path(stream.name)
                        os.replace(rollback, target)
                except OSError:
                    pass
            raise IntegrationError(f"Could not atomically apply proposal: {exc}") from exc
        finally:
            for _, temporary in staged:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return tuple(change.path for change, _ in targets)

    @staticmethod
    def render_diff(proposal: Proposal, baseline: str | Path) -> str:
        return render_changes(baseline, proposal.changes)
