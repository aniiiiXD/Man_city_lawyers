"""Worker provider contract and honest local/manual provider implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
import json

from .contracts import AgentTask


class ProviderUnavailable(RuntimeError):
    """Raised when the requested worker integration is not installed."""


@dataclass(frozen=True, slots=True)
class DispatchResult:
    state: str
    actor: str
    message: str
    packet_path: str | None = None


class WorkerProvider(Protocol):
    """Interface for a real coding worker adapter.

    Implementations must receive an isolated workspace and return only a
    completion state; proposals still pass through checks and human review.
    """

    provider_id: str

    def dispatch(self, task: AgentTask, workspace: Path, packet_dir: Path) -> DispatchResult:
        """Dispatch one task or return a clear unavailable/error result."""


class ManualQueueProvider:
    """Create a human-readable assignment packet; this is not an AI worker."""

    provider_id = "manual"

    def dispatch(self, task: AgentTask, workspace: Path, packet_dir: Path) -> DispatchResult:
        packet_dir.mkdir(parents=True, exist_ok=True)
        run_id = packet_dir.parent.name
        packet = {
            "run_id": run_id,
            "task": task.to_dict(),
            "workspace": str(workspace.resolve()),
            "instructions": (
                "Edit only files within file_scopes. Do not copy credentials into this workspace. "
                f"When finished, run `python3 -m dev_web resume {run_id}` from the source repository. "
                "The coordinator will capture a proposal; checks and an independent review are still required."
            ),
        }
        (packet_dir / "task.json").write_text(
            json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        dependencies = [f"- `{item}`" for item in task.dependencies] or ["- None"]
        checks = [f"- `{item}`" for item in task.required_checks] or [
            "- None configured; integration will remain blocked."
        ]
        workflow = [
            f"python3 -m dev_web resume {run_id}",
            f"python3 -m dev_web diff {run_id} {task.task_id}",
            f"python3 -m dev_web check {run_id} {task.task_id}",
            f"python3 -m dev_web review {run_id} {task.task_id} --reviewer REVIEWER --decision approve",
            f"python3 -m dev_web integrate {run_id} {task.task_id}",
        ]
        lines = [
            f"# {task.title}", "", f"Task: `{task.task_id}`", "", task.instructions, "",
            f"Run: `{run_id}`", "",
            "## File scope", *[f"- `{scope}`" for scope in task.file_scopes], "",
            "## Dependencies", *dependencies, "",
            "## Required checks", *checks, "",
            "## Workspace", str(workspace.resolve()), "",
            "## Completion workflow", "", *[f"1. `{command}`" for command in workflow], "",
            "Work only in this isolated snapshot. The coordinator rejects out-of-scope changes. "
            "The result is not integrated until required checks pass and a different reviewer approves it.",
        ]
        (packet_dir / "TASK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return DispatchResult(
            state="queued", actor="manual-queue",
            message="Manual assignment packet created; no agent was invoked.",
            packet_path=str(packet_dir / "TASK.md"),
        )


class UnavailableProvider:
    """Explicitly fail rather than falsely claiming to call native subagents."""

    def __init__(self, requested: str | None = None) -> None:
        self.provider_id = requested or "unconfigured"

    def ensure_available(self) -> None:
        raise ProviderUnavailable(
            f"No worker provider is configured for {self.provider_id!r}. This runtime cannot call "
            "native Codex conversation subagents. Use --provider manual for a local assignment queue, "
            "or install and explicitly register a real WorkerProvider adapter."
        )

    def dispatch(self, task: AgentTask, workspace: Path, packet_dir: Path) -> DispatchResult:
        self.ensure_available()
        raise AssertionError("ensure_available must raise")


_REGISTERED_PROVIDERS: dict[str, Callable[[], WorkerProvider]] = {}


def register_provider(provider_id: str, factory: Callable[[], WorkerProvider]) -> None:
    """Explicitly register an in-process provider adapter for embedding apps."""
    if provider_id in ("", "manual") or not provider_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Provider id must be an alphanumeric non-manual name")
    if not callable(factory):
        raise TypeError("Provider factory must be callable")
    _REGISTERED_PROVIDERS[provider_id] = factory


def get_provider(provider_id: str | None) -> WorkerProvider:
    if provider_id == "manual":
        return ManualQueueProvider()
    if provider_id in _REGISTERED_PROVIDERS:
        provider = _REGISTERED_PROVIDERS[provider_id]()
        if provider.provider_id != provider_id:
            raise ProviderUnavailable(
                f"Registered provider factory {provider_id!r} returned {provider.provider_id!r}."
            )
        return provider
    return UnavailableProvider(provider_id)
