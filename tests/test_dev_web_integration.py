from pathlib import Path
import sys

import pytest

from dev_web.contracts import AgentTask
from dev_web.events import EventStore
from dev_web.integration import IntegrationError, IntegrationQueue
from dev_web.repo_tools import CheckRunner, CheckSpec


def test_integration_requires_checks_and_independent_review(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    state = tmp_path / "state"
    baseline = state / "baselines" / "run-1" / "task-1"
    workspace = state / "workspaces" / "run-1" / "task-1"
    baseline.mkdir(parents=True)
    workspace.mkdir(parents=True)
    (baseline / "module.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    (workspace / "module.py").write_text("VALUE = 'after'\n", encoding="utf-8")
    task = AgentTask(
        task_id="task-1", title="Update value", instructions="Change value.",
        file_scopes=("module.py",), required_checks=("syntax",),
    )
    runner = CheckRunner((CheckSpec("syntax", (sys.executable, "-c", "print('ok')")),))

    with EventStore(state / "events.sqlite3") as events:
        queue = IntegrationQueue(repo, state, events, runner)
        proposal = queue.capture(
            run_id="run-1", task=task, baseline=baseline,
            workspace=workspace, actor="worker-a",
        )
        with pytest.raises(IntegrationError, match="checks must pass"):
            queue.integrate(run_id="run-1", task=task, workspace=workspace)
        with pytest.raises(IntegrationError, match="checks must pass"):
            queue.review(run_id="run-1", task=task, reviewer="reviewer-b", decision="approve")
        assert queue.run_checks(run_id="run-1", task=task, workspace=workspace)[0].passed
        with pytest.raises(IntegrationError, match="cannot provide its independent review"):
            queue.review(run_id="run-1", task=task, reviewer="worker-a", decision="approve")
        queue.review(run_id="run-1", task=task, reviewer="reviewer-b", decision="approve")
        assert queue.integrate(run_id="run-1", task=task, workspace=workspace) == ("module.py",)

    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 'after'\n"
