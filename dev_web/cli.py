"""Command-line interface for the local manual-first coding-agent web."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .contracts import AgentTask, TaskGraph
from .integration import IntegrationError
from .providers import ProviderUnavailable
from .runtime import DevWebRuntime, RuntimeErrorWithGuidance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m dev_web",
        description=(
            "Create a local coding-task DAG, dispatch a manual assignment, capture a proposal, "
            "run allowlisted checks, require independent review, then integrate."
        ),
        epilog=(
            "Example: python3 -m dev_web run --provider manual --task 'Add a verifier' "
            "--scope 'verifier.py' --scope 'tests/' --check pytest\n"
            "Checks are configured as fixed argv entries in .dev_web/config.json; this runtime "
            "does not invoke native Codex conversation subagents."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="create a DAG run and dispatch ready tasks")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--dag", help="JSON file containing a typed task DAG")
    source.add_argument("--task", help="instructions for a single task")
    run.add_argument("--title", help="short task title (defaults to --task text)")
    run.add_argument("--task-id", default="task-1")
    run.add_argument("--graph-id", default="local-run")
    run.add_argument("--scope", action="append", default=[],
                     help="writable file or directory scope; repeat this option")
    run.add_argument("--check", dest="checks", action="append", default=[],
                     help="required named check from .dev_web/config.json; repeat this option")
    run.add_argument("--max-attempts", type=int, default=2,
                     help="maximum submitted proposals before this task needs a new run (default: 2)")
    run.add_argument("--provider", help="worker provider; only 'manual' is built in")
    run.add_argument("--actor", default="coordinator", help="identity recorded for the task author")

    status = commands.add_parser("status", help="show run and task states")
    status.add_argument("run_id", nargs="?", help="defaults to the latest run")

    resume = commands.add_parser("resume", help="capture changed workspaces and dispatch newly ready work")
    resume.add_argument("run_id", nargs="?", help="defaults to the latest run")
    resume.add_argument("--actor", default="manual-worker", help="identity submitting the current proposal")

    check = commands.add_parser("check", help="run required allowlisted checks for a proposal")
    check.add_argument("run_id")
    check.add_argument("task_id")

    diff = commands.add_parser("diff", help="show the current proposal patch before review")
    diff.add_argument("run_id")
    diff.add_argument("task_id")

    review = commands.add_parser("review", help="record independent review for the current proposal")
    review.add_argument("run_id")
    review.add_argument("task_id")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", required=True, choices=("approve", "reject"))
    review.add_argument("--note", default="")

    integrate = commands.add_parser("integrate", help="integrate a checked and independently approved proposal")
    integrate.add_argument("run_id")
    integrate.add_argument("task_id")
    return parser


def _single_task_graph(args: argparse.Namespace) -> TaskGraph:
    if not args.scope:
        raise RuntimeErrorWithGuidance("A single task needs at least one --scope file or directory")
    title = args.title or args.task.strip().splitlines()[0][:120]
    return TaskGraph(
        graph_id=args.graph_id,
        tasks=(AgentTask(
            task_id=args.task_id, title=title, instructions=args.task,
            file_scopes=tuple(args.scope), required_checks=tuple(args.checks), author=args.actor,
            max_attempts=args.max_attempts,
        ),),
    )


def _run_id(runtime: DevWebRuntime, requested: str | None) -> str:
    value = requested or runtime.latest_run_id()
    if value is None:
        raise RuntimeErrorWithGuidance("There are no runs yet. Start one with `python3 -m dev_web run`.")
    return value


def _print_status(status: dict[str, object]) -> None:
    print(f"Run {status['run_id']}  provider={status['provider']}")
    for task in status["tasks"]:
        deps = ", ".join(task["dependencies"]) or "—"
        print(f"  {task['task_id']:<20} {task['status']:<20} deps={deps}")
        if task.get("proposal_id"):
            print(f"    proposal={task['proposal_id']} checks={','.join(task['required_checks']) or '(none)'}")
        if task.get("workspace"):
            print(f"    workspace={task['workspace']}")
        if task.get("packet_path"):
            print(f"    packet={task['packet_path']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        with DevWebRuntime(args.root) as runtime:
            if args.command == "run":
                graph = runtime.run_from_json(args.dag, provider_id=args.provider, actor=args.actor) \
                    if args.dag else _create_single(runtime, args)
                print(f"Run created: {graph.run_id}")
                print(f"Provider: {graph.provider_id}")
                for event in runtime.events.events(graph.run_id):
                    if event.event_type == "task_queued":
                        print(
                            f"Task {event.task_id} queued: packet={event.payload.get('packet_path')} "
                            f"workspace={event.payload.get('workspace')}"
                        )
                return 0
            if args.command == "status":
                run_id = _run_id(runtime, args.run_id)
                _print_status(runtime.status(run_id))
                return 0
            if args.command == "resume":
                run_id = _run_id(runtime, args.run_id)
                proposals = runtime.resume(run_id, actor=args.actor)
                if proposals:
                    for proposal in proposals:
                        print(
                            f"Captured {proposal.proposal_id} for {proposal.task_id}: "
                            f"{', '.join(change.path for change in proposal.changes)}"
                        )
                        print(f"Review artifact: {proposal.artifact_path}")
                        print(runtime.proposal_diff(run_id, proposal.task_id), end="")
                else:
                    print("No proposals captured. Check the task packets and pending tasks in status.")
                _print_status(runtime.status(run_id))
                return 0
            if args.command == "diff":
                diff_text = runtime.proposal_diff(args.run_id, args.task_id)
                print(diff_text, end="" if diff_text.endswith("\n") else "\n")
                return 0
            if args.command == "check":
                results = runtime.check_task(args.run_id, args.task_id)
                for result in results:
                    state = "PASS" if result.passed else "FAIL"
                    print(f"{state} {result.check_id}")
                    if result.stdout:
                        print(result.stdout.rstrip())
                    if result.stderr:
                        print(result.stderr.rstrip(), file=sys.stderr)
                    if result.error:
                        print(result.error, file=sys.stderr)
                return 0 if results and all(result.passed for result in results) else 1
            if args.command == "review":
                event = runtime.review_task(
                    args.run_id, args.task_id, reviewer=args.reviewer,
                    decision=args.decision, note=args.note,
                )
                print(f"Review recorded: {event.payload['decision']} by {event.actor}")
                return 0
            if args.command == "integrate":
                paths = runtime.integrate_task(args.run_id, args.task_id)
                print("Integrated: " + ", ".join(paths))
                return 0
    except (RuntimeErrorWithGuidance, ProviderUnavailable, IntegrationError, OSError, ValueError) as exc:
        print(f"dev_web: {exc}", file=sys.stderr)
        return 2
    return 2


def _create_single(runtime: DevWebRuntime, args: argparse.Namespace):
    graph = _single_task_graph(args)
    return runtime.create_run(graph, provider_id=args.provider, actor=args.actor)


if __name__ == "__main__":
    raise SystemExit(main())
