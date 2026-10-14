# Local coding-agent web

This package is a manual-first coordinator for coding work. It does not invoke
native Codex conversation subagents. A real provider must implement the
`WorkerProvider` protocol and be explicitly registered; until then `manual`
creates an isolated assignment folder for a human or a separately operated
agent.

## Quick start

From the repository root, create `.dev_web/config.json` with fixed argv checks
that the project owner is willing to run in a task snapshot:

```json
{
  "checks": [
    { "check_id": "pytest", "argv": ["python3", "-m", "pytest", "-q"], "timeout_seconds": 300 }
  ]
}
```

Create a task and queue it manually:

```sh
python3 -m dev_web run --provider manual \
  --task "Implement the verifier result schema" \
  --scope "verifier.py" --scope "tests/" --check pytest
```

The command prints a run id and a `TASK.md` packet under `.dev_web/queue/`.
Work only in the task's isolated snapshot under `.dev_web/workspaces/`. Then:

```sh
python3 -m dev_web resume RUN_ID
python3 -m dev_web diff RUN_ID task-1
python3 -m dev_web check RUN_ID task-1
python3 -m dev_web review RUN_ID task-1 --reviewer second-person --decision approve
python3 -m dev_web integrate RUN_ID task-1
python3 -m dev_web status RUN_ID
```

For a DAG, pass `--dag path/to/tasks.json` to `run`. The file shape is:

```json
{
  "graph_id": "implementation",
  "metadata": { "milestone": "M1" },
  "tasks": [
    {
      "task_id": "contracts",
      "title": "Define stable contracts",
      "instructions": "Add and document the task result types.",
      "file_scopes": ["dev_web/contracts.py", "tests/"],
      "dependencies": [],
      "required_checks": ["pytest"],
      "max_attempts": 2,
      "author": "coordinator",
      "expected_outputs": ["typed contracts"]
    },
    {
      "task_id": "scheduler",
      "title": "Add task scheduling",
      "instructions": "Implement the ready-task scheduler.",
      "file_scopes": ["dev_web/runtime.py"],
      "dependencies": ["contracts"],
      "required_checks": ["pytest"],
      "max_attempts": 2,
      "author": "coordinator",
      "expected_outputs": ["scheduler"]
    }
  ]
}
```

Directory scopes end in `/`; other scopes name an exact path. Changed files
outside the declared scopes, symlink paths, common secret files, and binary
changes are rejected. Proposal integration is refused unless every named
allowlisted check passes and a reviewer whose recorded identity differs from
the submitter approves that exact proposal. The repository is also checked for
concurrent edits before applying it. Reviewer and submitter names are local,
self-declared identities; a human process or authenticated provider must ensure
they represent separate people.

Checks are run as fixed argv vectors (`shell=False`) from the isolated snapshot.
Only a small execution environment is forwarded, and common credential files
are omitted from snapshots. This is a local coordination and review boundary,
not a hostile-code OS sandbox; configure checks conservatively. The package
itself makes no network calls and has no credential-bearing provider.
