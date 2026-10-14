# ArmorEnv System Plan: Governed RL for Solo Agents and Async Teams

## Implementation status

Core code and focused test files are now in the repository. Tests and benchmark
runs have **not** been executed yet, so implementation presence does not mean
an exit gate has passed.

| Milestone | Current state | Remaining exit work |
| --- | --- | --- |
| M0 | Experiment labels and metrics are documented. | Confirm fixed policy profiles and the live mock-MCP setup. |
| M1 | Seeded solo world, generic path, prompt-only path, verifier, and mirror path are implemented. | Run module and replay checks. |
| M2 | Solo and team Python SDK adapters use plan → token → invoke; team live ALLOW requires the remote result and ArmorIQ holds cannot be cleared locally. External hold resumption is not wired. | Verify SDK behavior against the registered mock MCP and compare mirror/live decisions. |
| M3 | Scripted solo and team baselines plus separate reward components are implemented. | Run pilots and freeze weights from evidence. |
| M4 | Solo and async-orchestrator GRPO/LoRA training code is implemented. | Run training, record checkpoints/configs, and inspect reward behavior. |
| M5 | Paired solo and team evaluators with bootstrap intervals are implemented. | Run held-out task suites and the smaller live evaluation. |
| M6 | Solo and team CLI demos exist. | Add a timeline UI and the later Manchester City showcase pack. |
| M7–M9 | Event-sourced Work Web, bounded scheduler, synthetic tool executor, worker-level mirror gate, approvals, and verifier are implemented. | Run concurrency/replay tests and verify live ArmorIQ worker calls. |
| M10 | Team reward, trajectory attribution, trainer, and evaluator are implemented. | Train/evaluate under matched compute and report uncertainty. |
| M11–M12 | Future work. | Build a timeline UI first; consider worker learning only after M10 is reliable. |
| E0 | Native Codex subagents implemented independent modules under root coordination. | Complete tests and record an integrated review trail. |
| E1 | `dev_web/` implements manual task DAGs, isolated snapshots, scoped repository tools, checks, review, and integration. | Connect a supported coding-agent provider; run the workflow end to end. |
| E2–E4 | Future work. | Dogfood only after a provider exists; then evaluate a governed coding-agent path. |

The Python `dev_web` runtime does not invoke native Codex conversation
subagents. Native agents were used interactively for this implementation; a
Python provider bridge is a separate E1 requirement.

## Product thesis

ArmorEnv is a set of reusable, stateful RL environments for teaching individual
agents and asynchronous agent teams to complete tool-using tasks inside
changing authority boundaries. Its ambitious environment is a shared
agent-operated work web: one learned coordinator directs a changing pool of
specialist agents while jobs, records, messages, and approvals evolve. The
first research question is:

> Does training with ArmorIQ's intent verification and runtime policy decisions
> help an agent complete more tasks safely and recover better from holds and
> blocks than training in a non-ArmorIQ environment?

ArmorIQ is the governed execution boundary: its SDK binds tool invocations to
captured plans, checks calls before they reach MCP tools, supports policy
outcomes such as allow / hold / block, attributes calls to users, and records
plan execution traces. ArmorIQ does not train the model or independently
determine task success. The environment and independent verifier own task
state and task reward.

## The distinction we will demonstrate

Both environments use the same task definitions, simulated world, MCP tools,
agent model, and task verifier. Their action paths differ deliberately.

```text
GENERIC ENVIRONMENT
Agent action -> MCP tool executes -> world changes -> task feedback

ARMORIQ ENVIRONMENT
Agent action -> intent / identity -> ArmorIQ policy + plan verification
                                      | allow -> MCP tool executes -> world changes
                                      | hold  -> approval flow; no execution yet
                                      | block -> no execution; world stays unchanged
                                      v
                          policy result + audit trace + task feedback
```

The non-ArmorIQ condition must not secretly use the ArmorIQ gate or return its
decision details. For the strongest control, the prompt-only arm receives the
same natural-language policy description as ArmorRL, while its simulated tools
remain ungated. That lets us test whether runtime enforcement and structured
decision feedback add value beyond telling the model the rules.

### Execution labels

Use explicit names in code, logs, charts, and the demo:

| Name | What happens | Permitted claim |
| --- | --- | --- |
| `generic` | Tools execute directly in an isolated simulated world; no policy feedback is shown to the agent. | Generic task-RL baseline |
| `prompt_policy` | Same world, with policy text in the prompt, but no runtime gate. | Tests whether written instructions are enough |
| `policy_mirror` | Fast local policy implementation based on the same versioned profile as ArmorIQ. | Training proxy only; label every decision as a mirror result |
| `armoriq_live` | Calls go through the ArmorIQ SDK and registered simulated MCP. | Live ArmorIQ enforcement and audit evidence |

Never label `policy_mirror` output an ArmorIQ decision. Before relying on the
mirror for training, compare its allow / hold / block outcomes with live
ArmorIQ on a fixed decision corpus and report agreement. Any mismatch is a
release blocker for that policy profile.

## Experiment design

### Training arms

Use a shared base model, tokenizer, task pool, seed list, rollout count,
optimizer settings, and training-token budget.

1. **TaskRL:** generic environment, task outcome reward, no policy observations.
2. **PromptPolicyRL:** policy text and role are visible, tools are ungated.
3. **ArmorRL:** same task and policy information as PromptPolicyRL; action
   execution is governed by ArmorIQ or its explicitly labelled mirror, and the
   agent observes allow / hold / block feedback.

PromptPolicyRL vs ArmorRL is the primary comparison. TaskRL shows how much
policy text alone changes behavior. Train a single policy model per arm; do not
add four collaborating models to the first experiment.

### Evaluation matrix

Evaluate every checkpoint in both isolated sandbox conditions. Run a smaller
paired evaluation through the live ArmorIQ MCP path after the sandbox results
are stable.

| Trained policy | Ungated isolated sandbox | Live ArmorIQ gate |
| --- | --- | --- |
| TaskRL | Task competence and unsafe side-effect rate | Enforcement-only baseline and recovery after a gate response |
| PromptPolicyRL | Whether prompted rules were learned | Runtime gate behavior for a prompt-only model |
| ArmorRL | Whether training internalized better action choices | Combined learned behavior and real ArmorIQ enforcement |

The sandbox is synthetic and has no external side effects. Never run the
ungated evaluation against real systems. The live evaluation MCP is also
simulated and registered with ArmorIQ for this project.

### Outcomes to report

- Verified task completion rate.
- Policy-compliant completion rate.
- Disallowed action attempt rate, before the gate acts.
- Disallowed side effects. ArmorIQ should prevent these at the MCP boundary.
- Recovery rate after hold / block, and successful approval use.
- Performance under held-out roles, task templates, and policy changes.
- Wasted actions, average steps, latency, and API/training cost.
- Policy-mirror to live-ArmorIQ agreement and live audit-trace coverage.

Keep task reward and governance signals separate in trajectory data. Do not
reward an `allow` outcome by itself; that could teach the policy to seek easy
allowed calls. Reward verified progress and completion, penalize unsafe
attempts and pointless retries, and make successful recovery from a legitimate
hold useful. Select exact reward weights using a pilot and document them before
the main comparison. Report uncertainty across task seeds; do not claim
ArmorIQ improves learning just because the runtime gate blocks calls.

## Ambitious extension: the Agentic Work Web

The larger environment is a synthetic, connected workplace that several agents
can operate in at once. Call it the **Agentic Work Web**. It is a network of
stateful apps, records, tasks, queues, and approval inboxes exposed through
structured tools and MCP APIs. Agents work in that shared world and exchange
artifacts and events. It is not pixel-based browser automation: the challenge
is coordinating actions and authority across a live, changing tool world.

### World contents

- Case / project workspace with goals, records, task boards, and versioned
  documents.
- Search and evidence index that returns source IDs and provenance.
- Messaging and review inboxes for agent-to-agent handoffs.
- Approval queue with asynchronous approve / reject / timeout decisions.
- Agent directory describing roles, capabilities, and current workload.
- Event log that records writes, messages, tool results, approvals, and
  policy decisions in causal order.

Every app is simulated and has deterministic state plus a verifier. New
verticals—legal matters, finance operations, research, or customer support—are
scenario packs over the same runtime.

### Async orchestrator

The trainable orchestrator receives a job and decides how to get it done. It
does not wait for every worker at every step. It dispatches bounded subtasks,
keeps working while independent workers run, reacts to completion / failure /
approval events, resolves conflicts, and decides when the final answer is
ready.

```text
User job
   |
   v
Orchestrator policy -- decomposes into a dependency graph
   |                  \
   | dispatch           \
   v                     v
Research agent       Records agent       Policy-review agent
   | async events       | async events       | async events
   +--------- shared Agentic Work Web / MCP tools ---------+
                              |
                 event queue + versioned workspace
                              |
                    merge / verify / finish
                              |
                ArmorIQ checks worker tool actions;
                child scopes depend on SDK support
```

The orchestration action space includes `spawn_subtask`, `delegate`,
`send_message`, `call_tool`, `wait_for_event`, `merge_artifact`, `retry`,
`cancel`, `request_approval`, and `finish`. A subtask names its goal, required
inputs, expected artifact, deadline, retry limit, and allowed tool / resource
scope. The orchestrator may dispatch independent subtasks concurrently and
must wait for declared dependencies before merging their outputs.

Use a bounded worker pool, event queue, and versioned shared workspace. Workers
return typed artifacts, citations, and provenance rather than unstructured
chat alone. A deterministic merge/verifier checks completeness, conflicting
facts, source support, and policy compliance. The controller must handle
out-of-order completion, stale reads, duplicate delivery, timeouts, worker
failure, retries, cancellation, and approval arriving after a wait begins.

### ArmorIQ in a team

The root job starts with an orchestrator identity and intent. A delegated
worker receives only the subtask scope it needs. Every worker's tool calls
remain subject to policy checks, identity attribution, and audit. The team
trace preserves parent job, delegation lineage, subtask, worker identity,
intent / plan IDs, tool action, policy result, artifact version, and final
verifier result.

ArmorIQ documents signed plan checking, per-user policy/audit, and approval
holds. The exact SDK surface for issuing child-agent authority must be confirmed
against our target SDK version before we commit the runtime to it. Keep a
`DelegationGate` interface: in live runs, use native child scopes only where the
chosen ArmorIQ integration supports them; otherwise use separately attributed
worker sessions and describe the limit plainly. A local scope simulator is
always labeled `policy_mirror`, never a live ArmorIQ delegation. Holds are
asynchronous events: one held worker must not stall unrelated work. A rejected
or timed-out delegation cannot be silently retried with broader authority.

### RL objective for the orchestrator

The primary trainable policy in this extension is the **orchestrator**, because
its decisions determine team size, delegation scope, parallelism, and when to
stop. Initially, worker agents use frozen policies or the same fixed model
prompt across conditions. This makes it possible to attribute gains to better
orchestration rather than changing every worker at once.

Use a team-level verified reward with separately logged components:

- `+` verified, complete, source-supported task outcome;
- `-` policy violation attempt, unsupported claim, or incorrect merge;
- `-` wasted worker spawn, redundant work, idle time, and repeated calls;
- `-` latency and token / tool cost, with a budget provided to the policy;
- `+` successful recovery from worker failure, approval hold, or conflict.

Avoid rewarding the raw number of agents spawned or messages sent. A one-agent
solution should beat an unnecessary swarm on cost when both produce the same
verified result. Record event-level causal traces for delayed credit assignment:
which dispatch or delegation led to which artifact, gate decision, and final
outcome. Start with team-level GRPO over complete asynchronous episodes; add
event / turn credit assignment only after replay and attribution are reliable.

### Multi-agent comparison

Treat this as a second experiment after the single-agent result. Use the same
Work Web tasks, workers, model budget, tool availability, and verifier:

1. **Solo PromptPolicy:** one agent with policy instructions; no runtime gate.
2. **Solo ArmorIQ:** one agent with policy instructions and live gate feedback.
3. **Async Team PromptPolicy:** orchestrator plus concurrent workers; policy
   instructions only, no runtime gate.
4. **Async Team ArmorIQ:** same orchestrator and workers; every worker MCP
   action is evaluated by ArmorIQ. Apply native child scopes if the selected
   integration supports them; otherwise use separately attributed worker
   sessions and report that boundary precisely.

Evaluate all four on isolated sandbox and live-gated mock MCP runs. For team
runs also report task success, compliant success, unsafe attempts and side
effects, approval recovery, worker utilization, duplicate work, conflict rate,
wall-clock latency, tokens / calls per successful task, delegation-scope
violations, and trace completeness. This can show whether ArmorIQ benefits
the team as well as the individual agent, and whether async coordination adds
value beyond simply spending more model calls.

Keep the causal comparisons explicit. ArmorIQ's value is the gated-vs-ungated
difference within the solo or team setup. The team's value is async-team vs
solo at a matched total model-token and tool-call budget. Also report wall-clock
time separately: parallel work may reduce elapsed time while using the same
total compute. Use identical task seeds, worker model, tool catalog, and
verifier in each paired comparison.

### A concrete episode: the work web in motion

The first showcase job is a synthetic evidence-review task, not a real legal
matter: “Prepare a sourced brief from this project folder, identify unresolved
facts, and submit it for review.” The orchestrator can assign independent
workers to source search, chronology extraction, and record validation. They
read a shared versioned workspace and return cited artifacts. A policy reviewer
checks the draft. If a worker requests access beyond its delegated scope, the
ArmorIQ path blocks that tool call. If a permitted action needs approval, that
worker enters `waiting_approval`; the other workers keep working. The
orchestrator receives events, resolves conflicting dates, retries a failed
search within budget, merges the evidence, and asks the independent verifier
whether the brief is complete and supported.

This episode makes the learning problem visible: the coordinator has to decide
what can run in parallel, which worker needs which authority, what to do with
late or conflicting results, whether approval is worth waiting for, and when
the evidence is good enough to finish. The team is a bounded pool, not an
unlimited crowd: start with four to eight worker slots, and expose the current
compute and tool-call budget in the observation.

### What counts as an RL step in an async episode

An orchestration step is a decision by the learned coordinator followed by the
next observation available to it. `spawn_subtask` returns immediately; worker
actions and tool results become timestamped events while other work continues.
`wait_for_event` advances simulated time to the next relevant event or deadline
instead of tying up a process. The observation includes pending jobs, completed
artifacts, failures, approvals, current budgets, and relevant shared-workspace
versions. Worker turns and tool actions are retained in the trajectory, but
the first learned policy is only the coordinator.

Use a deterministic virtual clock for training and replay, with an optional
wall-clock adapter for the demo. This keeps async ordering and timeout cases
reproducible. A delayed outcome is joined to the dispatch and follow-up choices
that caused it through event and task IDs. Start with a single coordinator per
episode and frozen role-conditioned worker policies. Later experiments can
train worker policies or a team policy only after credit assignment is sound;
that is a separate research milestone, not a hidden change to the first result.

## We build ArmorEnv with a coding-agent web

The agentic web is also the development method for this project. Keep two
systems distinct:

- **Coding web:** coding agents work on the real ArmorEnv repository. Their
  output is a patch, code, or review tied to an issue and a base commit.
- **RL Work Web:** agents operate inside the synthetic environment, and the
  learned orchestrator is trained and evaluated there.

The coding web is how we build the RL Work Web. The learned policy is not
allowed to rewrite its own training/evaluation rules as part of an episode.

```text
Product goal / issue
        |
        v
Coding coordinator -> dependency DAG, owners, acceptance checks
        |                         |
        +------ async dispatch ----+
        |             |            |
   env agent      ArmorIQ agent   RL/eval agent       independent reviewer
        |             |            |                    |
  isolated repo worktrees + task-scoped tools          |
        |             |            |                    |
        +------ patches + check logs + review ----------+
                              |
                    integration queue
                              |
                   shared project branch
```

From the first implementation milestone, use native Codex subagents as the
coding crew, with the root Codex session coordinating tasks, contracts, and
integration. This works before any project scheduler exists. The current
Codex session supports up to three worker subagents alongside the coordinator.
They share the workspace, so assign disjoint file ownership; use separate Git
worktrees when a tracked base is available and the task setup supports them.
Build the project scheduler and Work Web with this native crew. Later, connect
the repository adapter to a supported agent runtime/API and use the project
web to implement later modules. Native conversation subagents are not
automatically callable by the project's Python runtime; that dogfooding
handoff requires an explicit provider integration.

### Coding-agent roles and work protocol

Use a coordinator plus a bounded pool of specialist coding agents. The native
Codex crew starts with up to three workers here; the future standalone coding
web can target four to eight slots when its provider supports that capacity.
Assign work by independent module or
contract: environment/world, ArmorIQ adapter, RL/training, evaluation, and
interface/demo. Add a separate reviewer agent that did not author the patch.
One agent owns a file set at a time; parallel agents coordinate through
versioned contracts and artifacts rather than simultaneous edits to shared
files.

Every coding task includes: goal, base commit, owned paths, prerequisites,
expected patch/artifacts, acceptance commands, time/token/tool budget, and
allowed capabilities. The coordinator dispatches independent tasks
asynchronously, listens for `started`, `blocked`, `patch_ready`, `checks_done`,
and `review_done` events, and integrates only after dependencies and checks
pass. Failed checks return as a scoped repair task. The timeline preserves the
task, agent, patch, check log, review, and integration lineage.

Give each agent an isolated worktree or sandbox and a narrow repository-tools
MCP: read/search assigned source, edit owned paths, inspect diffs, and run an
allowlisted check. Do not give workers direct write access to the shared
integration branch. Route file and command actions through the ArmorIQ gate in
the governed condition. Denied actions make no change; a hold creates an
approval event while other coding tasks continue. Keep dependency changes,
secret access, network access, pushes, and deployment outside the default
worker capability set. Before relying on ArmorIQ for local repository tools,
verify that the chosen SDK/MCP integration supports those calls and preserve
the same generic-vs-live labels used elsewhere in the experiment.

### Coding-web modules and checks

Reuse the Work Web scheduler, event bus, worker registry, and trace format. Add
only the repository-specific adapter and integration controls:

| Module / target path | Responsibility | Required module checks |
| --- | --- | --- |
| `dev_web/task_adapter.py` | Convert project issues into typed task DAGs with file ownership and acceptance commands. | Reject missing owners, conflicting path ownership, invalid dependencies, and tasks without acceptance checks. |
| `dev_web/repo_workspace.py` | Create isolated worktrees/sandboxes from a recorded base commit and collect patch artifacts. | Agents cannot read/write outside their workspace; concurrent worktrees do not leak changes; patch records the correct base commit. |
| `dev_web/worker_runtime.py` | Launch or connect coding-agent workers through provider adapters; stream progress and stop/retry workers. | Worker crash/timeout is recorded; retries do not duplicate integration; configured concurrency is respected; provider IDs map to auditable worker identities. |
| `dev_web/event_store.py` | Persist run, task, worker, and tool events for status, resume, audit, and replay. | Restart resumes the same pending graph; duplicate events are idempotent; event history is append-only and isolated by run ID. |
| `dev_web/repo_tools_mcp.py` | Expose scoped read, search, patch, diff, and allowlisted test commands to coding agents. | Path traversal and disallowed commands are rejected; a denied/held action cannot mutate files; outputs redact configured secrets; tool results are replayable. |
| `dev_web/integration_queue.py` | Check dependencies, apply patches in order, surface conflicts, and update the shared branch. | Failed checks cannot merge; conflicts are explicit; applied patch hash matches reviewed artifact; repeated integration is idempotent. |
| `dev_web/review.py` | Request independent contract, diff, and check-log review before integration. | Reviewer identity differs from author; review is tied to exact patch hash; stale review is invalidated after patch changes. |
| `ui/dev_team_timeline.py` | Show coding tasks, active agents, waits, patches, checks, reviews, and merges. | Timeline matches event IDs and causal order; pending/failed work is visible; no unrecorded success is shown. |
| `dev_web/cli.py` and `configs/dev_web.yaml` | Configure providers, limits, gate mode, and run/status/resume/integrate commands. | Invalid provider or gate config fails before dispatch; resume restores event state; CLI status agrees with the event store. |

### Coding-web milestones

These milestones run alongside M0–M12. E0 starts with the first code change;
E2 is the point where our own orchestrator takes over part of the build.

#### E0 — Native Codex bootstrap

Split the initial solo environment into contract-first tasks and assign
independent modules to native Codex subagents with explicit file ownership.
The root session coordinates and reviews; agents share the workspace unless
separate worktrees are set up from a tracked base. **Exit:** at least two
independent patches integrate against agreed interfaces, with reproducible
acceptance checks and an attributable review trail.

#### E1 — Repository adapter

Build the task adapter, isolated repository workspace, narrow repo-tools MCP,
and patch integration queue on top of the project scheduler. Connect it to a
coding-agent runtime through a supported provider/API; native Codex subagents
remain available to us interactively but are not assumed callable from the
Python project. **Exit:** a coding-agent task can go from issue to isolated
patch to checks to review to integration, with a replayable event timeline.

#### E2 — Dogfood the agent web

Use the project orchestrator and its connected coding-agent provider to
implement later Work Web, benchmark, and UI modules. Keep native Codex agents
available as the interactive fallback and require an independent review for
changes to the orchestrator, governance adapter, or acceptance rules. **Exit:**
several parallel repo tasks complete without shared-file races, lost events,
or unreviewed integration.

#### E3 — Governed coding-agent comparison

Run paired repository tasks through generic coding tools and the ArmorIQ-gated
repo-tools MCP, using the same base commit, task, agent pool, model/token
budget, checks, and reviewer rubric. **Exit:** report acceptance rate, escaped
policy actions, defects found after merge, repair/rework, merge conflicts,
elapsed time, compute cost, and audit-trace coverage; separate real ArmorIQ
decisions from any mirror.

#### E4 — Optional coding-policy learning

Only after E3 has trustworthy traces, consider training a coordinator on
synthetic repository tasks. Keep this separate from the main Work Web RL result
and from actual project-source edits. **Exit:** held-out coding tasks improve
under fixed compute and acceptance checks without weakening governance or
review quality.

### How a coding run starts and finishes

1. **Freeze a base.** Start from a recorded commit and create a task file for
   each module or feature. Each task names owned paths, dependencies, expected
   outputs, acceptance checks, and budget. For Git worktrees, the base must be
   tracked so every worker receives the same files.
2. **Start the coordinator.** The coordinator validates the task DAG and
   provider/gate configuration, then dispatches independent tasks up to the
   worker limit. Workers receive isolated worktrees and only their task scope.
3. **Let agents run asynchronously.** A worker can publish progress, ask for a
   dependency, or submit a patch while other workers continue. The event log
   records assignments, tool calls, holds, failures, patch hashes, and check
   results. A failed worker is retried as a new attempt against the same
   acceptance contract.
4. **Validate each patch.** An automated check runner executes the task's
   allowlisted commands. A separate reviewer checks the exact patch hash,
   interface contract, test output, and scope. A changed patch invalidates the
   old review and checks.
5. **Integrate and close.** The integration queue applies only dependency-ready
   patches whose checks and review passed, reports conflicts as repair tasks,
   and produces one run report. A coding run can close with tasks in terminal
   success or failure states; the project is complete only when every required
   task succeeds, the full suite passes on the integrated branch, and no held
   action is misreported as executed.

The repository now has two separate pieces of this workflow. Native Codex
subagents can be assigned independent code modules interactively, as in the
initial implementation. The standalone `dev_web` package provides a manual
task queue with a SQLite event store, task snapshots, scoped work packets,
proposal capture, allowlisted checks, independent review, and gated
integration. It does not launch native Codex conversation agents and does not
create Git worktrees. A provider adapter and a tracked base commit are still
needed for autonomous isolated coding runs. The team Work Web is a separate
RL environment and is not the repository coding queue.

The current manual CLI surface is:

```bash
python -m dev_web run --provider manual --task "Implement a scoped module" \
  --scope learning/ --scope tests/ --check ruff
python -m dev_web status RUN_ID
python -m dev_web resume RUN_ID
python -m dev_web diff RUN_ID task-1
python -m dev_web check RUN_ID task-1
python -m dev_web review RUN_ID task-1 --reviewer independent-reviewer --decision approve
python -m dev_web integrate RUN_ID task-1
```

This implementation is intentionally manual-first: a person edits within the
task snapshot and submits the proposal. Reviewer identity is self-declared,
and allowlisted checks are not an operating-system sandbox for hostile code.
The command queue does not yet dispatch a coding-agent provider or apply
parallel Git worktree patches.

### Project completion sequence

The solo and team implementations, synthetic tool world, bounded asynchronous
scheduler, coding-web skeleton, and focused test suites are now present. The
test suites have not yet been run, and no training or held-out evaluation has
been performed. Before using Git worktrees, establish a tracked base commit.
The remaining sequence is:

1. Run the focused and full test suites; repair failures and verify seeded
   solo/team traces, holds, retries, event replay, and governance boundaries.
2. Connect and validate the registered ArmorIQ mock-MCP path for live team
   tool calls; keep SDK execution and local simulated execution mutually
   exclusive.
3. Run short solo and team training pilots, then the matched-budget training
   arms and held-out evaluations described above.
4. Connect a supported coding-agent provider to `dev_web`, add isolated Git
   worktrees and a true independent-review identity, and exercise the full
   issue-to-integrated-patch path.
5. Run paired generic-versus-live coding tasks only after the provider and
   ArmorIQ repository-tool boundary are validated; then add the timeline UI.

The completion bundle is reproducible configs and checkpoints, held-out
metrics with uncertainty, auditable live ArmorIQ traces, runnable solo and
team demos, and a final report that separates task, coordination, and
governance effects.

## System architecture and modules

The first implementation can remain a Python package. Keep interfaces narrow
so scenarios, training algorithms, and execution backends can be changed
without rewriting each other.

| Module / target path | Responsibility | Required module checks |
| --- | --- | --- |
| `models.py` / `core/contracts.py` | Typed action, observation, decision, task, and trajectory contracts; JSON serialization. | Round-trip every public type; reject unknown action kinds and malformed tool calls; deterministic serialization for equivalent inputs. |
| `tasks.py` / `scenarios/` | Versioned task templates, world fixtures, role assignment, seeded train / validation / test splits. | Same seed reproduces same episode; splits have no shared task IDs; every task has a verifier and reachable valid solution. |
| `policies.py` / `governance/profiles.py` | Canonical role and policy profiles used to configure ArmorIQ and create a local mirror. | Table-test allow / hold / block boundaries; unknown tools fail closed; profile versions are recorded; mirror decisions match the golden fixture. |
| `mcp_tools.py` / `worlds/` | Deterministic stateful MCP server and isolated world snapshots. | Test each tool’s valid and invalid transitions; denied calls do not mutate state; snapshots restore exact state; no real email, file, or external API effects. |
| `armoriq_gate.py` / `governance/` | `GenericGate`, `PromptPolicyGate`, `PolicyMirrorGate`, and live ArmorIQ session/SDK adapter behind one interface. | Unit-test gate mapping; fake SDK checks plan, token, MCP, params, user identity; block/hold never executes a tool; network / unknown SDK errors fail closed; a successful live invoke is not executed again locally. Add a separately marked live test against the registered mock MCP. |
| `orchestrator/armoriq_adapter.py` | Bind each worker tool action to an ArmorIQ plan, intent token, registered MCP action, user identity, and seeded task metadata. | Fake SDK checks plan/token/params/seed/user; live ALLOW uses returned remote data once; HOLD origin and delegation IDs survive into the approval trace; unknown calls fail closed. |
| `environment.py` / `envs/` | Episode reset/step lifecycle, observations, gate-mediated transitions, approval state, termination, trace assembly. | Deterministic seeded reset; Gymnasium API conformance; allow mutates as expected; hold pauses; block leaves world unchanged; correct approval/recovery; terminal/truncated episodes reject extra steps. |
| `rewards.py` / `learning/rewards.py` | Named task and governance reward components with versioned weights. | Exact component tests; allowed-but-useless calls do not gain progress reward; blocked attempts are visible; approval spam and loops cannot increase score; terminal reward requires verifier success. |
| `verifier.py` / `verification/` | Independent scoring from task spec and final world state, never agent assertion. | Correct completion passes; wrong recipient, missing evidence, forbidden attachment, bypassed approval, and premature finish fail. Property tests for invariants. |
| `agents.py` / `agents/` | Policy protocol, structured-action parsing, base model and LoRA adapter. | Parse valid JSON and fenced JSON; malformed outputs become safe no-op/replan; tool args validate against schemas; deterministic decoding works for evaluation. |
| `train.py` / `learning/` | Grouped multi-turn GRPO rollout and LoRA updates; experiment config, checkpoint, metrics, resume. | Group contains the same task; normalized advantages are finite; zero-variance groups are handled; only adapter weights update; checkpoint save/reload preserves outputs; sample trajectory updates parameters. |
| `evaluate.py` / `evaluation/` | Paired evaluation matrix, metrics, confidence intervals, audit joins, report export. | Identical episode seeds across policies; metric calculations match hand-computed fixtures; mirror/live outcomes remain distinguishable; missing traces are counted, never silently dropped. |
| `demo.py` / `ui/` | One memorable task timeline from proposed action through ArmorIQ decision, world result, reward, and final verification. | Offline demo is deterministic; live demo identifies itself and displays actual plan/decision IDs; no fabricated ArmorIQ result; approval pending state is visible. |
| `pyproject.toml`, CLI, configs | Reproducible install, optional live/RL dependencies, train/evaluate/demo commands. | Clean install in each extra profile; CLI help and config validation; secrets are never printed or committed. |

### Additional modules for the Agentic Work Web

Build these after the solo ArmorIQ environment and comparison work reliably.

| Module / target path | Responsibility | Required module checks |
| --- | --- | --- |
| `worlds/work_web.py` | Shared synthetic app graph: projects, records, messages, approvals, and event log. | Replay same seed to identical state; every event has one world mutation or is explicitly read-only; app state remains isolated between episodes. |
| `orchestrator/contracts.py` | Job, subtask, dependency edge, worker assignment, artifact, event, and budget types. | Serialization round-trips; reject cycles, missing dependencies, invalid scopes, and malformed artifact references. |
| `orchestrator/planner.py` | Turns a job into a typed dependency DAG and states expected deliverables. | DAG validity; all required outputs reachable; impossible or underspecified jobs are surfaced; plan does not grant tools by itself. |
| `orchestrator/scheduler.py` | Async bounded dispatch, priorities, deadlines, retries, cancellation, and backpressure. | Independent jobs run concurrently; dependency-gated jobs wait; enforce worker/concurrency budgets; retries are idempotent; cancellation prevents late results from mutating final state. |
| `orchestrator/event_bus.py` | Durable episode-scoped event stream and deterministic virtual clock for worker results, tool events, approvals, and timers. | Out-of-order and duplicate events handled; per-episode isolation; event replay reproduces scheduler state; a held task does not block unrelated event processing; timeout order is seed-reproducible. |
| `orchestrator/controller.py` | Trainable policy for dispatch / wait / merge / retry / cancel / finish decisions. | Invalid actions rejected; cannot exceed task budget; makes progress after success/failure/timeout/hold events; finish only accepted after verifier passes; observation never leaks hidden world state. |
| `orchestrator/worker_registry.py` | Worker capability, role, model, capacity, and policy identity registry. | Capability matching; capacity is respected; offline worker and stale registry are surfaced; no assignment silently escalates identity. |
| `orchestrator/workers/` | Researcher, evidence extractor, drafter, policy reviewer, and verifier worker interfaces. | Contract tests on each worker; outputs reference existing artifacts; injected instructions in documents cannot alter worker permissions. |
| `workspace/versioned_store.py` | Shared artifacts with immutable versions, provenance, and optimistic concurrency. | Concurrent writes detect conflicts; stale writes cannot overwrite newer artifacts; every artifact maps to producing worker and source events. |
| `governance/delegation.py` | Parent intent to scoped child tasks, policy mirror/live ArmorIQ delegation adapter, revocation. | Child scope is a subset of parent; out-of-scope action is blocked; revoked/expired scope fails closed; parent-child lineage appears in trace; test exact live SDK behavior separately. |
| `learning/orchestrator_rollout.py` | Capture asynchronous orchestration decisions and join delayed worker outcomes into RL trajectories. | Every decision links to spawned task IDs and resulting events; delayed reward joins correctly; dropped / late events are detected; virtual-clock replay works; coordinator and worker turns are distinguishable. |
| `learning/team_rewards.py` | Team-level success, compliance, quality, latency, and cost reward accounting. | No reward farming by spawning agents or sending messages; equivalent correct solutions rank by cost; task success and policy reward remain separately inspectable. |
| `evaluation/async_benchmark.py` | Solo-vs-team and generic-vs-ArmorIQ paired benchmark runner. | Paired task/seed/model budgets; concurrency and wall-time metrics are correct; report separates mirror from live ArmorIQ; bootstrap intervals are computed over tasks, not individual correlated events. |
| `ui/team_timeline.py` | Visual event graph showing parent job, concurrent workers, tools, ArmorIQ outcomes, artifacts, and merge. | Timeline order matches causal IDs; concurrent events display without implying false ordering; click-through exposes actual plan and audit references; no synthesized gate results. |

## Milestones and exit gates

### M0 — Experiment contract

Write this plan, freeze the comparison arms, metrics, task split rules, and
policy versions. Define which SDK mode represents live ArmorIQ and the exact
mock MCP deployment. **Exit:** team agrees on the primary PromptPolicyRL vs
ArmorRL comparison and the scorecard; no training claim is made yet.

### M1 — Reliable generic world

Implement typed contracts, scenario/task generator, MCP world, verifier, and
generic/prompt-policy environments. Use the E0 coding crew for independent
modules, with interface contracts and isolated worktrees. **Exit:** every task
has a deterministic valid route; module checks pass; full episodes replay from
seed; independently owned patches integrate through review.

### M2 — ArmorIQ boundary

Implement policy mirror and real SDK/session adapter, register the isolated
mock MCP, and capture plan IDs, decisions, user attribution, and audit trace.
**Exit:** block/hold stops execution; allow executes exactly once; mirror
agreement is measured against live ArmorIQ; no fake or ambiguous gate labels.

### M3 — Baselines and reward calibration

Run scripted/reference and untrained-model rollouts, inspect reward components,
fix reward loopholes, then freeze reward weights. **Exit:** verifier and
scorecard distinguish task success, policy compliance, and refusal.

### M4 — Real RL training

Train TaskRL, PromptPolicyRL, and ArmorRL using the same base model and budget.
Start with multi-turn GRPO over groups of complete trajectories; preserve
action boundaries and per-step feedback. Compare an outcome-only baseline with
turn-level credit assignment if compute allows. **Exit:** checkpoints,
configs, token budgets, and all rollout records are reproducible.

### M5 — Paired evaluation

Run the complete training/evaluation matrix on held-out tasks and policies,
then a paired live ArmorIQ evaluation against the registered mock MCP.
**Exit:** results include confidence intervals, mirror/live agreement, latency,
and audit coverage; demo claims match these results.

### M6 — Demo and domain showcase

Build the timeline UI and add a Manchester City legal-record scenario pack only
after the generic tasks work. Use public, properly attributed material and a
verifier tied to explicit task facts. **Exit:** the audience can see the same
task succeed in the generic path and the ArmorIQ path, including a blocked or
held attempt and the learned recovery.

### M7 — Agentic Work Web runtime

Implement the synthetic multi-app web, versioned workspace, typed artifacts,
and deterministic verifier using the E0 coding crew. **Exit:** two independent
workers can make concurrent changes; event replay reconstructs the episode;
conflict and stale-write scenarios are reproducible.

### M8 — Async orchestration without RL

Add the event bus, bounded scheduler, worker registry, and rule-based
orchestrator. Exercise parallel completion, one worker failing, a pending
approval, a stale artifact, cancellation, and retry. **Exit:** the orchestrator
finishes benchmark tasks end to end; all actions and worker lineage are
replayable; a hold never freezes unrelated workers. This unlocks E1: connect
the scheduler to isolated repository workspaces and coding-task acceptance
checks.

### M9 — Per-worker ArmorIQ governance

Connect root and worker identities, policy checks, and audit lineage to the
registered mock MCP. Verify SDK child-delegation support before relying on
native scopes; otherwise authorize each worker through its attributed session
and policy. Retain a clearly named mirror for fast local episodes. **Exit:** a
worker's out-of-policy tool action does not execute, and the team timeline
reconciles with real ArmorIQ records; any limitation in native delegation is
stated in the result. In parallel, E2 starts using the rule-based project
orchestrator to build later modules, while keeping independent review on
changes to the orchestrator and governance code.

### M10 — Train and evaluate the orchestrator

Freeze worker model/prompt, tasks, reward, and compute budget. Train the
orchestrator with grouped async rollouts; first compare task-level GRPO, then
test event-level credit assignment as a separate condition. Run solo/team ×
prompt/gated evaluation on held-out tasks. **Exit:** report quality, safety,
cost, latency, utilization, and uncertainty; claims distinguish coordination
gains from ArmorIQ governance gains.

### M11 — Agentic Web demo and showcase

Show a real-time team timeline: parent job decomposes, workers proceed in
parallel, one encounters a hold, the others continue, results merge, and the
verifier checks the final artifact. Add the Manchester City scenario pack
after the general work web is stable. **Exit:** every displayed worker, gate
decision, and result corresponds to a real recorded event.

### M12 — Optional worker-policy learning

Only after M10 has reliable replay, attribution, and credit assignment, train
role-conditioned worker policies as a separate experiment. Keep the frozen
worker and learned-worker results side by side, and hold total team compute
constant. **Exit:** worker-policy updates improve held-out verified outcomes
without increasing unsafe attempts, unsupported evidence, or cost beyond the
declared budget; orchestration and worker learning effects are reported
separately.

## Training and rollout record

Every trajectory stores: task and scenario version, seed, training arm, model
and adapter hash, exact prompt and generated action tokens, structured action,
world-state hash, gate type and source, ArmorIQ plan/decision/delegation IDs
when present, policy version, reward components, verifier result, and timing.
Sanitize sensitive data. This record enables replay, audit review, and
step-level credit assignment.

## Non-goals for the first experiment

- Pixel-based computer use or a live desktop.
- Four independently trained worker models in the first multi-agent pass; the
  async team experiment starts with a learned orchestrator and fixed workers.
- Real emails, customer data, case systems, or external side effects.
- Claiming that ArmorIQ makes every agent or every task better.

## References

- [ArmorIQ SDK](https://docs.armoriq.ai/sdk)
- [ArmorIQ sessions](https://docs.armoriq.ai/sdk/sessions)
- [ArmorIQ delegated trust tools (integration-specific)](https://docs.armoriq.ai/armorclaude/getting-started/mcp-tools)
- [ArmorIQ custom framework integration](https://docs.armoriq.ai/platform/sdk-cli/frameworks/custom)
- [ArmorIQ policy rules](https://docs.armoriq.ai/tools/policy-studio)
- [ArmorIQ plan detail and audit trail](https://docs.armoriq.ai/platform/intent-intelligence/plan-detail)
- [OpenEnv](https://huggingface.co/docs/openenv/index)
- [Agent-R1 step-level agent RL](https://github.com/AgentR1/Agent-R1)
- [Group Turn Policy Optimization (GTPO)](https://aclanthology.org/2026.acl-long.1962/)
