# ArmorEnv

ArmorEnv is a policy-aware RL testbed for agents that use tools. The solo
environment and the asynchronous **Agentic Work Web** use synthetic, seeded
tasks, a deterministic in-memory tool world, and independent outcome checks.
See [PLAN.md](PLAN.md) for the experiment design, milestones, module contracts,
and acceptance checks.

## What changes between conditions

```text
GENERIC
Agent -> synthetic tool -> world -> verifier

PROMPT_POLICY
Policy text -> agent -> synthetic tool (ungated) -> world -> verifier

POLICY_MIRROR
Policy text -> agent/worker -> local policy mirror -> allow / hold / block
                                         | allow -> synthetic tool -> world
                                         | hold/block -> no tool execution

ARMORIQ_LIVE
Plan -> intent token -> ArmorIQ invoke -> registered MCP -> result + audit trace
```

`generic` gives the learned coordinator no policy instructions and has no
runtime gate. `prompt_policy` gives the coordinator policy instructions while
tools remain ungated. `policy_mirror` gives it the same instructions plus
worker-tool enforcement and decision feedback, using a deterministic local
approximation for high-volume rollouts; mirror decisions are never called live
ArmorIQ decisions. The team experiment trains the coordinator while synthetic
specialist workers stay fixed. `armoriq_live` requires an explicit live
adapter and registered synthetic MCP.

The solo experiment trains a policy per condition. The team experiment trains
the orchestrator while specialist workers stay fixed. Its synthetic evidence
review contains a deliberate broad-export attempt, seeded facts, parallel
tasks, and a review hold. The mirror blocks the export before its simulated
side effect; the generic and prompt-only controls execute it in the isolated
world, where the verifier detects it.

## Setup

Install the optional training dependencies for model rollouts:

```bash
uv sync --extra rl
```

For local tests and linting, install `--extra dev`. The live solo SDK path also
needs `--extra armoriq`, an authenticated ArmorIQ account, and a registered
mock MCP. No live experiment or benchmark result is claimed by the code alone.

## Solo agent

Run the scripted trace without downloading a model:

```bash
uv run python demo.py --mode policy_mirror --profile analyst
uv run python demo.py --mode generic --profile analyst
```

Train a LoRA policy with grouped multi-turn GRPO. Use the same base model,
seeds, rollout count, and compute budget for each arm:

```bash
uv run --extra rl python train.py \
  --base-model Qwen/Qwen3-4B-Instruct \
  --condition generic --output-dir artifacts/solo-generic

uv run --extra rl python train.py \
  --base-model Qwen/Qwen3-4B-Instruct \
  --condition prompt_policy --output-dir artifacts/solo-prompt

uv run --extra rl python train.py \
  --base-model Qwen/Qwen3-4B-Instruct \
  --condition policy_mirror --output-dir artifacts/solo-mirror
```

Evaluate adapters on identical held-out tasks in one selected execution
condition. Repeat `--adapter LABEL=PATH` for each trained arm:

```bash
uv run --extra rl python evaluate.py \
  --base-model Qwen/Qwen3-4B-Instruct \
  --adapter TaskRL=artifacts/solo-generic/adapter \
  --adapter PromptPolicyRL=artifacts/solo-prompt/adapter \
  --adapter MirrorRL=artifacts/solo-mirror/adapter \
  --condition policy_mirror --report artifacts/solo-evaluation.json
```

The report includes verified completion, policy-compliant completion, unsafe
attempts, unsafe side effects, blocks, holds, and task-level paired bootstrap
intervals. Use `--condition generic`, `prompt_policy`, or `armoriq_live` to run
other evaluation paths. Live runs require `--user-email` and the configured
ArmorIQ account/MCP.

## Async Agentic Work Web

Run the readable rule-based team trace:

```bash
uv run python demo_team.py --condition policy_mirror --seed 7
uv run python demo_team.py --condition generic --seed 7
```

Train only the orchestrator; worker behaviors remain fixed across conditions:

```bash
uv run --extra rl python train_orchestrator.py \
  --base-model Qwen/Qwen3-4B-Instruct \
  --condition policy_mirror --output-dir artifacts/team-mirror
```

Evaluate one or more team adapters on matched seeded task instances:

```bash
uv run --extra rl python evaluate_orchestrator.py \
  --base-model Qwen/Qwen3-4B-Instruct \
  --adapter TeamMirror=artifacts/team-mirror/adapter \
  --condition generic --condition prompt_policy --condition policy_mirror \
  --report artifacts/team-evaluation.json
```

For a smaller live run, install and authenticate the ArmorIQ SDK, register the
synthetic Work Web MCP, and map its registered action names:

~~~bash
uv run --extra rl --extra armoriq python evaluate_orchestrator.py \
  --base-model Qwen/Qwen3-4B-Instruct \
  --adapter TeamMirror=artifacts/team-mirror/adapter \
  --condition policy_mirror --condition armoriq_live \
  --user-email agent@example.com \
  --tool-bindings-json configs/work-web-armoriq-bindings.example.json \
  --report artifacts/team-live-evaluation.json
~~~

The SDK adapter captures a one-action plan, obtains an intent token, and
invokes the registered MCP. A returned remote result is used directly and is
never executed by the in-memory tool executor. ArmorIQ holds stay pending and
time out because external approval resumption is not wired yet.
The example binding passes `episode_seed` to `read_source`; the registered
mock should use the same seeded fixture rules as the local world:
`launch_day = 10 + ((seed * 7) % 20)` and
`affected_accounts = 1 + ((seed * 5) % 9)`. That keeps the independent verifier
comparing the same task in mirror and live runs.
See [the registered mock MCP contract](configs/WORK_WEB_MCP.md) for every
action's parameters and expected response.

The episode reward records verified completion, policy-compliant completion,
source support, unsafe attempts, unauthorized side effects, worker and retry
costs, virtual time, tool calls, and recovery from holds. An `allow` decision or
extra worker spawn is never rewarded by itself. Live Work Web episodes require
an explicit `GovernanceAdapter` labeled `armoriq_live`; each allowed tool call
must return its already executed remote result, so the synthetic executor does
not repeat the side effect. Locally granting a synthetic approval cannot
resolve an ArmorIQ-originated hold.

## Coding-agent work web

This session used native Codex coding agents to implement separate modules
under root-agent coordination. The project's `dev_web/` package is a separate,
manual-first local coordinator: it can create a task DAG, isolated repository
snapshots, scoped work packets, proposals, allowlisted checks, independent
reviews, and gated integration. It does not call native Codex conversation
agents from Python; a real worker provider must be explicitly registered.

Configure only fixed check commands in `.dev_web/config.json`, then create a
manual assignment:

```bash
python -m dev_web run --provider manual \
  --task "Implement a scoped module" \
  --scope "learning/" --scope "tests/" --check ruff
```

The command prints a run ID and task packet. Edit inside the task snapshot,
then capture and review the proposal:

```bash
python -m dev_web resume RUN_ID
python -m dev_web diff RUN_ID task-1
python -m dev_web check RUN_ID task-1
python -m dev_web review RUN_ID task-1 --reviewer independent-reviewer --decision approve
python -m dev_web integrate RUN_ID task-1
python -m dev_web status RUN_ID
```

The local queue is an orchestration/review boundary, not an OS sandbox for
hostile code. Manual reviewer names are self-declared. Checks must be
conservative and allowlisted.

## Verify when ready

Tests and linting have not been run for this implementation. When you want to
verify it locally:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

## Current implementation limits

The Work Web currently has one seeded synthetic evidence-review scenario and
deterministic fixed specialists. It is enough to exercise asynchronous
dispatch, dependency scheduling, approval holds, tool gating, tracing, and
team-level RL code. It is not yet a broad benchmark, the real Manchester City
legal-record scenario, or an interactive timeline UI. Live team runs require
ArmorIQ authentication and a registered mock MCP with the configured action
names; external hold resumption is not wired, so unresolved holds stay closed
until timeout. The coding web has no native provider connection.

Focused unit tests are present, but have not been run yet. No training or
held-out evaluation has been run, so results and causal claims remain open.
