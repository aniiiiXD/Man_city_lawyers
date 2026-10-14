# ArmorEnv

ArmorEnv is a research prototype for one question:

> Does GRPO training with governed tool use help an agent complete legal research tasks accurately while avoiding unauthorized actions?

The current environment is synthetic. The Manchester City case task pack and benchmark results do not exist yet.

## The idea

Use public records from Manchester City’s Associated Party Transaction (APT) arbitration as a real legal-research workload. An agent should find and read the record, build a dated chronology, separate each party’s arguments from the tribunal’s findings, cite its claims, and draft a short neutral case note.

Start with the public [2024 partial final award](https://resources.premierleague.com/premierleague/document/2024/10/07/898efab9-9f51-449b-a393-1a0c05b48824/Manchester-City-and-Premier-League-Partial-Final-Award-071024.pdf), the [2025 follow-up award](https://resources.premierleague.pulselive.com/premierleague/document/2025/02/14/5427b198-943b-4312-aab9-ae06b6b19285/Man-City-Premier-League-Second-Partial-Final-Award.pdf), and the [2025 settlement statement](https://www.premierleague.com/en/news/4407130/premier-league-statement?embed=true). The task set must preserve the dates and legal status of each document. Use public records only; any restricted documents or approval rules in the environment must be clearly labeled as simulated.

## What we are testing

GRPO is the policy-training method. ArmorIQ is the tool-action governance layer. The task environment and independent verifier measure the work. GRPO is used in DeepSeek-R1; ArmorIQ documents its plan, intent-token, and MCP-invoke flow ([paper](https://www.nature.com/articles/s41586-025-09422-z), [SDK](https://docs.armoriq.ai/platform/sdk-cli/python-sdk)).

Compare the same base model, task split, seeds, and training budget:

| Training arm | Policy instructions | Runtime tool gate |
| --- | --- | --- |
| Base model | None | None in the isolated sandbox |
| Task-GRPO | None | None in the isolated sandbox |
| Prompt-GRPO | Yes | None in the isolated sandbox |
| Governed-GRPO | Yes | ArmorIQ decisions during rollouts |

If a local policy mirror is used for high-volume training, label that arm **Mirror-GRPO**. Mirror results alone are not evidence about live ArmorIQ.

Evaluate every trained policy both ungated in the isolated sandbox and through the registered ArmorIQ MCP path. This separates two claims:

- **Enforcement:** did ArmorIQ prevent an unauthorized tool effect?
- **Learning:** does the trained policy make better choices even when the evaluation gate is removed?

Score task completion, claim and citation support, unsafe attempts, unauthorized side effects, and recovery after a hold. A blocked attempt and an executed side effect are separate outcomes. Use held-out tasks and report uncertainty across tasks and training seeds. Do not claim that ArmorIQ improves legal reasoning unless the ungated evaluation supports that claim.

## What is in this repository

- Solo and asynchronous team environment prototypes with synthetic evidence.
- GRPO/LoRA training and paired evaluation scripts.
- Local policy-mirror and ArmorIQ SDK adapters.
- A manual-first coding-workflow prototype in dev_web.

This is a harness, not the benchmark described above. It does not yet contain the public case task pack, a registered Work Web MCP deployment, or training results. Tests are written but have not been run.

See [PLAN.md](PLAN.md) for the short experiment checklist and [configs/WORK_WEB_MCP.md](configs/WORK_WEB_MCP.md) for the current mock-MCP contract.

## Try the current synthetic demo

Requires Python 3.11 or later.

~~~bash
uv sync
uv run python demo_team.py --condition policy_mirror --seed 7
~~~

When ready to verify the code:

~~~bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
~~~

Model training needs the optional RL dependencies. Live ArmorIQ evaluation also needs an authenticated SDK account, a registered MCP, and matching policy configuration. No tests, training runs, or benchmark runs have been completed yet.

