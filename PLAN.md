# First experiment: GRPO for governed legal research

## Research question

For the same base model and matched legal-research tasks, does GRPO with ArmorIQ-governed tool use improve source-supported task success and reduce unauthorized actions compared with task-only or prompt-only GRPO?

This repo is currently a prototype harness. The public Manchester City task pack, live MCP deployment, training runs, and results are not complete.

## Workload

Use public primary records from Manchester City’s Premier League APT arbitration:

- [2024 partial final award](https://resources.premierleague.com/premierleague/document/2024/10/07/898efab9-9f51-449b-a393-1a0c05b48824/Manchester-City-and-Premier-League-Partial-Final-Award-071024.pdf)
- [2025 follow-up award](https://resources.premierleague.pulselive.com/premierleague/document/2025/02/14/5427b198-943b-4312-aab9-ae06b6b19285/Man-City-Premier-League-Second-Partial-Final-Award.pdf)
- [2025 settlement announcement](https://www.premierleague.com/en/news/4407130/premier-league-statement?embed=true)

Build tasks for chronology, argument-versus-finding distinctions, pinpoint citations, comparisons across the rulings, and a neutral case note. Gold answers must cite page or paragraph locations and be reviewed independently. Split train and held-out tasks by document or task family. Keep document dates and later settlement status explicit. Use public records only; simulated confidentiality and approval rules must be labeled synthetic.

## Agent and environment

Start with one learned orchestrator and fixed specialist roles for research, source auditing, cross-checking, and drafting. Agents use structured document tools through an MCP-style interface. The verifier scores the final work independently from the model and the governance adapter.

ArmorIQ governs tool invocation and records policy outcomes; it does not determine whether a legal claim is correct. The environment supplies tasks and state. GRPO optimizes the model from grouped rollout rewards.

## Comparison

Use the same base model, prompts, task split, seeds, token budget, and training settings for each arm:

1. Base model without RL.
2. Task-GRPO without policy instructions or runtime enforcement.
3. Prompt-GRPO with policy instructions but ungated sandbox tools.
4. Governed-GRPO with ArmorIQ decisions during rollouts.

If a local mirror supplies decisions at training scale, label that experiment Mirror-GRPO. Do not describe mirror results as live ArmorIQ results.

Evaluate every checkpoint in two settings:

- Ungated isolated sandbox: measures what the policy learned.
- Registered ArmorIQ MCP: measures runtime enforcement and live policy behavior.

Report enforcement and learned behavior separately. A gate blocking an action demonstrates enforcement; it does not by itself show improved task performance or learning.

## Measures

- Independently verified task completion.
- Claim and pinpoint-citation support.
- Unsafe tool attempts.
- Unauthorized tool effects that actually executed.
- Recovery and task completion after hold or block.
- Tool calls, virtual time, and model tokens.

Compare matched tasks and report uncertainty over held-out tasks and multiple training seeds. Do not claim improvement from a small pilot.

## Milestones

1. **Case task pack:** collect public source references, create cited gold answers, define train/dev/test splits, and get an independent review of the verifier.
2. **Tool path:** implement deterministic document search/read and draft tools; add a registered synthetic MCP with the same task fixtures and explicit policies.
3. **Harness validation:** establish base-model, task-only, and prompt-only results. Verify that blocked actions do not mutate the MCP state.
4. **GRPO pilot:** run matched grouped rollouts, inspect reward components and learning curves, then fix the training configuration before scaling.
5. **ArmorIQ comparison:** run the governed arm and live evaluation; compare mirror and live decisions on a fixed corpus; report policy, task, and resource metrics separately.

## Current implementation status

Present: synthetic solo and async team environments, GRPO/LoRA trainer prototypes, paired evaluators, a policy mirror, ArmorIQ SDK adapters, and focused test files.

Still needed: public case tasks and gold labels, a deployed registered MCP, independent task-set review, training/evaluation runs, and verified results. The focused tests have not been run. The dev_web package is manual-first and has no native Codex provider.

## Reading

- [DeepSeek-R1](https://www.nature.com/articles/s41586-025-09422-z) for GRPO.
- [ArmorIQ Python SDK](https://docs.armoriq.ai/platform/sdk-cli/python-sdk) for plan, intent-token, and MCP invocation.
- [AgentGym-RL](https://arxiv.org/abs/2509.08755) and [tau-bench](https://arxiv.org/abs/2406.12045) for multi-turn agent environments and outcome-based evaluation.

