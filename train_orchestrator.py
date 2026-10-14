"""Train the async work-web coordinator with grouped policy-gradient updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from learning.orchestrator_policy import HuggingFaceOrchestratorPolicy
from learning.orchestrator_training import (
    AsyncGRPOConfig,
    AsyncGRPOTrainer,
    append_trajectories,
)
from train import LogProbabilityModel, load_lora_policy, seed_everything
from work_web.scenarios import (
    build_generic_episode,
    build_policy_mirror_episode,
    build_prompt_policy_episode,
)


EPISODES = {
    "generic": build_generic_episode,
    "prompt_policy": build_prompt_policy_episode,
    "policy_mirror": build_policy_mirror_episode,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a trainable async orchestrator against a fixed synthetic worker team"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--condition",
        choices=tuple(EPISODES),
        required=True,
        help="High-throughput mirror training; live ArmorIQ is reserved for evaluation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-decisions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.iterations < 1 or args.max_decisions < 1:
        parser.error("--iterations and --max-decisions must be positive")
    seed_everything(args.seed)
    source_policy = load_lora_policy(args.base_model)
    policy = HuggingFaceOrchestratorPolicy(
        model=source_policy.model,
        tokenizer=source_policy.tokenizer,
        temperature=0.8,
    )
    logprobs = LogProbabilityModel(source_policy)
    config = AsyncGRPOConfig(group_size=args.group_size)
    trainer = AsyncGRPOTrainer(
        policy,
        config,
        episode_factory=EPISODES[args.condition],
        logprobs=logprobs,
        learning_rate=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    trajectory_path = args.output_dir / "trajectories.jsonl"

    for iteration in range(args.iterations):
        task_seed = args.seed + iteration
        trajectories = trainer.collect_group(
            task_id=f"synthetic-evidence-review:{task_seed}",
            condition=args.condition,
            task_seed=task_seed,
            max_decisions=args.max_decisions,
        )
        metrics = trainer.update(trajectories)
        metrics.update(
            {
                "iteration": iteration,
                "condition": args.condition,
                "task_seed": task_seed,
                "mean_verified_success_rate": fmean(
                    float(item.verified_complete) for item in trajectories
                ),
                "mean_policy_compliant_success_rate": fmean(
                    float(item.verified_complete and item.policy_compliant)
                    for item in trajectories
                ),
                "mean_worker_spawns": fmean(
                    float(item.team_outcome.get("worker_spawns", 0)) for item in trajectories
                ),
                "mean_unsafe_attempts": fmean(
                    float(item.team_outcome.get("unsafe_attempts", 0)) for item in trajectories
                ),
                "mean_unauthorized_side_effects": fmean(
                    float(item.team_outcome.get("unauthorized_side_effects", 0))
                    for item in trajectories
                ),
                "mean_generated_tokens": fmean(
                    float(item.team_outcome.get("generated_tokens", 0))
                    for item in trajectories
                ),
                "mean_environment_return": fmean(
                    item.environment_return for item in trajectories
                ),
            }
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        append_trajectories(trajectory_path, trajectories)
        print(json.dumps(metrics, sort_keys=True))

    adapter_dir = args.output_dir / "adapter"
    policy.model.save_pretrained(adapter_dir)
    policy.tokenizer.save_pretrained(adapter_dir)


if __name__ == "__main__":
    main()
