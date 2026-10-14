"""Multi-turn grouped RL training for ArmorEnv.

This is a compact GRPO implementation for JSON tool-calling policies. It
collects several complete trajectories for the *same* task, normalizes their
verified returns within that group, then performs clipped policy-gradient
updates on every agent turn. LoRA keeps the trainable parameter set small.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Iterable

from agents import AgentPolicy, HuggingFaceJSONPolicy
from environment import ArmorEnv
from models import AgentAction
from tasks import TaskSpec, build_task


@dataclass
class RolloutTurn:
    prompt: str
    completion: str
    action: dict[str, Any]
    reward: float
    old_logprob: float | None = None


@dataclass
class Trajectory:
    task_id: str
    condition: str
    seed: int
    turns: list[RolloutTurn] = field(default_factory=list)
    return_: float = 0.0
    completed: bool = False
    policy_compliant: bool = False
    blocks: int = 0
    holds: int = 0
    unsafe_attempts: int = 0
    unsafe_side_effects: int = 0
    tool_calls: int = 0
    governance_sources: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 4
    iterations: int = 100
    learning_rate: float = 2e-5
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    profiles: tuple[str, ...] = ("analyst", "reviewer", "contractor")


def run_rollout(
    policy: AgentPolicy,
    *,
    condition: str,
    task: TaskSpec,
    seed: int,
    gate_factory: Callable[[Any], Any] | None = None,
    record_logprobs: "LogProbabilityModel | None" = None,
) -> Trajectory:
    env = ArmorEnv(mode=condition, gate_factory=gate_factory)
    observation, _ = env.reset(seed=seed, profile_name=task.profile_name, task=task)
    trajectory = Trajectory(task_id=task.task_id, condition=condition, seed=seed)

    while True:
        prompt = env.render_for_agent(observation)
        action, completion = policy.act(prompt)
        before = env.world.snapshot() if action.kind.value == "tool" else None
        classified_unsafe = False
        if action.kind.value == "tool":
            trajectory.tool_calls += 1
            try:
                classified_unsafe = env.profile.decide(env._to_tool_call(action)).decision.value == "block"
            except (TypeError, ValueError):
                classified_unsafe = True
        old_logprob = record_logprobs.mean_log_probability(prompt, completion) if record_logprobs else None
        observation, reward, terminated, truncated, info = env.step(action)
        trajectory.turns.append(
            RolloutTurn(
                prompt=prompt,
                completion=completion,
                action=action.to_dict(),
                reward=reward,
                old_logprob=old_logprob,
            )
        )
        trajectory.return_ += reward
        gate = info.get("gate") or {}
        trajectory.blocks += int(gate.get("decision") == "block")
        trajectory.holds += int(gate.get("decision") == "hold")
        trajectory.unsafe_attempts += int(classified_unsafe)
        if classified_unsafe and info.get("result", {}).get("ok") and before != env.world.snapshot():
            trajectory.unsafe_side_effects += 1
        source = gate.get("source")
        if source:
            trajectory.governance_sources[str(source)] = (
                trajectory.governance_sources.get(str(source), 0) + 1
            )
        if terminated or truncated:
            verification = info.get("verification_detail") or {}
            trajectory.completed = bool(verification.get("complete", False))
            trajectory.policy_compliant = (
                bool(verification.get("policy_compliant", False))
                and trajectory.unsafe_attempts == 0
                and trajectory.unsafe_side_effects == 0
            )
            return trajectory


def seed_everything(seed: int) -> None:
    """Seed available RNGs so model sampling and task generation can be replayed."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def group_advantages(trajectories: Iterable[Trajectory]) -> list[float]:
    rewards = [trajectory.return_ for trajectory in trajectories]
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    sigma = pstdev(rewards)
    if sigma < 1e-8:
        return [0.0 for _ in rewards]
    mean = fmean(rewards)
    return [(reward - mean) / sigma for reward in rewards]


class LogProbabilityModel:
    """Computes average action-token log probability for clipped GRPO updates."""

    def __init__(self, policy: HuggingFaceJSONPolicy) -> None:
        self.policy = policy

    def mean_log_probability(self, prompt: str, completion: str) -> float:
        import torch

        tokenizer, model = self.policy.tokenizer, self.policy.model
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if completion_ids.numel() == 0:
            return -100.0
        device = next(model.parameters()).device
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            start = prompt_ids.shape[1] - 1
            action_logits = logits[:, start:, :]
            action_targets = targets[:, start:]
            token_logprobs = torch.log_softmax(action_logits, dim=-1).gather(
                -1, action_targets.unsqueeze(-1)
            )
        return float(token_logprobs.mean().item())

    def differentiable_mean_log_probability(self, prompt: str, completion: str) -> Any:
        import torch

        tokenizer, model = self.policy.tokenizer, self.policy.model
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if completion_ids.numel() == 0:
            return torch.tensor(-100.0, device=next(model.parameters()).device)
        device = next(model.parameters()).device
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1).to(device)
        logits = model(input_ids=input_ids).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        start = prompt_ids.shape[1] - 1
        action_logits = logits[:, start:, :]
        action_targets = targets[:, start:]
        token_logprobs = torch.log_softmax(action_logits, dim=-1).gather(-1, action_targets.unsqueeze(-1))
        return token_logprobs.mean()


class MultiTurnGRPOTrainer:
    """Online grouped RL trainer for a LoRA-adapted causal language model."""

    def __init__(self, policy: HuggingFaceJSONPolicy, config: GRPOConfig) -> None:
        import torch

        self.policy = policy
        self.config = config
        self.logprobs = LogProbabilityModel(policy)
        self.optimizer = torch.optim.AdamW(policy.model.parameters(), lr=config.learning_rate)

    def collect_group(self, *, condition: str, task: TaskSpec, seed: int) -> list[Trajectory]:
        self.policy.model.eval()
        return [
            run_rollout(
                self.policy,
                condition=condition,
                task=task,
                seed=seed,
                record_logprobs=self.logprobs,
            )
            for _ in range(self.config.group_size)
        ]

    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        import torch

        advantages = group_advantages(trajectories)
        losses: list[Any] = []
        # Keep dropout disabled so the new/old action probabilities use the
        # same policy distribution; gradients still flow in eval mode.
        self.policy.model.eval()
        for trajectory, advantage in zip(trajectories, advantages, strict=True):
            if advantage == 0.0:
                continue
            for turn in trajectory.turns:
                if turn.old_logprob is None:
                    continue
                new_logprob = self.logprobs.differentiable_mean_log_probability(turn.prompt, turn.completion)
                old_logprob = torch.tensor(turn.old_logprob, dtype=new_logprob.dtype, device=new_logprob.device)
                ratio = torch.exp(new_logprob - old_logprob)
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantage
                losses.append(-torch.minimum(unclipped, clipped))
        if not losses:
            return {"loss": 0.0, "mean_return": fmean(item.return_ for item in trajectories)}
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "mean_return": fmean(item.return_ for item in trajectories),
            "mean_advantage": fmean(advantages),
        }


def write_trajectory(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(trajectory)) + "\n")


def load_lora_policy(model_name: str) -> HuggingFaceJSONPolicy:
    """Load a causal LM and attach trainable LoRA adapters."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install torch, transformers, and peft to train.") from error

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return HuggingFaceJSONPolicy(model=get_peft_model(model, lora), tokenizer=tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a policy-aware tool agent with multi-turn GRPO")
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--condition",
        choices=("generic", "prompt_policy", "policy_mirror"),
        required=True,
        help="Training arm; live ArmorIQ is reserved for smaller paired evaluations.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    config = GRPOConfig(iterations=args.iterations, group_size=args.group_size)
    policy = load_lora_policy(args.base_model)
    trainer = MultiTurnGRPOTrainer(policy, config)
    metrics_path = args.output_dir / "metrics.jsonl"
    trajectory_path = args.output_dir / "trajectories.jsonl"

    for iteration in range(config.iterations):
        profile = config.profiles[iteration % len(config.profiles)]
        task, _ = build_task(profile, args.seed + iteration)
        trajectories = trainer.collect_group(condition=args.condition, task=task, seed=args.seed + iteration * 100)
        metrics = trainer.update(trajectories)
        metrics.update({"iteration": iteration, "condition": args.condition, "task_id": task.task_id})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        for trajectory in trajectories:
            write_trajectory(trajectory_path, trajectory)
        print(json.dumps(metrics))

    policy.model.save_pretrained(args.output_dir / "adapter")
    policy.tokenizer.save_pretrained(args.output_dir / "adapter")


if __name__ == "__main__":
    main()
