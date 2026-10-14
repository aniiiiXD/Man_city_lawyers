"""Grouped trajectory GRPO for a learned async-work orchestrator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Sequence

from learning.orchestrator_rollout import (
    AsyncEpisode,
    LogProbabilityRecorder,
    OrchestratorTrajectory,
    TextActionPolicy,
    run_orchestrator_rollout,
)


@dataclass(frozen=True)
class AsyncGRPOConfig:
    group_size: int = 4
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be between zero and one")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


def normalized_group_advantages(returns: Sequence[float]) -> list[float]:
    """Normalize one task group's episode returns; handle zero-variance groups."""

    if any(not isfinite(float(value)) for value in returns):
        raise ValueError("returns must be finite")
    if len(returns) < 2:
        return [0.0 for _ in returns]
    deviation = pstdev(returns)
    if deviation < 1e-8:
        return [0.0 for _ in returns]
    center = fmean(returns)
    return [(float(value) - center) / deviation for value in returns]


class AsyncGRPOTrainer:
    """Collect matched async episodes and update only trainable policy params."""

    def __init__(
        self,
        policy: TextActionPolicy,
        config: AsyncGRPOConfig,
        *,
        episode_factory: Callable[[], AsyncEpisode],
        logprobs: Any,
        learning_rate: float = 2e-5,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Install the rl extra to train an orchestrator") from error
        model = getattr(policy, "model", None)
        if model is None:
            raise TypeError("Trainable policy must expose a torch `model`")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("Policy has no trainable parameters")
        if learning_rate <= 0 or not isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        self.policy = policy
        self.config = config
        self.episode_factory = episode_factory
        self.logprobs = logprobs
        self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    def collect_group(
        self,
        *,
        task_id: str,
        condition: str,
        task_seed: int,
        max_decisions: int = 256,
    ) -> list[OrchestratorTrajectory]:
        """Roll out the same task/seed under independent policy samples."""

        trajectories: list[OrchestratorTrajectory] = []
        self.policy.model.eval()
        tokenizer = getattr(self.policy, "tokenizer", None)
        token_counter = None
        if tokenizer is not None:
            token_counter = lambda completion: len(
                tokenizer.encode(completion, add_special_tokens=False)
            )
        for _ in range(self.config.group_size):
            trajectory = run_orchestrator_rollout(
                self.policy,
                self.episode_factory(),
                task_id=task_id,
                condition=condition,
                seed=task_seed,
                logprobs=self.logprobs,
                token_counter=token_counter,
                max_decisions=max_decisions,
            )
            trajectories.append(trajectory)
        return trajectories

    def update(self, trajectories: Sequence[OrchestratorTrajectory]) -> dict[str, float]:
        import torch

        if len(trajectories) < 2:
            raise ValueError("GRPO update requires at least two trajectories")
        task_ids = {trajectory.task_id for trajectory in trajectories}
        seeds = {trajectory.seed for trajectory in trajectories}
        if len(task_ids) != 1 or len(seeds) != 1:
            raise ValueError("GRPO group must use the same task and seed")
        advantages = normalized_group_advantages([trajectory.return_ for trajectory in trajectories])
        losses: list[Any] = []
        # Keep dropout disabled so clipped ratios compare the same policy
        # distribution used to collect each completion.
        self.policy.model.eval()
        for trajectory, advantage in zip(trajectories, advantages, strict=True):
            if advantage == 0.0:
                continue
            for turn in trajectory.turns:
                if turn.old_logprob is None:
                    continue
                new_logprob = self.logprobs.differentiable_mean_log_probability(
                    turn.prompt, turn.completion
                )
                old_logprob = torch.tensor(
                    turn.old_logprob,
                    dtype=new_logprob.dtype,
                    device=new_logprob.device,
                )
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
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.policy.model.parameters() if parameter.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "mean_return": fmean(item.return_ for item in trajectories),
            "mean_advantage": fmean(advantages),
            "verified_success_rate": fmean(
                float(item.verified_complete and item.policy_compliant) for item in trajectories
            ),
        }


def append_trajectories(path: Path, trajectories: Sequence[OrchestratorTrajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(asdict(trajectory), sort_keys=True) + "\n")

