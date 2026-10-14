"""Paired held-out evaluation for solo-agent ArmorEnv training arms."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from agents import HuggingFaceJSONPolicy
from armoriq_gate import ArmorIQGate
from evaluation.metrics import paired_bootstrap_difference
from tasks import build_task
from train import Trajectory, run_rollout


@dataclass(frozen=True)
class EvaluationMetrics:
    label: str
    condition: str
    episodes: int
    verified_success_rate: float
    policy_compliant_success_rate: float
    mean_return: float
    mean_steps: float
    unsafe_attempt_rate: float
    unsafe_side_effects_per_episode: float
    blocked_action_attempts_per_episode: float
    approval_holds_per_episode: float


def evaluate_policy(
    policy: HuggingFaceJSONPolicy,
    *,
    label: str,
    condition: str,
    episodes: int,
    seed: int,
    gate_factory=None,
) -> tuple[EvaluationMetrics, list[Trajectory]]:
    """Evaluate on identical deterministic task seeds for every trained arm."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    profiles = ("analyst", "reviewer", "contractor")
    trajectories: list[Trajectory] = []
    for episode_index in range(episodes):
        episode_seed = seed + 10_000 + episode_index
        profile = profiles[episode_index % len(profiles)]
        task, _ = build_task(profile, episode_seed)
        trajectories.append(
            run_rollout(
                policy,
                condition=condition,
                task=task,
                seed=seed + 20_000 + episode_index,
                gate_factory=gate_factory,
            )
        )
    total_tool_calls = max(sum(item.tool_calls for item in trajectories), 1)
    metrics = EvaluationMetrics(
        label=label,
        condition=condition,
        episodes=episodes,
        verified_success_rate=fmean(float(item.completed) for item in trajectories),
        policy_compliant_success_rate=fmean(
            float(item.completed and item.policy_compliant) for item in trajectories
        ),
        mean_return=fmean(item.return_ for item in trajectories),
        mean_steps=fmean(len(item.turns) for item in trajectories),
        unsafe_attempt_rate=sum(item.unsafe_attempts for item in trajectories) / total_tool_calls,
        unsafe_side_effects_per_episode=fmean(
            item.unsafe_side_effects for item in trajectories
        ),
        blocked_action_attempts_per_episode=fmean(item.blocks for item in trajectories),
        approval_holds_per_episode=fmean(item.holds for item in trajectories),
    )
    return metrics, trajectories


def load_adapter_policy(base_model: str, adapter_path: Path) -> HuggingFaceJSONPolicy:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install transformers and peft before evaluation.") from error
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return HuggingFaceJSONPolicy(model=model, tokenizer=tokenizer, temperature=0.0)


def _adapter_specs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for raw in args.adapter or ():
        if "=" not in raw:
            parser.error("--adapter must be LABEL=PATH")
        label, path = raw.split("=", 1)
        if not label.strip() or not path.strip():
            parser.error("--adapter must be LABEL=PATH")
        specs.append((label.strip(), Path(path)))
    if not specs:
        # Keep the first scaffold's two-adapter flags usable for existing runs.
        if args.plain_adapter is None or args.armor_adapter is None:
            parser.error("Provide at least two --adapter LABEL=PATH values")
        specs.extend(
            [("TaskRL", args.plain_adapter), ("ArmorRL", args.armor_adapter)]
        )
    if len(specs) < 2:
        parser.error("Paired evaluation needs at least two adapters")
    labels = [label for label, _ in specs]
    if len(set(labels)) != len(labels):
        parser.error("Adapter labels must be unique")
    return specs


def _save_report(
    path: Path,
    metrics: list[EvaluationMetrics],
    traces: dict[str, list[Trajectory]],
    paired_differences: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": [asdict(item) for item in metrics],
        "paired_success_differences": paired_differences,
        "traces": {
            label: [asdict(trace) for trace in values]
            for label, values in traces.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired held-out ArmorEnv evaluation")
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter", action="append", metavar="LABEL=PATH",
        help="Adapter to evaluate; repeat to compare multiple training arms.",
    )
    parser.add_argument("--plain-adapter", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--armor-adapter", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--condition",
        choices=("generic", "prompt_policy", "policy_mirror", "armoriq_live"),
        default="policy_mirror",
    )
    parser.add_argument("--user-email", help="Required for --condition armoriq_live")
    parser.add_argument("--llm-name", default="qwen3")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    if args.condition == "armoriq_live" and not args.user_email:
        parser.error("--user-email is required with --condition armoriq_live")

    adapter_specs = _adapter_specs(args, parser)
    loaded = [(label, load_adapter_policy(args.base_model, path)) for label, path in adapter_specs]
    gate_factory = None
    if args.condition == "armoriq_live":
        gate_factory = lambda _profile: ArmorIQGate(
            user_email=args.user_email,
            llm_name=args.llm_name,
        )

    all_metrics: list[EvaluationMetrics] = []
    all_traces: dict[str, list[Trajectory]] = {}
    success_by_label: dict[str, dict[str, float]] = {}
    for label, policy in loaded:
        metrics, trajectories = evaluate_policy(
            policy,
            label=label,
            condition=args.condition,
            episodes=args.episodes,
            seed=args.seed,
            gate_factory=gate_factory,
        )
        all_metrics.append(metrics)
        all_traces[label] = trajectories
        success_by_label[label] = {
            trajectory.task_id: float(trajectory.completed)
            for trajectory in trajectories
        }

    baseline_label = all_metrics[0].label
    paired = {
        f"{label}_minus_{baseline_label}": asdict(
            paired_bootstrap_difference(
                success_by_label[baseline_label],
                success_by_label[label],
                resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
        )
        for label, _ in loaded[1:]
    }
    _save_report(args.report, all_metrics, all_traces, paired)
    print(json.dumps({"metrics": [asdict(item) for item in all_metrics], "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
