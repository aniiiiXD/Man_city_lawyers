"""Paired held-out evaluation for learned async-work coordinators."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from evaluate import load_adapter_policy
from evaluation.metrics import paired_bootstrap_difference
from orchestrator.armoriq_adapter import (
    DEFAULT_TOOL_BINDINGS,
    ArmorIQGovernanceAdapter,
    ArmorIQToolBinding,
)
from learning.orchestrator_policy import HuggingFaceOrchestratorPolicy
from learning.orchestrator_rollout import OrchestratorTrajectory, run_orchestrator_rollout
from work_web.scenarios import (
    build_armoriq_live_episode,
    build_generic_episode,
    build_policy_mirror_episode,
    build_prompt_policy_episode,
)


EPISODES = {
    "generic": build_generic_episode,
    "prompt_policy": build_prompt_policy_episode,
    "policy_mirror": build_policy_mirror_episode,
}
CONDITIONS = (*EPISODES, "armoriq_live")


@dataclass(frozen=True)
class TeamEvaluationMetrics:
    label: str
    condition: str
    episodes: int
    verified_success_rate: float
    policy_compliant_success_rate: float
    evidence_supported_rate: float
    mean_team_return: float
    mean_worker_spawns: float
    mean_redundant_workers: float
    mean_unsafe_attempts: float
    mean_unauthorized_side_effects: float
    mean_retries: float
    mean_virtual_ticks: float
    mean_generated_tokens: float
    mean_tool_calls: float
    mean_recovered_holds: float


def _metric(label: str, condition: str, trajectories: list[OrchestratorTrajectory]) -> TeamEvaluationMetrics:
    outcomes = [item.team_outcome for item in trajectories]
    completed = [item.verified_complete for item in trajectories]
    return TeamEvaluationMetrics(
        label=label,
        condition=condition,
        episodes=len(trajectories),
        verified_success_rate=fmean(float(value) for value in completed),
        policy_compliant_success_rate=fmean(
            float(item.verified_complete and item.policy_compliant) for item in trajectories
        ),
        evidence_supported_rate=fmean(float(row.get("evidence_supported", False)) for row in outcomes),
        mean_team_return=fmean(item.return_ for item in trajectories),
        mean_worker_spawns=fmean(float(row.get("worker_spawns", 0)) for row in outcomes),
        mean_redundant_workers=fmean(float(row.get("redundant_workers", 0)) for row in outcomes),
        mean_unsafe_attempts=fmean(float(row.get("unsafe_attempts", 0)) for row in outcomes),
        mean_unauthorized_side_effects=fmean(
            float(row.get("unauthorized_side_effects", 0)) for row in outcomes
        ),
        mean_retries=fmean(float(row.get("retries", 0)) for row in outcomes),
        mean_virtual_ticks=fmean(float(row.get("virtual_ticks", 0)) for row in outcomes),
        mean_generated_tokens=fmean(
            float(row.get("generated_tokens", 0)) for row in outcomes
        ),
        mean_tool_calls=fmean(float(row.get("tool_calls", 0)) for row in outcomes),
        mean_recovered_holds=fmean(float(row.get("recovered_holds", 0)) for row in outcomes),
    )


def _parse_adapters(raw_values: list[str], parser: argparse.ArgumentParser) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for raw in raw_values:
        if "=" not in raw:
            parser.error("--adapter must be LABEL=PATH")
        label, path = raw.split("=", 1)
        if not label.strip() or not path.strip():
            parser.error("--adapter must be LABEL=PATH")
        result.append((label.strip(), Path(path)))
    if not result:
        parser.error("Provide at least one --adapter LABEL=PATH")
    if len({label for label, _ in result}) != len(result):
        parser.error("Adapter labels must be unique")
    return result


def _load_tool_bindings(
    path: Path | None, *, default_mcp: str, parser: argparse.ArgumentParser
) -> dict[str, ArmorIQToolBinding]:
    if path is None:
        return {
            name: ArmorIQToolBinding(
                default_mcp,
                binding.action,
                seed_parameter=binding.seed_parameter,
            )
            for name, binding in DEFAULT_TOOL_BINDINGS.items()
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"Could not read --tool-bindings-json: {error}")
    if not isinstance(raw, dict) or not raw:
        parser.error("--tool-bindings-json must contain a non-empty object")
    bindings: dict[str, ArmorIQToolBinding] = {}
    for tool, value in raw.items():
        if (
            not isinstance(tool, str)
            or not isinstance(value, dict)
            or not isinstance(value.get("mcp"), str)
            or not isinstance(value.get("action"), str)
            or (
                value.get("seed_parameter") is not None
                and not isinstance(value.get("seed_parameter"), str)
            )
        ):
            parser.error(
                "Each binding needs mcp/action strings and optional seed_parameter string"
            )
        try:
            bindings[tool] = ArmorIQToolBinding(
                value["mcp"],
                value["action"],
                seed_parameter=value.get("seed_parameter"),
            )
        except ValueError as error:
            parser.error(str(error))
    return bindings


def _build_episode(condition: str, live_adapter: ArmorIQGovernanceAdapter | None):
    if condition == "armoriq_live":
        if live_adapter is None:
            raise RuntimeError("Live condition requires an ArmorIQ adapter")
        return build_armoriq_live_episode(governance_adapter=live_adapter)
    return EPISODES[condition]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate async orchestrator adapters on matched seeds")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--user-email", help="Required for live ArmorIQ evaluation")
    parser.add_argument("--llm-name", default="qwen3")
    parser.add_argument("--mcp-name", default="work-web-mcp")
    parser.add_argument(
        "--tool-bindings-json",
        type=Path,
        help="JSON map from Work Web tool names to registered {mcp, action} names",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-decisions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--report", type=Path, default=Path("artifacts/team-evaluation.json"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_decisions < 1:
        parser.error("--episodes and --max-decisions must be positive")

    adapters = _parse_adapters(args.adapter, parser)
    conditions = args.condition or list(EPISODES)
    if "armoriq_live" in conditions and not args.user_email:
        parser.error("--user-email is required when --condition armoriq_live is selected")
    live_adapter = None
    if "armoriq_live" in conditions:
        bindings = _load_tool_bindings(
            args.tool_bindings_json,
            default_mcp=args.mcp_name,
            parser=parser,
        )
        try:
            live_adapter = ArmorIQGovernanceAdapter(
                user_email=args.user_email,
                llm_name=args.llm_name,
                tool_bindings=bindings,
            )
        except RuntimeError as error:
            parser.error(str(error))
    results: list[dict[str, object]] = []
    traces: dict[str, list[OrchestratorTrajectory]] = {}
    success_by_run: dict[str, dict[str, float]] = {}
    try:
        for label, adapter_path in adapters:
            base_policy = load_adapter_policy(args.base_model, adapter_path)
            policy = HuggingFaceOrchestratorPolicy(
                model=base_policy.model,
                tokenizer=base_policy.tokenizer,
                temperature=0.0,
            )
            policy.model.eval()
            for condition in conditions:
                run_key = f"{label}:{condition}"
                trajectories: list[OrchestratorTrajectory] = []
                for episode_index in range(args.episodes):
                    episode_seed = args.seed + 10_000 + episode_index
                    trajectory = run_orchestrator_rollout(
                        policy,
                        _build_episode(condition, live_adapter),
                        task_id=f"synthetic-evidence-review:{episode_seed}",
                        condition=condition,
                        seed=episode_seed,
                        max_decisions=args.max_decisions,
                        token_counter=lambda completion, tokenizer=policy.tokenizer: len(
                            tokenizer.encode(completion, add_special_tokens=False)
                        ),
                    )
                    trajectories.append(trajectory)
                metrics = _metric(label, condition, trajectories)
                results.append(asdict(metrics))
                traces[run_key] = trajectories
                success_by_run[run_key] = {
                    item.task_id: float(item.verified_complete) for item in trajectories
                }
    finally:
        if live_adapter is not None:
            live_adapter.close()

    paired: dict[str, object] = {}
    if len(adapters) > 1:
        control_label = adapters[0][0]
        for condition in conditions:
            control_key = f"{control_label}:{condition}"
            for label, _ in adapters[1:]:
                treatment_key = f"{label}:{condition}"
                paired[f"model:{treatment_key}-minus-{control_key}"] = asdict(
                    paired_bootstrap_difference(
                        success_by_run[control_key],
                        success_by_run[treatment_key],
                        resamples=args.bootstrap_resamples,
                        seed=args.seed,
                    )
                )
    if len(conditions) > 1:
        control_condition = conditions[0]
        for label, _ in adapters:
            control_key = f"{label}:{control_condition}"
            for condition in conditions[1:]:
                treatment_key = f"{label}:{condition}"
                paired[f"condition:{treatment_key}-minus-{control_key}"] = asdict(
                    paired_bootstrap_difference(
                        success_by_run[control_key],
                        success_by_run[treatment_key],
                        resamples=args.bootstrap_resamples,
                        seed=args.seed,
                    )
                )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "metrics": results,
                "paired_success_differences": paired,
                "traces": {
                    key: [asdict(item) for item in values]
                    for key, values in traces.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"metrics": results, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
