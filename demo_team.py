"""Print one complete async work-web episode and its governance trace."""

from __future__ import annotations

import argparse
import json

from orchestrator.episode import RuleBasedEpisodePolicy
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
    parser = argparse.ArgumentParser(description="Show async coordination, approvals, and gate events")
    parser.add_argument("--condition", choices=tuple(EPISODES), default="policy_mirror")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=128)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    episode = EPISODES[args.condition]()
    policy = RuleBasedEpisodePolicy()
    observation, reset_info = episode.reset(seed=args.seed)
    print(json.dumps({"event": "job.started", **reset_info}, indent=2))
    for decision_index in range(args.max_steps):
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = episode.step(action)
        print(
            json.dumps(
                {
                    "event": "orchestrator.decision",
                    "decision": decision_index + 1,
                    "action": action,
                    "reward": reward,
                    "virtual_time": observation.get("virtual_time"),
                    "active_tasks": observation.get("active_task_ids"),
                    "pending_approvals": observation.get("pending_approvals"),
                    "events": info.get("events", []),
                },
                ensure_ascii=False,
            )
        )
        if terminated or truncated:
            print(
                json.dumps(
                    {
                        "event": "job.finished",
                        "condition": args.condition,
                        "terminated": terminated,
                        "truncated": truncated,
                        "verification": info.get("verification"),
                        "artifacts": observation.get("artifacts", []),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
    print(json.dumps({"event": "job.stopped", "reason": "demo decision budget reached"}))


if __name__ == "__main__":
    main()
