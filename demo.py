"""CLI demo that prints a complete policy-governed action trace."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from agents import ScriptedReferencePolicy
from armoriq_gate import ArmorIQGate
from environment import ArmorEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a visible ArmorEnv episode")
    parser.add_argument("--mode", choices=("plain", "armored"), default="armored")
    parser.add_argument("--profile", choices=("analyst", "reviewer", "contractor"), default="analyst")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--live-armoriq", action="store_true")
    parser.add_argument("--user-email")
    parser.add_argument("--llm-name", default="qwen3")
    args = parser.parse_args()

    gate_factory = None
    if args.live_armoriq:
        if not args.user_email:
            parser.error("--user-email is required with --live-armoriq")
        gate_factory = lambda _profile: ArmorIQGate(user_email=args.user_email, llm_name=args.llm_name)

    env = ArmorEnv(mode=args.mode, gate_factory=gate_factory)
    policy = ScriptedReferencePolicy()
    observation, metadata = env.reset(seed=args.seed, profile_name=args.profile)
    print(json.dumps({"event": "reset", **metadata, "observation": observation}, indent=2))

    while True:
        prompt = env.render_for_agent(observation)
        action, _ = policy.act(prompt)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            json.dumps(
                {
                    "event": "step",
                    "action": action.to_dict(),
                    "reward": reward,
                    "gate": info["gate"],
                    "result": info["result"],
                    "verification": info["verification"],
                },
                indent=2,
            )
        )
        if terminated or truncated:
            print(json.dumps({"event": "complete", "trace": [asdict(item) for item in env.trace]}, indent=2))
            return


if __name__ == "__main__":
    main()
