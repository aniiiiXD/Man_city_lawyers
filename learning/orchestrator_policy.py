"""JSON-action language-model policy for the async coordinator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


def parse_object(completion: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model completion, if well formed."""

    candidate = completion.strip()
    if "```" in candidate:
        blocks = [piece.strip() for piece in candidate.split("```") if "{" in piece]
        if blocks:
            candidate = blocks[0]
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip()
    start = candidate.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        return None
    return dict(value)


@dataclass
class HuggingFaceOrchestratorPolicy:
    """Generate one structured orchestrator action from a causal LM."""

    model: Any
    tokenizer: Any
    max_new_tokens: int = 192
    temperature: float = 0.7

    def act(self, prompt: str) -> tuple[dict[str, Any], str]:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation.update(do_sample=True, temperature=self.temperature)
        else:
            generation["do_sample"] = False
        with torch.no_grad():
            output = self.model.generate(**encoded, **generation)
        completion_ids = output[0, encoded["input_ids"].shape[1] :]
        completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        action = parse_object(completion)
        # The environment treats this explicit invalid action as a penalized
        # no-op, keeping malformed generations inside the trajectory.
        return action or {"kind": "invalid"}, completion

