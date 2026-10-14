"""Agent policies used for rollouts, including a trainable Hugging Face policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from models import ActionKind, AgentAction


class AgentPolicy(Protocol):
    def act(self, prompt: str) -> tuple[AgentAction, str]: ...


def parse_action(text: str) -> AgentAction:
    """Parse a model completion, tolerating Markdown fenced JSON."""
    candidate = text.strip()
    if "```" in candidate:
        chunks = [chunk.strip() for chunk in candidate.split("```") if "{" in chunk]
        candidate = chunks[-1].removeprefix("json").strip() if chunks else candidate
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        return AgentAction(ActionKind.REPLAN, rationale="Could not parse a structured action")
    try:
        return AgentAction.from_dict(json.loads(candidate[start : end + 1]))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return AgentAction(ActionKind.REPLAN, rationale="Invalid structured action")


def action_json(action: AgentAction) -> str:
    return json.dumps(action.to_dict(), separators=(",", ":"), sort_keys=True)


class ScriptedReferencePolicy:
    """Deterministic smoke/demo policy. It is not used as an RL baseline."""

    def __init__(self) -> None:
        self._phase = 0

    def act(self, prompt: str) -> tuple[AgentAction, str]:
        if "Approval pending: True" in prompt:
            action = AgentAction(ActionKind.REQUEST_APPROVAL, rationale="A held action needs human approval")
        elif "'read_documents': []" in prompt:
            target = "incident_42" if "incident" in prompt.lower() else "weekly_metrics"
            if "contract" in prompt.lower():
                target = "contract_terms"
            action = AgentAction(
                ActionKind.TOOL,
                tool="records.read",
                arguments={"document_id": target},
                rationale="Read the document required by the task",
            )
        elif "'drafted': False" in prompt:
            action = AgentAction(ActionKind.TOOL, tool="workspace.draft", rationale="Create the requested brief")
        elif "'attached_documents': []" in prompt:
            target = "incident_42" if "incident" in prompt.lower() else "weekly_metrics"
            if "contract" in prompt.lower():
                target = "contract_terms"
            action = AgentAction(
                ActionKind.TOOL,
                tool="workspace.attach",
                arguments={"document_id": target},
                rationale="Attach the verified supporting document",
            )
        else:
            recipient = "operations@acme.test"
            if "security" in prompt.lower():
                recipient = "security@acme.test"
            if "legal" in prompt.lower():
                recipient = "legalops@acme.test"
            action = AgentAction(
                ActionKind.TOOL,
                tool="workspace.send",
                arguments={"recipient": recipient},
                rationale="Deliver the completed internal brief",
            )
        return action, action_json(action)


@dataclass
class HuggingFaceJSONPolicy:
    """Causal-LM policy that emits one JSON action per environment turn."""

    model: Any
    tokenizer: Any
    max_new_tokens: int = 192
    temperature: float = 0.8

    def act(self, prompt: str) -> tuple[AgentAction, str]:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = generated[0, encoded["input_ids"].shape[1] :]
        completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return parse_action(completion), completion
