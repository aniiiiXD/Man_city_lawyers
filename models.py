"""Shared data contracts for the ArmorIQ agent-RL environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class Decision(StrEnum):
    ALLOW = "allow"
    HOLD = "hold"
    BLOCK = "block"


class ExecutionMode(StrEnum):
    """Observable execution conditions used by the solo experiment."""

    GENERIC = "generic"
    PROMPT_POLICY = "prompt_policy"
    POLICY_MIRROR = "policy_mirror"
    ARMORIQ_LIVE = "armoriq_live"


class ActionKind(StrEnum):
    TOOL = "tool"
    REQUEST_APPROVAL = "request_approval"
    REPLAN = "replan"
    FINISH = "finish"


@dataclass(frozen=True)
class AgentAction:
    """One structured decision made by the policy at an environment step."""

    kind: ActionKind
    tool: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""
    revised_goal: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentAction":
        if not isinstance(raw, Mapping):
            raise TypeError("Action must be a JSON object")
        kind_value = raw.get("kind")
        if not isinstance(kind_value, str):
            raise ValueError("Action must include string field 'kind'")
        kind = ActionKind(kind_value)
        arguments = raw.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise TypeError("Action arguments must be a JSON object")
        tool = raw.get("tool")
        if tool is not None and not isinstance(tool, str):
            raise TypeError("Action tool must be a string")
        if kind == ActionKind.TOOL and not tool:
            raise ValueError("Tool action needs a qualified tool name")
        rationale = raw.get("rationale", "")
        revised_goal = raw.get("revised_goal")
        if not isinstance(rationale, str):
            raise TypeError("Action rationale must be a string")
        if revised_goal is not None and not isinstance(revised_goal, str):
            raise TypeError("Revised goal must be a string")
        return cls(
            kind=kind,
            tool=tool,
            arguments=arguments,
            rationale=rationale,
            revised_goal=revised_goal,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    """A fully qualified MCP-style tool invocation."""

    mcp: str
    action: str
    params: Mapping[str, Any]

    @property
    def qualified_name(self) -> str:
        return f"{self.mcp}.{self.action}"


@dataclass(frozen=True)
class GateDecision:
    decision: Decision
    reason: str
    plan_id: str | None = None
    delegation_id: str | None = None
    remote_result: Any | None = None
    source: str = "unspecified"


@dataclass(frozen=True)
class StepRecord:
    episode_id: str
    step: int
    observation: Mapping[str, Any]
    action: AgentAction
    tool_call: ToolCall | None
    gate: GateDecision | None
    reward: float
    reward_parts: Mapping[str, float]
    result: Mapping[str, Any]
    terminated: bool
    truncated: bool

