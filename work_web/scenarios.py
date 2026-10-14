"""Public factories for reproducible work-web example jobs."""

from __future__ import annotations

from typing import Any

from orchestrator.contracts import GovernanceAdapter, GovernanceMode
from orchestrator.governance import AllowAllGovernanceAdapter, ScopePolicyMirror
from orchestrator.scenario import (
    EvidenceReviewBundle,
    EvidenceReviewWorker,
    InMemoryToolExecutor,
    build_evidence_review_scenario as build_evidence_review_bundle,
)


def build_evidence_review_scenario(**kwargs: Any):
    """Return a ready-to-reset RL episode for the synthetic evidence review."""
    from orchestrator.episode import AsyncEpisode

    return AsyncEpisode(**kwargs)


def build_evidence_review_episode(**kwargs: Any):
    """Explicitly named alias for ``build_evidence_review_scenario``."""
    return build_evidence_review_scenario(**kwargs)


def _set_condition_adapter(
    kwargs: dict[str, Any], mode: GovernanceMode, adapter: GovernanceAdapter
) -> None:
    configured = kwargs.get("governance_adapter")
    if configured is not None:
        mode_value = getattr(configured, "mode", None)
        if mode_value is None:
            raise TypeError("governance adapter must expose a mode label")
        configured_mode = getattr(mode_value, "value", str(mode_value))
        if configured_mode != mode.value:
            raise ValueError(
                f"{mode.value} episode cannot use an adapter labeled {configured_mode}"
            )
    kwargs["governance_adapter"] = configured if configured is not None else adapter


def build_generic_episode(**kwargs: Any):
    """Unrestricted runtime condition, with deterministic synthetic tools."""
    _set_condition_adapter(
        kwargs,
        GovernanceMode.GENERIC,
        AllowAllGovernanceAdapter(GovernanceMode.GENERIC),
    )
    return build_evidence_review_scenario(**kwargs)


def build_prompt_policy_episode(**kwargs: Any):
    """Prompt-instruction condition with no runtime gate (clearly labeled)."""
    _set_condition_adapter(
        kwargs,
        GovernanceMode.PROMPT_POLICY,
        AllowAllGovernanceAdapter(GovernanceMode.PROMPT_POLICY),
    )
    return build_evidence_review_scenario(**kwargs)


def build_policy_mirror_episode(**kwargs: Any):
    """Deterministic local policy mirror; this is not a live ArmorIQ session."""
    _set_condition_adapter(
        kwargs,
        GovernanceMode.POLICY_MIRROR,
        ScopePolicyMirror(
            allowed_tools_by_role={
                "researcher": frozenset({"case.read_source"}),
                "auditor": frozenset({"case.audit_sources"}),
                "reviewer": frozenset({"case.compare_sources"}),
                "writer": frozenset({"workspace.create_draft"}),
                "publisher": frozenset({"workspace.submit_brief"}),
                "archivist": frozenset({"audit.archive_source"}),
            }
        ),
    )
    return build_evidence_review_scenario(**kwargs)


def build_armoriq_live_episode(
    *, governance_adapter: GovernanceAdapter, **kwargs: Any
):
    """Construct a live integration episode only with an explicit SDK adapter.

    Pass the concrete ArmorIQGovernanceAdapter or an explicitly labeled test
    adapter. The adapter must keep live results remote and report native
    child-delegation support only when the selected SDK confirms it.
    """
    if governance_adapter.mode != GovernanceMode.ARMIQ_LIVE:
        raise ValueError("live episode requires an adapter labeled armoriq_live")
    if not callable(getattr(governance_adapter, "authorize", None)):
        raise TypeError("live adapter must implement authorize(request)")
    if not callable(getattr(governance_adapter, "report", None)):
        raise TypeError("live adapter must implement report(request, decision, result)")
    kwargs["governance_adapter"] = governance_adapter
    return build_evidence_review_scenario(**kwargs)


__all__ = [
    "EvidenceReviewBundle",
    "EvidenceReviewWorker",
    "InMemoryToolExecutor",
    "build_evidence_review_bundle",
    "build_evidence_review_episode",
    "build_evidence_review_scenario",
    "build_armoriq_live_episode",
    "build_generic_episode",
    "build_policy_mirror_episode",
    "build_prompt_policy_episode",
]
