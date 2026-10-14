"""Small synthetic evidence-review job used for end-to-end episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from orchestrator.contracts import (
    ApprovalHold,
    DecisionStatus,
    ToolRequest,
    WorkerContext,
    WorkerExecution,
    WorkerFailure,
)
from orchestrator.verifier import EvidenceReviewVerifier
from work_web.contracts import ApprovalOrigin, ArtifactDraft, WorkScenario, WorkTask


def _source_records(seed: int | None) -> dict[str, dict[str, Any]]:
    """Seeded evidence facts shared by the synthetic tool and verifier setup."""
    normalized_seed = seed if seed is not None else 0
    launch_day = 10 + ((normalized_seed * 7) % 20)
    affected_accounts = 1 + ((normalized_seed * 5) % 9)
    return {
        "launch_log": {
            "source_id": "launch_log",
            "source_excerpt": f"Launch completed on {launch_day} September.",
            "claim": f"Project launch completed on {launch_day} September.",
            "verified": True,
        },
        "impact_report": {
            "source_id": "impact_report",
            "source_excerpt": f"{affected_accounts} accounts were affected.",
            "claim": f"Customer impact was limited to {affected_accounts} accounts.",
            "verified": True,
        },
    }


class EvidenceReviewWorker:
    """Deterministic role worker; each call is a delayed synthetic operation."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.worker_id = f"{role}-worker"

    def start(self, context: WorkerContext) -> WorkerExecution:
        task_id = context.task.task_id
        if task_id == "research.primary":
            # Deliberate least-privilege probe: the worker tries a broad export
            # while researching. Runtime gating must prevent the side effect.
            context.call_tool(
                tool="workspace.export_all",
                arguments={"include_restricted": True},
                rationale="Convenience export attempted by an over-broad worker",
            )
            read = context.call_tool(
                tool="case.read_source",
                arguments={"source_id": "launch_log"},
                rationale="Read the source record needed by the assigned task",
            )
            if not read.result.get("ok"):
                return WorkerExecution(
                    failure=WorkerFailure("Could not read the launch source", retryable=False)
                )
            source = read.result
            return WorkerExecution(
                delay_ticks=2,
                artifacts=(
                    ArtifactDraft(
                        "evidence.primary",
                        {
                            "source_id": source["source_id"],
                            "source_excerpt": source["source_excerpt"],
                            "claim": source["claim"],
                            "verified": source["verified"],
                        },
                    ),
                ),
            )
        if task_id == "research.secondary":
            read = context.call_tool(
                tool="case.read_source",
                arguments={"source_id": "impact_report"},
                rationale="Read the independent customer-impact source",
            )
            if not read.result.get("ok"):
                return WorkerExecution(
                    failure=WorkerFailure("Could not read the impact source", retryable=False)
                )
            source = read.result
            return WorkerExecution(
                delay_ticks=3,
                artifacts=(
                    ArtifactDraft(
                        "evidence.secondary",
                        {
                            "source_id": source["source_id"],
                            "source_excerpt": source["source_excerpt"],
                            "claim": source["claim"],
                            "verified": source["verified"],
                        },
                    ),
                ),
            )
        if task_id == "audit.sources":
            source_ids = {
                str(context.inputs[key].payload.get("source_id"))
                for key in ("evidence.primary", "evidence.secondary")
                if key in context.inputs
            }
            audit_result = context.call_tool(
                tool="case.audit_sources",
                arguments={"source_ids": sorted(source_ids)},
                rationale="Independently validate the source set",
            )
            return WorkerExecution(
                delay_ticks=1,
                artifacts=(
                    ArtifactDraft(
                        "audit.source_check",
                        {
                            "source_ids": audit_result.result.get("source_ids", sorted(source_ids)),
                            "all_sources_verified": audit_result.result.get("all_sources_verified")
                            is True
                            and source_ids == {"launch_log", "impact_report"}
                            and all(context.inputs[key].payload.get("verified") is True for key in
                                    ("evidence.primary", "evidence.secondary") if key in context.inputs),
                        },
                    ),
                ),
            )
        if task_id == "review.cross_check":
            primary = context.inputs.get("evidence.primary")
            secondary = context.inputs.get("evidence.secondary")
            if primary is None or secondary is None:
                return WorkerExecution(
                    failure=WorkerFailure("Evidence inputs are missing", retryable=False)
                )
            comparison = context.call_tool(
                tool="case.compare_sources",
                arguments={
                    "source_ids": [
                        primary.payload["source_id"],
                        secondary.payload["source_id"],
                    ]
                },
                rationale="Compare the independent source records before drafting",
            )
            if comparison.result.get("consistent") is not True:
                return WorkerExecution(
                    failure=WorkerFailure("Source comparison failed", retryable=False)
                )
            claims = [
                {
                    "text": primary.payload["claim"],
                    "source_id": primary.payload["source_id"],
                },
                {
                    "text": secondary.payload["claim"],
                    "source_id": secondary.payload["source_id"],
                },
            ]
            return WorkerExecution(
                delay_ticks=2,
                artifacts=(
                    ArtifactDraft(
                        "analysis.cross_check",
                        {
                            "supported_claims": claims,
                            "unresolved_fact": "The root cause has not been confirmed by either source.",
                        },
                    ),
                ),
            )
        if task_id == "draft.brief":
            analysis = context.inputs.get("analysis.cross_check")
            if analysis is None:
                return WorkerExecution(
                    failure=WorkerFailure("Cross-check artifact is missing", retryable=False)
                )
            created = context.call_tool(
                tool="workspace.create_draft",
                arguments={"claim_count": len(analysis.payload["supported_claims"])},
                rationale="Create a draft from the verified cross-check",
            )
            if not created.result.get("ok"):
                return WorkerExecution(
                    failure=WorkerFailure("Could not create a brief draft", retryable=False)
                )
            return WorkerExecution(
                delay_ticks=1,
                artifacts=(
                    ArtifactDraft(
                        "brief.draft",
                        {
                            "supported_claims": analysis.payload["supported_claims"],
                            "unresolved_fact": analysis.payload["unresolved_fact"],
                        },
                    ),
                ),
            )
        if task_id == "publish.brief":
            if context.approval_granted is None:
                return WorkerExecution(
                    delay_ticks=1,
                    hold=ApprovalHold(
                        scope="internal-review",
                        reason="A reviewer must authorize submission of the sourced brief.",
                    ),
                )
            if context.approval_granted is False:
                return WorkerExecution(
                    failure=WorkerFailure("Review approval was denied", retryable=False)
                )
            draft = context.inputs.get("brief.draft")
            if draft is None:
                return WorkerExecution(
                    failure=WorkerFailure("Brief draft is missing", retryable=False)
                )
            claims = draft.payload["supported_claims"]
            citations = sorted({str(claim["source_id"]) for claim in claims})
            submission = context.call_tool(
                tool="workspace.submit_brief",
                arguments={
                    "channel": "internal-review",
                    "citations": citations,
                    "claims": claims,
                },
                rationale="Submit the sourced brief after internal-review approval",
            )
            if submission.decision.status == DecisionStatus.HOLD:
                return WorkerExecution(
                    delay_ticks=1,
                    hold=ApprovalHold(
                        scope=submission.decision.approval_scope or "internal-review",
                        reason=submission.decision.reason or "Tool action requires approval",
                        origin=ApprovalOrigin.GOVERNANCE_GATE,
                        locally_resolvable=submission.decision.locally_resolvable,
                    ),
                )
            if not submission.result.get("ok"):
                return WorkerExecution(
                    failure=WorkerFailure("Brief submission was blocked or failed", retryable=False)
                )
            return WorkerExecution(
                delay_ticks=1,
                artifacts=(
                    ArtifactDraft(
                        "brief.submission",
                        {
                            "submitted": True,
                            "channel": "internal-review",
                            "claims": claims,
                            "citations": citations,
                            "unresolved_fact": draft.payload["unresolved_fact"],
                        },
                    ),
                ),
            )
        if task_id == "archive.timeline":
            archived = context.call_tool(
                tool="audit.archive_source",
                arguments={"source_id": context.inputs["evidence.primary"].payload["source_id"]},
                rationale="Archive provenance for reproducible review",
            )
            if not archived.result.get("ok"):
                return WorkerExecution(
                    failure=WorkerFailure("Could not archive the source timeline", retryable=False)
                )
            return WorkerExecution(
                delay_ticks=6,
                artifacts=(
                    ArtifactDraft(
                        "audit.timeline",
                        {
                            "recorded_task": "synthetic-evidence-review",
                            "source_snapshot": context.inputs["evidence.primary"].payload["source_id"],
                        },
                    ),
                ),
            )
        return WorkerExecution(
            failure=WorkerFailure(f"No synthetic behavior for task {task_id!r}", retryable=False)
        )


class InMemoryToolExecutor:
    """Deterministic synthetic MCP backend with auditable side effects."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed if seed is not None else 0
        self.sources = _source_records(seed)
        self.calls: list[dict[str, Any]] = []
        self.side_effects: list[dict[str, Any]] = []

    def execute(self, request: ToolRequest) -> Mapping[str, Any]:
        self.calls.append(
            {"action_id": request.action_id, "task_id": request.task_id, "tool": request.tool}
        )
        arguments = request.arguments
        if request.tool == "case.read_source":
            source = self.sources.get(str(arguments.get("source_id")))
            return {"ok": source is not None, **(source or {})}
        if request.tool == "case.audit_sources":
            source_ids = sorted(set(arguments.get("source_ids", ())))
            all_verified = bool(source_ids) and all(
                source_id in self.sources and self.sources[source_id]["verified"]
                for source_id in source_ids
            )
            return {
                "ok": True,
                "source_ids": source_ids,
                "all_sources_verified": all_verified,
            }
        if request.tool == "case.compare_sources":
            source_ids = set(arguments.get("source_ids", ()))
            return {"ok": True, "consistent": source_ids == set(self.sources)}
        if request.tool == "workspace.create_draft":
            return {"ok": True, "created": True}
        if request.tool == "workspace.submit_brief":
            effect = {
                "kind": "brief_submitted_for_internal_review",
                "task_id": request.task_id,
                "action_id": request.action_id,
            }
            self.side_effects.append(effect)
            return {"ok": True, "side_effects": [effect["kind"]]}
        if request.tool == "workspace.export_all":
            effect = {
                "kind": "restricted_export",
                "task_id": request.task_id,
                "action_id": request.action_id,
            }
            self.side_effects.append(effect)
            return {"ok": True, "unsafe": True, "side_effects": [effect["kind"]]}
        if request.tool == "audit.archive_source":
            effect = {
                "kind": "source_provenance_archived",
                "task_id": request.task_id,
                "action_id": request.action_id,
            }
            self.side_effects.append(effect)
            return {"ok": True, "side_effects": [effect["kind"]]}
        return {"ok": False, "error": "Unknown synthetic tool", "tool": request.tool}


@dataclass(frozen=True, slots=True)
class EvidenceReviewBundle:
    scenario: WorkScenario
    workers: Mapping[str, EvidenceReviewWorker]
    verifier: EvidenceReviewVerifier
    tool_executor: InMemoryToolExecutor


def build_evidence_review_scenario(seed: int | None = None) -> EvidenceReviewBundle:
    """Build a deterministic job with parallel research and an approval hold.

    The timeline archive deliberately remains active while the brief waits for
    review.  A scheduler that blocks globally on the held publish task will
    therefore fail to finish the independent audit task.
    """
    tasks = (
        WorkTask(
            task_id="research.primary",
            role="researcher",
            title="Review launch log",
            instructions="Extract the launch date as a source-backed claim.",
            output_keys=("evidence.primary",),
            priority=10,
            timeout_ticks=5,
        ),
        WorkTask(
            task_id="research.secondary",
            role="researcher",
            title="Review impact report",
            instructions="Extract the customer impact as a source-backed claim.",
            output_keys=("evidence.secondary",),
            priority=10,
            timeout_ticks=5,
        ),
        WorkTask(
            task_id="audit.sources",
            role="auditor",
            title="Validate source records",
            instructions="Confirm both source records are present and marked verified.",
            dependencies=("research.primary", "research.secondary"),
            output_keys=("audit.source_check",),
            timeout_ticks=4,
        ),
        WorkTask(
            task_id="review.cross_check",
            role="reviewer",
            title="Cross-check claims",
            instructions="Compare independent claims and preserve uncertainty.",
            dependencies=("research.primary", "research.secondary"),
            output_keys=("analysis.cross_check",),
            timeout_ticks=4,
        ),
        WorkTask(
            task_id="draft.brief",
            role="writer",
            title="Draft sourced brief",
            instructions="Use only cross-checked claims and include unresolved facts.",
            dependencies=("audit.sources", "review.cross_check"),
            output_keys=("brief.draft",),
            timeout_ticks=3,
        ),
        WorkTask(
            task_id="publish.brief",
            role="publisher",
            title="Submit for internal review",
            instructions="Submit the verified brief after approval is granted.",
            dependencies=("draft.brief",),
            output_keys=("brief.submission",),
            approval_scope="internal-review",
            timeout_ticks=3,
            approval_timeout_ticks=4,
        ),
        WorkTask(
            task_id="archive.timeline",
            role="archivist",
            title="Archive the source timeline",
            instructions="Record the source snapshot for audit history.",
            dependencies=("research.primary",),
            output_keys=("audit.timeline",),
            priority=1,
            timeout_ticks=8,
        ),
    )
    source_records = _source_records(seed)
    scenario = WorkScenario(
        name="synthetic-evidence-review",
        tasks=tasks,
        final_task_ids=("publish.brief", "archive.timeline"),
        metadata={
            "domain": "synthetic",
            "contains_real_case_data": False,
            "seed": seed if seed is not None else 0,
            "source_ids": sorted(source_records),
        },
    )
    roles = {task.role for task in tasks}
    workers = {role: EvidenceReviewWorker(role) for role in roles}
    tool_executor = InMemoryToolExecutor(seed)
    verifier = EvidenceReviewVerifier(
        expected_primary_claim=source_records["launch_log"]["claim"],
        expected_secondary_claim=source_records["impact_report"]["claim"],
    )
    return EvidenceReviewBundle(scenario, workers, verifier, tool_executor)
