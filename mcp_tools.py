"""A stateful, in-memory MCP-like tool world used for RL rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from models import ToolCall
from tasks import Document, TaskSpec


@dataclass
class WorkspaceState:
    read_documents: set[str] = field(default_factory=set)
    redacted_documents: set[str] = field(default_factory=set)
    drafted: bool = False
    attached_documents: list[str] = field(default_factory=list)
    sent_to: str | None = None
    exported: bool = False


class SimulatedMCPWorld:
    """Executes deterministic tools. It has no policy logic; the gate owns that."""

    mcp_name = "case-mcp"

    def __init__(self, task: TaskSpec, documents: tuple[Document, ...]):
        self.task = task
        self.documents = {document.document_id: document for document in documents}
        self.state = WorkspaceState()

    def tool_catalog(self) -> list[dict[str, Any]]:
        return [
            {"name": "records.search", "description": "Search documents by title."},
            {"name": "records.read", "description": "Read one document by id."},
            {"name": "workspace.draft", "description": "Create the requested internal brief."},
            {"name": "workspace.attach", "description": "Attach a document to the brief."},
            {"name": "workspace.send", "description": "Send the prepared brief."},
            {"name": "workspace.export", "description": "Export a document bundle."},
            {"name": "records.redact", "description": "Create a redacted copy of a document."},
        ]

    def call(self, call: ToolCall) -> dict[str, Any]:
        if call.mcp != self.mcp_name:
            return {"ok": False, "error": f"Unknown MCP server: {call.mcp}"}
        handler = getattr(self, f"_tool_{call.action.replace('.', '_')}", None)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool: {call.action}"}
        try:
            return handler(dict(call.params))
        except (TypeError, ValueError, KeyError) as error:
            # Malformed arguments become a normal tool failure, never an
            # unhandled exception that could skip reward / trace recording.
            return {"ok": False, "error": f"Invalid arguments: {type(error).__name__}"}

    def _tool_records_search(self, params: Mapping[str, Any]) -> dict[str, Any]:
        query = str(params.get("query", "")).strip().lower()
        matches = [
            {"document_id": doc.document_id, "title": doc.title, "classification": doc.classification}
            for doc in self.documents.values()
            if query in doc.title.lower() or query in doc.text.lower()
        ]
        return {"ok": True, "documents": matches}

    def _tool_records_read(self, params: Mapping[str, Any]) -> dict[str, Any]:
        document_id = str(params.get("document_id", ""))
        document = self.documents.get(document_id)
        if document is None:
            return {"ok": False, "error": "Document not found"}
        self.state.read_documents.add(document_id)
        return {
            "ok": True,
            "document_id": document.document_id,
            "classification": document.classification,
            "text": document.text,
        }

    def _tool_workspace_draft(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if not self.state.read_documents:
            return {"ok": False, "error": "Read supporting evidence before drafting"}
        self.state.drafted = True
        return {"ok": True, "message": "Draft created"}

    def _tool_workspace_attach(self, params: Mapping[str, Any]) -> dict[str, Any]:
        document_id = str(params.get("document_id", ""))
        if not self.state.drafted:
            return {"ok": False, "error": "Create a draft before attaching documents"}
        if document_id not in self.state.read_documents:
            return {"ok": False, "error": "Read a document before attaching it"}
        if document_id not in self.state.attached_documents:
            self.state.attached_documents.append(document_id)
        return {"ok": True, "message": f"Attached {document_id}"}

    def _tool_workspace_send(self, params: Mapping[str, Any]) -> dict[str, Any]:
        recipient = str(params.get("recipient", ""))
        if not self.state.drafted:
            return {"ok": False, "error": "No draft exists"}
        self.state.sent_to = recipient
        return {"ok": True, "message": f"Brief sent to {recipient}"}

    def _tool_workspace_export(self, params: Mapping[str, Any]) -> dict[str, Any]:
        raw_ids = params.get("document_ids", [])
        if not isinstance(raw_ids, (list, tuple)):
            return {"ok": False, "error": "document_ids must be a list"}
        ids = [str(item) for item in raw_ids]
        if not ids:
            return {"ok": False, "error": "At least one document is required for export"}
        if any(document_id not in self.documents for document_id in ids):
            return {"ok": False, "error": "One or more documents were not found"}
        if any(document_id not in self.state.read_documents for document_id in ids):
            return {"ok": False, "error": "Read each document before exporting it"}
        self.state.exported = True
        return {"ok": True, "message": "Bundle exported", "document_ids": ids}

    def _tool_records_redact(self, params: Mapping[str, Any]) -> dict[str, Any]:
        document_id = str(params.get("document_id", ""))
        if document_id not in self.documents:
            return {"ok": False, "error": "Document not found"}
        if document_id not in self.state.read_documents:
            return {"ok": False, "error": "Read a document before redacting it"}
        self.state.redacted_documents.add(document_id)
        return {"ok": True, "message": f"Redacted copy created for {document_id}"}

    def snapshot(self) -> dict[str, Any]:
        return {
            "read_documents": sorted(self.state.read_documents),
            "redacted_documents": sorted(self.state.redacted_documents),
            "drafted": self.state.drafted,
            "attached_documents": list(self.state.attached_documents),
            "sent_to": self.state.sent_to,
            "exported": self.state.exported,
        }
