"""Local, review-gated coding-agent work queue.

The package deliberately does not connect to the native Codex conversation
subagent runtime. It provides a manual queue and a provider protocol so a real
worker adapter can be added without pretending that one is configured.
"""

from .contracts import AgentTask, TaskGraph, TaskStatus
from .providers import WorkerProvider, register_provider

__all__ = ["AgentTask", "TaskGraph", "TaskStatus", "WorkerProvider", "register_provider"]
__version__ = "0.1.0"
