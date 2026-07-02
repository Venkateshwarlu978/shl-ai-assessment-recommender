"""FastAPI dependency providers."""

from functools import lru_cache

from app.agent.orchestrator import AgentOrchestrator


@lru_cache(maxsize=1)
def get_orchestrator() -> AgentOrchestrator:
    """Return a cached orchestrator instance for request handling."""

    return AgentOrchestrator()
