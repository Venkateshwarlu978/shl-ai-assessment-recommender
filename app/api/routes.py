"""FastAPI routes for the SHL assessment recommender."""

from fastapi import APIRouter, Depends

from app.agent.orchestrator import AgentOrchestrator
from app.api.dependencies import get_orchestrator
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Handle a stateless chat turn."""

    return orchestrator.handle(request.messages)
