from fastapi.testclient import TestClient

from app.main import app
from app.agent.orchestrator import AgentOrchestrator
from app.models.chat import ChatMessage


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendation_flow_uses_catalog_data() -> None:
    orchestrator = AgentOrchestrator()
    request = [
        ChatMessage(
            role="user",
            content="We need a Python backend developer for a junior role. Recommend SHL assessments.",
        )
    ]

    response = orchestrator.handle(request)

    assert response.reply
    assert len(response.recommendations) >= 1
    assert response.recommendations[0].name
