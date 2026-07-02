"""Chat API and conversation context models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.recommendation import Recommendation


MessageRole = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    """Single stateless chat message supplied by the client."""

    role: MessageRole
    content: str = Field(min_length=1)

    model_config = ConfigDict(str_strip_whitespace=True)


class ConversationContext(BaseModel):
    """Structured hiring context extracted from the conversation history."""

    role: str | None = None
    industry: str | None = None
    programming_language: str | None = None
    experience: str | None = None
    seniority: str | None = None
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    assessment_type: str | None = None
    language: str | None = None
    user_constraints: list[str] = Field(default_factory=list)
    requested_assessment_names: list[str] = Field(default_factory=list)
    wants_comparison: bool = False
    wants_refinement: bool = False
    latest_user_message: str = ""
    conversation_summary: str = ""

    def has_minimum_recommendation_context(self) -> bool:
        """Return whether there is enough context to recommend assessments."""

        has_role_or_skill = bool(self.role or self.technical_skills or self.assessment_type)
        return has_role_or_skill and bool(self.latest_user_message)


class ChatRequest(BaseModel):
    """POST /chat request body."""

    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    """POST /chat response body. Keep this schema stable."""

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
