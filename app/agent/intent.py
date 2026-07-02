"""Intent detection for SHL assessment conversations."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from app.models.chat import ConversationContext


class Intent(StrEnum):
    """Supported conversation intents."""

    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    COMPARE = "compare"
    REFINE = "refine"
    REFUSE = "refuse"


class IntentDetectionResult(BaseModel):
    """Detected intent with a concise explanation for observability."""

    intent: Intent
    confidence: float
    reason: str


class IntentDetector:
    """Rule-based intent detector for deterministic orchestration."""

    OFF_TOPIC_PATTERNS = (
        r"\blegal advice\b",
        r"\bwrite (?:my )?resume\b",
        r"\bsalary\b",
        r"\binterview questions\b",
        r"\bgeneral hiring advice\b",
        r"\bweather\b",
        r"\bstock\b",
    )

    def detect(self, context: ConversationContext) -> IntentDetectionResult:
        """Detect the user's current intent from structured context."""

        latest = context.latest_user_message.casefold()
        if self._matches_any(latest, self.OFF_TOPIC_PATTERNS):
            return IntentDetectionResult(
                intent=Intent.REFUSE,
                confidence=0.9,
                reason="The latest request is outside SHL assessment recommendation scope.",
            )

        if context.wants_comparison:
            return IntentDetectionResult(
                intent=Intent.COMPARE,
                confidence=0.9,
                reason="The user asked to compare assessments.",
            )

        if context.wants_refinement and context.has_minimum_recommendation_context():
            return IntentDetectionResult(
                intent=Intent.REFINE,
                confidence=0.8,
                reason="The user is modifying existing hiring criteria.",
            )

        if context.has_minimum_recommendation_context():
            return IntentDetectionResult(
                intent=Intent.RECOMMEND,
                confidence=0.85,
                reason="The conversation contains enough role or skill context to retrieve assessments.",
            )

        return IntentDetectionResult(
            intent=Intent.CLARIFY,
            confidence=0.75,
            reason="The request is assessment-related but lacks enough hiring context.",
        )

    def _matches_any(self, text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
