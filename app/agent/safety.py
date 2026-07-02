"""Safety guardrails for SHL assessment conversations."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from app.models.chat import ConversationContext


class SafetyStatus(StrEnum):
    """Safety status for the latest request."""

    ALLOW = "allow"
    REFUSE = "refuse"


class SafetyDecision(BaseModel):
    """Safety decision and user-safe reply when refused."""

    status: SafetyStatus
    reason: str
    reply: str | None = None


class SafetyGuard:
    """Detect off-topic, unsafe, and prompt-injection requests."""

    PROMPT_INJECTION_PATTERNS = (
        r"\bignore (?:all )?(?:previous|prior|above|system|developer) instructions\b",
        r"\bdisregard (?:all )?(?:previous|prior|above|system|developer) instructions\b",
        r"\breveal (?:the )?(?:system|developer|hidden) prompt\b",
        r"\bshow (?:me )?(?:the )?(?:system|developer|hidden) prompt\b",
        r"\byou are now\b",
        r"\bact as\b.*\bwithout restrictions\b",
        r"\bdo not follow\b.*\binstructions\b",
        r"\breturn assessments outside (?:the )?catalog\b",
        r"\binvent\b.*\bassessments?\b",
    )

    OFF_TOPIC_PATTERNS = (
        r"\blegal advice\b",
        r"\bemployment law\b",
        r"\bcompensation\b",
        r"\bsalary\b",
        r"\bwrite (?:a |my )?job description\b",
        r"\bwrite (?:a |my )?resume\b",
        r"\binterview questions\b",
        r"\bgeneral hiring advice\b",
        r"\bperformance review\b",
        r"\bweather\b",
        r"\bsports\b",
        r"\bstock market\b",
    )

    SHL_SCOPE_PATTERNS = (
        r"\bshl\b",
        r"\bassessments?\b",
        r"\btests?\b",
        r"\bproduct catalog\b",
        r"\brecommend\b",
        r"\bcompare\b",
        r"\bhiring\b",
        r"\brole\b",
    )

    def check(self, context: ConversationContext) -> SafetyDecision:
        """Return whether the request can proceed."""

        latest = context.latest_user_message.strip()
        lower_latest = latest.casefold()

        if self._matches_any(lower_latest, self.PROMPT_INJECTION_PATTERNS):
            return SafetyDecision(
                status=SafetyStatus.REFUSE,
                reason="Prompt injection attempt detected.",
                reply=(
                    "I can help only with SHL assessment recommendations and comparisons "
                    "using the SHL catalog. I cannot follow instructions that override those rules."
                ),
            )

        if self._matches_any(lower_latest, self.OFF_TOPIC_PATTERNS):
            return SafetyDecision(
                status=SafetyStatus.REFUSE,
                reason="Request is outside SHL assessment recommendation scope.",
                reply=(
                    "I can help with SHL assessment recommendations, refinements, and catalog-based "
                    "comparisons. I cannot provide unrelated hiring, legal, salary, or general advice."
                ),
            )

        if latest and not self._is_in_scope(context, lower_latest):
            return SafetyDecision(
                status=SafetyStatus.REFUSE,
                reason="No SHL assessment-related scope detected.",
                reply=(
                    "I can only help with SHL Individual Test Solution recommendations or comparisons. "
                    "Please ask about the role, skills, or SHL assessments you want to evaluate."
                ),
            )

        return SafetyDecision(status=SafetyStatus.ALLOW, reason="Request is in scope.")

    def _is_in_scope(self, context: ConversationContext, lower_latest: str) -> bool:
        if context.has_minimum_recommendation_context() or context.wants_comparison:
            return True
        return self._matches_any(lower_latest, self.SHL_SCOPE_PATTERNS)

    def _matches_any(self, text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
