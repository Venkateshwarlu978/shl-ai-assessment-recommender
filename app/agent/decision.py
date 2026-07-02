"""Decision engine for agent actions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from app.agent.intent import Intent, IntentDetectionResult
from app.agent.safety import SafetyDecision, SafetyStatus
from app.models.chat import ConversationContext


class AgentAction(StrEnum):
    """Actions supported by the orchestrator."""

    CLARIFY = "clarify"
    RETRIEVE = "retrieve"
    RECOMMEND = "recommend"
    COMPARE = "compare"
    REFUSE = "refuse"


class Decision(BaseModel):
    """Decision engine output."""

    action: AgentAction
    reason: str


class DecisionEngine:
    """Choose the next agent action from safety, intent, and context."""

    def decide(
        self,
        context: ConversationContext,
        intent: IntentDetectionResult,
        safety: SafetyDecision,
    ) -> Decision:
        """Decide which path the orchestrator should execute."""

        if safety.status == SafetyStatus.REFUSE:
            return Decision(action=AgentAction.REFUSE, reason=safety.reason)

        if intent.intent == Intent.COMPARE:
            if len(context.requested_assessment_names) < 2:
                return Decision(action=AgentAction.CLARIFY, reason="Comparison needs at least two assessment names.")
            return Decision(action=AgentAction.COMPARE, reason=intent.reason)

        if intent.intent == Intent.CLARIFY:
            return Decision(action=AgentAction.CLARIFY, reason=intent.reason)

        if intent.intent == Intent.REFUSE:
            return Decision(action=AgentAction.REFUSE, reason=intent.reason)

        if intent.intent in {Intent.RECOMMEND, Intent.REFINE}:
            if not context.has_minimum_recommendation_context():
                return Decision(action=AgentAction.CLARIFY, reason="Recommendation context is incomplete.")
            return Decision(action=AgentAction.RECOMMEND, reason=intent.reason)

        return Decision(action=AgentAction.CLARIFY, reason="Unable to route request confidently.")
