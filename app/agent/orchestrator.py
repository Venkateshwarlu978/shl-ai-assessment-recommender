"""Agent orchestrator for stateless SHL assessment conversations."""

from __future__ import annotations

import structlog

from app.agent.clarification import ClarificationEngine
from app.agent.comparison import ComparisonEngine
from app.agent.context import ConversationContextBuilder
from app.agent.decision import AgentAction, DecisionEngine
from app.agent.intent import IntentDetector
from app.agent.recommendation import RecommendationEngine
from app.agent.safety import SafetyGuard
from app.models.chat import ChatMessage, ChatResponse
from app.retrieval.engine import HybridRetriever

logger = structlog.get_logger(__name__)


class AgentOrchestrator:
    """Coordinate context, intent, safety, retrieval, and response generation."""

    def __init__(
        self,
        context_builder: ConversationContextBuilder | None = None,
        intent_detector: IntentDetector | None = None,
        safety_guard: SafetyGuard | None = None,
        decision_engine: DecisionEngine | None = None,
        clarification_engine: ClarificationEngine | None = None,
        retriever: HybridRetriever | None = None,
        recommendation_engine: RecommendationEngine | None = None,
        comparison_engine: ComparisonEngine | None = None,
    ) -> None:
        self.context_builder = context_builder or ConversationContextBuilder()
        self.intent_detector = intent_detector or IntentDetector()
        self.safety_guard = safety_guard or SafetyGuard()
        self.decision_engine = decision_engine or DecisionEngine()
        self.clarification_engine = clarification_engine or ClarificationEngine()
        self.retriever = retriever or HybridRetriever()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.comparison_engine = comparison_engine or ComparisonEngine()

    def handle(self, messages: list[ChatMessage]) -> ChatResponse:
        """Handle one stateless chat request."""

        context = self.context_builder.build(messages)
        intent = self.intent_detector.detect(context)
        safety = self.safety_guard.check(context)
        decision = self.decision_engine.decide(context, intent, safety)

        logger.info(
            "agent_decision",
            action=decision.action,
            intent=intent.intent,
            reason=decision.reason,
        )

        if decision.action == AgentAction.REFUSE:
            return ChatResponse(reply=safety.reply or self._refusal_reply())

        if decision.action == AgentAction.CLARIFY:
            return ChatResponse(reply=self.clarification_engine.ask(context))

        if decision.action == AgentAction.COMPARE:
            try:
                assessments = self.retriever.retrieve_by_names(context.requested_assessment_names)
            except (FileNotFoundError, ImportError, ValueError) as exc:
                logger.warning("comparison_retrieval_unavailable", error=str(exc))
                return ChatResponse(
                    reply="The SHL catalog indexes are not ready yet, so I cannot compare assessments right now."
                )
            return ChatResponse(
                reply=self.comparison_engine.compare(assessments, context.requested_assessment_names),
                recommendations=self.comparison_engine.to_public_recommendations(assessments),
            )

        if decision.action == AgentAction.RECOMMEND:
            query = self._retrieval_query(context)
            try:
                retrieved = self.retriever.retrieve(query, top_k=10, rerank=True)
            except (FileNotFoundError, ImportError, ValueError) as exc:
                logger.warning("recommendation_retrieval_unavailable", error=str(exc))
                return ChatResponse(
                    reply="The SHL catalog indexes are not ready yet, so I cannot recommend assessments right now."
                )
            ranked = self.recommendation_engine.rank(context, retrieved, limit=10)
            return ChatResponse(
                reply=self.recommendation_engine.build_reply(context, ranked),
                recommendations=self.recommendation_engine.to_public_recommendations(ranked),
            )

        return ChatResponse(reply=self.clarification_engine.ask(context))

    def _retrieval_query(self, context) -> str:
        parts = [
            context.role or "",
            context.assessment_type or "",
            context.programming_language or "",
            context.seniority or "",
            context.experience or "",
            " ".join(context.technical_skills),
            " ".join(context.soft_skills),
            " ".join(context.user_constraints),
            context.latest_user_message,
        ]
        return " ".join(part for part in parts if part).strip()

    def _refusal_reply(self) -> str:
        return "I can only help with SHL assessment recommendations and catalog-based comparisons."
