"""Recommendation engine for retrieved SHL assessments."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.assessment import Assessment
from app.models.chat import ConversationContext
from app.models.recommendation import Recommendation
from app.retrieval.engine import RetrievalResult
from app.retrieval.tokenization import tokenize


@dataclass(frozen=True)
class RankedAssessment:
    """Assessment with recommendation ranking score."""

    assessment: Assessment
    score: float
    reason: str


class RecommendationEngine:
    """Rank retrieved SHL assessments against structured hiring context."""

    def rank(
        self,
        context: ConversationContext,
        retrieved: list[RetrievalResult],
        limit: int = 10,
    ) -> list[RankedAssessment]:
        """Rank retrieved assessments and keep the top 1-10."""

        bounded_limit = min(max(limit, 1), 10)
        ranked = [
            RankedAssessment(
                assessment=result.assessment,
                score=result.score + self._context_score(context, result.assessment),
                reason=self._reason(context, result.assessment),
            )
            for result in retrieved
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:bounded_limit]

    def to_public_recommendations(self, ranked: list[RankedAssessment]) -> list[Recommendation]:
        """Convert ranked assessments to the required public schema."""

        return [
            Recommendation(
                name=item.assessment.name,
                url=item.assessment.url,
                test_type=item.assessment.test_type,
            )
            for item in ranked[:10]
        ]

    def build_reply(self, context: ConversationContext, ranked: list[RankedAssessment]) -> str:
        """Create a concise, catalog-grounded recommendation reply."""

        if not ranked:
            return (
                "I could not find matching SHL assessments in the catalog for that request. "
                "Please provide a role, skills, or assessment type to narrow the search."
            )

        role_phrase = f" for {context.role}" if context.role else ""
        lines = [f"I found {len(ranked)} SHL assessment recommendation(s){role_phrase}:"]
        for index, item in enumerate(ranked, start=1):
            test_type = f" ({item.assessment.test_type})" if item.assessment.test_type else ""
            reason = f" - {item.reason}" if item.reason else ""
            lines.append(f"{index}. {item.assessment.name}{test_type}{reason}")
        return "\n".join(lines)

    def _context_score(self, context: ConversationContext, assessment: Assessment) -> float:
        text = assessment.search_text.casefold()
        score = 0.0

        if context.role:
            score += 0.8 * self._term_overlap(context.role, text)
        score += 0.7 * self._list_overlap(context.technical_skills, text)
        score += 0.6 * self._list_overlap(context.soft_skills, text)

        if context.assessment_type and context.assessment_type.casefold() in text:
            score += 0.7
        if context.programming_language and context.programming_language.casefold() in text:
            score += 0.6
        if context.language and context.language.casefold() in " ".join(assessment.languages).casefold():
            score += 0.3
        if self._constraint_match(context, assessment):
            score += 0.4

        return score

    def _reason(self, context: ConversationContext, assessment: Assessment) -> str:
        reasons: list[str] = []
        text = assessment.search_text.casefold()
        matched_skills = [skill for skill in context.technical_skills if skill.casefold() in text]
        matched_soft = [skill for skill in context.soft_skills if skill.casefold() in text]

        if matched_skills:
            reasons.append("matches " + ", ".join(matched_skills[:3]))
        if matched_soft:
            reasons.append("covers " + ", ".join(matched_soft[:2]))
        if context.assessment_type and context.assessment_type.casefold() in text:
            reasons.append(f"fits the {context.assessment_type} assessment focus")
        if assessment.duration_minutes:
            reasons.append(f"{assessment.duration_minutes} minutes")
        return "; ".join(reasons)

    def _constraint_match(self, context: ConversationContext, assessment: Assessment) -> bool:
        constraint_text = " ".join(context.user_constraints).casefold()
        if "remote" in constraint_text and assessment.remote_testing:
            return True
        if "adaptive" in constraint_text and assessment.adaptive_support:
            return True
        return False

    def _list_overlap(self, terms: list[str], text: str) -> float:
        if not terms:
            return 0.0
        matches = sum(1 for term in terms if term.casefold() in text)
        return matches / len(terms)

    def _term_overlap(self, phrase: str, text: str) -> float:
        tokens = tokenize(phrase)
        if not tokens:
            return 0.0
        return sum(1 for token in tokens if token in text) / len(tokens)
