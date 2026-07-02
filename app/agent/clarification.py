"""Clarification engine for incomplete SHL assessment requests."""

from __future__ import annotations

from app.models.chat import ConversationContext


class ClarificationEngine:
    """Ask one intelligent follow-up question when context is incomplete."""

    def ask(self, context: ConversationContext) -> str:
        """Return exactly one clarification question."""

        if not context.role and not context.technical_skills and not context.assessment_type:
            return "What type of role are you hiring for?"

        if not context.role and context.technical_skills:
            skills = ", ".join(context.technical_skills[:3])
            return f"What role should the assessment support for candidates with {skills} skills?"

        if context.role and not (context.technical_skills or context.assessment_type):
            return "Which skills or assessment type should the SHL tests focus on?"

        if not context.seniority and not context.experience:
            return "What seniority or experience level are you hiring for?"

        if context.wants_comparison and len(context.requested_assessment_names) < 2:
            return "Which two or more SHL assessments would you like to compare?"

        return "What constraint matters most for these SHL assessments, such as duration, remote testing, or assessment type?"
