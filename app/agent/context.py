"""Stateless conversation context builder."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.models.chat import ChatMessage, ConversationContext


PROGRAMMING_LANGUAGES = {
    "python",
    "java",
    "javascript",
    "typescript",
    "c#",
    "c++",
    "go",
    "golang",
    "ruby",
    "php",
    "sql",
    "scala",
    "kotlin",
    "swift",
}

TECHNICAL_SKILLS = {
    "api",
    "aws",
    "azure",
    "data analysis",
    "debugging",
    "devops",
    "django",
    "excel",
    "fastapi",
    "frontend",
    "git",
    "machine learning",
    "react",
    "rest",
    "selenium",
    "sql",
    "statistics",
    "testing",
}

SOFT_SKILLS = {
    "communication",
    "collaboration",
    "leadership",
    "problem solving",
    "teamwork",
    "personality",
    "situational judgement",
    "judgment",
}

ASSESSMENT_TYPES = {
    "ability",
    "behavioral",
    "cognitive",
    "coding",
    "personality",
    "skills",
    "situational judgement",
    "technical",
}

SENIORITY_PATTERNS = {
    "intern": r"\b(intern|internship|graduate|entry[- ]level|fresher)\b",
    "junior": r"\b(junior|jr\.?)\b",
    "mid-level": r"\b(mid[- ]level|intermediate)\b",
    "senior": r"\b(senior|sr\.?|lead|principal|manager)\b",
}


class ConversationContextBuilder:
    """Build structured context from the client-provided message history."""

    def build(self, messages: list[ChatMessage]) -> ConversationContext:
        """Extract hiring context without relying on server-side sessions."""

        user_messages = [message.content for message in messages if message.role == "user"]
        latest_user_message = user_messages[-1] if user_messages else ""
        combined_user_text = "\n".join(user_messages)
        lower_text = combined_user_text.casefold()

        return ConversationContext(
            role=self._extract_role(combined_user_text),
            industry=self._extract_industry(combined_user_text),
            programming_language=self._first_match(lower_text, PROGRAMMING_LANGUAGES),
            experience=self._extract_experience(combined_user_text),
            seniority=self._extract_seniority(lower_text),
            technical_skills=self._matches(lower_text, TECHNICAL_SKILLS | PROGRAMMING_LANGUAGES),
            soft_skills=self._matches(lower_text, SOFT_SKILLS),
            assessment_type=self._first_match(lower_text, ASSESSMENT_TYPES),
            language=self._extract_assessment_language(combined_user_text),
            user_constraints=self._extract_constraints(combined_user_text),
            requested_assessment_names=self._extract_requested_assessment_names(latest_user_message),
            wants_comparison=bool(re.search(r"\b(compare|comparison|versus|vs\.?)\b", latest_user_message, re.I)),
            wants_refinement=bool(re.search(r"\b(also|instead|include|exclude|change|refine|more|less)\b", latest_user_message, re.I)),
            latest_user_message=latest_user_message,
            conversation_summary=self._summarize(messages),
        )

    def _extract_role(self, text: str) -> str | None:
        patterns = (
            r"\b(?:hiring|hire|recruiting|recruit)\s+(?:for\s+)?(?:an?\s+)?([A-Za-z0-9+#./ -]{2,60}?)(?:\s+(?:with|who|for|in|at|having)\b|[.,;]|$)",
            r"\b(?:role|position|job)\s*(?:is|:|-)?\s*(?:an?\s+)?([A-Za-z0-9+#./ -]{2,60}?)(?:[.,;]|$)",
            r"\b(?:need|want)\s+(?:an?\s+)?(?:assessment|test)\s+for\s+(?:an?\s+)?([A-Za-z0-9+#./ -]{2,60}?)(?:[.,;]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._clean(match.group(1))
        return None

    def _extract_industry(self, text: str) -> str | None:
        match = re.search(r"\b(?:industry|domain|sector)\s*(?:is|:|-)?\s*([A-Za-z &-]{2,50})", text, re.I)
        return self._clean(match.group(1)) if match else None

    def _extract_experience(self, text: str) -> str | None:
        match = re.search(r"\b(\d+\+?\s*(?:years|yrs|year)\s+(?:of\s+)?experience)\b", text, re.I)
        return self._clean(match.group(1)) if match else None

    def _extract_seniority(self, lower_text: str) -> str | None:
        for seniority, pattern in SENIORITY_PATTERNS.items():
            if re.search(pattern, lower_text):
                return seniority
        return None

    def _extract_assessment_language(self, text: str) -> str | None:
        match = re.search(r"\b(?:test|assessment)\s+language\s*(?:is|:|-)?\s*([A-Za-z -]{2,30})", text, re.I)
        return self._clean(match.group(1)) if match else None

    def _extract_constraints(self, text: str) -> list[str]:
        constraints: list[str] = []
        for pattern in (
            r"\b(?:under|less than|within)\s+\d+\s*(?:minutes|mins|hours|hrs)\b",
            r"\bremote(?:ly)?\b",
            r"\badaptive\b",
            r"\b(?:must|should|need to)\s+[^.]{3,80}",
        ):
            constraints.extend(self._clean(match.group(0)) for match in re.finditer(pattern, text, re.I))
        return self._unique(constraints)

    def _extract_requested_assessment_names(self, text: str) -> list[str]:
        compare_match = re.search(r"\bcompare\s+(.+)", text, re.I)
        if not compare_match:
            return []
        raw = compare_match.group(1)
        parts = re.split(r"\s+and\s+|\s+vs\.?\s+|,|;", raw, flags=re.I)
        return self._unique(self._clean(part) for part in parts if self._clean(part))

    def _matches(self, lower_text: str, candidates: Iterable[str]) -> list[str]:
        matches = [candidate for candidate in candidates if re.search(rf"\b{re.escape(candidate)}\b", lower_text)]
        return self._unique(matches)

    def _first_match(self, lower_text: str, candidates: Iterable[str]) -> str | None:
        matches = self._matches(lower_text, candidates)
        return matches[0] if matches else None

    def _summarize(self, messages: list[ChatMessage]) -> str:
        snippets = [f"{message.role}: {message.content}" for message in messages[-6:]]
        return "\n".join(snippets)

    def _clean(self, value: str) -> str:
        return " ".join(value.split()).strip(" .,:;-")

    def _unique(self, values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.casefold()
            if value and key not in seen:
                result.append(value)
                seen.add(key)
        return result
