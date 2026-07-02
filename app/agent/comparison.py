"""Catalog-grounded comparison engine for SHL assessments."""

from __future__ import annotations

from app.models.assessment import Assessment
from app.models.recommendation import Recommendation


class ComparisonEngine:
    """Generate comparisons only from retrieved SHL catalog records."""

    def compare(self, assessments: list[Assessment], requested_names: list[str]) -> str:
        """Return a catalog-grounded comparison reply."""

        if len(assessments) < 2:
            requested = ", ".join(requested_names) if requested_names else "those assessments"
            return (
                f"I could not find at least two matching SHL catalog assessments for {requested}. "
                "Please use the exact SHL assessment names from the catalog."
            )

        lines = ["Here is a catalog-based comparison of the matching SHL assessments:"]
        for assessment in assessments:
            lines.extend(
                [
                    "",
                    f"{assessment.name}",
                    f"Test type: {assessment.test_type or 'Not specified'}",
                    f"Duration: {assessment.duration or 'Not specified'}",
                    f"Skills measured: {', '.join(assessment.skills_measured) if assessment.skills_measured else 'Not specified'}",
                    f"Languages: {', '.join(assessment.languages) if assessment.languages else 'Not specified'}",
                    f"Remote testing: {self._bool_text(assessment.remote_testing)}",
                    f"Adaptive support: {self._bool_text(assessment.adaptive_support)}",
                    f"Catalog URL: {assessment.url}",
                ]
            )
        return "\n".join(lines)

    def to_public_recommendations(self, assessments: list[Assessment]) -> list[Recommendation]:
        """Expose compared assessments through the required recommendation schema."""

        return [
            Recommendation(
                name=assessment.name,
                url=assessment.url,
                test_type=assessment.test_type,
            )
            for assessment in assessments[:10]
        ]

    def _bool_text(self, value: bool | None) -> str:
        if value is True:
            return "Yes"
        if value is False:
            return "No"
        return "Not specified"
