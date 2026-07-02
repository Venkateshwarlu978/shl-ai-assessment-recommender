"""Assessment domain models."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Assessment(BaseModel):
    """Validated SHL assessment record used by retrieval and recommendation."""

    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    description: str = ""
    duration: str | None = None
    duration_minutes: int | None = Field(default=None, ge=1)
    skills_measured: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    remote_testing: bool | None = None
    adaptive_support: bool | None = None
    test_type: str = ""
    search_text: str = ""

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("url")
    @classmethod
    def validate_shl_url(cls, value: str) -> str:
        """Accept only absolute SHL URLs."""

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Assessment URL must be absolute HTTP(S)")
        if not parsed.netloc.lower().endswith("shl.com"):
            raise ValueError("Assessment URL must belong to shl.com")
        return value.rstrip("/")

    @field_validator("skills_measured", "languages", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> list[str]:
        """Normalize scraped strings or lists into clean unique lists."""

        if value is None:
            return []
        if isinstance(value, str):
            items = re.split(r",|;|\||\band\b", value, flags=re.IGNORECASE)
        elif isinstance(value, list):
            items = value
        else:
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = re.sub(r"\s+", " ", str(item)).strip(" .:-")
            key = cleaned.casefold()
            if cleaned and key not in seen:
                normalized.append(cleaned)
                seen.add(key)
        return normalized

    @model_validator(mode="after")
    def populate_search_fields(self) -> "Assessment":
        """Populate duration minutes and retrieval text after validation."""

        if self.duration and self.duration_minutes is None:
            self.duration_minutes = extract_duration_minutes(self.duration)

        self.search_text = " ".join(
            part
            for part in (
                self.name,
                self.description,
                self.test_type,
                " ".join(self.skills_measured),
                " ".join(self.languages),
                f"duration {self.duration_minutes} minutes" if self.duration_minutes else "",
                "remote testing" if self.remote_testing else "",
                "adaptive testing" if self.adaptive_support else "",
            )
            if part
        )
        return self


def extract_duration_minutes(value: str) -> int | None:
    """Extract a best-effort minute duration from SHL duration text."""

    normalized = value.lower()
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|hrs|hours)", normalized)
    minute_match = re.search(r"(\d+)\s*(?:minute|min|mins|minutes)", normalized)

    total = 0
    if hour_match:
        total += round(float(hour_match.group(1)) * 60)
    if minute_match:
        total += int(minute_match.group(1))

    if total > 0:
        return total

    bare_number = re.search(r"\b(\d{1,3})\b", normalized)
    if bare_number:
        return int(bare_number.group(1))
    return None
