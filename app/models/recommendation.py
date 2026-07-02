"""Recommendation response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    """Public recommendation item required by the chat API schema."""

    name: str = Field(default="")
    url: str = Field(default="")
    test_type: str = Field(default="")
