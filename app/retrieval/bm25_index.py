"""BM25 indexing for the validated SHL catalog."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from app.catalog.parser import CatalogParser
from app.models.assessment import Assessment
from app.retrieval.tokenization import tokenize
from app.utils.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BM25Artifacts:
    """Files and counts produced by BM25 indexing."""

    index_path: Path
    count: int


class BM25IndexBuilder:
    """Build and persist a BM25 index for SHL assessment search text."""

    def __init__(
        self,
        settings: Settings | None = None,
        parser: CatalogParser | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.parser = parser or CatalogParser(self.settings)

    def build(
        self,
        catalog_path: Path | None = None,
        index_path: Path | None = None,
    ) -> BM25Artifacts:
        """Build and save a BM25 index from parsed catalog records."""

        assessments = self.parser.parse_file(catalog_path)
        if not assessments:
            raise ValueError("Cannot build BM25 index for an empty catalog")

        corpus = [tokenize(self._document_text(assessment)) for assessment in assessments]
        if any(not tokens for tokens in corpus):
            raise ValueError("All catalog records must produce at least one BM25 token")

        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(corpus)
        destination = index_path or self.settings.bm25_index_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": bm25,
            "corpus": corpus,
            "items": [assessment.model_dump() for assessment in assessments],
        }

        with destination.open("wb") as file:
            pickle.dump(payload, file)

        logger.info("bm25_index_built", count=len(assessments), index_path=str(destination))
        return BM25Artifacts(index_path=destination, count=len(assessments))

    def load(self, index_path: Path | None = None) -> dict[str, Any]:
        """Load a persisted BM25 index payload."""

        source = index_path or self.settings.bm25_index_path
        with source.open("rb") as file:
            payload = pickle.load(file)
        if not {"bm25", "corpus", "items"}.issubset(payload):
            raise ValueError(f"Invalid BM25 index payload: {source}")
        return payload

    def _document_text(self, assessment: Assessment) -> str:
        return assessment.search_text or assessment.name


def build_bm25_index(
    catalog_path: Path | None = None,
    index_path: Path | None = None,
) -> BM25Artifacts:
    """Build a BM25 index using default dependencies."""

    return BM25IndexBuilder().build(catalog_path=catalog_path, index_path=index_path)
