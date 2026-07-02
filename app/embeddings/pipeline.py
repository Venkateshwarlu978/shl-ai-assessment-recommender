"""Embedding generation pipeline for validated SHL catalog records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog

from app.catalog.parser import CatalogParser
from app.models.assessment import Assessment
from app.utils.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


class TextEmbedder(Protocol):
    """Protocol for replaceable text embedding backends."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        """Encode text into a numeric matrix."""


@dataclass(frozen=True)
class EmbeddingArtifacts:
    """Files produced by the embedding pipeline."""

    embeddings_path: Path
    metadata_path: Path
    count: int
    dimension: int


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: TextEmbedder | None = None

    @property
    def model(self) -> TextEmbedder:
        """Load the embedding model only when generation is requested."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        """Encode text using the configured sentence-transformer."""

        return self.model.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )


class EmbeddingPipeline:
    """Build dense embeddings and metadata from the parsed SHL catalog."""

    def __init__(
        self,
        settings: Settings | None = None,
        parser: CatalogParser | None = None,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.parser = parser or CatalogParser(self.settings)
        self.embedder = embedder or SentenceTransformerEmbedder(self.settings.embedding_model_name)

    def build(
        self,
        catalog_path: Path | None = None,
        embeddings_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> EmbeddingArtifacts:
        """Generate and persist embeddings for every validated assessment."""

        assessments = self.parser.parse_file(catalog_path)
        if not assessments:
            raise ValueError("Cannot build embeddings for an empty catalog")

        texts = [self._embedding_text(assessment) for assessment in assessments]
        vectors = self.embedder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        import numpy as np

        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != len(assessments):
            raise ValueError("Embedding model returned an invalid matrix shape")

        destination = embeddings_path or self.settings.embeddings_path
        metadata_destination = metadata_path or self.settings.embeddings_metadata_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_destination.parent.mkdir(parents=True, exist_ok=True)

        np.save(destination, matrix)
        metadata_destination.write_text(
            json.dumps(self._metadata_payload(assessments, matrix.shape[1]), indent=2),
            encoding="utf-8",
        )

        logger.info(
            "embeddings_built",
            count=len(assessments),
            dimension=int(matrix.shape[1]),
            embeddings_path=str(destination),
            metadata_path=str(metadata_destination),
        )
        return EmbeddingArtifacts(
            embeddings_path=destination,
            metadata_path=metadata_destination,
            count=len(assessments),
            dimension=int(matrix.shape[1]),
        )

    def _embedding_text(self, assessment: Assessment) -> str:
        """Return the retrieval text used for dense embedding."""

        return assessment.search_text or " ".join(
            part
            for part in (
                assessment.name,
                assessment.description,
                assessment.test_type,
                " ".join(assessment.skills_measured),
            )
            if part
        )

    def _metadata_payload(self, assessments: list[Assessment], dimension: int) -> dict[str, object]:
        """Create metadata that maps vector rows back to catalog records."""

        return {
            "embedding_model": self.settings.embedding_model_name,
            "dimension": dimension,
            "count": len(assessments),
            "items": [
                {
                    "row": index,
                    "name": assessment.name,
                    "url": assessment.url,
                    "test_type": assessment.test_type,
                }
                for index, assessment in enumerate(assessments)
            ],
        }


def build_embeddings(
    catalog_path: Path | None = None,
    embeddings_path: Path | None = None,
    metadata_path: Path | None = None,
) -> EmbeddingArtifacts:
    """Build dense embeddings using default dependencies."""

    return EmbeddingPipeline().build(
        catalog_path=catalog_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )
