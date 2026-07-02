"""FAISS dense-vector indexing for SHL catalog embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from app.embeddings.pipeline import EmbeddingPipeline
from app.utils.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FAISSArtifacts:
    """Files and counts produced by FAISS indexing."""

    index_path: Path
    count: int
    dimension: int


class FAISSIndexBuilder:
    """Build and load FAISS indexes over normalized catalog embeddings."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_pipeline: EmbeddingPipeline | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline(self.settings)

    def build(
        self,
        embeddings_path: Path | None = None,
        metadata_path: Path | None = None,
        index_path: Path | None = None,
        rebuild_embeddings: bool = False,
    ) -> FAISSArtifacts:
        """Build and persist a FAISS inner-product index."""

        vector_path = embeddings_path or self.settings.embeddings_path
        meta_path = metadata_path or self.settings.embeddings_metadata_path

        if rebuild_embeddings or not vector_path.exists() or not meta_path.exists():
            artifacts = self.embedding_pipeline.build(
                embeddings_path=vector_path,
                metadata_path=meta_path,
            )
            logger.info("embeddings_generated_for_faiss", count=artifacts.count)

        import faiss
        import numpy as np

        vectors = np.load(vector_path).astype("float32")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError(f"Invalid embeddings matrix: {vector_path}")

        metadata = self._load_metadata(meta_path)
        if metadata.get("count") != int(vectors.shape[0]):
            raise ValueError("Embedding metadata count does not match vector rows")

        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)

        destination = index_path or self.settings.faiss_index_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(destination))

        logger.info(
            "faiss_index_built",
            count=int(vectors.shape[0]),
            dimension=int(vectors.shape[1]),
            index_path=str(destination),
        )
        return FAISSArtifacts(
            index_path=destination,
            count=int(vectors.shape[0]),
            dimension=int(vectors.shape[1]),
        )

    def load(self, index_path: Path | None = None):
        """Load a persisted FAISS index."""

        import faiss

        source = index_path or self.settings.faiss_index_path
        if not source.exists():
            raise FileNotFoundError(f"FAISS index not found: {source}")
        return faiss.read_index(str(source))

    def _load_metadata(self, metadata_path: Path) -> dict[str, Any]:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Embedding metadata not found: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "items" not in payload:
            raise ValueError(f"Invalid embedding metadata: {metadata_path}")
        return payload


def build_faiss_index(
    embeddings_path: Path | None = None,
    metadata_path: Path | None = None,
    index_path: Path | None = None,
    rebuild_embeddings: bool = False,
) -> FAISSArtifacts:
    """Build a FAISS index using default dependencies."""

    return FAISSIndexBuilder().build(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        index_path=index_path,
        rebuild_embeddings=rebuild_embeddings,
    )
