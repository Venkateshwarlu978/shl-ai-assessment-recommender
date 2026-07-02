"""Hybrid retrieval engine for SHL assessment recommendations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog

from app.catalog.parser import CatalogParser
from app.embeddings.pipeline import SentenceTransformerEmbedder, TextEmbedder
from app.models.assessment import Assessment
from app.retrieval.bm25_index import BM25IndexBuilder
from app.retrieval.faiss_index import FAISSIndexBuilder
from app.retrieval.tokenization import tokenize
from app.utils.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


class Reranker(Protocol):
    """Protocol for cross-encoder rerankers."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return relevance scores for query-document pairs."""


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved assessment and its scoring details."""

    assessment: Assessment
    score: float
    bm25_score: float
    dense_score: float
    rerank_score: float | None = None


class CrossEncoderReranker:
    """Lazy sentence-transformers CrossEncoder wrapper."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Reranker | None = None

    @property
    def model(self) -> Reranker:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(pairs)
        return [float(score) for score in scores]


class HybridRetriever:
    """Hybrid BM25 + dense vector retrieval with CrossEncoder reranking."""

    def __init__(
        self,
        settings: Settings | None = None,
        bm25_builder: BM25IndexBuilder | None = None,
        faiss_builder: FAISSIndexBuilder | None = None,
        embedder: TextEmbedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bm25_builder = bm25_builder or BM25IndexBuilder(self.settings)
        self.faiss_builder = faiss_builder or FAISSIndexBuilder(self.settings)
        self.embedder = embedder or SentenceTransformerEmbedder(self.settings.embedding_model_name)
        self.reranker = reranker or CrossEncoderReranker(self.settings.cross_encoder_model_name)
        self._bm25_payload: dict | None = None
        self._faiss_index = None
        self._metadata: dict | None = None

    def retrieve(self, query: str, top_k: int | None = None, rerank: bool = True) -> list[RetrievalResult]:
        """Retrieve catalog assessments relevant to the query."""

        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            return []

        limit = min(top_k or self.settings.retrieval_top_k, self.settings.retrieval_top_k)
        candidate_limit = max(limit, self.settings.retrieval_candidate_limit)

        try:
            bm25_scores = self._bm25_scores(cleaned_query)
            dense_scores = self._dense_scores(cleaned_query, candidate_limit)
            candidates = self._combine_scores(bm25_scores, dense_scores, candidate_limit)
        except (FileNotFoundError, ImportError, ValueError, OSError, EOFError) as exc:
            logger.warning("retrieval_fallback_to_catalog", error=str(exc))
            candidates = self._fallback_candidates(cleaned_query, candidate_limit)

        if rerank and candidates:
            try:
                candidates = self._rerank(cleaned_query, candidates)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("rerank_failed", error=str(exc))

        return candidates[:limit]

    def retrieve_by_names(self, names: list[str]) -> list[Assessment]:
        """Retrieve assessments whose catalog names match requested names."""

        normalized_names = [self._normalize_name(name) for name in names if name.strip()]
        if not normalized_names:
            return []

        assessments = self._items()
        matches: list[Assessment] = []
        for requested in normalized_names:
            exact = [
                assessment
                for assessment in assessments
                if self._normalize_name(assessment.name) == requested
            ]
            partial = [
                assessment
                for assessment in assessments
                if requested in self._normalize_name(assessment.name)
                or self._normalize_name(assessment.name) in requested
            ]
            for assessment in exact or partial:
                if assessment not in matches:
                    matches.append(assessment)
        return matches

    def _bm25_scores(self, query: str) -> dict[int, float]:
        payload = self._load_bm25()
        raw_scores = payload["bm25"].get_scores(tokenize(query))
        return self._normalize_score_map({index: float(score) for index, score in enumerate(raw_scores)})

    def _dense_scores(self, query: str, candidate_limit: int) -> dict[int, float]:
        index = self._load_faiss()
        query_vector = self.embedder.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        import numpy as np

        vector = np.asarray(query_vector, dtype="float32")
        scores, indices = index.search(vector, min(candidate_limit, index.ntotal))
        dense = {
            int(row_index): float(score)
            for row_index, score in zip(indices[0], scores[0], strict=False)
            if int(row_index) >= 0
        }
        return self._normalize_score_map(dense)

    def _combine_scores(
        self,
        bm25_scores: dict[int, float],
        dense_scores: dict[int, float],
        candidate_limit: int,
    ) -> list[RetrievalResult]:
        row_ids = set(bm25_scores) | set(dense_scores)
        assessments = self._items()
        results: list[RetrievalResult] = []

        for row_id in row_ids:
            if row_id >= len(assessments):
                continue
            bm25_score = bm25_scores.get(row_id, 0.0)
            dense_score = dense_scores.get(row_id, 0.0)
            score = (
                self.settings.retrieval_bm25_weight * bm25_score
                + self.settings.retrieval_dense_weight * dense_score
            )
            results.append(
                RetrievalResult(
                    assessment=assessments[row_id],
                    score=score,
                    bm25_score=bm25_score,
                    dense_score=dense_score,
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:candidate_limit]

    def _rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        pairs = [(query, result.assessment.search_text) for result in candidates]
        rerank_scores = self.reranker.predict(pairs)
        reranked = [
            RetrievalResult(
                assessment=result.assessment,
                score=float(rerank_score),
                bm25_score=result.bm25_score,
                dense_score=result.dense_score,
                rerank_score=float(rerank_score),
            )
            for result, rerank_score in zip(candidates, rerank_scores, strict=False)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)

    def _load_bm25(self) -> dict:
        if self._bm25_payload is None:
            self._bm25_payload = self.bm25_builder.load()
        return self._bm25_payload

    def _load_faiss(self):
        if self._faiss_index is None:
            self._faiss_index = self.faiss_builder.load()
        return self._faiss_index

    def _load_metadata(self) -> dict:
        if self._metadata is None:
            source = self.settings.embeddings_metadata_path
            if not source.exists():
                raise FileNotFoundError(f"Embedding metadata not found: {source}")
            self._metadata = json.loads(source.read_text(encoding="utf-8"))
        return self._metadata

    def _items(self) -> list[Assessment]:
        try:
            payload = self._load_bm25()
            return [Assessment.model_validate(item) for item in payload["items"]]
        except (FileNotFoundError, ValueError, OSError, EOFError) as exc:
            logger.warning("catalog_fallback_for_items", error=str(exc))
            return self._catalog_assessments()

    def _normalize_score_map(self, scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score == min_score:
            return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}
        return {
            key: (value - min_score) / (max_score - min_score)
            for key, value in scores.items()
        }

    def _fallback_candidates(self, query: str, candidate_limit: int) -> list[RetrievalResult]:
        assessments = self._catalog_assessments()
        if not assessments:
            return []

        tokens = set(tokenize(query))
        scored: list[RetrievalResult] = []
        for assessment in assessments:
            text = assessment.search_text.casefold()
            overlap = sum(1 for token in tokens if token in text)
            if overlap == 0:
                continue
            scored.append(
                RetrievalResult(
                    assessment=assessment,
                    score=float(overlap),
                    bm25_score=float(overlap),
                    dense_score=0.0,
                )
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:candidate_limit]

    def _catalog_assessments(self) -> list[Assessment]:
        return CatalogParser(self.settings).parse_file()

    def _normalize_name(self, name: str) -> str:
        return " ".join(tokenize(name))


def retrieve_assessments(query: str, top_k: int | None = None) -> list[RetrievalResult]:
    """Retrieve assessments using the default hybrid retriever."""

    return HybridRetriever().retrieve(query=query, top_k=top_k)
