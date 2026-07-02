"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings for API, retrieval, scraping, and LLM access."""

    app_name: str = "SHL Assessment Recommender"
    app_env: str = "development"
    log_level: str = "INFO"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    shl_catalog_base_url: str = "https://www.shl.com"
    shl_catalog_start_url: str = "https://www.shl.com/solutions/products/product-catalog/"
    scraper_timeout_seconds: int = Field(default=30, ge=1, le=120)
    scraper_max_concurrency: int = Field(default=5, ge=1, le=20)

    data_dir: Path = Path("data")
    catalog_json_path: Path = Path("data/shl_catalog.json")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_path: Path = Path("data/catalog_embeddings.npy")
    embeddings_metadata_path: Path = Path("data/catalog_embeddings_metadata.json")
    faiss_index_path: Path = Path("data/faiss.index")
    bm25_index_path: Path = Path("data/bm25_index.pkl")

    retrieval_bm25_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    retrieval_dense_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    retrieval_candidate_limit: int = Field(default=30, ge=1, le=100)
    retrieval_top_k: int = Field(default=10, ge=1, le=10)
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize log level names while rejecting unsupported values."""

        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(sorted(allowed))}")
        return normalized

    @field_validator("retrieval_dense_weight")
    @classmethod
    def validate_weight_sum(cls, dense_weight: float, values) -> float:
        """Keep hybrid retrieval weights interpretable."""

        bm25_weight = values.data.get("retrieval_bm25_weight", 0.45)
        if round(bm25_weight + dense_weight, 6) != 1.0:
            raise ValueError("RETRIEVAL_BM25_WEIGHT and RETRIEVAL_DENSE_WEIGHT must sum to 1.0")
        return dense_weight


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings for dependency injection."""

    return Settings()
