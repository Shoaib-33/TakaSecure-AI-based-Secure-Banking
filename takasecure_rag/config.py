from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "local-vllm"
    vllm_model: str = "takasecure"

    policy_pdf: Path = Path(
        "data/rag/v4/TakaSecure_Enterprise_Banking_Policy_Corpus_v4.pdf"
    )
    corpus_version: str = "4.0"
    qdrant_location: str = ":memory:"
    qdrant_collection: str = "takasecure_policy_v4"
    dense_embedding_model: str = "BAAI/bge-m3"
    sparse_embedding_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-base"
    retrieval_k: int = Field(default=20, ge=4, le=100)
    rerank_top_n: int = Field(default=8, ge=2, le=20)

    upstash_redis_rest_url: str | None = None
    upstash_redis_rest_token: str | None = None
    cache_ttl_seconds: int = Field(default=86_400, ge=60)
    cache_namespace: str = "takasecure:v1"

    max_corrections: int = Field(default=1, ge=0, le=2)
    max_regenerations: int = Field(default=1, ge=0, le=2)


@lru_cache
def get_settings() -> Settings:
    return Settings()
