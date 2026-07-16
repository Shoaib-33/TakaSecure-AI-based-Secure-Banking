import hashlib
import json
from typing import Any

from langchain_community.storage import UpstashRedisByteStore

from .config import Settings
from .schemas import ChatRequest


class VerifiedResponseCache:
    """Exact-match response cache backed by LangChain's Upstash byte store."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(
            settings.upstash_redis_rest_url and settings.upstash_redis_rest_token
        )
        self.store = (
            UpstashRedisByteStore(
                url=settings.upstash_redis_rest_url,
                token=settings.upstash_redis_rest_token,
                ttl=settings.cache_ttl_seconds,
                namespace=settings.cache_namespace,
            )
            if self.enabled
            else None
        )

    def key(self, request: ChatRequest) -> str:
        identity = {
            "request": request.model_dump(mode="json"),
            "corpus_version": self.settings.corpus_version,
            "model": self.settings.vllm_model,
            "pipeline": self.settings.cache_namespace,
        }
        canonical = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.store:
            return None
        value = self.store.mget([key])[0]
        return json.loads(value.decode("utf-8")) if value else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.store:
            self.store.mset([(key, json.dumps(value, ensure_ascii=False).encode("utf-8"))])
