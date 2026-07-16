import hashlib
import json
import logging
from typing import Any

from langchain_community.storage import UpstashRedisByteStore

from .config import Settings
from .schemas import ChatRequest


logger = logging.getLogger(__name__)


class VerifiedResponseCache:
    """Exact-match response cache backed by LangChain's Upstash byte store."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(
            settings.upstash_redis_rest_url and settings.upstash_redis_rest_token
        )
        self.store = None
        self.status = "miss" if self.enabled else "disabled"
        self.last_error: str | None = None
        if self.enabled:
            try:
                self.store = UpstashRedisByteStore(
                    url=settings.upstash_redis_rest_url,
                    token=settings.upstash_redis_rest_token,
                    ttl=settings.cache_ttl_seconds,
                    namespace=settings.cache_namespace,
                )
            except Exception as error:
                self._fail_open("initialization", error)

    def _fail_open(self, operation: str, error: Exception) -> None:
        self.last_error = f"{operation}: {type(error).__name__}"
        self.status = "error"
        self.enabled = False
        self.store = None
        logger.warning("Optional response cache disabled after %s failure: %s", operation, error)

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
        try:
            value = self.store.mget([key])[0]
            self.status = "hit" if value else "miss"
            return json.loads(value.decode("utf-8")) if value else None
        except Exception as error:  # The cache must never take down policy answering.
            self._fail_open("read", error)
            return None

    def put(self, key: str, value: dict[str, Any]) -> bool:
        if self.store:
            try:
                self.store.mset(
                    [(key, json.dumps(value, ensure_ascii=False).encode("utf-8"))]
                )
                return True
            except Exception as error:  # The answer remains valid if cache storage fails.
                self._fail_open("write", error)
        return False
