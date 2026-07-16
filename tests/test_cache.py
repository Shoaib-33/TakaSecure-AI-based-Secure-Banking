import takasecure_rag.cache as cache_module
from takasecure_rag.cache import VerifiedResponseCache
from takasecure_rag.config import Settings
from takasecure_rag.schemas import ChatRequest


def test_exact_cache_key_is_stable():
    cache = VerifiedResponseCache(Settings())
    request = ChatRequest(question="What evidence is required?")
    assert cache.key(request) == cache.key(request)
    assert len(cache.key(request)) == 64


def test_cache_key_changes_with_authorization_context():
    cache = VerifiedResponseCache(Settings())
    employee = ChatRequest(question="What evidence is required?", user_role="employee")
    auditor = ChatRequest(question="What evidence is required?", user_role="internal_auditor")
    assert cache.key(employee) != cache.key(auditor)


def test_cache_key_changes_with_corpus_version():
    request = ChatRequest(question="What evidence is required?")
    old = VerifiedResponseCache(Settings(corpus_version="4.0"))
    new = VerifiedResponseCache(Settings(corpus_version="4.1"))
    assert old.key(request) != new.key(request)


class BrokenStore:
    def mget(self, _keys):
        raise RuntimeError("cache unavailable")

    def mset(self, _items):
        raise RuntimeError("cache unavailable")


def test_cache_initialization_failure_is_optional(monkeypatch):
    def fail_to_initialize(**_kwargs):
        raise RuntimeError("invalid cache configuration")

    monkeypatch.setattr(cache_module, "UpstashRedisByteStore", fail_to_initialize)
    cache = VerifiedResponseCache(
        Settings(
            upstash_redis_rest_url="https://example.invalid",
            upstash_redis_rest_token="test-token",
        )
    )
    assert not cache.enabled
    assert cache.status == "error"
    assert cache.store is None


def test_cache_read_failure_disables_cache_and_returns_miss():
    cache = VerifiedResponseCache(Settings())
    cache.enabled = True
    cache.store = BrokenStore()
    assert cache.get("key") is None
    assert cache.status == "error"
    assert cache.store is None


def test_cache_write_failure_does_not_raise():
    cache = VerifiedResponseCache(Settings())
    cache.enabled = True
    cache.store = BrokenStore()
    assert not cache.put("key", {"answer": "safe"})
    assert cache.status == "error"
