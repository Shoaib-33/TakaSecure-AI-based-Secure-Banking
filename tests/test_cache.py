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
