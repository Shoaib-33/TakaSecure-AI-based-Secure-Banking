from langchain_core.documents import Document

from takasecure_rag.authorization import PolicyCatalog, canonical_policy_id
from takasecure_rag.config import Settings


def catalog() -> PolicyCatalog:
    return PolicyCatalog(Settings().policy_catalog)


def test_unknown_employee_role_is_denied():
    decision = catalog().authorize_request("employee", None)
    assert not decision.allowed


def test_department_role_mismatch_is_denied():
    decision = catalog().authorize_request("credit_analyst", "compliance")
    assert not decision.allowed


def test_authorized_role_and_department_are_allowed():
    decision = catalog().authorize_request("credit_analyst", "credit")
    assert decision.allowed


def test_document_filter_applies_current_policy_access_to_legacy_version():
    documents = [
        Document(page_content="TSB-CREDIT-02-1 current policy"),
        Document(page_content="TSB-CREDIT-02-1-LEGACY superseded policy"),
        Document(page_content="TSB-COMPLIANCE-02-1 unrelated restricted policy"),
        Document(page_content="Unclassified appendix"),
    ]
    filtered = catalog().filter_documents(documents, "credit_analyst", "credit")
    assert len(filtered) == 2
    assert all("TSB-CREDIT" in document.page_content for document in filtered)
    assert canonical_policy_id("TSB-CREDIT-02-1-LEGACY") == "TSB-CREDIT-02-1"


def test_approved_tool_comes_from_catalog_metadata():
    documents = [Document(page_content="Policy TSB-CREDIT-01-1 applies.")]
    assert catalog().approved_tools(documents) == {
        "TSB-CREDIT-01-1": "calculate_document_age"
    }
