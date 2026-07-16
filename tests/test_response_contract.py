from langchain_core.documents import Document

from takasecure_rag.graph import _display_page, _length_failure_verification
from takasecure_rag.schemas import ChatResponse, GroundedAnswer, Verification


def test_backend_normalizes_zero_based_pdf_page():
    assert _display_page(Document(page_content="x", metadata={"page": 129})) == 130


def test_pdf_page_label_wins_when_present():
    document = Document(
        page_content="x",
        metadata={"page": 129, "page_label": "A-12"},
    )
    assert _display_page(document) == "A-12"


def test_truncated_verifier_fails_closed():
    verification = _length_failure_verification()
    assert not verification.passed
    assert verification.unsupported_claims
    assert "withheld" in verification.reasoning


def test_tool_routing_fields_survive_response_contract():
    answer = GroundedAnswer(
        answer="Use the approved calculator.",
        citations=["TSB-CREDIT-01-1"],
        grounded=True,
        requires_tool=True,
        tool_name="calculate_document_age",
        tool_inputs={"document_date": "2026-01-01"},
    )
    response = ChatResponse(
        answer=answer.answer,
        citations=answer.citations,
        grounded=True,
        escalation_required=False,
        cache_hit=False,
        retrieval_strategy="direct",
        correction_attempts=0,
        verification=Verification(passed=True, reasoning="Evidence checks passed."),
        sources=[],
        cache_status="miss",
        requires_tool=answer.requires_tool,
        tool_name=answer.tool_name,
        tool_inputs=answer.tool_inputs,
    )
    assert response.tool_name == "calculate_document_age"
    assert response.tool_inputs["document_date"] == "2026-01-01"
