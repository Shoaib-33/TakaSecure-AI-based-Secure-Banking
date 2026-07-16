from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    conversation_context: list[str] = Field(default_factory=list, max_length=10)
    user_role: str = "employee"
    department: str | None = None
    response_format: Literal["text", "json"] = "text"
    as_of_date: str | None = None


class QueryPlan(BaseModel):
    """A retrieval plan chosen by the language model."""

    strategy: Literal["direct", "multi_query"]
    retrieval_question: str
    reasoning: str


class RetrievalGrade(BaseModel):
    """The language model's decision about evidence sufficiency."""

    sufficient: bool
    corrective_query: str | None = None
    reasoning: str


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[str]
    grounded: bool
    escalation_required: bool = False


class Verification(BaseModel):
    """The language model's decision about whether an answer is publishable."""

    passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    grounded: bool
    escalation_required: bool
    cache_hit: bool
    retrieval_strategy: str
    correction_attempts: int
    verification: Verification
    sources: list[dict]
