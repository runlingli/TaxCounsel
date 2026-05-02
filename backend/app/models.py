from typing import Literal

from pydantic import BaseModel, Field


class TaxAnswer(BaseModel):
    answer: str = Field(description="Direct answer, or refusal if out of scope")
    page_citations: list[str] = Field(
        description="Page-level citations: 'IRS Pub 17 · Chapter 1 · p.12'"
    )
    tax_year: int = Field(description="Tax year this answer applies to")
    confidence: float = Field(description="Top reranker score")
    disclaimer: str = Field(
        default="For informational purposes only. Consult a licensed tax professional."
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    tax_year: int = 2025
    chat_history: list[ChatMessage] = Field(default_factory=list)


class RetrievedDoc(BaseModel):
    source: str
    section: str
    page: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: TaxAnswer
    original_query: str
    rewritten_query: str
    retrieved_docs: list[RetrievedDoc]
    attempts: int


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    tax_year: int
    is_helpful: bool
    comment: str = ""


class CacheStats(BaseModel):
    size: int
    max_size: int
