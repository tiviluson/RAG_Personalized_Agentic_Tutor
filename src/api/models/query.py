"""Pydantic schemas for chat/query endpoints."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AnswerMode(StrEnum):
    """Supported answer styles."""

    LONG = "long"
    SHORT = "short"
    ELI5 = "eli5"


class MetadataFilters(BaseModel):
    """Optional metadata filters narrowing retrieval scope."""

    course_id: str | None = None
    module_week: str | None = None
    module_name: str | None = None
    uploaded_by: str | None = None
    source_filename: str | None = None


class CreateSessionRequest(BaseModel):
    """Body for POST /api/chat/session."""

    student_id: str
    course_id: str


class CreateSessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str


class ChatQueryRequest(BaseModel):
    """Body for POST /api/chat/query."""

    session_id: str
    query: str
    answer_mode: AnswerMode = AnswerMode.LONG
    filters: MetadataFilters | None = None


class Citation(BaseModel):
    """Source metadata for a single inline citation marker."""

    index: int
    source_filename: str
    page_num: int | None = None
    section: str | None = None
    chapter: str | None = None
    module_week: str | None = None
    collection: str = ""
    content_category: str | None = None
    relevance_score: float = 0.0
    text_preview: str = Field(default="")


class PipelineMetrics(BaseModel):
    """Timing and count metrics from the query pipeline."""

    preprocessing_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    context_assembly_ms: float = 0.0
    total_candidates: int = 0
    deduped_candidates: int = 0
    final_candidates: int = 0
    strategy_used: str = "simple"


class ChatEvent(BaseModel):
    """A single SSE event in the streaming response.

    Types:
        status: Pipeline stage update ("Thinking...", "Searching...", "Reranking...").
        chunk: A text fragment of the response.
        done: Final event with citations and metrics.
        error: An error occurred.
    """

    type: str  # "chunk" | "done" | "error"
    data: str | dict = ""
