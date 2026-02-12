"""Pydantic schemas for upload endpoints."""

from pydantic import BaseModel

from src.ingestion.types import DocType


class UploadResponse(BaseModel):
    """Response for a single file upload."""

    job_id: str
    filename: str
    status: str = "queued"


class BatchUploadResponse(BaseModel):
    """Response for a batch file upload."""

    jobs: list[UploadResponse]


class JobStatusResponse(BaseModel):
    """Response for job status queries."""

    job_id: str
    status: str
    progress: int
    filename: str
    doc_type: str
    error: str = ""
    doc_id: str = ""
    chunks_indexed: int = 0
    created_at: str = ""
    updated_at: str = ""


class CollectionStatsResponse(BaseModel):
    """Response for collection statistics."""

    course_content: dict
    student_notes: dict
