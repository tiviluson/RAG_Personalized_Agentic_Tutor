"""Upload endpoints for the ingestion pipeline."""

import uuid
from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from loguru import logger

from src.api.models.upload import (
    BatchUploadResponse,
    JobStatusResponse,
    UploadResponse,
)
from src.config import settings
from src.db.job_store import get_job_status, set_job_status
from src.db.qdrant import get_qdrant_client
from src.ingestion.pipeline import run_ingestion
from src.ingestion.types import DocType

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".md"}


def _save_upload(file: UploadFile, job_id: str) -> Path:
    """Save an uploaded file to the configured upload directory.

    Args:
        file: The uploaded file from FastAPI.
        job_id: Job ID used as a filename prefix to avoid collisions.

    Returns:
        The path where the file was saved.
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{job_id}_{file.filename}"
    dest.write_bytes(file.file.read())
    return dest


def _validate_upload(file: UploadFile) -> None:
    """Validate file extension and size.

    Args:
        file: The uploaded file from FastAPI.

    Raises:
        HTTPException: If the file extension or size is invalid.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    if file.size and file.size > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
        )


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    doc_type: DocType = Form(...),
    role: Literal["lecturer", "student"] = Form(...),
    course_id: int = Form(...),
    student_id: str | None = Form(None),
    module_name: str = Form(""),
    module_week: int | None = Form(None),
) -> BatchUploadResponse:
    """Upload one or more documents for ingestion.

    Validates files, saves them to disk, and queues background
    ingestion tasks. Returns immediately with job IDs.

    Args:
        background_tasks: FastAPI background task manager.
        files: One or more uploaded files (``.pdf`` or ``.md``).
        doc_type: The document type for all uploaded files.
        role: Uploader role (``lecturer`` or ``student``).
        course_id: The course this upload belongs to.
        student_id: Required if ``role`` is ``student``.
        module_name: Optional module/topic label.
        module_week: Optional week number.

    Returns:
        A ``BatchUploadResponse`` containing a job entry per file.
    """
    if role == "student" and not student_id:
        raise HTTPException(
            status_code=422,
            detail="student_id is required when role is 'student'.",
        )

    jobs = []
    client = get_qdrant_client()

    for file in files:
        _validate_upload(file)
        job_id = str(uuid.uuid4())
        file_path = _save_upload(file, job_id)

        metadata = {
            "course_id": course_id,
            "uploaded_by": student_id if role == "student" else "lecturer",
            "student_id": student_id if role == "student" else None,
            "module_name": module_name,
            "module_week": module_week,
        }

        set_job_status(
            job_id,
            status="queued",
            progress=0,
            filename=file.filename,
            doc_type=doc_type.value,
        )

        background_tasks.add_task(
            run_ingestion,
            file_path=file_path,
            doc_type=doc_type,
            metadata=metadata,
            job_id=job_id,
            client=client,
        )

        logger.info("Queued job {} for {}", job_id, file.filename)
        jobs.append(UploadResponse(job_id=job_id, filename=file.filename))

    return BatchUploadResponse(jobs=jobs)


@router.get("/upload/status/{job_id}", response_model=JobStatusResponse)
async def get_upload_status(job_id: str) -> JobStatusResponse:
    """Query the processing status of an upload job.

    Args:
        job_id: The job identifier returned by the upload endpoint.

    Returns:
        A ``JobStatusResponse`` with current status and progress.

    Raises:
        HTTPException: If the job ID is not found.
    """
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(job_id=job_id, **vars(job))
