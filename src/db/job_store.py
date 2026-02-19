"""In-memory thread-safe job status store for tracking background ingestion jobs."""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class JobStatus:
    """Status record for a single ingestion job.

    Attributes:
        status: Current processing stage (queued, loading, chunking,
            embedding, storing, complete, or failed).
        progress: Completion percentage (0--100).
        error: Error message if the job failed, else empty string.
        filename: Original uploaded filename.
        doc_type: Document type label.
        doc_id: Assigned document ID (set on completion).
        chunks_indexed: Number of chunks stored (set on completion).
        created_at: timestamp of job creation.
        updated_at: timestamp of last status update.
    """

    status: str = "queued"
    progress: int = 0
    error: str = ""
    filename: str = ""
    doc_type: str = ""
    doc_id: str = ""
    chunks_indexed: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


JOB_TTL_HOURS = 24

_store: dict[str, JobStatus] = {}
_lock = threading.Lock()


def set_job_status(job_id: str, **fields: object) -> None:
    """Create or update a job status entry.

    Args:
        job_id: Unique identifier for the job.
        **fields: Keyword arguments matching ``JobStatus`` field names
            to create or update.
    """
    with _lock:
        if job_id not in _store:
            _store[job_id] = JobStatus(**fields)  # type: ignore[arg-type]
        else:
            job = _store[job_id]
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat()


def get_job_status(job_id: str) -> JobStatus | None:
    """Retrieve the status of a job.

    Args:
        job_id (str): Unique identifier for the job.

    Returns:
        JobStatus | None: The job status record, or ``None`` if the job ID is not found.
    """
    with _lock:
        return _store.get(job_id)


def cleanup_old_jobs(max_age_hours: int = JOB_TTL_HOURS) -> int:
    """Remove completed or failed jobs older than ``max_age_hours``.

    Args:
        max_age_hours (int): Maximum age in hours for terminal jobs.

    Returns:
        int: Number of jobs removed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    terminal = {"complete", "failed"}
    with _lock:
        to_remove = [
            jid
            for jid, job in _store.items()
            if job.status in terminal
            and datetime.fromisoformat(job.updated_at) < cutoff
        ]
        for jid in to_remove:
            del _store[jid]
    return len(to_remove)
