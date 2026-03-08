"""Pipeline orchestrator: load -> chunk -> embed -> store.

Dispatches to the correct loader and chunker based on document type,
tracks job status at each stage, and stores results in Qdrant.
"""

import gc
import uuid
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient

from src.config import settings
from src.db.job_store import set_job_status
from src.ingestion.chunkers import (
    chunk_markdown,
    chunk_scanned_pages,
    chunk_slides,
    chunk_textbook,
)
from src.ingestion.embedders import embed_chunks
from src.ingestion.enrichment import enrich_chunks
from src.ingestion.loaders import (
    load_markdown,
    load_scanned_pdf,
    load_slide_pdf,
    load_textbook_pdf,
)
from src.ingestion.storage import find_doc_by_hash, file_hash, upsert_chunks
from src.ingestion.types import DocType


def _embed_and_store_batched(
    chunks: list[dict],
    client: QdrantClient,
    collection: str,
    doc_id: str,
    extra_meta: dict,
    job_id: str,
) -> int:
    """Embed and store chunks in batches to limit peak memory usage.

    Processes chunks through embedding and Qdrant storage in fixed-size
    batches, updating job progress after each batch. If a batch fails,
    all previously stored batches remain in Qdrant and the job status
    reflects the last successful count.

    Args:
        chunks (list[dict]): Chunk dicts with text + metadata (no embeddings yet).
        client (QdrantClient): An initialised Qdrant client.
        collection (str): Target Qdrant collection name.
        doc_id (str): Document-level unique identifier.
        extra_meta (dict): Additional payload fields for every point.
        job_id (str): Job ID for progress tracking.

    Returns:
        int: Total number of chunks stored.
    """
    embed_batch_size = 8  # Small for frequent progress updates
    store_batch_size = settings.upsert_batch_size  # 64 for efficient Qdrant upserts
    total = len(chunks)
    stored = 0
    pending: list[dict] = []

    for i in range(0, total, embed_batch_size):
        sub = chunks[i : i + embed_batch_size]

        # Update progress before embedding (55-80% range)
        progress = 55 + int((i / total) * 25)
        set_job_status(job_id, status="embedding", progress=progress)

        sub = embed_chunks(sub)
        pending.extend(sub)
        del sub
        gc.collect()

        # Store when pending reaches store batch size or last sub-batch
        is_last = (i + embed_batch_size) >= total
        if len(pending) >= store_batch_size or is_last:
            progress = 80 + int((stored / max(total, 1)) * 15)
            set_job_status(job_id, status="storing", progress=progress)

            n = upsert_chunks(
                client, collection, pending, doc_id=doc_id, extra_meta=extra_meta
            )
            stored += n
            pending = []

            logger.info("[ingest] Stored {} chunks ({}/{})", n, stored, total)

    set_job_status(
        job_id, status="embedding_storing", progress=95, chunks_indexed=stored
    )

    return stored


def run_ingestion(
    file_path: Path,
    doc_type: DocType,
    metadata: dict,
    job_id: str,
    client: QdrantClient,
) -> dict:
    """Execute the full ingestion pipeline for a single document.

    Dispatches to the appropriate loader and chunker based on
    ``doc_type``, embeds all chunks, and stores them in Qdrant.
    Updates job status at each stage.

    Args:
        file_path: Path to the uploaded file on disk.
        doc_type: The type of document being ingested.
        metadata: Upload metadata dict containing ``course_id``,
            ``uploaded_by``, and optionally ``student_id``,
            ``module_name``, ``module_week``.
        job_id: Unique job identifier for status tracking.
        client: An initialised Qdrant client.

    Returns:
        A dict with ``doc_id``, ``chunks_indexed``, ``doc_type``,
        and ``collection``.
    """
    doc_id = str(uuid.uuid4())
    # Strip the "{job_id}_" prefix added by _save_upload
    source_filename = file_path.name.split("_", 1)[1]
    logger.info(
        "[ingest] {} | type={} | doc_id={}", source_filename, doc_type, doc_id
    )

    try:
        # -- Dedup check --
        content_hash = file_hash(file_path)
        collection = _get_collection(metadata)
        existing_doc_id = find_doc_by_hash(client, collection, content_hash)
        if existing_doc_id:
            logger.info(
                "[ingest] Skipping duplicate file (hash={}, existing doc_id={})",
                content_hash[:12],
                existing_doc_id,
            )
            set_job_status(
                job_id,
                status="complete",
                progress=100,
                doc_id=existing_doc_id,
                chunks_indexed=0,
            )
            return {
                "doc_id": existing_doc_id,
                "chunks_indexed": 0,
                "doc_type": doc_type.value,
                "collection": collection,
                "skipped": "duplicate",
            }

        # -- Load --
        set_job_status(job_id, status="loading", progress=10)
        raw = _load(file_path, doc_type)

        # -- Chunk --
        set_job_status(job_id, status="chunking", progress=25)
        chunks = _chunk(raw, doc_type, source_filename)
        for chunk in chunks:
            if "source_filename" in chunk:
                chunk["source_filename"] = source_filename

        # -- Enrich (LLM-based content_category + topic_tags) --
        set_job_status(job_id, status="enriching", progress=40)
        chunks = enrich_chunks(chunks)

        # -- Embed & Store (batched to limit peak memory) --
        set_job_status(job_id, status="embedding_storing", progress=55)
        extra_meta = _build_extra_meta(metadata)
        extra_meta["file_hash"] = content_hash
        n = _embed_and_store_batched(
            chunks, client, collection, doc_id, extra_meta, job_id
        )

        # -- Cleanup intermediate data to free memory --
        del raw, chunks
        gc.collect()

        # -- Done --
        set_job_status(
            job_id, status="complete", progress=100, doc_id=doc_id, chunks_indexed=n
        )
        result = {
            "doc_id": doc_id,
            "chunks_indexed": n,
            "doc_type": doc_type.value,
            "collection": collection,
        }
        logger.info("[ingest] Complete: {}", result)
        return result

    except Exception as e:
        logger.error("[ingest] Failed for {}: {}", source_filename, e)
        set_job_status(job_id, status="failed", error=str(e))
        raise

    finally:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("[ingest] Could not delete uploaded file: {}", file_path)


def _load(file_path: Path, doc_type: DocType):
    """Dispatch to the appropriate loader.

    Args:
        file_path: Path to the file.
        doc_type: Document type determining which loader to use.

    Returns:
        Loader output (type varies by doc_type).

    Raises:
        ValueError: If ``doc_type`` is not recognised.
    """
    if doc_type in (DocType.TEXTBOOK, DocType.PAPER):
        return load_textbook_pdf(file_path)
    if doc_type == DocType.LECTURE_SLIDES:
        return load_slide_pdf(file_path)
    if doc_type == DocType.LECTURE_NOTES_TYPED:
        return load_scanned_pdf(file_path, is_handwritten=False)
    if doc_type == DocType.LECTURE_NOTES_HANDWRITTEN:
        return load_scanned_pdf(file_path, is_handwritten=True)
    if doc_type == DocType.MARKDOWN:
        return load_markdown(file_path)
    raise ValueError(f"Unknown doc_type: {doc_type}")


def _chunk(raw, doc_type: DocType, source_filename: str) -> list[dict]:
    """Dispatch to the appropriate chunker.

    Args:
        raw: Output from the loader (type varies by doc_type).
        doc_type: Document type determining which chunker to use.
        source_filename: Original filename for chunk metadata.

    Returns:
        A list of chunk dicts.

    Raises:
        ValueError: If ``doc_type`` is not recognised.
    """
    if doc_type in (DocType.TEXTBOOK, DocType.PAPER):
        return chunk_textbook(raw)
    if doc_type == DocType.LECTURE_SLIDES:
        return chunk_slides(raw)
    if doc_type in (DocType.LECTURE_NOTES_TYPED, DocType.LECTURE_NOTES_HANDWRITTEN):
        return chunk_scanned_pages(raw)
    if doc_type == DocType.MARKDOWN:
        return chunk_markdown(raw, source_filename)
    raise ValueError(f"Unknown doc_type: {doc_type}")


def _get_collection(metadata: dict) -> str:
    """Determine the target Qdrant collection.

    Args:
        metadata: Upload metadata. If ``student_id`` is present,
            routes to ``student_notes``; otherwise ``course_content``.

    Returns:
        The collection name.
    """
    return "student_notes" if metadata.get("student_id") else "course_content"


def _build_extra_meta(metadata: dict) -> dict:
    """Build extra metadata to attach to every chunk payload.

    Args:
        metadata: Upload metadata dict.

    Returns:
        A dict of extra payload fields for Qdrant.
    """
    extra = {
        "course_id": metadata.get("course_id"),
        "uploaded_by": metadata.get("uploaded_by"),
        "module_name": metadata.get("module_name", ""),
        "module_week": metadata.get("module_week"),
    }
    if metadata.get("student_id"):
        extra["student_id"] = metadata["student_id"]
    return extra
