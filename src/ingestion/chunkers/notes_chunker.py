"""Chunker for scanned and handwritten notes (post-OCR text)."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.ingestion.chunkers.base import _token_len
from src.ingestion.types import PageData

NOTES_MAX_TOKENS = 512
NOTES_OVERLAP = 100


def chunk_scanned_pages(pages_data: list[PageData]) -> list[dict]:
    """Chunk post-OCR text from scanned or handwritten pages.

    Splits each page independently so that page boundaries are
    preserved in the chunk metadata.

    Args:
        pages_data: A list of ``PageData`` dicts from
            ``load_scanned_pdf``.

    Returns:
        A list of chunk dicts with keys: ``text``, ``chunk_index``,
        ``page_num``, ``doc_type``, ``source_filename``,
        ``ocr_method``, ``token_count``.
    """
    # Bug fix #3: use token values directly (not * 4) since
    # length_function=_token_len already measures in tokens
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=NOTES_MAX_TOKENS,
        chunk_overlap=NOTES_OVERLAP,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_chunks = []
    chunk_idx = 0

    for page in pages_data:
        page_splits = splitter.split_text(page["text"])
        for split_text in page_splits:
            all_chunks.append({
                "text": split_text,
                "chunk_index": chunk_idx,
                "page_num": page["page_num"],
                "doc_type": page["doc_type"],
                "source_filename": page["source_filename"],
                "ocr_method": page["ocr_method"],
                "token_count": _token_len(split_text),
            })
            chunk_idx += 1

    logger.info(
        "Chunked {} pages into {} chunks", len(pages_data), len(all_chunks)
    )
    return all_chunks
