"""Per-slide chunker for lecture slide PDFs."""

from loguru import logger

from src.ingestion.chunkers.base import _get_tiktoken_encoder
from src.ingestion.types import SlideData

SLIDE_MAX_TOKENS = 1024


def chunk_slides(slide_raw: list[SlideData]) -> list[dict]:
    """Produce one chunk per slide, splitting if a slide exceeds the token limit.

    Most slides are 100-300 tokens and remain as a single chunk. Slides
    that exceed ``SLIDE_MAX_TOKENS`` are split rather than truncated to
    avoid silently discarding content.

    Args:
        slide_raw: A list of ``SlideData`` dicts from ``load_slide_pdf``.

    Returns:
        A list of chunk dicts with keys: ``text``, ``chunk_index``,
        ``slide_number``, ``doc_type``, ``source_filename``,
        ``extraction_method``.
    """
    enc = _get_tiktoken_encoder()
    chunks = []
    chunk_index = 0

    for slide in slide_raw:
        text = slide["text"]
        used_vision = slide.get("used_vision", False)
        if used_vision and slide.get("visual_text"):
            text = f"{text}\n\n{slide['visual_text']}".strip()
            extraction_method = "hybrid" if slide["text"].strip() else "generated"
        else:
            extraction_method = "extracted (raw)"
        tokens = enc.encode(text)

        if len(tokens) <= SLIDE_MAX_TOKENS:
            chunks.append({
                **slide,
                "text": text,
                "chunk_index": chunk_index,
                "extraction_method": extraction_method,
            })
            chunk_index += 1
        else:
            for start in range(0, len(tokens), SLIDE_MAX_TOKENS):
                part_tokens = tokens[start : start + SLIDE_MAX_TOKENS]
                part_text = enc.decode(part_tokens)
                chunks.append({
                    **slide,
                    "text": part_text,
                    "chunk_index": chunk_index,
                    "extraction_method": extraction_method,
                })
                chunk_index += 1

    logger.info("Prepared {} slide chunks from {} slides", len(chunks), len(slide_raw))
    return chunks
