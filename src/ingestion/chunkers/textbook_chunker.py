"""Structure-aware chunker for typeset textbook and paper PDFs."""

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc.document import DoclingDocument
from loguru import logger

TEXTBOOK_MAX_TOKENS = 1024

_chunker_enc = tiktoken.get_encoding("cl100k_base")
_chunker_tokenizer = OpenAITokenizer(
    tokenizer=_chunker_enc, max_tokens=TEXTBOOK_MAX_TOKENS
)


def _get_page_no(chunk) -> int | None:
    """Extract the page number from a chunk's doc_items provenance.

    Args:
        chunk: A docling ``Chunk`` object with optional provenance
            metadata.

    Returns:
        The 1-based page number, or ``None`` if unavailable.
    """
    if chunk.meta and chunk.meta.doc_items:
        for item in chunk.meta.doc_items:
            if item.prov:
                return item.prov[0].page_no
    return None


def chunk_textbook(doc: DoclingDocument) -> list[dict]:
    """Chunk a DoclingDocument using docling's HybridChunker.

    Produces structure-aware chunks that respect section boundaries,
    keep tables atomic, and merge short adjacent elements.

    Args:
        doc: A ``DoclingDocument`` returned by ``load_textbook_pdf``.

    Returns:
        A list of chunk dicts with keys: ``text``, ``chunk_index``,
        ``page_num``, ``section_headings``, ``chapter``, ``section``,
        ``token_count``, ``doc_type``.
    """
    chunker = HybridChunker(
        tokenizer=_chunker_tokenizer,
        merge_peers=True,
    )

    chunks = []
    for i, chunk in enumerate(chunker.chunk(doc)):
        page_no = _get_page_no(chunk)

        headings = []
        if chunk.meta and chunk.meta.headings:
            headings = list(chunk.meta.headings)

        token_count = _chunker_tokenizer.count_tokens(chunk.text)

        chunks.append({
            "text": chunk.text,
            "chunk_index": i,
            "page_num": page_no,
            "section_headings": headings,
            "chapter": headings[0] if headings else None,
            "section": headings[1] if len(headings) > 1 else None,
            "token_count": token_count,
            "doc_type": "textbook",
        })

    logger.info(
        "Chunked into {} textbook chunks (max {} tokens)",
        len(chunks),
        TEXTBOOK_MAX_TOKENS,
    )
    if chunks:
        token_counts = [c["token_count"] for c in chunks]
        logger.info(
            "  Token stats: min={}, max={}, mean={:.0f}",
            min(token_counts),
            max(token_counts),
            sum(token_counts) / len(token_counts),
        )
    return chunks
