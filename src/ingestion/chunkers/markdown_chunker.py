"""Header-aware chunker for Markdown documents.

Splits by header hierarchy first, then applies recursive character
splitting within large sections. Protects LaTeX display blocks and
fenced code blocks from being split mid-content.
"""

import re
import uuid

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from src.ingestion.chunkers.base import _token_len

MARKDOWN_MAX_TOKENS = 1024
MARKDOWN_OVERLAP = 100

# Regex patterns for protected blocks (LaTeX display math, math
# environments, and fenced code blocks)
_PROTECTED_BLOCKS = re.compile(
    r"(\$\$.+?\$\$)"                              # $$...$$
    r"|(\\\[.+?\\\])"                              # \[...\]
    r"|(\\begin\{[^}]+\}.+?\\end\{[^}]+\})"       # \begin{env}...\end{env}
    r"|(```.*?```)",                                # fenced code blocks
    re.DOTALL,
)


def _protect_blocks(text: str) -> tuple[str, dict[str, str]]:
    """Replace special blocks with unique placeholders.

    Protects LaTeX display math (``$$...$$``, ``\\[...\\]``),
    LaTeX environments (``\\begin{...}...\\end{...}``), and fenced
    code blocks from being split by the text splitter.

    Args:
        text: Raw markdown text.

    Returns:
        A tuple of (text with placeholders, mapping from placeholder
        to original block content).
    """
    placeholders: dict[str, str] = {}

    def _replace(match: re.Match) -> str:
        original = match.group(0)
        key = f"__PROTECTED_{uuid.uuid4().hex[:12]}__"
        placeholders[key] = original
        return key

    protected_text = _PROTECTED_BLOCKS.sub(_replace, text)
    return protected_text, placeholders


def _restore_blocks(text: str, placeholders: dict[str, str]) -> str:
    """Restore original blocks from placeholders.

    Args:
        text: Text containing placeholder tokens.
        placeholders: Mapping from placeholder to original content.

    Returns:
        Text with all placeholders replaced by original content.
    """
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


def chunk_markdown(
    md_text: str, source_filename: str = "notes.md"
) -> list[dict]:
    """Chunk markdown by header hierarchy, then split large sections.

    Each chunk carries header metadata (``h1``, ``h2``, ``h3``).
    LaTeX display blocks, math environments, and fenced code blocks
    are protected from being split mid-content.

    Args:
        md_text: Raw markdown text content.
        source_filename: Original filename for metadata.

    Returns:
        A list of chunk dicts with keys: ``text``, ``chunk_index``,
        ``section_headers``, ``chapter``, ``section``, ``token_count``,
        ``doc_type``, ``source_filename``.
    """
    # Protect special blocks before splitting
    protected_text, placeholders = _protect_blocks(md_text)

    # Stage 1: split by header hierarchy
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_splits = md_splitter.split_text(protected_text)

    # Stage 2: further split any section that exceeds the token limit
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MARKDOWN_MAX_TOKENS,
        chunk_overlap=MARKDOWN_OVERLAP,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " "],
    )
    final_docs = char_splitter.split_documents(header_splits)

    # Restore protected blocks and build chunk dicts
    chunks = []
    for i, doc in enumerate(final_docs):
        restored_text = _restore_blocks(doc.page_content, placeholders)
        chunks.append({
            "text": restored_text,
            "chunk_index": i,
            "section_headers": doc.metadata,
            "chapter": doc.metadata.get("h1"),
            "section": doc.metadata.get("h2"),
            "doc_type": "markdown_note",
            "source_filename": source_filename,
            "extraction_method": "extracted (raw)",
        })

    logger.info("Chunked markdown into {} chunks", len(chunks))
    return chunks
