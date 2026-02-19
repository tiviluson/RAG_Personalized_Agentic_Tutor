"""Markdown document loader."""

from pathlib import Path

from loguru import logger


def load_markdown(md_path: str | Path) -> str:
    """Read a Markdown file and return its raw text content.

    The text may contain LaTeX (``$...$``, ``$$...$$``), fenced code
    blocks, image references, and hyperlinks -- all are preserved as-is.
    Special-block handling is deferred to the chunker.

    Args:
        md_path: Path to the ``.md`` file.

    Returns:
        The full text content of the file.
    """
    text = Path(md_path).read_text(encoding="utf-8")
    logger.info("Loaded markdown: {} ({} chars)", Path(md_path).name, len(text))
    return text
