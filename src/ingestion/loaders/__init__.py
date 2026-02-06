"""Document loaders for the ingestion pipeline."""

from src.ingestion.loaders.markdown import load_markdown
from src.ingestion.loaders.pdf_textbook import load_textbook_pdf

__all__ = [
    "load_markdown",
    "load_textbook_pdf",
]
