"""Document loaders for the ingestion pipeline."""

from src.ingestion.loaders.markdown import load_markdown
from src.ingestion.loaders.pdf_scan import load_scanned_pdf
from src.ingestion.loaders.pdf_slides import load_slide_pdf
from src.ingestion.loaders.pdf_textbook import load_textbook_pdf

__all__ = [
    "load_markdown",
    "load_scanned_pdf",
    "load_slide_pdf",
    "load_textbook_pdf",
]
