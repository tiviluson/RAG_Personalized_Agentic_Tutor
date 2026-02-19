"""Loader for typeset textbook and paper PDFs with structured extraction."""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument, PictureItem
from loguru import logger


def load_textbook_pdf(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
) -> DoclingDocument:
    """Load a textbook PDF and return a structured document representation.

    Extracts text, tables, and figure descriptions from a typeset PDF,
    preserving structural metadata such as headings, sections, and
    page boundaries.

    Args:
        pdf_path: Path to the PDF file.
        page_range: Optional ``(start, end)`` page range (1-based,
            inclusive) to limit processing.

    Returns:
        A ``DoclingDocument`` containing the structured content.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        prompt=(
            "This is an academic document."
            "1. List ALL text visible on the slide, preserving bullet structure. "
            "2. Describe any diagrams, graphs, figures, or tables in detail. "
            "3. Use LaTeX notation ($...$) for any mathematical expressions. "
        ),
    )
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    if page_range:
        result = converter.convert(str(pdf_path), page_range=page_range)
    else:
        result = converter.convert(str(pdf_path))

    doc = result.document
    figures = [el for el, _ in doc.iterate_items() if isinstance(el, PictureItem)]
    logger.info(
        "Loaded: {} | Pages: {} | Tables: {} | Figures: {}",
        Path(pdf_path).name,
        len(list(doc.pages)),
        len(list(doc.tables)),
        len(figures),
    )
    return doc
