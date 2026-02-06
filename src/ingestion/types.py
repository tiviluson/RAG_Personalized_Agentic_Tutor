"""Shared types for the ingestion pipeline.

Defines the ``DocType`` enum and TypedDicts used across loaders,
chunkers, embedders, and storage modules.
"""

from enum import StrEnum
from typing import TypedDict


class DocType(StrEnum):
    """Supported document types for ingestion.

    Attributes:
        TEXTBOOK: Typeset textbook PDF (processed via docling).
        PAPER: Academic paper PDF (same pipeline as textbook).
        LECTURE_SLIDES: Slide-deck PDF (PyMuPDF + optional Gemini Vision).
        LECTURE_NOTES_TYPED: Typed/printed scanned notes (surya-ocr).
        LECTURE_NOTES_HANDWRITTEN: Handwritten scanned notes (Gemini Vision).
        MARKDOWN: Plain Markdown file.
    """

    TEXTBOOK = "textbook"
    PAPER = "paper"
    LECTURE_SLIDES = "lecture_slides"
    LECTURE_NOTES_TYPED = "lecture_notes_typed"
    LECTURE_NOTES_HANDWRITTEN = "lecture_notes_handwritten"
    MARKDOWN = "markdown"


class PageData(TypedDict, total=True):
    """Per-page output from scan/handwritten loaders.

    Attributes:
        text: Extracted text content for the page.
        page_num: Zero-based page index in the source PDF.
        doc_type: Document type label (e.g. ``"lecture_note"``).
        source_filename: Original filename of the uploaded document.
        ocr_method: OCR method used (``"native"``, ``"surya"``,
            ``"gemini_vision"``, or ``"skipped"``).
    """

    text: str
    page_num: int
    doc_type: str
    source_filename: str
    ocr_method: str


class SlideData(TypedDict, total=True):
    """Per-slide output from the slide loader.

    Attributes:
        text: Native text extracted from the slide.
        visual_text: Gemini Vision description (empty if not used).
        slide_number: One-based slide index.
        doc_type: Always ``"lecture_slide"``.
        source_filename: Original filename of the uploaded document.
        used_vision: Whether Gemini Vision was invoked for this slide.
    """

    text: str
    visual_text: str
    slide_number: int
    doc_type: str
    source_filename: str
    used_vision: bool
