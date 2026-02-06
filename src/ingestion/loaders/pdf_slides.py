"""Loader for lecture slide PDFs with optional Gemini Vision fallback."""

from pathlib import Path

import fitz
from google import genai
from google.genai import types
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.ingestion.types import SlideData

SLIDE_SPARSE_THRESHOLD = 100  # chars -- below this, use Gemini Vision

_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client | None:
    """Return a lazily-initialised Gemini client singleton.

    Returns:
        A ``genai.Client`` if the API key is configured, else ``None``.
    """
    global _genai_client
    if _genai_client is None and settings.google_api_key:
        _genai_client = genai.Client(api_key=settings.google_api_key)
    return _genai_client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def _describe_slide_with_gemini(image_bytes: bytes) -> str:
    """Generate a text description of a visually-heavy slide.

    Args:
        image_bytes: PNG-encoded image of the slide.

    Returns:
        Plain-text description of the slide content.

    Raises:
        Exception: Propagated after 3 retry attempts.
    """
    client = _get_genai_client()
    prompt = (
        "This is a university lecture slide. "
        "1. List ALL text visible on the slide, preserving bullet structure. "
        "2. Describe any diagrams, graphs, figures, or tables in detail. "
        "3. Use LaTeX notation ($...$) for any mathematical expressions. "
        "Output plain text only."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
    )
    return response.text


def load_slide_pdf(pdf_path: str | Path) -> list[SlideData]:
    """Load a lecture slide PDF as one record per slide page.

    Extracts native text from each slide via PyMuPDF. Slides with fewer
    than ``SLIDE_SPARSE_THRESHOLD`` characters of native text are
    rendered to PNG and described by Gemini Flash Vision.

    Args:
        pdf_path: Path to the slide PDF file.

    Returns:
        A list of ``SlideData`` dicts, one per slide.
    """
    doc = fitz.open(str(pdf_path))
    slide_chunks: list[SlideData] = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        used_vision = False
        visual_text = ""  # bug fix #1: reset per slide (was outside loop)

        if len(text) < SLIDE_SPARSE_THRESHOLD:
            client = _get_genai_client()
            if client:
                pixmap = page.get_pixmap(dpi=150)
                image_bytes = pixmap.tobytes("png")
                visual_text = _describe_slide_with_gemini(image_bytes)
                used_vision = True
            else:
                text = (
                    f"[Slide {page_num + 1}: visual content"
                    " -- set GOOGLE_API_KEY for description]"
                )

        slide_chunks.append(
            SlideData(
                text=text,
                visual_text=visual_text,
                slide_number=page_num + 1,
                doc_type="lecture_slide",
                source_filename=Path(pdf_path).name,
                used_vision=used_vision,
            )
        )

    doc.close()
    vision_count = sum(1 for c in slide_chunks if c["used_vision"])
    logger.info(
        "Loaded {} slides from {} ({} used Gemini Vision)",
        len(slide_chunks),
        Path(pdf_path).name,
        vision_count,
    )
    return slide_chunks
