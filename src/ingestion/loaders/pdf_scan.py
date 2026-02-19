"""Loader for scanned and handwritten PDF documents."""

import io
from pathlib import Path

import fitz
from google.genai import types
from loguru import logger
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.clients import get_genai_client
from src.ingestion.types import PageData

NATIVE_TEXT_THRESHOLD = 100  # chars per page -- below this, treat as scanned

_det_predictor = None
_rec_predictor = None


def _get_surya_predictors():
    """Return lazily-initialised surya detection and recognition predictors.

    Model is loaded once and cached at module level to avoid re-instantiation.

    Returns:
        A ``(DetectionPredictor, RecognitionPredictor)`` tuple.
    """
    global _det_predictor, _rec_predictor
    if _det_predictor is None:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        _det_predictor = DetectionPredictor()
        _rec_predictor = RecognitionPredictor()
    return _det_predictor, _rec_predictor


def _ocr_printed_image(image: Image.Image) -> str:
    """OCR a printed/typed scan using surya-ocr.

    Args:
        image: PIL image of the scanned page.

    Returns:
        Extracted text with lines joined by newlines.
    """
    det, rec = _get_surya_predictors()
    results = rec([image], det, langs=[["en"]])
    lines = [line.text for line in results[0].text_lines]
    return "\n".join(lines)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def _ocr_handwritten_image(image_bytes: bytes) -> str:
    """Transcribe handwritten notes using Gemini Flash Vision.

    Args:
        image_bytes: PNG-encoded image of the handwritten page.

    Returns:
        Plain-text transcription of the handwritten content.

    Raises:
        Exception: Propagated after 3 retry attempts.
    """
    client = get_genai_client()
    prompt = (
        "Transcribe ALL handwritten text from this note page exactly as written. "
        "Preserve the structure, headings, and bullet points. "
        "For mathematical expressions, use LaTeX notation ($...$). "
        "Output plain text only -- do not add any commentary."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
    )
    return response.text


def load_scanned_pdf(
    pdf_path: str | Path,
    is_handwritten: bool = False,
) -> list[PageData]:
    """Load a scanned or handwritten PDF, applying OCR per page.

    Pages with sufficient native text (>= ``NATIVE_TEXT_THRESHOLD``
    chars) are used as-is. Otherwise:

    - Printed scans: surya-ocr (local, no API key needed).
    - Handwritten notes: Gemini Flash Vision.

    Args:
        pdf_path: Path to the PDF file.
        is_handwritten: If ``True``, use Gemini Vision for OCR
            instead of surya-ocr.

    Returns:
        A list of ``PageData`` dicts, one per page.
    """
    doc = fitz.open(str(pdf_path))
    pages_data: list[dict] = []

    for page_num, page in enumerate(doc):
        native_text = page.get_text("text").strip()

        if len(native_text) >= NATIVE_TEXT_THRESHOLD:
            pages_data.append({
                "text": native_text,
                "page_num": page_num + 1,
                "extraction_method": "extracted (raw)",
            })
            continue

        pixmap = page.get_pixmap(dpi=200)
        image_bytes = pixmap.tobytes("png")

        if is_handwritten:
            client = get_genai_client()
            if client:
                text = _ocr_handwritten_image(image_bytes)
                extraction_method = "generated"
            else:
                text = f"[Handwritten page {page_num + 1}: set GOOGLE_API_KEY]"
                extraction_method = "extracted (raw)"
        else:
            pil_image = Image.open(io.BytesIO(image_bytes))
            text = _ocr_printed_image(pil_image)
            extraction_method = "extracted (ocr)"

        pages_data.append({
            "text": text,
            "page_num": page_num + 1,
            "extraction_method": extraction_method,
        })

    doc.close()

    source_filename = Path(pdf_path).name
    doc_type = "handwritten_note" if is_handwritten else "lecture_note"
    for p in pages_data:
        p["doc_type"] = doc_type
        p["source_filename"] = source_filename

    methods = {p["extraction_method"] for p in pages_data}
    logger.info(
        "Loaded {} pages from {} -- extraction methods: {}",
        len(pages_data),
        source_filename,
        methods,
    )
    return pages_data
