"""Utilities for extracting and preparing text from uploaded documents."""

from typing import BinaryIO, TypedDict


class PageData(TypedDict):
    """Text extracted from one PDF page."""

    page_number: int
    text: str


class TextChunk(TypedDict):
    """A text segment and its source PDF page."""

    page_number: int
    text: str


def extract_text_from_pdf(pdf_file: BinaryIO) -> list[PageData]:
    """Extract text from every page of a PDF file object.

    Empty or image-only pages are included with an empty ``text`` value.
    Invalid or unreadable PDFs raise ``ValueError`` with a helpful message.
    """
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction requires pypdf; install the project dependencies."
        ) from exc

    try:
        reader = PdfReader(pdf_file)
        pages: list[PageData] = []
        for page_number, page in enumerate(reader.pages, start=1):
            # pypdf may return None when a page has no extractable text.
            pages.append(
                {"page_number": page_number, "text": page.extract_text() or ""}
            )
        return pages
    except (PdfReadError, OSError, TypeError) as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc


def split_text_into_chunks(
    pages: list[PageData], chunk_size: int = 500, overlap: int = 100
) -> list[TextChunk]:
    """Split page text into overlapping character chunks.

    Each returned chunk keeps the page number it came from. Empty or
    whitespace-only pages do not produce chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[TextChunk] = []
    for page in pages:
        text = page["text"]
        if not text or not text.strip():
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {"page_number": page["page_number"], "text": chunk_text}
                )

            if end == len(text):
                break
            start = end - overlap

    return chunks
