"""Helpers for extracting full-text content from user-provided attachments.

Unlike the document ingestion pipeline (which chunks aggressively for
vectorization), chat attachments must stay intact so the LLM receives the
complete context the user supplied. This module centralises the lightweight
parsing needed for PDFs and plain-text files.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pdfplumber

from backend.utils.ai_workflow_utils.document_processing import has_cid_corruption


class AttachmentProcessingError(ValueError):
    """Raised when attachment text cannot be extracted (e.g., CID corruption)."""


def _extract_pdf_text(file_bytes: bytes, filename: str) -> str:
    """Return the concatenated text of all PDF pages."""
    pages_text: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        if pdf.pages:
            first_page_text = pdf.pages[0].extract_text() or ""
            if has_cid_corruption(first_page_text):
                raise AttachmentProcessingError(
                    f"CID encoding corruption detected in '{filename}'"
                )

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            if has_cid_corruption(page_text):
                raise AttachmentProcessingError(
                    f"CID encoding corruption detected on page {page.page_number} of '{filename}'"
                )
            pages_text.append(page_text.strip())

    return "\n\n".join(pages_text)


def _extract_text_file(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def extract_attachment_text(
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str] = None,
) -> str:
    """Extract the raw text for supported attachment types.

    Args:
        file_bytes: Raw attachment bytes.
        filename: Original filename (used for type inference).
        content_type: Optional MIME type supplied by the client.
    """

    normalized_type = (content_type or "").lower()
    extension = Path(filename).suffix.lower()

    if normalized_type == "application/pdf" or extension == ".pdf":
        return _extract_pdf_text(file_bytes, filename)

    if normalized_type == "text/plain" or extension == ".txt":
        return _extract_text_file(file_bytes)

    raise AttachmentProcessingError(
        f"Unsupported attachment type for '{filename}' ({content_type or 'unknown'})"
    )
