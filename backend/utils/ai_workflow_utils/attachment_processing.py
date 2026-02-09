"""Helpers for extracting full-text content from user-provided attachments.

Unlike the document ingestion pipeline (which chunks aggressively for
vectorization), chat attachments must stay intact so the LLM receives the
complete context the user supplied.

This module reuses the document processors from ``document_processing`` so
that every file type supported for KB ingestion is also supported as a chat
attachment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from backend.utils.ai_workflow_utils.document_processing import (
    get_document_processor,
)

logger = logging.getLogger(__name__)


class AttachmentProcessingError(ValueError):
    """Raised when attachment text cannot be extracted (e.g., CID corruption)."""


def _passthrough_chunker(doc_metadata: dict, text: str) -> list[dict]:
    """A no-op chunker that captures the full text as a single chunk.

    The document processors require a *chunk_document* callable.  By using
    this trivial implementation we get back the complete extracted text
    without any splitting, which is what we need for chat context.
    """
    return [{"text": text}]


def extract_attachment_text(
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str] = None,
) -> str:
    """Extract the raw text for supported attachment types.

    Delegates to the same document processors used by the KB ingestion
    pipeline, but collects the full text instead of chunking it.

    Args:
        file_bytes: Raw attachment bytes.
        filename: Original filename (used for type inference).
        content_type: Optional MIME type supplied by the client.

    Raises:
        AttachmentProcessingError: If the file type is unsupported or
            the content cannot be extracted.
    """
    try:
        processor = get_document_processor(content_type, filename)
    except ValueError:
        raise AttachmentProcessingError(
            f"Unsupported attachment type for '{filename}' "
            f"({content_type or 'unknown'})"
        )

    try:
        chunks = processor.process(
            file_bytes,
            chunk_document=_passthrough_chunker,
            storage_path=filename,
            doc_name=Path(filename).name,
        )
    except ValueError as exc:
        raise AttachmentProcessingError(str(exc)) from exc

    if not chunks:
        return ""

    return "\n\n".join(chunk["text"] for chunk in chunks if chunk.get("text"))
