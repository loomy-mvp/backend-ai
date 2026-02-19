"""Helpers for extracting full-text content from user-provided attachments.

Unlike the document ingestion pipeline (which chunks aggressively for
vectorization), chat attachments must stay intact so the LLM receives the
complete context the user supplied.

This module reuses the document processors from ``document_processing`` so
that every file type supported for KB ingestion is also supported as a chat
attachment.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, List, Optional

from backend.config.prompts import format_attachment_block
from backend.utils.ai_workflow_utils.document_processing import (
    get_document_processor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attachment type constants
# ---------------------------------------------------------------------------

TEXT_ATTACHMENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/xml",
    "text/xml",
}
TEXT_ATTACHMENT_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx", ".xlsx", ".pptx", ".xml"}
IMAGE_ATTACHMENT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


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


# ---------------------------------------------------------------------------
# Attachment decoding / classification helpers
# ---------------------------------------------------------------------------


def _decode_attachment_payload(data: str) -> tuple[bytes, Optional[str]]:
    """Decode a base64 payload or data URL and return bytes plus inferred MIME type."""
    if not data:
        raise ValueError("Attachment payload is empty")

    payload = data.strip()
    if payload.startswith("data:"):
        try:
            header, encoded = payload.split(",", 1)
        except ValueError as exc:
            raise ValueError("Malformed data URL for attachment") from exc
        mime = header.split(";")[0].replace("data:", "", 1)
        return base64.b64decode(encoded), mime or None
    return base64.b64decode(payload), None


def _is_text_attachment(content_type: Optional[str], filename: str) -> bool:
    if content_type and content_type.lower() in TEXT_ATTACHMENT_TYPES:
        return True
    extension = Path(filename).suffix.lower()
    return extension in TEXT_ATTACHMENT_EXTENSIONS


def _is_image_attachment(content_type: Optional[str], filename: str) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    extension = Path(filename).suffix.lower()
    return extension in IMAGE_ATTACHMENT_EXTENSIONS


def _encode_image_data_url(content_type: Optional[str], data: bytes, filename: str) -> str:
    # TODO: To reference a GCS location after persisting the image, modify this function accordingly.
    guessed_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{guessed_type};base64,{b64}"


def _build_attachment_context(attachments: Optional[List[dict]]) -> tuple[str, int, List[dict]]:
    """Create textual context and collect image payloads for attachments."""
    if not attachments:
        return "", 0, []

    text_sections: list[str] = []
    image_inputs: list[dict] = []

    for attachment in attachments:
        filename = attachment.get("filename") or "attachment"
        content_type = attachment.get("content_type")
        payload = attachment.get("data")
        if not payload:
            raise AttachmentProcessingError(
                f"Attachment '{filename}' has an empty payload and could not be processed"
            )

        try:
            file_bytes, inferred_type = _decode_attachment_payload(payload)
        except Exception as exc:
            raise AttachmentProcessingError(
                f"Failed to decode attachment '{filename}': {exc}"
            ) from exc
        finally:
            # Free the base64 string as soon as it has been decoded
            attachment.pop("data", None)

        resolved_type = (content_type or inferred_type or mimetypes.guess_type(filename)[0] or "").lower()

        if _is_text_attachment(resolved_type, filename):
            try:
                text_payload = extract_attachment_text(file_bytes, filename, resolved_type)
            except AttachmentProcessingError as exc:
                raise AttachmentProcessingError(
                    f"Failed to extract text from attachment '{filename}': {exc}"
                ) from exc
            finally:
                # Free raw bytes immediately after text extraction
                del file_bytes

            normalized_text = text_payload.strip()
            if not normalized_text:
                raise AttachmentProcessingError(
                    f"No text could be extracted from attachment '{filename}'"
                )

            text_sections.append(format_attachment_block(filename, normalized_text))
        elif _is_image_attachment(resolved_type, filename):
            try:
                data_url = _encode_image_data_url(resolved_type, file_bytes, filename)
            except Exception as exc:
                raise AttachmentProcessingError(
                    f"Failed to encode image attachment '{filename}': {exc}"
                ) from exc
            finally:
                del file_bytes
            image_inputs.append({
                "filename": filename,
                "data_url": data_url,
            })
        else:
            del file_bytes
            raise AttachmentProcessingError(
                f"Unsupported attachment type for '{filename}' ({resolved_type or 'unknown'})"
            )

    text_context = "\n----------\n".join(text_sections)
    return text_context, len(text_sections), image_inputs


def _guess_attachment_filename(content_type: Optional[str], index: int) -> str:
    """Derive a fallback filename when the client does not provide one."""
    extension = mimetypes.guess_extension(content_type or "") or ""
    return f"attachment_{index + 1}{extension}"


def _string_attachment_to_dict(raw_attachment: str, index: int) -> dict:
    """Convert a simple string payload into the structured attachment shape."""
    payload = raw_attachment.strip()
    if not payload:
        raise ValueError("Attachment string is empty")

    content_type: Optional[str] = None
    data = payload

    if payload.startswith("data:"):
        # Preserve the full data URL so downstream decoding works unchanged.
        header = payload.split(";", 1)[0]
        content_type = header.replace("data:", "", 1) or None
    elif ":" in payload:
        possible_type, remainder = payload.split(":", 1)
        if "/" in possible_type:
            content_type = possible_type
            data = remainder

    return {
        "filename": _guess_attachment_filename(content_type, index),
        "content_type": content_type,
        "data": data,
    }


def _normalize_request_attachments(raw_attachments: Optional[List[Any]]) -> List[dict]:
    """Ensure every attachment passed downstream is a dict with expected keys."""
    if not raw_attachments:
        return []

    normalized: List[dict] = []

    for idx, attachment in enumerate(raw_attachments):
        if not isinstance(attachment, str):
            logger.warning(
                "[attachments] Expected string payload but received %s at index %s",
                type(attachment),
                idx,
            )
            continue

        try:
            payload = _string_attachment_to_dict(attachment, idx)
            normalized.append(payload)
        except Exception as exc:
            logger.warning(
                "[attachments] Failed to normalize attachment at index %s: %s",
                idx,
                exc,
            )

    return normalized
