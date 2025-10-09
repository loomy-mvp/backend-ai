"""Utilities for processing different document types prior to embedding.

The module exposes a simple factory that maps MIME content types (or file
extensions) to dedicated processors. Each processor is responsible for
transforming the raw file bytes into semantic chunks that downstream
components (e.g., embeddings, vector stores) can consume.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional, Type

import pdfplumber

Chunk = Dict[str, object]
Chunker = Callable[[Dict[str, object], str], list[Chunk]]


class DocumentProcessor(ABC):
    """Base interface for document processors."""

    @abstractmethod
    def process(
        self,
        file_bytes: bytes,
        *,
        chunk_document: Chunker,
        storage_path: str,
        doc_name: str,
    ) -> list[Chunk]:
        """Return a list of chunks extracted from the document."""


class PDFDocumentProcessor(DocumentProcessor):
    """Process PDF documents into text chunks."""

    def process(
        self,
        file_bytes: bytes,
        *,
        chunk_document: Chunker,
        storage_path: str,
        doc_name: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                doc_metadata = {
                    "name": doc_name,
                    "page": page_number,
                    "storage_path": storage_path,
                }
                chunks.extend(chunk_document(doc_metadata, page_text))
        return chunks


class TextDocumentProcessor(DocumentProcessor):
    def process(self, *args, **kwargs):  # type: ignore[override]
        pass


class DocxDocumentProcessor(DocumentProcessor):
    def process(self, *args, **kwargs):  # type: ignore[override]
        pass


class XlsxDocumentProcessor(DocumentProcessor):
    def process(self, *args, **kwargs):  # type: ignore[override]
        pass


class PptxDocumentProcessor(DocumentProcessor):
    def process(self, *args, **kwargs):  # type: ignore[override]
        pass


class XlmDocumentProcessor(DocumentProcessor):
    def process(self, *args, **kwargs):  # type: ignore[override]
        pass


_CONTENT_TYPE_PROCESSORS: Dict[str, Type[DocumentProcessor]] = {
    "application/pdf": PDFDocumentProcessor,
    "text/plain": TextDocumentProcessor,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxDocumentProcessor,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": XlsxDocumentProcessor,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": PptxDocumentProcessor,
    "application/xml": XlmDocumentProcessor,
}

_EXTENSION_FALLBACKS: Dict[str, Type[DocumentProcessor]] = {
    ".pdf": PDFDocumentProcessor,
    ".txt": TextDocumentProcessor,
    ".docx": DocxDocumentProcessor,
    ".xlsx": XlsxDocumentProcessor,
    ".pptx": PptxDocumentProcessor,
    ".xlm": XlmDocumentProcessor,
    ".xml": XlmDocumentProcessor,
}


def get_document_processor(content_type: Optional[str], storage_path: str) -> DocumentProcessor:
    """Return a processor that matches the given content type or file extension."""

    processor_cls: Optional[Type[DocumentProcessor]] = None

    if content_type:
        processor_cls = _CONTENT_TYPE_PROCESSORS.get(content_type.lower())

    if processor_cls is None:
        extension = Path(storage_path).suffix.lower()
        processor_cls = _EXTENSION_FALLBACKS.get(extension)

    if processor_cls is None:
        raise ValueError(f"No document processor available for type '{content_type}' or extension '{storage_path}'.")

    return processor_cls()
