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


def has_cid_corruption(text: str) -> bool:
    """
    Check if extracted text contains CID encoding corruption.
    
    CID (Character Identifier) corruption appears as patterns like:
    (cid:6)(cid:7)(cid:1) etc., indicating symbolic fonts that can't be decoded.
    
    Args:
        text: Extracted text to check
    
    Returns:
        True if CID corruption is detected
    """
    return "(cid:" in text.lower()


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
    """Process PDF documents into text chunks.
    
    Skips documents with CID encoding corruption (symbolic fonts).
    """

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
            # Check first page for CID corruption before processing entire document
            if pdf.pages:
                first_page_text = pdf.pages[0].extract_text() or ""
                if has_cid_corruption(first_page_text):
                    print(f"[PDF] Skipping document '{doc_name}': CID encoding corruption detected")
                    print(f"[PDF] This document uses symbolic fonts and requires OCR processing")
                    raise ValueError(f"CID encoding corruption detected in '{doc_name}' - document skipped")
            
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                
                # Double-check each page (in case corruption appears later)
                if has_cid_corruption(page_text):
                    print(f"[PDF] CID corruption detected on page {page_number} of '{doc_name}'")
                    raise ValueError(f"CID encoding corruption detected on page {page_number} - document skipped")
                
                doc_metadata = {
                    "name": doc_name,
                    "page": page_number,
                    "storage_path": storage_path,
                }
                chunks.extend(chunk_document(doc_metadata, page_text))
        return chunks


class TextDocumentProcessor(DocumentProcessor):
    """Process plain text documents into chunks."""
    
    def process(
        self,
        file_bytes: bytes,
        *,
        chunk_document: Chunker,
        storage_path: str,
        doc_name: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        
        # Decode text file (try UTF-8 first, fallback to latin-1)
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = file_bytes.decode('latin-1')
        
        if not text.strip():
            print(f"[TXT] Skipping empty document '{doc_name}'")
            return chunks
        
        # Text files don't have pages, so we treat the whole file as page 1
        doc_metadata = {
            "name": doc_name,
            "page": 1,
            "storage_path": storage_path,
        }
        chunks.extend(chunk_document(doc_metadata, text))
        return chunks


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
