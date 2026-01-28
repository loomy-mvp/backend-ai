"""Utilities for processing different document types prior to embedding.

The module exposes a simple factory that maps MIME content types (or file
extensions) to dedicated processors. Each processor is responsible for
transforming the raw file bytes into semantic chunks that downstream
components (e.g., embeddings, vector stores) can consume.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import pdfplumber
from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from openpyxl import load_workbook
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

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
    """Process DOCX documents into text chunks.
    
    Extracts text from paragraphs, tables, headers, and footers.
    Skips sections that contain images (to be handled separately via OCR).
    """

    def _has_images_in_paragraph(self, paragraph) -> bool:
        """Check if a paragraph contains embedded images."""
        # Check for inline images (drawings and pictures)
        drawing_elements = paragraph._element.findall('.//' + qn('w:drawing'))
        picture_elements = paragraph._element.findall('.//' + qn('w:pict'))
        return len(drawing_elements) > 0 or len(picture_elements) > 0

    def _extract_table_text(self, table: Table) -> str:
        """Extract text from a table, preserving structure."""
        table_text_parts: List[str] = []
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells]
            # Join cells with tab separator for structure
            table_text_parts.append(" | ".join(row_cells))
        return "\n".join(table_text_parts)

    def _has_images_in_table(self, table: Table) -> bool:
        """Check if a table contains any images."""
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    if self._has_images_in_paragraph(paragraph):
                        return True
        return False

    def process(
        self,
        file_bytes: bytes,
        *,
        chunk_document: Chunker,
        storage_path: str,
        doc_name: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        
        try:
            doc = DocxDocument(io.BytesIO(file_bytes))
        except Exception as e:
            print(f"[DOCX] Failed to open document '{doc_name}': {e}")
            raise ValueError(f"Failed to open DOCX document '{doc_name}': {e}")

        # Track if document has any extractable content
        has_content = False
        
        # Extract text from main body
        # We'll process the document as sections, treating each logical block
        # Note: DOCX doesn't have page numbers directly, we simulate with sections
        section_number = 1
        current_section_text: List[str] = []
        section_has_images = False
        
        # Iterate through body elements in order (paragraphs and tables)
        for element in doc.element.body:
            # Check if element is a paragraph
            if element.tag == qn('w:p'):
                for paragraph in doc.paragraphs:
                    if paragraph._element == element:
                        # Check for images in this paragraph
                        if self._has_images_in_paragraph(paragraph):
                            section_has_images = True
                            print(f"[DOCX] Image detected in section {section_number} of '{doc_name}' - skipping section")
                            break
                        
                        text = paragraph.text.strip()
                        if text:
                            current_section_text.append(text)
                        break
            
            # Check if element is a table
            elif element.tag == qn('w:tbl'):
                for table in doc.tables:
                    if table._tbl == element:
                        if self._has_images_in_table(table):
                            section_has_images = True
                            print(f"[DOCX] Image detected in table in section {section_number} of '{doc_name}' - skipping section")
                            break
                        
                        table_text = self._extract_table_text(table)
                        if table_text.strip():
                            current_section_text.append(table_text)
                        break
            
            # Check for section breaks to split content
            if element.tag == qn('w:sectPr') or (
                element.tag == qn('w:p') and 
                element.find('.//' + qn('w:sectPr')) is not None
            ):
                # End of section - process accumulated text
                if current_section_text and not section_has_images:
                    section_text = "\n\n".join(current_section_text)
                    if section_text.strip():
                        has_content = True
                        doc_metadata = {
                            "name": doc_name,
                            "page": section_number,
                            "storage_path": storage_path,
                        }
                        chunks.extend(chunk_document(doc_metadata, section_text))
                
                section_number += 1
                current_section_text = []
                section_has_images = False
        
        # Process any remaining content
        if current_section_text and not section_has_images:
            section_text = "\n\n".join(current_section_text)
            if section_text.strip():
                has_content = True
                doc_metadata = {
                    "name": doc_name,
                    "page": section_number,
                    "storage_path": storage_path,
                }
                chunks.extend(chunk_document(doc_metadata, section_text))
        
        # Fallback: if structured parsing didn't work, try simple extraction
        if not has_content:
            all_text_parts: List[str] = []
            has_any_images = False
            
            for paragraph in doc.paragraphs:
                if self._has_images_in_paragraph(paragraph):
                    has_any_images = True
                    continue  # Skip paragraphs with images
                text = paragraph.text.strip()
                if text:
                    all_text_parts.append(text)
            
            for table in doc.tables:
                if self._has_images_in_table(table):
                    has_any_images = True
                    continue
                table_text = self._extract_table_text(table)
                if table_text.strip():
                    all_text_parts.append(table_text)
            
            if all_text_parts:
                full_text = "\n\n".join(all_text_parts)
                doc_metadata = {
                    "name": doc_name,
                    "page": 1,
                    "storage_path": storage_path,
                }
                chunks.extend(chunk_document(doc_metadata, full_text))
            elif has_any_images:
                print(f"[DOCX] Document '{doc_name}' contains only images - requires OCR processing")
            else:
                print(f"[DOCX] No extractable text found in '{doc_name}'")
        
        return chunks


class XlsxDocumentProcessor(DocumentProcessor):
    """Process XLSX spreadsheets into text chunks.
    
    Treats each worksheet as a separate "page" for chunking purposes.
    Extracts data from cells, preserving row/column structure.
    """

    def _format_cell_value(self, cell) -> str:
        """Convert cell value to string, handling different types."""
        if cell.value is None:
            return ""
        
        # Handle different cell types
        if cell.is_date and cell.value:
            try:
                return str(cell.value)
            except Exception:
                return str(cell.value)
        
        return str(cell.value).strip()

    def _extract_sheet_text(self, sheet) -> str:
        """Extract all text from a worksheet, preserving structure."""
        rows_text: List[str] = []
        
        for row in sheet.iter_rows():
            cell_values = [self._format_cell_value(cell) for cell in row]
            # Skip completely empty rows
            if not any(cell_values):
                continue
            # Join cells with separator
            rows_text.append(" | ".join(cell_values))
        
        return "\n".join(rows_text)

    def process(
        self,
        file_bytes: bytes,
        *,
        chunk_document: Chunker,
        storage_path: str,
        doc_name: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        
        try:
            # data_only=True gets computed values instead of formulas
            workbook = load_workbook(
                io.BytesIO(file_bytes), 
                data_only=True, 
                read_only=True
            )
        except Exception as e:
            print(f"[XLSX] Failed to open workbook '{doc_name}': {e}")
            raise ValueError(f"Failed to open XLSX workbook '{doc_name}': {e}")
        
        try:
            sheet_names = workbook.sheetnames
            
            if not sheet_names:
                print(f"[XLSX] No worksheets found in '{doc_name}'")
                return chunks
            
            for page_number, sheet_name in enumerate(sheet_names, start=1):
                try:
                    sheet = workbook[sheet_name]
                except Exception as e:
                    print(f"[XLSX] Failed to access sheet '{sheet_name}' in '{doc_name}': {e}")
                    continue
                
                sheet_text = self._extract_sheet_text(sheet)
                
                if not sheet_text.strip():
                    print(f"[XLSX] Skipping empty sheet '{sheet_name}' in '{doc_name}'")
                    continue
                
                # Include sheet name as context for the chunk
                sheet_header = f"[Sheet: {sheet_name}]\n\n"
                full_text = sheet_header + sheet_text
                
                doc_metadata = {
                    "name": doc_name,
                    "page": page_number,
                    "sheet_name": sheet_name,
                    "storage_path": storage_path,
                }
                chunks.extend(chunk_document(doc_metadata, full_text))
        
        finally:
            workbook.close()
        
        if not chunks:
            print(f"[XLSX] No extractable content found in '{doc_name}'")
        
        return chunks


class PptxDocumentProcessor(DocumentProcessor):
    """Process PPTX presentations into text chunks.
    
    Treats each slide as a separate "page" for chunking purposes.
    Extracts text from shapes, text frames, and tables.
    Skips slides that contain images (to be handled separately via OCR).
    """

    def _shape_has_image(self, shape) -> bool:
        """Check if a shape is or contains an image."""
        # Check for picture shapes
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            return True
        
        # Check for linked/embedded images
        if shape.shape_type == MSO_SHAPE_TYPE.LINKED_PICTURE:
            return True
        
        # Check for placeholders that might contain images
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format:
            # Picture placeholders
            if shape.placeholder_format.type and 'PICTURE' in str(shape.placeholder_format.type):
                return True
        
        return False

    def _extract_shape_text(self, shape) -> Optional[str]:
        """Extract text from a shape if it has a text frame."""
        if not shape.has_text_frame:
            return None
        
        text_parts: List[str] = []
        for paragraph in shape.text_frame.paragraphs:
            paragraph_text = "".join(run.text for run in paragraph.runs)
            if paragraph_text.strip():
                text_parts.append(paragraph_text.strip())
        
        return "\n".join(text_parts) if text_parts else None

    def _extract_table_text(self, table) -> str:
        """Extract text from a PowerPoint table."""
        rows_text: List[str] = []
        for row in table.rows:
            cell_texts: List[str] = []
            for cell in row.cells:
                # Extract text from cell's text frame
                cell_text_parts: List[str] = []
                for paragraph in cell.text_frame.paragraphs:
                    para_text = "".join(run.text for run in paragraph.runs)
                    if para_text.strip():
                        cell_text_parts.append(para_text.strip())
                cell_texts.append(" ".join(cell_text_parts))
            rows_text.append(" | ".join(cell_texts))
        return "\n".join(rows_text)

    def _slide_has_images(self, slide) -> bool:
        """Check if a slide contains any images."""
        for shape in slide.shapes:
            if self._shape_has_image(shape):
                return True
            # Check grouped shapes
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                for sub_shape in shape.shapes:
                    if self._shape_has_image(sub_shape):
                        return True
        return False

    def process(
        self,
        file_bytes: bytes,
        *,
        chunk_document: Chunker,
        storage_path: str,
        doc_name: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        
        try:
            presentation = Presentation(io.BytesIO(file_bytes))
        except Exception as e:
            print(f"[PPTX] Failed to open presentation '{doc_name}': {e}")
            raise ValueError(f"Failed to open PPTX presentation '{doc_name}': {e}")
        
        if not presentation.slides:
            print(f"[PPTX] No slides found in '{doc_name}'")
            return chunks
        
        slides_with_images: List[int] = []
        
        for slide_number, slide in enumerate(presentation.slides, start=1):
            # Check if slide contains images - skip for now
            if self._slide_has_images(slide):
                slides_with_images.append(slide_number)
                print(f"[PPTX] Slide {slide_number} of '{doc_name}' contains images - skipping")
                # For now, skip slides with images
                continue
            
            slide_text_parts: List[str] = []
            
            # Extract slide title if available
            if slide.shapes.title and slide.shapes.title.has_text_frame:
                title_text = self._extract_shape_text(slide.shapes.title)
                if title_text:
                    slide_text_parts.append(f"# {title_text}")
            
            # Extract text from all shapes
            for shape in slide.shapes:
                # Skip title as we already processed it
                if shape == slide.shapes.title:
                    continue
                
                # Handle tables
                if shape.has_table:
                    table_text = self._extract_table_text(shape.table)
                    if table_text.strip():
                        slide_text_parts.append(table_text)
                    continue
                
                # Handle grouped shapes
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    for sub_shape in shape.shapes:
                        if sub_shape.has_text_frame:
                            text = self._extract_shape_text(sub_shape)
                            if text:
                                slide_text_parts.append(text)
                    continue
                
                # Handle regular shapes with text
                if shape.has_text_frame:
                    text = self._extract_shape_text(shape)
                    if text:
                        slide_text_parts.append(text)
            
            if not slide_text_parts:
                print(f"[PPTX] Slide {slide_number} of '{doc_name}' has no extractable text")
                continue
            
            slide_text = "\n\n".join(slide_text_parts)
            
            doc_metadata = {
                "name": doc_name,
                "page": slide_number,
                "storage_path": storage_path,
            }
            chunks.extend(chunk_document(doc_metadata, slide_text))
        
        if slides_with_images:
            print(f"[PPTX] '{doc_name}': {len(slides_with_images)} slide(s) skipped due to images: {slides_with_images}")
        
        if not chunks:
            if slides_with_images:
                print(f"[PPTX] Document '{doc_name}' contains only image slides - requires OCR processing")
            else:
                print(f"[PPTX] No extractable content found in '{doc_name}'")
        
        return chunks


class XmlDocumentProcessor(DocumentProcessor):
    """Process XML documents into text chunks.
    
    Extracts text content from XML elements, handling different XML structures.
    Preserves element hierarchy as context where meaningful.
    """

    def _clean_tag(self, tag: str) -> str:
        """Remove namespace prefix from XML tag."""
        # Handle {namespace}tag format
        if '}' in tag:
            return tag.split('}', 1)[1]
        return tag

    def _extract_text_recursive(
        self, 
        element: ET.Element, 
        depth: int = 0,
        max_depth: int = 50
    ) -> List[str]:
        """Recursively extract text from XML elements."""
        if depth > max_depth:
            return []
        
        text_parts: List[str] = []
        
        # Get element's direct text
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())
        
        # Process child elements
        for child in element:
            child_texts = self._extract_text_recursive(child, depth + 1, max_depth)
            text_parts.extend(child_texts)
            
            # Get tail text (text after child element)
            if child.tail and child.tail.strip():
                text_parts.append(child.tail.strip())
        
        return text_parts

    def _extract_structured_text(self, root: ET.Element) -> str:
        """Extract text while preserving some structure from XML."""
        # Common document-like XML structures
        document_tags = {'document', 'doc', 'article', 'content', 'body', 'text'}
        section_tags = {'section', 'chapter', 'part', 'div', 'paragraph', 'p', 'para'}
        
        output_parts: List[str] = []
        
        def process_element(element: ET.Element, level: int = 0) -> None:
            tag = self._clean_tag(element.tag).lower()
            
            # Check if this is a section-like element
            is_section = tag in section_tags
            
            element_text_parts: List[str] = []
            
            # Get direct text
            if element.text and element.text.strip():
                element_text_parts.append(element.text.strip())
            
            # Process children
            for child in element:
                process_element(child, level + 1)
                # Get tail text
                if child.tail and child.tail.strip():
                    element_text_parts.append(child.tail.strip())
            
            # Add text from this element
            if element_text_parts:
                element_text = " ".join(element_text_parts)
                if is_section:
                    output_parts.append("\n" + element_text)
                elif element_text not in " ".join(output_parts):
                    # Avoid duplicating text already added by children
                    pass
        
        # Try structured extraction first
        process_element(root)
        
        if output_parts:
            return "\n\n".join(part.strip() for part in output_parts if part.strip())
        
        # Fallback to simple recursive extraction
        all_texts = self._extract_text_recursive(root)
        return "\n".join(all_texts)

    def process(
        self,
        file_bytes: bytes,
        *,
        chunk_document: Chunker,
        storage_path: str,
        doc_name: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        
        # Decode XML content
        try:
            xml_content = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                xml_content = file_bytes.decode('latin-1')
            except UnicodeDecodeError:
                print(f"[XML] Failed to decode '{doc_name}' with UTF-8 or Latin-1")
                raise ValueError(f"Failed to decode XML document '{doc_name}'")
        
        # Parse XML
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"[XML] Failed to parse XML document '{doc_name}': {e}")
            raise ValueError(f"Failed to parse XML document '{doc_name}': {e}")
        
        # Extract text content
        extracted_text = self._extract_structured_text(root)
        
        if not extracted_text.strip():
            # Fallback: try simple text extraction
            all_texts = self._extract_text_recursive(root)
            extracted_text = "\n".join(all_texts)
        
        if not extracted_text.strip():
            print(f"[XML] No text content found in '{doc_name}'")
            return chunks
        
        # Add root element info as context
        root_tag = self._clean_tag(root.tag)
        
        doc_metadata = {
            "name": doc_name,
            "page": 1,
            "root_element": root_tag,
            "storage_path": storage_path,
        }
        chunks.extend(chunk_document(doc_metadata, extracted_text))
        
        return chunks


_CONTENT_TYPE_PROCESSORS: Dict[str, Type[DocumentProcessor]] = {
    "application/pdf": PDFDocumentProcessor,
    "text/plain": TextDocumentProcessor,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxDocumentProcessor,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": XlsxDocumentProcessor,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": PptxDocumentProcessor,
    "application/xml": XmlDocumentProcessor,
    "text/xml": XmlDocumentProcessor,
}

_EXTENSION_FALLBACKS: Dict[str, Type[DocumentProcessor]] = {
    ".pdf": PDFDocumentProcessor,
    ".txt": TextDocumentProcessor,
    ".docx": DocxDocumentProcessor,
    ".xlsx": XlsxDocumentProcessor,
    ".pptx": PptxDocumentProcessor,
    ".xml": XmlDocumentProcessor,
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
