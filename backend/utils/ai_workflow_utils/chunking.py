"""
Document chunking utilities for semantic text splitting.
Separated from kb_api to avoid FastAPI dependencies in batch jobs.
"""

import logging
import uuid
import numpy as np
import cohere
import os
import unicodedata

from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.config.chatbot_config import EMBEDDING_CONFIG

logger = logging.getLogger(__name__)

# Initialize Cohere client
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(cohere_api_key)
embedding_model_name = get_config_value(config_set=EMBEDDING_CONFIG, key="model")


def make_ascii_safe(text: str) -> str:
    """Convert text to ASCII-safe format for use in Pinecone vector IDs.
    
    Pinecone requires vector IDs to be ASCII only. This function:
    1. Normalizes Unicode characters (NFD decomposition)
    2. Removes combining marks (accents, diacritics)
    3. Encodes to ASCII with error handling
    
    Example:
        "dell'irregolare" -> "dell'irregolare" (apostrophe normalized)
        "Café" -> "Cafe"
    """
    # Normalize Unicode to decomposed form (NFD)
    normalized = unicodedata.normalize('NFD', text)
    # Remove combining marks and encode to ASCII
    ascii_text = ''.join(
        char for char in normalized
        if unicodedata.category(char) != 'Mn'  # Mn = Mark, nonspacing (accents)
    )
    # Encode to ASCII, replacing any remaining non-ASCII chars
    return ascii_text.encode('ascii', errors='replace').decode('ascii')


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

_PINECONE_METADATA_LIMIT_BYTES = 40_960
_METADATA_OVERHEAD_BYTES = 2_048
_MAX_TEXT_BYTES = _PINECONE_METADATA_LIMIT_BYTES - _METADATA_OVERHEAD_BYTES

def _split_oversized_text(text: str, max_bytes: int = _MAX_TEXT_BYTES) -> list[str]:
    """Split *text* into parts whose UTF-8 size does not exceed *max_bytes*.

    Prefers splitting at newline (``\\n``) boundaries so that table rows
    (single-``\\n``-separated lines produced by the Excel processor) stay
    intact.  Falls back to a hard byte-level split only when a single line
    already exceeds the limit.

    Args:
        text: The raw text of a chunk – may be arbitrarily large.
        max_bytes: Maximum UTF-8 byte length per returned part.

    Returns:
        A list of non-empty string parts, each within the byte budget.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]

    parts: list[str] = []
    lines = text.split("\n")
    current_bytes: list[bytes] = []
    current_size = 0

    for line in lines:
        line_enc = (line + "\n").encode("utf-8")

        if len(line_enc) > max_bytes:
            # A single line is bigger than the budget – hard-split it.
            # Flush whatever we were building first.
            if current_bytes:
                parts.append(b"".join(current_bytes).rstrip(b"\n").decode("utf-8"))
                current_bytes = []
                current_size = 0

            # Hard-split the oversized line by byte slices, then decode safely.
            raw = line.encode("utf-8")
            pos = 0
            while pos < len(raw):
                slice_bytes = raw[pos: pos + max_bytes]
                # Trim to a valid UTF-8 boundary.
                decoded = slice_bytes.decode("utf-8", errors="ignore")
                if decoded:
                    parts.append(decoded)
                pos += max_bytes
            continue

        if current_size + len(line_enc) > max_bytes:
            # Current part would overflow – flush it and start a new one.
            if current_bytes:
                parts.append(b"".join(current_bytes).rstrip(b"\n").decode("utf-8"))
            current_bytes = [line_enc]
            current_size = len(line_enc)
        else:
            current_bytes.append(line_enc)
            current_size += len(line_enc)

    if current_bytes:
        parts.append(b"".join(current_bytes).rstrip(b"\n").decode("utf-8"))

    return [p for p in parts if p.strip()]


def _make_chunks(doc_metadata: dict, text: str) -> list[dict]:
    """Build one-or-more chunk dicts from *text*, enforcing the Pinecone byte limit.

    If *text* fits within the budget a single chunk is returned.  Otherwise
    ``_split_oversized_text`` is called and each part becomes its own chunk
    (with a fresh UUID so IDs remain unique).
    """
    parts = _split_oversized_text(text)
    if len(parts) > 1:
        logger.warning(
            "[chunking] Text for '%s' page %s exceeded Pinecone metadata limit "
            "(%d bytes) and was split into %d sub-chunks.",
            doc_metadata.get("name"),
            doc_metadata.get("page"),
            len(text.encode("utf-8")),
            len(parts),
        )
    chunks = []
    for part in parts:
        chunk_id = f"{make_ascii_safe(doc_metadata['name'])}-{str(uuid.uuid4())}"
        chunk: dict = {
            "chunk_id": chunk_id[:512],
            "page": doc_metadata["page"],
            "text": part,
            "storage_path": doc_metadata["storage_path"],
        }
        if doc_metadata.get("has_images"):
            chunk["has_images"] = True
        chunks.append(chunk)
    return chunks


def chunk_document(doc_metadata: dict, content: str, max_similarity: float = 0.70, max_tokens: int = 1000, min_tokens: int = 150) -> list:
    """
    Split document content into semantically-merged chunks.
    Chunks are split when:
    - Token count is at least min_tokens (default 150) AND
      (Similarity falls below max_similarity (default 0.70) OR token count exceeds max_tokens (default 1000))
    Token count is estimated as word_count / 1.33
    
    For very large documents (>800 paragraphs), falls back to simple token-based chunking
    to avoid memory issues from thousands of Cohere embedding calls.
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    
    # Helper function to estimate token count
    def estimate_tokens(text: str) -> int:
        """Estimate tokens as word_count / 1.33"""
        word_count = len(text.split())
        return int(word_count / 1.33)
    
    # For very large documents, skip semantic chunking to avoid OOM from thousands of embeddings
    # Use simple token-based chunking instead
    if len(paragraphs) > 800:
        logger.warning(
            f"Document '{doc_metadata.get('name')}' has {len(paragraphs)} paragraphs. "
            f"Using simple token-based chunking to avoid memory issues."
        )
        chunks = []
        current_chunk = paragraphs[0]
        
        for para in paragraphs[1:]:
            potential_chunk = current_chunk + "\n\n" + para
            potential_tokens = estimate_tokens(potential_chunk)
            
            if potential_tokens <= max_tokens:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it meets minimum size
                if estimate_tokens(current_chunk) >= min_tokens:
                    chunks.extend(_make_chunks(doc_metadata, current_chunk))
                    current_chunk = para
                else:
                    # Chunk too small, force merge even if exceeds max
                    current_chunk = potential_chunk
        
        # Add last chunk
        if current_chunk:
            chunks.extend(_make_chunks(doc_metadata, current_chunk))
        
        return chunks
    
    # If only one paragraph, return it directly without embedding (will be embedded later in _embed_doc)
    if len(paragraphs) == 1:
        return _make_chunks(doc_metadata, paragraphs[0])
    
    chunks = []
    current_chunk = paragraphs[0]
    current_chunk_texts = [current_chunk]
    
    # Get embedding for the first paragraph
    current_embedding = np.array(
        co.embed(
            texts=[current_chunk],
            model=embedding_model_name,
            input_type="search_document",
            embedding_types=["float"],
        ).embeddings.float_[0]
    )
    
    for i in range(1, len(paragraphs)):
        para = paragraphs[i]
        para_embedding = np.array(
            co.embed(
                texts=[para],
                model=embedding_model_name,
                input_type="search_document",
                embedding_types=["float"],
            ).embeddings.float_[0]
        )
        
        sim = cosine_similarity(current_embedding, para_embedding)
        
        # Check if merging would exceed token limit
        potential_chunk = current_chunk + "\n\n" + para
        potential_tokens = estimate_tokens(potential_chunk)
        current_tokens = estimate_tokens(current_chunk)
        
        # Decision logic:
        # 1. If below min_tokens, always merge (unless exceeding max_tokens)
        # 2. If at or above min_tokens, split if similarity is low OR max_tokens would be exceeded
        should_merge = False
        if current_tokens < min_tokens:
            # Below minimum, keep merging unless we'd exceed max
            should_merge = potential_tokens <= max_tokens
        else:
            # At or above minimum, apply similarity and max token checks
            should_merge = sim >= max_similarity and potential_tokens <= max_tokens
        
        if should_merge:
            # Merge with current chunk
            current_chunk = potential_chunk
            current_chunk_texts.append(para)
            # Update current embedding as the mean of embeddings (calculated by multiplying the mean by the n of current para in the chunk)
            current_embedding = (current_embedding * len(current_chunk_texts) + para_embedding) / (len(current_chunk_texts) + 1)
        else:
            # Save current chunk (split due to low similarity or token limit exceeded)
            chunks.extend(_make_chunks(doc_metadata, current_chunk))
            # Start new chunk
            current_chunk = para
            current_chunk_texts = [para]
            current_embedding = para_embedding
        
        # Explicit cleanup to avoid accumulation
        del para_embedding
        del potential_chunk
    
    # Add last chunk
    chunks.extend(_make_chunks(doc_metadata, current_chunk))
    
    return chunks
