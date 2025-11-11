"""
Minimal helper functions for batch embedding without FastAPI dependencies.
Replicates only the necessary functions from backend.services.kb_api
"""

import json
import logging
import os
import re
import mimetypes
import uuid
import unicodedata
import gc
from typing import Any, Dict
from pathlib import Path
import sys

from google.cloud import storage
from google.oauth2 import service_account
import cohere
from pinecone import Pinecone, Vector
from pydantic import BaseModel
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.config.chatbot_config import EMBEDDING_CONFIG
from backend.utils.ai_workflow_utils.document_processing import get_document_processor

logger = logging.getLogger(__name__)

# Initialize clients
gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
if gcp_credentials_info:
    gcp_credentials_info = json.loads(gcp_credentials_info)
    gcp_service_account_credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)
    storage_client = storage.Client(credentials=gcp_service_account_credentials)
else:
    storage_client = storage.Client()

cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(cohere_api_key)
embedding_model_name = get_config_value(config_set=EMBEDDING_CONFIG, key="model")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def sanitize_to_ascii(text: str) -> str:
    """Convert text to ASCII-safe string by removing or replacing non-ASCII characters."""
    # Normalize unicode characters to their closest ASCII equivalents
    text = unicodedata.normalize('NFKD', text)
    # Encode to ASCII, ignoring characters that can't be converted
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Replace any remaining problematic characters with underscores
    text = re.sub(r'[^a-zA-Z0-9\-_.]', '_', text)
    return text

# Helper function to estimate token count
def estimate_tokens(text: str) -> int:
    """Estimate tokens as word_count / 1.33"""
    word_count = len(text.split()) + 1
    return int(word_count / 1.33)

def chunk_document(doc_metadata: dict, content: str, max_similarity: float = 0.65, max_tokens: int = 1000, min_tokens: int = 150) -> list:
    """
    Split document content into semantically-merged chunks.
    Chunks are split when:
    - Token count is at least min_tokens (default 150) AND
      (Similarity falls below max_similarity (default 0.65) OR token count exceeds max_tokens (default 1000))
    Token count is estimated as word_count / 1.33
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    
    chunks = []
    current_chunk = paragraphs[0]
    current_chunk_texts = [current_chunk]
    
    # Sanitize doc name for ASCII-only IDs
    safe_doc_name = sanitize_to_ascii(doc_metadata['name'])
    
    # Get embedding for the first paragraph
    current_embedding = co.embed(
        texts=[current_chunk],
        model=embedding_model_name,
        input_type="search_document",
        embedding_types=["float"]
    ).embeddings.float_[0]
    
    for i in range(1, len(paragraphs)):
        para = paragraphs[i]
        para_embedding = co.embed(
            texts=[para],
            model=embedding_model_name,
            input_type="search_document",
            embedding_types=["float"]
        ).embeddings.float_[0]
        
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
            # Update current embedding as the mean of embeddings
            current_embedding = (current_embedding * len(current_chunk_texts) + para_embedding) / (len(current_chunk_texts) + 1)
        else:
            # Save current chunk (split due to low similarity or token limit exceeded)
            chunks.append({
                "chunk_id": f"{safe_doc_name}-{str(uuid.uuid4())}",
                "page": doc_metadata["page"],
                "text": current_chunk,
                "storage_path": doc_metadata["storage_path"]
            })
            # Start new chunk
            current_chunk = para
            current_chunk_texts = [para]
            current_embedding = para_embedding
    
    # Add last chunk
    chunks.append({
        "chunk_id": f"{safe_doc_name}-{str(uuid.uuid4())}",
        "page": doc_metadata["page"],
        "text": current_chunk,
        "storage_path": doc_metadata["storage_path"]
    })
    
    return chunks


class EmbedRequest(BaseModel):
    library: str
    organization_id: str
    bucket_name: str
    user_id: str = None
    storage_path: str | None = None
    content_type: str | None = None
    overwrite: bool = False


class UpsertRequest(BaseModel):
    index_name: str
    vectors: list


def _embed_doc(embed_request: EmbedRequest) -> Dict[str, Any]:
    """Embed a single stored document and prepare vectors for upsert."""
    
    storage_path = embed_request.storage_path
    
    if not storage_path:
        raise ValueError("storage_path is required to embed a document")
    
    if embed_request.library not in ["organization", "private", "public"]:
        raise ValueError("library must be 'organization', 'private', or 'public'")

    if embed_request.library == "private" and not embed_request.user_id:
        raise ValueError("user_id is required for private library")
    
    # For batch embedding of public documents:
    # - Bucket name comes from the request (e.g., "loomy-public-documents")
    # - Index is always "public"
    # - No namespace used
    bucket_name = embed_request.bucket_name
    index_name = "public"
    namespace = None
    
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(storage_path)
    
    if not blob.exists():
        return {
            "status": "error",
            "message": f"Document {storage_path} not found in bucket {bucket_name}",
            "chunks": 0,
            "vectors": [],
            "index_name": index_name,
            "namespace": namespace,
        }
    
    # Check if document is already embedded when overwrite=False
    if not embed_request.overwrite and pc.has_index(index_name):
        try:
            index = pc.Index(name=index_name)
            # Query for any vector with this storage_path
            query_result = index.query(
                vector=[0] * 1536,  # Dummy vector just to check metadata
                filter={"storage_path": {"$eq": storage_path}},
                top_k=1,
                include_metadata=True
            )
            
            if query_result.matches and len(query_result.matches) > 0:
                logger.info(f"Document {storage_path} already embedded in index, skipping")
                return {
                    "status": "skipped",
                    "message": f"Document already embedded (overwrite=False)",
                    "chunks": 0,
                    "vectors": [],
                    "index_name": index_name,
                    "namespace": namespace,
                }
        except Exception as e:
            logger.warning(f"Unable to check if document exists in index: {e}")
            # Continue with embedding if check fails
    
    if pc.has_index(index_name) and embed_request.overwrite:
        try:
            index = pc.Index(name=index_name)
            index.delete(filter={"storage_path": {"$eq": storage_path}})
        except Exception as e:
            logger.warning(f"Unable to delete existing vectors for {storage_path}: {e}")
    
    file_bytes = blob.download_as_bytes()
    content_type = (
        embed_request.content_type
        or blob.content_type
        or mimetypes.guess_type(storage_path)[0]
    )
    
    doc_name = storage_path.split("/")[-1]
    
    try:
        processor = get_document_processor(content_type, storage_path)
    except ValueError as exc:
        return {
            "status": "error",
            "message": str(exc),
            "chunks": 0,
            "vectors": [],
            "index_name": index_name,
            "namespace": namespace,
        }
    
    try:
        chunks = processor.process(
            file_bytes,
            chunk_document=chunk_document,
            storage_path=storage_path,
            doc_name=doc_name,
        )
    except ValueError as e:
        # CID corruption detected - skip this document
        logger.warning(f"Skipping document {storage_path}: {e}")
        return {
            "status": "skipped",
            "message": f"Document skipped: {str(e)}",
            "chunks": 0,
            "vectors": [],
            "index_name": index_name,
            "namespace": namespace,
        }
    
    # Free up memory immediately after processing
    del file_bytes
    del processor
    
    if not chunks:
        return {
            "status": "error",
            "message": "No chunks generated",
            "chunks": 0,
            "vectors": [],
            "index_name": index_name,
            "namespace": namespace,
        }
    
    texts = [chunk["text"] for chunk in chunks]
    
    # Cohere embed API has a limit of 96 texts per call split into batches if needed
    batch_size = 96
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = co.embed(
            texts=batch,
            model=embedding_model_name,
            input_type="search_document",
            embedding_types=["float"]
        ).embeddings.float_
        embeddings.extend(batch_embeddings)
    
    # Free up memory after embedding
    del texts
    
    vectors = [
        Vector(
            id=chunk["chunk_id"],
            values=embedding,
            metadata={
                "page": chunk["page"],
                "chunk_text": chunk["text"],
                "storage_path": storage_path,
                "doc_name": doc_name,
                "library": embed_request.library,
                "source": storage_path.split("/")[0],
            }
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    
    # Store counts before deleting
    num_chunks = len(chunks)
    
    # Free up memory after creating vectors
    del chunks
    del embeddings
    gc.collect()  # Force garbage collection
    
    return {
        "status": "success",
        "chunks": num_chunks,
        "vectors": vectors,
        "index_name": index_name,
        "namespace": namespace,
    }


def _upsert_to_vector_store(upsert_request: UpsertRequest) -> Dict[str, Any]:
    """Upsert vectors to Pinecone."""
    
    try:
        # Create index if it doesn't exist
        if not pc.has_index(upsert_request.index_name):
            logger.info(f"Creating index {upsert_request.index_name}")
            from pinecone import ServerlessSpec
            pc.create_index(
                name=upsert_request.index_name,
                dimension=1536,  # Cohere embed-v4.0 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        index = pc.Index(name=upsert_request.index_name)
        
        upsert_response = index.upsert(
            vectors=upsert_request.vectors
        )
        
        return {
            "status": "success",
            "upserted": upsert_response.upserted_count if hasattr(upsert_response, 'upserted_count') else len(upsert_request.vectors)
        }
        
    except Exception as e:
        logger.error(f"Error upserting to vector store: {e}")
        return {"status": "error", "message": str(e)}
