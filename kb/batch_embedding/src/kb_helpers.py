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

from backend.utils.get_config_value import get_config_value
from backend.config.chatbot_config import EMBEDDING_CONFIG
from backend.utils.document_processing import get_document_processor

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


def chunk_document(doc_metadata: dict, content: str) -> list:
    """Split document content into semantically-merged chunks using a similarity threshold of 0.95."""
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
        
        if sim >= 0.95:
            # Merge with current chunk
            current_chunk += "\n\n" + para
            current_chunk_texts.append(para)
            # Update current embedding as the mean of embeddings
            current_embedding = (current_embedding * len(current_chunk_texts) + para_embedding) / (len(current_chunk_texts) + 1)
        else:
            # Save current chunk
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
    namespace: str = None


def _embed_doc(embed_request: EmbedRequest) -> Dict[str, Any]:
    """Embed a single stored document and prepare vectors for upsert."""
    
    storage_path = embed_request.storage_path
    
    if not storage_path:
        raise ValueError("storage_path is required to embed a document")
    
    if embed_request.library not in ["organization", "private"]:
        raise ValueError("library must be 'organization' or 'private'")
    
    if embed_request.library == "private" and not embed_request.user_id:
        raise ValueError("user_id is required for private library")
    
    # For batch embedding of public documents:
    # - Bucket name comes from the request (e.g., "loomy-public-documents")
    # - Index is always "public"
    # - Namespace is the first folder in the path (e.g., "circolari" from "circolari/2024/doc.pdf")
    bucket_name = embed_request.bucket_name
    index_name = "public"
    
    # Extract namespace from storage path (first folder)
    path_parts = storage_path.split('/')
    namespace = path_parts[0] if len(path_parts) > 0 else "default"
    
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
    
    if pc.has_index(index_name) and embed_request.overwrite:
        try:
            index = pc.Index(name=index_name)
            index.delete(namespace=namespace, filter={"storage_path": {"$eq": storage_path}})
        except Exception as e:
            logger.warning(f"Unable to delete existing vectors for {storage_path}: {e}")
    
    file_bytes = blob.download_as_bytes()
    content_type = (
        embed_request.content_type
        or blob.content_type
        or mimetypes.guess_type(storage_path)[0]
    )
    
    if re.search(r"/.*-(.*)", storage_path):
        doc_name = re.search(r"/.*-(.*)", storage_path).group(1)
    else:
        doc_name = storage_path.split("/")[-1].rsplit(".", 1)[0]
    
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
    
    chunks = processor.process(
        file_bytes,
        chunk_document=chunk_document,
        storage_path=storage_path,
        doc_name=doc_name,
    )
    
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
    embeddings = co.embed(
        texts=texts,
        model=embedding_model_name,
        input_type="search_document",
        embedding_types=["float"]
    ).embeddings.float_
    
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
            }
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    
    return {
        "status": "success",
        "chunks": len(chunks),
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
            vectors=upsert_request.vectors,
            namespace=upsert_request.namespace
        )
        
        return {
            "status": "success",
            "upserted": upsert_response.upserted_count if hasattr(upsert_response, 'upserted_count') else len(upsert_request.vectors)
        }
        
    except Exception as e:
        logger.error(f"Error upserting to vector store: {e}")
        return {"status": "error", "message": str(e)}
