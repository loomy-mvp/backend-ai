"""
Minimal helper functions for batch embedding without FastAPI dependencies.
Provides streaming embed+upsert to avoid OOM on large documents.
"""

import gc
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from google.cloud import storage
from google.oauth2 import service_account
import cohere
from pinecone import Pinecone
from pydantic import BaseModel

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.config.chatbot_config import EMBEDDING_CONFIG
from backend.utils.ai_workflow_utils.document_processing import get_document_processor
from backend.utils.ai_workflow_utils.chunking import chunk_document

logger = logging.getLogger(__name__)

PINECONE_MAX_REQUEST_BYTES = 2 * 1024 * 1024  # Pinecone hard limit per request
PINECONE_SAFE_REQUEST_BYTES = int(PINECONE_MAX_REQUEST_BYTES * 0.9)  # Stay well under the cap

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


def _estimate_payload_size(payload: Dict[str, Any]) -> int:
    """Approximate payload size in bytes by serializing it to JSON."""
    return len(json.dumps(payload, ensure_ascii=False))


class EmbedRequest(BaseModel):
    library: str
    organization_id: str | None = None
    bucket_name: str
    user_id: str | None = None
    storage_path: str | None = None
    content_type: str | None = None
    overwrite: bool = True


def _ensure_index(index_name: str):
    """Create index if it doesn't exist and return an Index handle."""
    if not pc.has_index(index_name):
        logger.info(f"Creating index {index_name}")
        from pinecone import ServerlessSpec
        pc.create_index(
            name=index_name,
            dimension=1536,  # Cohere embed-v4.0 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(name=index_name)


def _upsert_vectors_batch(index, vectors: List[Dict[str, Any]], namespace: str | None) -> int:
    """Upsert a list of dict-vectors to Pinecone, respecting payload size limits.

    Returns the number of upserted vectors.
    """
    if not vectors:
        return 0

    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_batch_bytes = 0

    for payload in vectors:
        payload_size = _estimate_payload_size(payload)

        if payload_size > PINECONE_SAFE_REQUEST_BYTES:
            logger.warning(
                "Vector %s (~%d bytes) exceeds safe payload size; sending as single-vector batch",
                payload.get("id"),
                payload_size,
            )
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_bytes = 0
            batches.append([payload])
            continue

        if current_batch and current_batch_bytes + payload_size > PINECONE_SAFE_REQUEST_BYTES:
            batches.append(current_batch)
            current_batch = []
            current_batch_bytes = 0

        current_batch.append(payload)
        current_batch_bytes += payload_size

    if current_batch:
        batches.append(current_batch)

    total_upserted = 0
    for idx, batch in enumerate(batches, start=1):
        upsert_response = index.upsert(vectors=batch, namespace=namespace)
        if hasattr(upsert_response, "upserted_count"):
            total_upserted += upsert_response.upserted_count
        elif isinstance(upsert_response, dict) and "upserted_count" in upsert_response:
            total_upserted += upsert_response["upserted_count"]
        else:
            total_upserted += len(batch)

    return total_upserted


def _embed_and_upsert_doc(embed_request: EmbedRequest) -> Dict[str, Any]:
    """Embed a document and upsert vectors in streaming sub-batches.

    Unlike the two-step ``_embed_doc`` + ``_upsert_to_vector_store`` flow,
    this function embeds and upserts in batches of 96 chunks so that peak
    memory stays bounded.  Essential for large documents (700+ page PDFs)
    where holding all vectors at once can exceed the container memory limit.
    """

    storage_path = embed_request.storage_path

    if not storage_path:
        raise ValueError("storage_path is required to embed a document")

    if embed_request.library not in ["organization", "private", "public"]:
        raise ValueError("library must be 'organization', 'private', or 'public'")

    if embed_request.library == "private" and not embed_request.user_id:
        raise ValueError("user_id is required for private library")

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
            "upserted": 0,
        }

    # Delete existing vectors when overwriting
    if pc.has_index(index_name) and embed_request.overwrite:
        try:
            idx = pc.Index(name=index_name)
            idx.delete(filter={"storage_path": {"$eq": storage_path}})
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
        return {"status": "error", "message": str(exc), "chunks": 0, "upserted": 0}

    try:
        chunks = processor.process(
            file_bytes,
            chunk_document=chunk_document,
            storage_path=storage_path,
            doc_name=doc_name,
        )
    except ValueError as e:
        logger.warning(f"Skipping document {storage_path}: {e}")
        return {"status": "skipped", "message": str(e), "chunks": 0, "upserted": 0}

    # Free heavy objects immediately
    del file_bytes
    del processor
    gc.collect()

    if not chunks:
        return {"status": "error", "message": "No chunks generated", "chunks": 0, "upserted": 0}

    # ------- streaming embed + upsert in sub-batches -------
    index = _ensure_index(index_name)
    batch_size = 96
    total_chunks = len(chunks)
    total_upserted = 0

    for batch_start in range(0, total_chunks, batch_size):
        batch_chunks = chunks[batch_start : batch_start + batch_size]
        texts = [c["text"] for c in batch_chunks]

        # Embed this sub-batch
        batch_embeddings = co.embed(
            texts=texts,
            model=embedding_model_name,
            input_type="search_document",
            embedding_types=["float"],
        ).embeddings.float_

        del texts

        # Build lightweight dict vectors (skip Pinecone Vector objects)
        dict_vectors = [
            {
                "id": chunk["chunk_id"],
                "values": list(embedding),
                "metadata": {
                    "page": chunk["page"],
                    "chunk_text": chunk["text"],
                    "storage_path": storage_path,
                    "doc_name": doc_name,
                    "library": embed_request.library,
                    "source": storage_path.split("/")[0],
                },
            }
            for chunk, embedding in zip(batch_chunks, batch_embeddings)
        ]

        del batch_embeddings

        # Upsert immediately and free
        total_upserted += _upsert_vectors_batch(index, dict_vectors, namespace)

        del dict_vectors

        logger.debug(
            "  Sub-batch %d–%d / %d embedded & upserted",
            batch_start + 1,
            min(batch_start + batch_size, total_chunks),
            total_chunks,
        )

    # Free the full chunks list and force GC
    del chunks
    gc.collect()

    return {
        "status": "success",
        "chunks": total_chunks,
        "upserted": total_upserted,
    }
