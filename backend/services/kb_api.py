from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Response, Depends, BackgroundTasks
import re
import uuid
import json
import numpy as np
import httpx
import logging
from pydantic import BaseModel
from typing import Any
from google.cloud import storage
from google.oauth2 import service_account
import cohere
from pinecone import Pinecone, ServerlessSpec, Vector
from fastapi.responses import JSONResponse
import os
import mimetypes
from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from backend.utils.auth import verify_token
from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.config.chatbot_config import EMBEDDING_CONFIG
from backend.utils.ai_workflow_utils.document_processing import get_document_processor

kb_router = APIRouter(dependencies=[Depends(verify_token)])

# Google Cloud Storage - handle both explicit credentials and default (Cloud Run)
gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
if gcp_credentials_info:
    gcp_credentials_info = json.loads(gcp_credentials_info)
    gcp_service_account_credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)
    storage_client = storage.Client(credentials=gcp_service_account_credentials)
else:
    # Use default credentials (e.g., Cloud Run service account)
    storage_client = storage.Client()

# Cohere
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(cohere_api_key)
embedding_model_name = get_config_value(config_set=EMBEDDING_CONFIG, key="model")
# Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
document_webhook_url = os.getenv("DOCUMENT_WEBHOOK_URL")
PUBLIC_BUCKET_NAME = "loomy-public-documents"
PUBLIC_INDEX_NAME = "public"

class StorageRequest(BaseModel):
    user_id: str
    organization_id: str
    library: str
    document_id: str
    filename: str
    file: Any
    content_type: str
    overwrite: str = "false"

class EmbedRequest(BaseModel):
    library: str  # "organization", "private", or "public"
    organization_id: str
    user_id: str | None = None  # Only required if library is "private"
    storage_path: str | None = None
    content_type: str | None = None
    overwrite: bool = False # Whether to overwrite existing vectors

class UpsertRequest(BaseModel):
    index_name: str
    vectors: list  # list of [id, embedding]
    namespace: str | None = None  # Namespace for the vectors

class UploadRequest(BaseModel):
    user_id: str
    organization_id: str
    document_id: str
    filename: str
    file_content: Any
    library: str
    content_type: str
    overwrite: str | bool = "false"

class DeleteFileRequest(BaseModel):
    organization_id: str
    user_id: str | None = None  # Required when library is "private"
    library: str  # "organization", "private", or "public"
    storage_path: str | None = None
    filename: str | None = None


def delete_public_document(
    storage_path: str | None = None,
    *,
    source: str | None = None,
    bucket_name: str = "loomy-public-documents",
) -> None:
    """Delete public content from GCS (when storage_path provided) and matching vectors from Pinecone."""
    if not storage_path and not source:
        raise ValueError("Either storage_path or source is required")
    if storage_path and source:
        raise ValueError("Provide only storage_path or source, not both")

    normalized_path: str | None = None

    # GCS (only when deleting a specific file)
    if storage_path:
        if not storage_path.strip():
            raise ValueError("storage_path is required")

        normalized_path = storage_path.lstrip("/")
        if not normalized_path:
            raise ValueError("storage_path cannot be empty")

        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(normalized_path)
        if not blob.exists():
            raise FileNotFoundError(f"Document {normalized_path} not found in {bucket_name}")

        blob.delete()
        logger.info(f"[delete_public_document] Deleted file from GCS: {normalized_path}")

    normalized_source: str | None = None
    if source:
        normalized_source = source.strip()
        if not normalized_source:
            raise ValueError("source cannot be empty")

    # Pinecone
    index_name = "public"
    if not pc.has_index(index_name):
        logger.warning(
            "[delete_public_document] Pinecone index '%s' does not exist; skipping vector deletion",
            index_name,
        )
        return

    try:
        index = pc.Index(name=index_name)
        if normalized_path:
            filter_clause = {"storage_path": {"$eq": normalized_path}}
        else:
            filter_clause = {"source": {"$eq": normalized_source}}

        delete_response = index.delete(filter=filter_clause)
        pinecone_deleted = (
            delete_response.get("deleted_count", 0)
            if isinstance(delete_response, dict)
            else 0
        )
        logger.info(
            "[delete_public_document] Deleted %s vectors from Pinecone using %s=%s",
            pinecone_deleted,
            "storage_path" if normalized_path else "source",
            normalized_path or normalized_source,
        )
    except Exception as exc:
        logger.error("[delete_public_document] Error deleting from Pinecone: %s", exc, exc_info=True)
        raise

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def chunk_document(doc_metadata: str, content: str, max_similarity: float = 0.70, max_tokens: int = 1000, min_tokens: int = 150) -> list:
    """
    Split document content into semantically-merged chunks.
    Chunks are split when:
    - Token count is at least min_tokens (default 150) AND
      (Similarity falls below max_similarity (default 0.70) OR token count exceeds max_tokens (default 1000))
    Token count is estimated as word_count / 1.33
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    
    chunks = []
    current_chunk = paragraphs[0]
    current_chunk_texts = [current_chunk]
    
    # Helper function to estimate token count
    def estimate_tokens(text: str) -> int:
        """Estimate tokens as word_count / 1.33"""
        word_count = len(text.split())
        return int(word_count / 1.33)
    
    # Get embedding for the first paragraph
    current_embedding = co.embed(texts=[current_chunk],
                                 model=embedding_model_name,
                                 input_type="search_document",
                                 embedding_types=["float"]).embeddings.float_[0]
    
    for i in range(1, len(paragraphs)):
        para = paragraphs[i]
        para_embedding = co.embed(texts=[para],
                                  model=embedding_model_name,
                                  input_type="search_document",
                                  embedding_types=["float"]).embeddings.float_[0]
        
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
            chunks.append({
                "chunk_id": f"{doc_metadata['name']}-{str(uuid.uuid4())}",
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
        "chunk_id": f"{doc_metadata['name']}-{str(uuid.uuid4())}",
        "page": doc_metadata["page"],
        "text": current_chunk,
        "storage_path": doc_metadata["storage_path"]
    })
    
    return chunks

async def send_document_webhook(document_webhook_payload: dict):
    """Send document processing status to the configured webhook."""

    if not document_webhook_url:
        print("[webhook] DOCUMENT_WEBHOOK_URL not configured; skipping notification")
        return

    try:
        webhook_token = os.getenv("WEBHOOK_TOKEN")
        headers = {"Authorization": f"Bearer {webhook_token}"} if webhook_token else None

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                document_webhook_url,
                json=document_webhook_payload,
                headers=headers,
            )
            response.raise_for_status()
        print(f"[webhook] Notification sent: status={document_webhook_payload.get('status')} path={document_webhook_payload.get('storage_path')}")
    except Exception as exc:
        print(f"[webhook] Failed to send notification: {exc}")

def _store_file(storage_request: StorageRequest) -> dict:
    """Store a file in Google Cloud Storage following library conventions."""

    overwrite_flag = str(storage_request.overwrite).lower() == "true"

    if storage_request.library == "public":
        bucket_name = PUBLIC_BUCKET_NAME
    else:
        bucket_name = f"bucket-{storage_request.organization_id}"

    try:
        bucket_obj = storage_client.get_bucket(bucket_name)
    except Exception as exc:
        if storage_request.library == "public":
            logger.error("[_store_file] Public bucket %s unavailable: %s", bucket_name, exc)
            raise
        bucket_obj = storage_client.create_bucket(bucket_name)

    # Determine folder prefix based on library type
    if storage_request.library == "organization":
        folder_prefix = "organization/"
    elif storage_request.library == "private":
        folder_prefix = f"private/{storage_request.user_id}/"
    elif storage_request.library == "public":
        folder_prefix = "manual_upload/"
    else:
        folder_prefix = ""

    # storage_path = folder_prefix + storage_request.document_id + "-" + storage_request.filename
    storage_path = folder_prefix + storage_request.filename
    blob = bucket_obj.blob(storage_path)

    if blob.exists() and not overwrite_flag:
        return {
            "status": "error",
            "storage_path": storage_path,
            "reason": "file_exists",
        }

    # Read file content - handle both UploadFile and bytes
    if isinstance(storage_request.file, bytes):
        file_content = storage_request.file
    elif hasattr(storage_request.file, 'file'):
        file_content = storage_request.file.file.read()
    else:
        file_content = storage_request.file
    
    blob.upload_from_string(file_content, content_type=storage_request.content_type)
    return {"status": "uploaded", "storage_path": storage_path}

def _embed_doc(embed_request: EmbedRequest):
    """Embed a single stored document and prepare vectors for upsert."""

    storage_path = embed_request.storage_path

    if not storage_path:
        raise ValueError("storage_path is required to embed a document")

    # Validate library type
    if embed_request.library not in ["organization", "private", "public"]:
        raise ValueError("library must be 'organization', 'private', or 'public'")

    # Validate user_id for private library
    if embed_request.library == "private" and not embed_request.user_id:
        raise ValueError("user_id is required for private library")

    # Determine bucket, namespace, and index name
    if embed_request.library == "public":
        bucket_name = PUBLIC_BUCKET_NAME
        namespace = None
        index_name = PUBLIC_INDEX_NAME
    else:
        bucket_name = f"bucket-{embed_request.organization_id}"
        namespace = "organization" if embed_request.library == "organization" else embed_request.user_id
        index_name = embed_request.organization_id

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
    
    # If vectors for this document already exist, delete them to avoid duplicates
    if pc.has_index(index_name) and embed_request.overwrite:
        try:
            index = pc.Index(name=index_name)
            index.delete(namespace=namespace, filter={"storage_path": {"$eq": storage_path}})
        except Exception as e:
            print(f"[_embed_doc] Warning: unable to delete existing vectors for {storage_path}: {e}")
    else:
        index = None

    file_bytes = blob.download_as_bytes()
    content_type = (
        embed_request.content_type # ? Need to keep?
        or blob.content_type
        or mimetypes.guess_type(storage_path)[0]
    )

    if re.search(r"/.*-(.*)", storage_path): # .* can be changed following the document-id pattern
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

    try:
        chunks = processor.process(
            file_bytes,
            chunk_document=chunk_document,
            storage_path=storage_path,
            doc_name=doc_name,
        )
    except ValueError as e:
        # CID corruption detected - skip this document
        print(f"[_embed_doc] Skipping document {storage_path}: {e}")
        return {
            "status": "skipped",
            "message": f"Document skipped due to CID corruption: {str(e)}",
            "chunks": 0,
            "vectors": [],
            "index_name": index_name,
            "namespace": namespace,
        }

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
    embeddings = co.embed(texts=texts,
                          model=embedding_model_name,
                          input_type="search_document",
                          embedding_types=["float"]).embeddings.float_

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = {
            "page": chunk["page"],
            "chunk_text": chunk["text"],
            "storage_path": chunk["storage_path"],
            "doc_name": doc_name,
            "library": embed_request.library,
            "organization_id": embed_request.organization_id,
        }

        if embed_request.library == "private":
            metadata["user_id"] = embed_request.user_id

        vectors.append(
            Vector(
                id=chunk["chunk_id"],
                values=embedding,
                metadata=metadata,
            )
        )

    return {
        "status": "success",
        "chunks": len(chunks),
        "vectors": vectors,
        "index_name": index_name,
        "namespace": namespace,
        "doc_name": doc_name,
    }

def _upsert_to_vector_store(upsert_request: UpsertRequest):
    """Internal version of upsert_to_vector_store for use within upload-doc."""
    if not pc.has_index(upsert_request.index_name):
        pc.create_index(
            name=upsert_request.index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # TODO: To be changed to EU
            )
        )
    index = pc.Index(name=upsert_request.index_name)
    # Use the namespace from request, or default to None
    index.upsert(upsert_request.vectors, namespace=upsert_request.namespace)
    return {"status": "success", "upserted": len(upsert_request.vectors), "namespace": upsert_request.namespace}

async def process_doc_upload(upload_data: UploadRequest | dict):
    """Background task to process complete document upload, embedding, and storage with file content."""
    if isinstance(upload_data, UploadRequest):
        upload_data = upload_data.model_dump()
    try:
        logger.info(f"[process_doc_upload] Starting document processing for {upload_data['filename']}")
        
        # Step 1: Store the file
        logger.info("[process_doc_upload] Step 1: Storing file")

        file_bytes = upload_data.get("file_content", b"")
        file_size = len(file_bytes)

        storage_request = StorageRequest(
            user_id=upload_data["user_id"],
            organization_id=upload_data["organization_id"],
            library=upload_data.get("library"),
            document_id=upload_data.get("document_id"),
            filename=upload_data["filename"],
            file=upload_data["file_content"],
            content_type=upload_data.get("content_type"),
            overwrite=upload_data.get("overwrite", "false"),
        )

        storage_result = _store_file(storage_request)
        # Release large byte payload once persisted
        upload_data["file_content"] = b""
        logger.info(f"[process_doc_upload] Storage result: {storage_result}")

        storage_path = storage_result.get("storage_path")
        if storage_result.get("status") == "error": # TODO: implement SKIPPING, file existing
            logger.warning("[process_doc_upload] Storage failed; aborting embedding and upsert")
            await send_document_webhook({
                "storage_path": storage_path or upload_data.get("filename"),
                "size_bytes": file_size,
                "status": "error",
                "details": {
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "reason": storage_result.get("reason"),
                },
            })
            return

        if not storage_path:
            logger.error("[process_doc_upload] Missing stored filename; aborting")
            await send_document_webhook({
                "storage_path": upload_data.get("filename"),
                "size_bytes": file_size,
                "status": "error",
                "details": {
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "error": "missing_storage_path",
                },
            })
            return

        # Step 2: Embed the document
        logger.info("[process_doc_upload] Step 2: Embedding document")

        embed_request = EmbedRequest(
            library=upload_data['library'],
            organization_id=upload_data['organization_id'],
            user_id=upload_data['user_id'] if upload_data['library'] == "private" else None,
            storage_path=storage_path,
            content_type=upload_data.get("content_type"),
            overwrite=str(upload_data.get("overwrite", "false")).lower() == "true"
        )

        embed_result = _embed_doc(embed_request)
        logger.info(f"[process_doc_upload] Embed result: status={embed_result.get('status')} chunks={embed_result.get('chunks', 0)}")
        
        embed_status = embed_result.get("status", "unknown")
        if embed_status != "success":
            await send_document_webhook({
                "storage_path": storage_path,
                "size_bytes": file_size,
                "status": "processing", # Webhook status is different then the internal embed status
                # ? Why to use internal status?
                "details": {
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "chunks": embed_result.get("chunks", 0),
                    "message": "Document embedding finished",
                },
            })
            return

        # Step 3: Upsert to vector store (only if we have vectors)
        if embed_result.get("vectors") and len(embed_result["vectors"]) > 0:
            logger.info("[process_doc_upload] Step 3: Upserting to vector store")

            upsert_request = UpsertRequest(
                index_name=embed_result["index_name"],
                namespace=embed_result["namespace"],
                vectors=embed_result["vectors"]
            )

            upsert_result = _upsert_to_vector_store(upsert_request)

            logger.info(f"[process_doc_upload] Upsert result: upserted={upsert_result.get('upserted', 0)}")
            await send_document_webhook({
                "storage_path": storage_path,
                "size_bytes": file_size,
                "status": "ready",
                "details": {
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "chunks": embed_result.get("chunks", 0),
                    "vectors_upserted": upsert_result.get("upserted", 0),
                },
            })
        else:
            logger.warning("[process_doc_upload] No vectors to upsert")
            upsert_result = {"status": "error", "reason": "no_vectors"}
            await send_document_webhook({
                "storage_path": storage_path,
                "size_bytes": file_size,
                "status": "error",
                "details": {
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "chunks": embed_result.get("chunks", 0),
                    "message": embed_result.get("message"),
                },
            })
            return
        
        logger.info(f"[process_doc_upload] Document processing completed for {upload_data['filename']}")
        
    except Exception as e:
        logger.error(f"[process_doc_upload] Error processing document {upload_data['filename']}: {str(e)}", exc_info=True)
        storage_path = locals().get("storage_path") or upload_data.get("filename")
        details = {
            "document_id": upload_data.get("document_id"),
            "library": upload_data.get("library"),
            "error": str(e),
        }
        size_bytes = locals().get("file_size")
        if size_bytes is None:
            size_bytes = len(upload_data.get("file_content", b""))
        await send_document_webhook({
            "status": "error",
            "storage_path": storage_path,
            "size_bytes": size_bytes,
            "details": details,
        })

@kb_router.post("/upload-doc")
async def upload_doc(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    organization_id: str = Form(...),
    document_id: str = Form(...),
    filename: str = Form(...),
    library: str = Form(...),  # "organization" or "private"
    content_type: str = Form(...),
    overwrite: str = Form("false"),
    test: bool = Form(False)
):
    """
    Complete document upload endpoint that handles:
    1. File storage to GCS
    2. Document embedding
    3. Vector upsert to Pinecone
    
    This endpoint processes everything asynchronously and returns immediately.
    Use webhooks (when implemented) to get notification of completion.
    
    Args:
        user_id: User identifier
        organization_id: Organization identifier  
        document_id: Unique document identifier
        library: "organization" or "private" - determines storage location and access
        file: The PDF file to upload
        content_type: MIME type of the file
        overwrite: "true" to overwrite existing files, "false" to skip
    """
    # Temp: log the entire payload
    logger.info(f"Received upload request: user_id={user_id}, organization_id={organization_id}, document_id={document_id}, filename={filename}, library={library}, content_type={content_type}, overwrite={overwrite}, test={test}")
    if test:
        await send_document_webhook({
            "storage_path": f"test-{filename}",
            "size_bytes": 0,
            "status": "request_received",
            "details": {},
        })

        return {
            "user_id": user_id,
            "organization_id": organization_id,
            "document_id": document_id,
            "storage_path": f"test-{filename}",
            "library": library,
            "status": "test_mode",
            "message": "Test mode - no processing performed."
        }

    logger.info(f"[upload_doc] File {filename} received for upload by user {user_id} in organization {organization_id}")
    # Validate library type
    if library not in ["organization", "private", "public"]:
        return JSONResponse(status_code=400, content={"error": "library must be 'organization', 'private', or 'public'"})
    
    # Validate user_id for private library
    if library == "private" and not user_id:
        return JSONResponse(status_code=400, content={"error": "user_id is required for private library"})
    
    # Read file content before passing to background task
    file_content = await file.read()
    
    # Create upload request object for background processing
    upload_data = UploadRequest(
        user_id=user_id,
        organization_id=organization_id,
        document_id=document_id,
        filename=filename,
        file_content=file_content,
        library=library,
        content_type=content_type,
        overwrite=overwrite,
    )
    
    # Add complete document processing to background tasks
    background_tasks.add_task(process_doc_upload, upload_data)
    
    # Return immediately with receipt
    return {
        "user_id": user_id,
        "organization_id": organization_id,
        "document_id": document_id,
        "storage_path": filename,
        "library": library,
        "status": "document_received",
        "message": "Document upload and processing started. You will receive a webhook notification when complete."
    }

@kb_router.post("/delete-file")
def delete_file(delete_request: DeleteFileRequest):
    """Delete a file from GCS and remove all its chunks from Pinecone vector store."""
    try:
        if delete_request.library not in ["organization", "private", "public"]:
            return JSONResponse(status_code=400, content={"error": "library must be 'organization', 'private', or 'public'"})

        if delete_request.library == "private" and not delete_request.user_id:
            return JSONResponse(status_code=400, content={"error": "user_id is required for private library"})

        # Use the same bucket naming convention as upload
        bucket_name = (
            PUBLIC_BUCKET_NAME
            if delete_request.library == "public"
            else f"bucket-{delete_request.organization_id}"
        )
        
        # Determine folder prefix based on library type (same as upload)
        if delete_request.library == "organization":
            folder_prefix = "organization/"
        elif delete_request.library == "private":
            folder_prefix = f"private/{delete_request.user_id}/"
        elif delete_request.library == "public":
            folder_prefix = "manual_upload/"
        else:
            folder_prefix = ""

        storage_path = delete_request.storage_path or delete_request.filename
        if not storage_path:
            return JSONResponse(status_code=400, content={"error": "storage_path or filename is required"})

        # If storage_path doesn't include the folder prefix, add it
        normalized_path = storage_path.lstrip("/")
        if not normalized_path.startswith(folder_prefix):
            normalized_path = folder_prefix + normalized_path

        # Delete from GCS
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(normalized_path)
        if not blob.exists():
            return JSONResponse(status_code=404, content={"error": "File not found in storage."})
        
        blob.delete()
        logger.info(f"[delete_file] Deleted file from GCS: {normalized_path}")

        # Delete chunks from Pinecone
        if delete_request.library == "public":
            index_name = PUBLIC_INDEX_NAME
            namespace = None
        else:
            index_name = delete_request.organization_id
            namespace = "organization" if delete_request.library == "organization" else delete_request.user_id
        
        pinecone_deleted = 0
        if pc.has_index(index_name):
            try:
                index = pc.Index(name=index_name)
                # Delete all vectors with matching storage_path metadata
                delete_response = index.delete(
                    namespace=namespace,
                    filter={"storage_path": {"$eq": normalized_path}}
                )
                if isinstance(delete_response, dict) and not delete_response: # If delete_response is empty dict delete is successful
                    logger.info(f"[delete_file] Deleted {normalized_path} from Pinecone")
            except Exception as e:
                logger.error(f"[delete_file] Error deleting from Pinecone: {e}")
                raise

        return {
            "status": "success",
            "deleted_from_vectordb": pinecone_deleted > 0,
            "vectors_deleted": pinecone_deleted,
            "storage_path": normalized_path
        }
    except Exception as e:
        logger.error(f"[delete_file] Unexpected error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    


@kb_router.post("/download-file")
def download_file(download_request: DeleteFileRequest):
    """Download a file from the user's GCS bucket and return its contents."""
    try:
        if download_request.library not in ["organization", "private", "public"]:
            return JSONResponse(status_code=400, content={"error": "library must be 'organization', 'private', or 'public'"})

        if download_request.library == "private" and not download_request.user_id:
            return JSONResponse(status_code=400, content={"error": "user_id is required for private library"})

        if download_request.library == "public":
            bucket_name = PUBLIC_BUCKET_NAME
            folder_prefix = "manual_upload/"
        else:
            bucket_suffix = "org" if download_request.library == "organization" else download_request.user_id
            folder_prefix = "organization/" if download_request.library == "organization" else f"private/{download_request.user_id}/"
            bucket_name = f"bucket-{download_request.organization_id}-{bucket_suffix}"

        storage_path = download_request.storage_path or download_request.filename
        if not storage_path:
            return JSONResponse(status_code=400, content={"error": "storage_path or filename is required"})

        normalized_path = storage_path.lstrip("/")
        if not normalized_path.startswith(folder_prefix):
            normalized_path = folder_prefix + normalized_path

        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(normalized_path)
        if not blob.exists():
            return JSONResponse(status_code=404, content={"error": "File not found."})
        data = blob.download_as_bytes()
        content_type = blob.content_type or ("application/json" if normalized_path.lower().endswith(".json") else "application/octet-stream")
        return Response(content=data, media_type=content_type)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})