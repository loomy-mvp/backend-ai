from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Response, Depends, BackgroundTasks
import pdfplumber
import io
import re
import uuid
import json
import numpy as np
import httpx
from pydantic import BaseModel
from google.cloud import storage
from google.oauth2 import service_account
import cohere
from pinecone import Pinecone, ServerlessSpec, Vector
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from backend.utils.auth import verify_token

kb_router = APIRouter(dependencies=[Depends(verify_token)])

# Google Cloud Storage
gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
gcp_credentials_info = json.loads(gcp_credentials_info)
gcp_service_account_credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)
storage_client = storage.Client(credentials=gcp_service_account_credentials)
# Cohere
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)
embedding_model_name = "embed-v4.0"
# Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
document_webhook_url = os.getenv("DOCUMENT_WEBHOOK_URL")

class StorageRequest(BaseModel):
    user_id: str = Form(...),
    organization_id: str = Form(...),
    library: str = Form(...),
    document_id: str = Form(...),
    filename: str = Form(...),
    file: UploadFile = File(...),
    content_type: str = Form(...),
    overwrite: str = Form("false")

class EmbedRequest(BaseModel):
    library: str  # "organization" or "private"
    organization_id: str
    user_id: str = None  # Only required if library is "private"
    storage_path: str | None = None
    overwrite: bool = False # Whether to overwrite existing vectors

class UpsertRequest(BaseModel):
    index_name: str
    vectors: list  # list of [id, embedding]
    namespace: str = None  # Namespace for the vectors

class UploadRequest(BaseModel):
    user_id: str
    organization_id: str
    document_id: str
    filename: str
    file_content: bytes
    library: str
    content_type: str
    overwrite: str | bool = "false"

class DeleteFileRequest(BaseModel):
    organization_id: str
    user_id: str | None = None  # Required when library is "private"
    library: str  # "organization" or "private"
    storage_path: str | None = None
    filename: str | None = None

class RetrieveRequest(BaseModel):
    query: str
    organization_id: str
    library: str  # "organization" or "private"
    user_id: str | None = None  # Required when library is "private"
    top_k: int = 5

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def chunk_document(doc_metadata: str, content: str, ) -> list:
    """Split document content into semantically-merged chunks using a similarity threshold of 0.95."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks = []
    current_chunk = paragraphs[0]
    current_chunk_texts = [current_chunk]
    # Get embedding for the first paragraph
    current_embedding = co.embed(texts=[current_chunk], model=embedding_model_name).embeddings[0]
    for i in range(1, len(paragraphs)):
        para = paragraphs[i]
        para_embedding = co.embed(texts=[para], model=embedding_model_name).embeddings[0]
        sim = cosine_similarity(current_embedding, para_embedding)
        if sim >= 0.95:
            # Merge with current chunk
            current_chunk += "\n\n" + para
            current_chunk_texts.append(para)
            # Update current embedding as the mean of embeddings (calculated by multiplying the mean by the n of current para in the chunk)
            current_embedding = (current_embedding * len(current_chunk_texts) + para_embedding) / (len(current_chunk_texts) + 1)
        else:
            # Save current chunk
            chunks.append({
                "id": f"{doc_metadata['name']}-{str(uuid.uuid4())}",
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
        "id": f"{doc_metadata['name']}-{str(uuid.uuid4())}",
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
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(document_webhook_url, json=document_webhook_payload)
            response.raise_for_status()
        print(f"[webhook] Notification sent: status={document_webhook_payload.get('status')} path={document_webhook_payload.get('storage_path')}")
    except Exception as exc:
        print(f"[webhook] Failed to send notification: {exc}")

def _store_file(storage_request: StorageRequest) -> dict:
    """Store a file in Google Cloud Storage following library conventions."""

    overwrite_flag = str(storage_request.overwrite).lower() == "true"

    bucket_name = f"bucket-{storage_request.organization_id}"

    try:
        bucket_obj = storage_client.get_bucket(bucket_name)
    except Exception:
        bucket_obj = storage_client.create_bucket(bucket_name)

    # Determine folder prefix based on library type
    if storage_request.library == "organization":
        folder_prefix = "organization/"
    elif storage_request.library == "private":
        folder_prefix = f"private/{storage_request.user_id}/"
    else:
        folder_prefix = ""

    storage_path = folder_prefix + storage_request.document_id + "-" + storage_request.filename
    blob = bucket_obj.blob(storage_path)

    if blob.exists() and not overwrite_flag:
        return {
            "status": "skipped",
            "storage_path": storage_path,
            "reason": "file_exists",
        }

    blob.upload_from_string(storage_request.file, content_type=storage_request.content_type) # ! This is GCP Storage content_type, we shall use the same
    return {"status": "uploaded", "storage_path": storage_path}

def _embed_doc(embed_request: EmbedRequest):
    """Embed a single stored document and prepare vectors for upsert."""

    storage_path = embed_request.storage_path

    if not storage_path:
        raise ValueError("storage_path is required to embed a document")

    # Validate library type
    if embed_request.library not in ["organization", "private"]:
        raise ValueError("library must be 'organization' or 'private'")

    # Validate user_id for private library
    if embed_request.library == "private" and not embed_request.user_id:
        raise ValueError("user_id is required for private library")

    # Determine bucket, namespace, and index name
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
    
    # Temporarily only support PDF documents
    if not storage_path.lower().endswith(".pdf"):
        return {
            "status": "skipped",
            "message": "Only PDF documents are currently supported",
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
    if re.search(r"/.*-(.*)", storage_path): # .* can be changed following the document-id pattern
        doc_name = re.search(r"/.*-(.*)", storage_path).group(1)
    else:
        doc_name = storage_path.split("/")[-1].rsplit(".", 1)[0]

    chunks = []
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
    embeddings = co.embed(texts=texts, model=embedding_model_name).embeddings

    vectors = [
        Vector(
            id=chunk["id"],
            values=embedding,
            metadata={
                "page": chunk["page"],
                "chunk_text": chunk["text"],
                "storage_path": chunk["storage_path"],
                "library": embed_request.library,
                "organization_id": embed_request.organization_id,
                "user_id": embed_request.user_id if embed_request.library == "private" else None,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

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
        print(f"[process_doc_upload] Starting document processing for {upload_data['filename']}")
        
        # Step 1: Store the file
        print("[process_doc_upload] Step 1: Storing file")

        file_bytes = upload_data.get("file_content", b"")
        file_size = len(file_bytes)

        async def notify(status: str, *, storage_path: str | None, details: dict | None = None):
            await send_document_webhook({
                "storage_path": storage_path,
                "size_bytes": file_size,
                "status": status,
                "details": details or {}
            })

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
        print(f"[process_doc_upload] Storage result: {storage_result}")

        storage_path = storage_result.get("storage_path")
        if storage_result.get("status") == "skipped":
            print("[process_doc_upload] Storage skipped; aborting embedding and upsert")
            await notify(
                status="skipped",
                storage_path=storage_path or upload_data.get("filename"),
                details={
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "reason": storage_result.get("reason"),
                },
            )
            return

        if not storage_path:
            print("[process_doc_upload] Missing stored filename; aborting")
            await notify(
                "error",
                storage_path=upload_data.get("filename"),
                details={
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "error": "missing_storage_path",
                },
            )
            return

        # Step 2: Embed the document
        print("[process_doc_upload] Step 2: Embedding document")
        embed_request = EmbedRequest(
            library=upload_data['library'],
            organization_id=upload_data['organization_id'],
            user_id=upload_data['user_id'] if upload_data['library'] == "private" else None,
            storage_path=storage_path,
            overwrite=str(upload_data.get("overwrite", "false")).lower() == "true"
        )

        embed_result = _embed_doc(embed_request)
        print(f"[process_doc_upload] Embed result: status={embed_result.get('status')} chunks={embed_result.get('chunks', 0)}")
        
        embed_status = embed_result.get("status", "unknown")
        if embed_status != "success":
            await notify(
                embed_status,
                storage_path=storage_path,
                # ? Do i really need details for upload?
                details={
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "chunks": embed_result.get("chunks", 0),
                    "message": embed_result.get("message"),
                },
            )
            return

        # Step 3: Upsert to vector store (only if we have vectors)
        if embed_result.get("vectors") and len(embed_result["vectors"]) > 0:
            print("[process_doc_upload] Step 3: Upserting to vector store")

            upsert_request = UpsertRequest(
                index_name=embed_result["index_name"],
                namespace=embed_result["namespace"],
                vectors=embed_result["vectors"]
            )

            upsert_result = _upsert_to_vector_store(upsert_request)

            print(f"[process_doc_upload] Upsert result: upserted={upsert_result.get('upserted', 0)}")
            await notify(
                "document_ready",
                storage_path=storage_path,
                details={
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "chunks": embed_result.get("chunks", 0),
                    "vectors_upserted": upsert_result.get("upserted", 0),
                },
            )
        else:
            print("[process_doc_upload] No vectors to upsert")
            upsert_result = {"status": "skipped", "reason": "no_vectors"}
            await notify(
                "no_vectors",
                storage_path=storage_path,
                details={
                    "document_id": upload_data.get("document_id"),
                    "library": upload_data.get("library"),
                    "chunks": embed_result.get("chunks", 0),
                    "message": embed_result.get("message"),
                },
            )
            return
        
        print(f"[process_doc_upload] Document processing completed for {upload_data['filename']}")
        
    except Exception as e:
        print(f"[process_doc_upload] Error processing document {upload_data['filename']}: {str(e)}")
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
            "storage_path": storage_path,
            "size_bytes": size_bytes,
            "status": "error",
            "details": details,
        })

@kb_router.post("/upload-doc")
async def upload_doc(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    organization_id: str = Form(...),
    document_id: str = Form(...),
    library: str = Form(...),  # "organization" or "private"
    file: UploadFile = File(...),
    content_type: str = Form(...),
    overwrite: str = Form("false")
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
    # Validate library type
    if library not in ["organization", "private"]:
        return JSONResponse(status_code=400, content={"error": "library must be 'organization' or 'private'"})
    
    # Validate user_id for private library
    if library == "private" and not user_id:
        return JSONResponse(status_code=400, content={"error": "user_id is required for private library"})
    
    # Read file content before passing to background task
    file_content = await file.read()
    filename = file.filename # ? Check that this is the entire path or just the name
    
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
    try:
        if delete_request.library not in ["organization", "private"]:
            return JSONResponse(status_code=400, content={"error": "library must be 'organization' or 'private'"})

        if delete_request.library == "private" and not delete_request.user_id:
            return JSONResponse(status_code=400, content={"error": "user_id is delete_requestuired for private library"})

        bucket_suffix = "org" if delete_request.library == "organization" else delete_request.user_id
        bucket_name = f"bucket-{delete_request.organization_id}-{bucket_suffix}"
        folder_prefix = "organization/" if delete_request.library == "organization" else f"private/{delete_request.user_id}/"

        storage_path = delete_request.storage_path or delete_request.filename
        if not storage_path:
            return JSONResponse(status_code=400, content={"error": "storage_path or filename is delete_requestuired"})

        normalized_path = storage_path.lstrip("/")
        if not normalized_path.startswith(folder_prefix):
            normalized_path = folder_prefix + normalized_path

        bucket_name = f"bucket-{delete_request.organization_id}-{bucket_suffix}"
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(normalized_path)
        if blob.exists():
            blob.delete()
            return {"status": "success", "deleted": normalized_path}
        return JSONResponse(status_code=404, content={"error": "File not found."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@kb_router.post("/download-file")
def download_file(req: DeleteFileRequest):
    """Download a file from the user's GCS bucket and return its contents."""
    try:
        if req.library not in ["organization", "private"]:
            return JSONResponse(status_code=400, content={"error": "library must be 'organization' or 'private'"})

        if req.library == "private" and not req.user_id:
            return JSONResponse(status_code=400, content={"error": "user_id is required for private library"})

        bucket_suffix = "org" if req.library == "organization" else req.user_id
        folder_prefix = "organization/" if req.library == "organization" else f"private/{req.user_id}/"

        storage_path = req.storage_path or req.filename
        if not storage_path:
            return JSONResponse(status_code=400, content={"error": "storage_path or filename is required"})

        normalized_path = storage_path.lstrip("/")
        if not normalized_path.startswith(folder_prefix):
            normalized_path = folder_prefix + normalized_path

        bucket_name = f"bucket-{req.organization_id}-{bucket_suffix}"
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(normalized_path)
        if not blob.exists():
            return JSONResponse(status_code=404, content={"error": "File not found."})
        data = blob.download_as_bytes()
        content_type = blob.content_type or ("application/json" if normalized_path.lower().endswith(".json") else "application/octet-stream")
        return Response(content=data, media_type=content_type)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@kb_router.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """Retrieve similar documents from the Pinecone vector store."""
    try:
        print("[retrieve] Start retrieval process")
        if req.library not in ["organization", "private"]:
            return JSONResponse(status_code=400, content={"error": "library must be 'organization' or 'private'"})

        if req.library == "private" and not req.user_id:
            return JSONResponse(status_code=400, content={"error": "user_id is required for private library"})

        index_name = req.organization_id
        namespace = "organization" if req.library == "organization" else req.user_id

        # Check if index exists
        if not pc.has_index(index_name):
            print("[retrieve] Index not found", index_name)
            return JSONResponse(status_code=404, content={"error": f"Index '{index_name}' not found"})

        # Get the index
        print(f"[retrieve] Getting index: {index_name}")
        index = pc.Index(name=index_name)

        # Build metadata filter to stay within the proper library scope
        metadata_filter = {"library": {"$eq": req.library}}
        if req.library == "private":
            metadata_filter["user_id"] = {"$eq": req.user_id}

        # Embed the query
        print(f"[retrieve] Embedding query: {req.query}")
        query_embedding = co.embed(texts=[req.query], model=embedding_model_name).embeddings[0]

        # Prepare the Pinecone search query
        print(f"[retrieve] Querying Pinecone index: {index_name} namespace: {namespace}")
        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=req.top_k,
            include_metadata=True,
            include_values=False,
            filter=metadata_filter
        )

        print("[retrieve] Query completed, formatting results")
        # Format results
        matches = results.get("matches", [])
        retrieved_docs = []
        for match in matches:
            metadata = match.get("metadata", {})
            print(f"[retrieve] Match metadata: {metadata}")
            retrieved_docs.append({
                "id": match.get("id"),
                "score": match.get("score"),
                "chunk_text": metadata.get("chunk_text", ""),
                "category": metadata.get("category", ""),
                "metadata": metadata # TODO: return the docs info
            })

        print(f"[retrieve] Returning {len(retrieved_docs)} results")
        # TODO: Update to return source documents when ready
        return {
            "status": "success",
            "query": req.query,
            "results": retrieved_docs,
            "total_results": len(retrieved_docs)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Standalone FastAPI app for running this module directly
kb_api = FastAPI(title="KB API", description="Knowledge Base API", version="1.0.0")
kb_api.include_router(kb_router)