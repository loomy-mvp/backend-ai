from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Response, Depends
import pdfplumber
import io
import uuid
import json
import numpy as np
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

class EmbedRequest(BaseModel):
    bucket: str

class BucketRequest(BaseModel):
    bucket: str

class UpsertRequest(BaseModel):
    index_name: str
    vectors: list  # list of [id, embedding]

class DeleteFileRequest(BaseModel):
    bucket: str
    file_path: str

class RetrieveRequest(BaseModel):
    query: str
    index_name: str
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
                "text": current_chunk
            })
            # Start new chunk
            current_chunk = para
            current_chunk_texts = [para]
            current_embedding = para_embedding
    # Add last chunk
    chunks.append({
        "id": f"{doc_metadata['name']}-{str(uuid.uuid4())}",
        "page": doc_metadata["page"],
        "text": current_chunk
    })
    return chunks

@kb_router.post("/get-or-create-bucket")
def get_or_create_bucket(req: BucketRequest):
    bucket_name = "rag-suite-bucket-" + req.bucket
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception:
        bucket = storage_client.create_bucket(bucket_name)
    files = [blob.name for blob in bucket.list_blobs()]
    return {"bucket": req.bucket, "files": files}

@kb_router.post("/upload-file")
def upload_file(
    bucket: str = Form(...),
    overwrite: str = Form("false"),
    file: UploadFile = File(...)
):
    bucket_name = "rag-suite-bucket-" + bucket
    try:
        bucket_obj = storage_client.get_bucket(bucket_name)
    except Exception:
        bucket_obj = storage_client.create_bucket(bucket_name)
    blob = bucket_obj.blob(file.filename)
    if not blob.exists() or overwrite == "true":
        blob.upload_from_file(file.file, rewind=True)
        return {"status": "success", "filename": file.filename}
    else:
        return JSONResponse(status_code=409, content={"error": "File already exists. Use overwrite to replace it."})

@kb_router.post("/embed-docs")
def embed_docs(req: EmbedRequest):
    bucket_name = "rag-suite-bucket-" + req.bucket
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs())
    docs = []
    for blob in blobs:
        # This is valid only for pdfs
        content = blob.download_as_bytes()
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                docs.append({
                    "name": blob.name,
                    "page": i + 1,
                    "text": page_text
                })

    # 2. Chunk documents
    chunks = []
    for doc in docs:
        doc_metadata = {"name": doc["name"], "page": doc["page"]}
        chunks.extend(chunk_document(doc_metadata, doc["text"]))

    # 3. Embed with Cohere
    texts = [chunk["text"] for chunk in chunks]
    embeddings = co.embed(texts=texts, model="embed-v4.0").embeddings

    # Return chunk IDs and embeddings for upsert
    # vectors = [(chunk["id"], emb) for chunk, emb in zip(chunks, embeddings)]
    vectors = [Vector(id=chunk["id"], values=emb, metadata={"page": chunk["page"], "chunk_text": chunk["text"]}) for chunk, emb in zip(chunks, embeddings)]
    return {"status": "success", "chunks": len(chunks), "vectors": vectors}

@kb_router.post("/upsert-to-vector-store")
def upsert_to_vector_store(req: UpsertRequest):
    if not pc.has_index(req.index_name):
        pc.create_index(
            name=req.index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    index = pc.Index(name=req.index_name)
    index.upsert(req.vectors, namespace=None) # TODO: Deep down namespace
    return {"status": "success", "upserted": len(req.vectors)}

@kb_router.post("/delete-file")
def delete_file(req: DeleteFileRequest):
    bucket_name = "rag-suite-bucket-" + req.bucket
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(req.file_path)
        if blob.exists():
            blob.delete()
            return {"status": "success", "deleted": req.file_path}
        else:
            return JSONResponse(status_code=404, content={"error": "File not found."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@kb_router.post("/download-file")
def download_file(req: DeleteFileRequest):
    """Download a file from the user's GCS bucket and return its contents."""
    bucket_name = "rag-suite-bucket-" + req.bucket
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(req.file_path)
        if not blob.exists():
            return JSONResponse(status_code=404, content={"error": "File not found."})
        data = blob.download_as_bytes()
        content_type = blob.content_type or ("application/json" if req.file_path.lower().endswith(".json") else "application/octet-stream")
        return Response(content=data, media_type=content_type)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@kb_router.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """Retrieve similar documents from the Pinecone vector store."""
    try:
        print("[retrieve] Start retrieval process")
        # Check if index exists
        if not pc.has_index(req.index_name):
            print("[retrieve] Index name not found")
            return JSONResponse(status_code=404, content={"error": f"Index '{req.index_name}' not found"})
        
        # Get the index
        print(f"[retrieve] Getting index: {req.index_name}")
        index = pc.Index(name=req.index_name)
        
        # Embed the query
        print(f"[retrieve] Embedding query: {req.query}")
        query_embedding = co.embed(texts=[req.query], model=embedding_model_name).embeddings[0]

        # Prepare the Pinecone search query
        print(f"[retrieve] Querying Pinecone index: {req.index_name}")
        index = pc.Index(req.index_name)

        results = index.query(
            namespace="__default__", # Namespaces might be the right way to handle different users
            vector=query_embedding, 
            top_k=req.top_k,
            include_metadata=True,
            include_values=False
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
                "metadata": metadata # TODO: This should be removed and only keep the metadata needed not the entire object
            })

        print(f"[retrieve] Returning {len(retrieved_docs)} results")
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