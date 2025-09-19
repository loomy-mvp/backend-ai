# RAG Suite: PDF Knowledge Base Chatbot

Build a Retrieval-Augmented Generation (RAG) chatbot over your PDFs. The backend is powered by FastAPI, Cohere embeddings, and Pinecone vector search. Documents can live locally under `kb/` and/or in Google Cloud Storage (GCS).

## Architecture Overview

- Backend: FastAPI services in `backend/services/`
  - `kb_api.py` (port 8000): Knowledge base — upload, parse, chunk, embed, vector upsert, retrieve
  - `chatbot_api.py` (port 8001): RAG chatbot — chat, sessions, providers
  - `extract_api.py` (port 8002): Stateless parsing/extraction — parse files and extract structured data
  - Launcher: `backend/start_apis.py` to start individual services
- Knowledge Base: PDFs in `kb/` and/or a GCS bucket; embeddings via Cohere; vectors in Pinecone
 

## Quickstart (Windows PowerShell)

### 1) Backend setup
```powershell
cd c:\Users\leoac\Work\Companies\Loomy\backend
python -m venv .venvinspiron14
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` in the repo root or `backend/` with the relevant keys (values are examples/placeholders):
```env
# GCS service account JSON as a single line (or use GOOGLE_APPLICATION_CREDENTIALS below)
GCP_SERVICE_ACCOUNT_CREDENTIALS={"type":"service_account", ...}

# Embeddings and vector DB
COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional LLM providers used by chatbot/extract services
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...

# Optional overrides
KB_API_BASE_URL=http://localhost:8000
```

Then start the APIs (choose one of the options):

Option A — launcher:
```powershell
python start_apis.py kb       # :8000
python start_apis.py chatbot  # :8001
python start_apis.py extract  # :8002
```

Option B — uvicorn directly:
```powershell
uvicorn services.kb_api:kb_api --host 0.0.0.0 --port 8000 --reload
uvicorn services.chatbot_api:chatbot_api --host 0.0.0.0 --port 8001 --reload
uvicorn services.extract_api:extract_api --host 0.0.0.0 --port 8002 --reload
```

 

## Typical Workflow

1) Upload documents to the KB (local `kb/` or via API to GCS)
2) Embed: extract → chunk → embed with Cohere → upsert to Pinecone
3) Chat: query with the Chatbot API using your `index_name` and `top_k`
4) Optional: Use Extract API to parse files and extract structured fields

## API Overview

Knowledge Base API (`:8000`)
- POST `/get-or-create-bucket`: Create/access a GCS bucket scoped by user key
- POST `/upload-file`: Upload files (PDF supported for embedding flow)
- POST `/embed-docs`: Parse, chunk, and embed documents with Cohere
- POST `/upsert-to-vector-store`: Upsert embeddings into Pinecone
- POST `/delete-file`: Delete a file from GCS
- POST `/download-file`: Download a file from GCS
- POST `/retrieve`: Semantic retrieve from Pinecone for a query

Chatbot API (`:8001`)
- POST `/chat`: RAG conversation endpoint
- GET `/sessions`: List active sessions
- GET `/sessions/{session_id}/history`: Session history
- DELETE `/sessions/{session_id}`: Delete a session
- DELETE `/sessions`: Clear all sessions
- GET `/providers`: Available LLM providers and defaults
- GET `/health`: Health check

Example `/chat` body:
```json
{
  "message": "What are the key points?",
  "index_name": "my-documents",
  "top_k": 5
}
```

Extract API (`:8002`)
- POST `/parse_document`: Parse an uploaded file (pdf/docx/txt) to text + images
- POST `/extract`: Stateless extraction using text/images, a template, and instructions
- GET `/providers`: Available LLM providers and defaults
- GET `/health`: Health check
- GET `/`: Basic API info

## Knowledge Base Layout

- Local docs: put PDFs under `kb/` for quick experiments
- GCS: use `/get-or-create-bucket` and `/upload-file` to manage remote files
- Vector store: Pinecone index is referenced by `index_name` in requests

## Troubleshooting

- Virtual env: ensure you activated the venv before installing and running
- Missing deps: run `pip install -r backend/requirements.txt`
- Service account: either set `GCP_SERVICE_ACCOUNT_CREDENTIALS` as single-line JSON or set `GOOGLE_APPLICATION_CREDENTIALS` to a JSON file path
- Module paths: starting from repo root vs `backend/` changes module prefixes; when in doubt, run commands from `backend/`

## License

MIT