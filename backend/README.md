# Backend APIs

This backend provides multiple FastAPI services under the `services` package:

- `services/kb_api.py`: Knowledge Base API for document processing, chunking, embedding, and vector storage
- `services/chatbot_api.py`: Chatbot API that performs RAG over the KB
- `services/extract_api.py`: Stateless extraction API for parsing files and extracting structured info

The `start_apis.py` helper lets you run individual services quickly.

## Setup

1) Create and activate a virtual environment (Windows PowerShell):
```powershell
cd c:\Users\leoac\Work\Companies\Loomy\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:
```powershell
pip install -r requirements.txt
```

3) Configure environment variables in a `.env` file at repo root or `backend`:
```env
# Google Cloud Storage service account JSON (single line)
GCP_SERVICE_ACCOUNT_CREDENTIALS={"type":"service_account",...}

# Cohere (embeddings) and Pinecone (vector DB) for KB
COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional: override default endpoints used by services
KB_API_BASE_URL=http://localhost:8000

# Optional: LLM providers (used by chatbot/extract services)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

## Run Services

You can use the launcher or run with `uvicorn` directly.

### Option A: Using the launcher
```powershell
# From backend directory
python start_apis.py kb       # Starts Knowledge Base API on :8000
python start_apis.py chatbot  # Starts Chatbot API on :8001
python start_apis.py extract  # Starts Extract API on :8002
```

### Option B: Using uvicorn directly (from backend directory)
```powershell
uvicorn services.kb_api:kb_api --host 0.0.0.0 --port 8000 --reload
uvicorn services.chatbot_api:chatbot_api --host 0.0.0.0 --port 8001 --reload
uvicorn services.extract_api:extract_api --host 0.0.0.0 --port 8002 --reload
```

If you run from the repo root, prefix modules with `backend.` (e.g., `backend.services.kb_api:kb_api`).

## API Endpoints

### Knowledge Base API (`:8000`)
- `POST /get-or-create-bucket` – Create or access a GCS bucket scoped by user key
- `POST /upload-file` – Upload files to GCS (PDF supported for embedding flow)
- `POST /embed-docs` – Extract PDF text, chunk, embed with Cohere
- `POST /upsert-to-vector-store` – Upsert embeddings into Pinecone
- `POST /delete-file` – Delete a file from GCS
- `POST /download-file` – Download a file from GCS
- `POST /retrieve` – Semantic retrieve from Pinecone for a query

### Chatbot API (`:8001`)
- `POST /chat` – RAG conversation endpoint
- `GET /sessions` – List active sessions
- `GET /sessions/{session_id}/history` – Get session history
- `DELETE /sessions/{session_id}` – Delete a session
- `DELETE /sessions` – Clear all sessions
- `GET /providers` – Available LLM providers and defaults
- `GET /health` – Health check

Sample request:
```json
{
  "message": "What are the key points?",
  "index_name": "my-documents",
  "top_k": 5
}
```

### Extract API (`:8002`)
- `POST /parse_document` – Parse an uploaded file (pdf/docx/txt) to text + images
- `POST /extract` – Stateless extraction using text/images, template, and instructions
- `GET /providers` – Available LLM providers and defaults
- `GET /health` – Health check
- `GET /` – Basic API info

## Typical Workflow
1) Upload and embed documents via KB API (`upload-file` → `embed-docs` → `upsert-to-vector-store`)
2) Ask questions via Chatbot API (`/chat`, providing `index_name` and `top_k`)
3) Optionally use Extract API to parse and extract structured information from files

## Frontend (optional)
If you have a Streamlit frontend, start it after services are up:
```powershell
cd frontend
streamlit run Home.py
```

## Troubleshooting
- Missing packages (e.g., `cohere`) – ensure `pip install -r requirements.txt` ran successfully in the active venv.
- If `.env` values contain JSON, keep them as a single-line string.
- Running from repo root vs `backend` affects module paths; use the noted prefixes accordingly.