# Batch Embedding for GCS Documents

Batch processing tool to embed multiple documents from Google Cloud Storage buckets and store their vector representations in Pinecone.

## Overview

This tool:
1. Scans specified folders in a GCS bucket
2. Processes each document through the embedding pipeline
3. Stores vector embeddings in Pinecone

Useful for:
- Initial bulk loading of documents into your knowledge base
- Periodic batch updates of large document collections
- Re-indexing documents after configuration changes

## Environment Variables

Required:
```bash
GCP_SERVICE_ACCOUNT_CREDENTIALS=<JSON credentials>
COHERE_API_KEY=<your-cohere-api-key>
PINECONE_API_KEY=<your-pinecone-api-key>
```

## Usage

### Basic Command

```bash
python run_batch_embed.py \
  --bucket-name <BUCKET_NAME> \
  --folders <FOLDER1> <FOLDER2> \
  --organization-id <ORG_ID> \
  --library <organization|private> \
  [--user-id <USER_ID>] \
  [--overwrite] \
  [--log-level INFO]
```

### Examples

**Organization documents:**
```bash
python run_batch_embed.py \
  --bucket-name bucket-org123 \
  --folders organization/contracts organization/policies \
  --organization-id org123 \
  --library organization
```

**Private user documents:**
```bash
python run_batch_embed.py \
  --bucket-name bucket-org123 \
  --folders private/user456/documents \
  --organization-id org123 \
  --library private \
  --user-id user456
```

**Re-index with overwrite:**
```bash
python run_batch_embed.py \
  --bucket-name bucket-org123 \
  --folders organization/legal \
  --organization-id org123 \
  --library organization \
  --overwrite
```

## Docker

Build:
```bash
docker build -t batch-embedding:latest -f batch_embedding/Dockerfile .
```

Run:
```bash
docker run --rm \
  -e GCP_SERVICE_ACCOUNT_CREDENTIALS="${GCP_CREDENTIALS}" \
  -e COHERE_API_KEY="${COHERE_API_KEY}" \
  -e PINECONE_API_KEY="${PINECONE_API_KEY}" \
  batch-embedding:latest \
  --bucket-name bucket-org123 \
  --folders organization/docs \
  --organization-id org123 \
  --library organization
```

## Cloud Run Job

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed Cloud Run deployment instructions.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--bucket-name` | Yes | GCS bucket name |
| `--folders` | Yes | Space-separated list of folder paths |
| `--organization-id` | Yes | Organization identifier |
| `--library` | Yes | `organization` or `private` |
| `--user-id` | Conditional | Required if `--library private` |
| `--overwrite` | No | Overwrite existing vectors |
| `--log-level` | No | Logging level (default: INFO) |
