# Bedrock Batch Process Job

This job reads all `.txt` files from a GCS folder in the `loomy-jobs` bucket, processes each file using AWS Bedrock's Amazon Nova Micro model via the Converse API, and saves the outputs as `{original_filename}_output.txt`.

Designed to run on **Google Cloud Run** with **GCS** for storage and **AWS Bedrock** for LLM inference.

## Features

- **Prompt Caching**: Uses AWS Bedrock's prompt caching feature to cache the user-defined prompt across multiple file processing calls, reducing costs and latency.
- **Batch Processing**: Automatically processes all `.txt` files in the specified folder.
- **Output Management**: Saves outputs with `_output.txt` suffix, and excludes already-processed output files from subsequent runs.

## Usage

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--folder` | Yes | Folder name within the `loomy-jobs` bucket to process |
| `--prompt` | Yes | User-defined prompt to use for processing each file |
| `--dry-run` | No | List files without processing them |

### Example

```bash
python bedrock_batch_process.py \
    --folder "my-documents" \
    --prompt "Summarize the following document in bullet points:"
```

### Dry Run

To see which files would be processed without actually processing them:

```bash
python bedrock_batch_process.py \
    --folder "my-documents" \
    --prompt "Your prompt here" \
    --dry-run
```

## Prompt Caching

The job uses AWS Bedrock's prompt caching feature to optimize costs and latency. The user-defined prompt is placed at the beginning of each message with a cache checkpoint marker. This allows the prompt to be cached and reused across all file processing calls.

**How it works:**
1. The user prompt is sent first in the message content
2. A `cachePoint` marker is placed after the prompt
3. The file content follows the cache point
4. On subsequent calls, the cached prompt is reused, and only the file content varies

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_BEDROCK_REGION` | `eu-central-1` | AWS region for Bedrock |
| `AWS_ACCESS_KEY_ID` | - | AWS access key (required for Bedrock) |
| `AWS_SECRET_ACCESS_KEY` | - | AWS secret key (required for Bedrock) |
| `GOOGLE_APPLICATION_CREDENTIALS` | - | Path to GCP service account JSON (or use default Cloud Run identity) |

## Docker

### Build

```bash
docker build -f backend/jobs/bedrock_batch_process/Dockerfile -t bedrock-batch-process .
```

### Run

```bash
docker run \
    -e AWS_ACCESS_KEY_ID=your-aws-key \
    -e AWS_SECRET_ACCESS_KEY=your-aws-secret \
    -e AWS_BEDROCK_REGION=eu-central-1 \
    -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-sa.json \
    -v /path/to/gcp-sa.json:/secrets/gcp-sa.json \
    bedrock-batch-process \
    python bedrock_batch_process.py --folder "my-folder" --prompt "Your prompt"
```

## Cloud Run Deployment

When deployed on Cloud Run:
- GCS authentication is automatic via the service account attached to the Cloud Run service
- AWS credentials must be provided via environment variables or Secret Manager

## GCS Structure

```
loomy-jobs/
└── {folder}/
    ├── document1.txt          # Input file
    ├── document1_output.txt   # Output (generated)
    ├── document2.txt          # Input file
    └── document2_output.txt   # Output (generated)
```

## Model

Uses **Amazon Nova Micro (EU)** (`eu.amazon.nova-micro-v1:0`) - a fast, cost-effective model suitable for batch text processing tasks.
