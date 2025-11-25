# Sentenze Cassazione Scraper

Scraper dedicated to the ForoEuropeo archive of civil law "massime" published by the Corte di Cassazione. Each entry is downloaded, converted to UTF-8 plain text, and uploaded directly to Google Cloud Storage so it can run as a Cloud Run Job like the other scrapers in this repository.

## Features

- Discovers every subject/category exposed on the ForoEuropeo landing page
- Visits all articles for a category (no client-side pagination required)
- Extracts title + body, appends the source URL, and stores the result as `.txt`
- Uploads data to GCS via the shared `upload_to_storage` helper
- Allows throttling between requests and category limiting for tests
- Emits structured metrics (categories processed, files uploaded, etc.)

## Project Layout

```
sentenze-cassazione/
├── Dockerfile
├── README.md
├── requirements.txt
├── run_scraper.py
└── src/
    ├── __init__.py
    └── scrape_sentenze.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the usual environment variables before running locally or in Cloud Run:

```bash
# Mandatory: bucket that will host the txt files
export GCS_BUCKET_NAME="loomy-public-documents"

# Optional: service account credentials for local runs
export GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}'
```

## Usage

### Local execution

```bash
python run_scraper.py \
  --bucket-name loomy-public-documents \
  --base-folder sentenze_cassazione/test \
  --limit-categories 3 \
  --request-delay 1.0
```

### Cloud Run Job

1. Build and push the container using the provided Dockerfile (identical structure to the other scrapers).
2. Create a Cloud Run Job that sets `--bucket-name` and `--base-folder` as needed.
3. Supply the service account credentials via a mounted secret or workload identity.

## Output Layout

```
gs://<bucket>/<base_folder>/
├── 1038-acque-pubbliche/
│   ├── 63795-acque-derivazioni.txt
│   ├── ...
├── 1102-appalto/
│   └── ...
└── ...
```

Each file contains the article title, the full body (paragraphs separated by blank lines) and the original source URL appended at the bottom for traceability.

## CLI Flags

| Flag | Description |
| ---- | ----------- |
| `--bucket-name` | Target GCS bucket (default `loomy-public-documents`). |
| `--base-folder` | Root folder inside the bucket (default `sentenze_cassazione`). |
| `--limit-categories` | Optional cap on the number of categories for quick tests. |
| `--request-delay` | Seconds to wait between HTTP calls (default `0.5`). |
| `--log-level` | Standard Python log level (`INFO` by default). |

## Logging & Metrics

The runner prints a Cloud Run–friendly summary block that highlights how many categories were processed, how many files were uploaded, how many duplicates were skipped, and a per-category breakdown. Errors are logged with stack traces to simplify debugging.
