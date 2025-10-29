# Notariato Scraper

This scraper downloads all PDF documents from the Agenzia delle Entrate "Provvedimenti non soggetti a pubblicità" archive and uploads them to Google Cloud Storage.

## Features

- Asynchronous scraping for high performance
- Automatic retry logic with exponential backoff
- Preserves hierarchy by year
- Skips already uploaded files
- Tracks statistics by year
- Respects rate limits with semaphore-based concurrency control
- Handles multiple entry points (2014-2025)

## Structure

The scraper navigates through multiple entry points covering years 2014-2025:
1. **Archive page**: Historical documents (2006-2016)
2. **Year pages**: Documents organized by year (2017-2025)
3. **Month pages**: Monthly document listings for recent years

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
# GCP Service Account Credentials (JSON string)
export GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}'

# Optional: Override default bucket name
export GCS_BUCKET_NAME="your-bucket-name"
```

## Usage

### Run the scraper

```bash
python run_scraper.py
```

### Command-line Arguments

```bash
python run_scraper.py \
  --bucket-name "your-bucket-name" \
  --base-folder "notariato" \
  --max-concurrent 10 \
  --log-level INFO
```

### Use as a module

```python
import asyncio
from google.cloud import storage
from src.scrape_notariato import NotariatoScraper

async def main():
    scraper = NotariatoScraper(
        bucket_name="your-bucket-name",
        storage_client=storage.Client(),
        max_concurrent=10,
        base_folder="notariato"
    )
    results = await scraper.scrape()
    print(f"Downloaded {results['files_downloaded']} files")

asyncio.run(main())
```

## Output Structure

Files are stored in GCS with the following structure:

```
gs://bucket-name/notariato/
├── 2025/
│   ├── provvedimento_gennaio.pdf
│   └── ...
├── 2024/
│   └── ...
├── 2023/
│   └── ...
└── ...
```

## Logging

The scraper uses Python's `logging` module. Logs include:
- Pages being crawled
- Files being downloaded
- Upload confirmations
- Errors and retries
- Final statistics by year

## Error Handling

- Network errors: Automatic retry with exponential backoff (up to 3 attempts)
- File name too long: Automatic truncation with separator
- Already uploaded files: Skipped automatically
- Invalid URLs: Logged and skipped

## Performance

- **Concurrent requests**: Configurable (default: 10)
- **Timeout**: 30s for pages, 60s for file downloads
- **Rate limiting**: Semaphore-based concurrency control

## Target File Types

- PDF (`.pdf`)

## Notes

- The scraper automatically discovers documents from multiple year ranges
- Files are organized by year extracted from the URL
- Duplicate files (same URL) are only downloaded once
- The scraper handles both archive and current document sections

## Docker Deployment

### Build the Docker image

From the `kb/scraping` directory:

```bash
docker build -f notariato/Dockerfile -t notariato-scraper .
```

### Run with Docker

```bash
docker run --rm \
  -e GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}' \
  notariato-scraper \
  --bucket-name "your-bucket-name" \
  --base-folder "notariato" \
  --max-concurrent 10 \
  --log-level INFO
```

### Deploy to Google Cloud Run

```bash
# Build and push to GCR
docker build -f notariato/Dockerfile -t gcr.io/your-project/notariato-scraper .
docker push gcr.io/your-project/notariato-scraper

# Deploy as Cloud Run Job
gcloud run jobs create notariato-scraper \
  --image gcr.io/your-project/notariato-scraper \
  --region europe-west1 \
  --set-env-vars GCS_BUCKET_NAME=your-bucket-name \
  --args="--bucket-name,your-bucket-name,--base-folder,notariato,--max-concurrent,10,--log-level,INFO"
```

## Entry Points

The scraper covers the following sections:

- Archive (2006-2016): Base archive page
- 2017-2020: Year-specific pages under "altri-provvedimenti-non-soggetti-attuale"
- 2021-2025: Individual year and month pages
- Current year months: January, March 2025 (and any new months)

## Coverage

This scraper is designed to be comprehensive and capture all documents from:
- Historical archive
- Year-by-year listings
- Month-by-month current year updates

The multiple entry points ensure complete coverage across all time periods.
