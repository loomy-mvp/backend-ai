# Agenzia Entrate Risoluzioni Scraper

This scraper downloads all documents (PDF, ZIP, XLSX, XLS, XSD, XML) from the Agenzia delle Entrate risoluzioni archive and uploads them to Google Cloud Storage.

## Features

- Asynchronous scraping for high performance
- Automatic retry logic with exponential backoff
- Preserves full hierarchy (year/month/filename)
- Skips already uploaded files
- Tracks statistics by year
- Handles nested folders within month pages
- Respects rate limits with semaphore-based concurrency control

## Structure

The scraper navigates through:
1. **Main archive page**: Lists all years
2. **Year pages**: List months (or direct files)
3. **Month pages**: Contains files and potentially subfolders
4. **Subfolders**: Additional files organized by topic

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
python scrape_risoluzioni.py
```

### Use as a module

```python
import asyncio
from google.cloud import storage
from scrape_risoluzioni import AgenziaEntrateRisoluzioniScraper

async def main():
    scraper = AgenziaEntrateRisoluzioniScraper(
        bucket_name="your-bucket-name",
        storage_client=storage.Client(),
        max_concurrent=10,
        base_folder="risoluzioni"
    )
    results = await scraper.scrape()
    print(f"Downloaded {results['files_downloaded']} files")

asyncio.run(main())
```

## Output Structure

Files are stored in GCS with the following structure:

```
gs://bucket-name/risoluzioni/
├── 2025/
│   ├── Gennaio/
│   │   ├── Risoluzione n. 1 - pdf.pdf
│   │   └── ...
│   └── Febbraio/
│       └── ...
├── 2024/
│   ├── Gennaio/
│   ├── Febbraio/
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
- ZIP (`.zip`)
- Excel (`.xlsx`, `.xls`)
- XML (`.xml`, `.xsd`)

## Notes

- The scraper automatically handles Italian month names in URLs
- Files are named using the link text from the webpage for better readability
- Duplicate files (same URL) are only downloaded once
- The hierarchy is preserved even when files appear in nested subfolders
