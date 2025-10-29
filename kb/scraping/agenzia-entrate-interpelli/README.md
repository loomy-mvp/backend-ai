# Agenzia Entrate Interpelli Scraper

This scraper downloads all documents (PDF, ZIP, XLSX, XLS, XSD, XML) from the Agenzia delle Entrate interpelli archives and uploads them to Google Cloud Storage.

## Features

- Asynchronous scraping for high performance
- Automatic retry logic with exponential backoff
- Preserves full hierarchy (year/month/filename)
- Skips already uploaded files
- Tracks statistics by year
- Handles nested folders within month pages
- Respects rate limits with semaphore-based concurrency control
- Scrapes multiple interpelli archive sections

## Archive Sections Covered

The scraper automatically navigates through all of these interpelli archive sections:

1. **Archivio Interpelli** - Main interpelli archive
   - `https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/risposte-agli-interpelli/interpelli/archivio-interpelli`

2. **Archivio istanze di interpello sui nuovi investimenti**
   - `https://www.agenziaentrate.gov.it/portale/web/guest/archivio-istanze-di-interpello-sui-nuovi-investimenti`

3. **Archivio principi di diritto**
   - `https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/risposte-agli-interpelli/principi-di-diritto/archivio-principi-di-diritto`

4. **Archivio risposte alle istanze di consulenza giuridica**
   - `https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/risposte-agli-interpelli/risposte-alle-istanze-di-consulenza-giuridica/archivio-risposte-alle-istanze-di-consulenza-giuridica`

## Structure

The scraper navigates through:
1. **Main archive pages**: Lists all years for each archive section
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
python run_scraper.py
```

### Run with custom parameters

```bash
python run_scraper.py --bucket-name your-bucket --base-folder interpelli --max-concurrent 15 --log-level DEBUG
```

### Use as a module

```python
import asyncio
from google.cloud import storage
from src.scrape_interpelli import InterpelliScraper

async def main():
    scraper = InterpelliScraper(
        bucket_name="your-bucket-name",
        storage_client=storage.Client(),
        max_concurrent=10,
        base_folder="interpelli"
    )
    results = await scraper.scrape()
    print(f"Downloaded {results['files_downloaded']} files")

asyncio.run(main())
```

## Output Structure

Files are stored in GCS with the following structure:

```
gs://bucket-name/interpelli/
├── 2025/
│   ├── Gennaio/
│   │   ├── Interpello n. 1 - pdf.pdf
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

## Docker Build and Run

### Build the Docker image

From the parent scraping directory:

```bash
docker build -f agenzia-entrate-interpelli/Dockerfile -t interpelli-scraper .
```

### Run the container

```bash
docker run --rm \
  -e GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}' \
  interpelli-scraper \
  --bucket-name loomy-public-documents \
  --base-folder interpelli \
  --max-concurrent 10
```

## Notes

- The scraper automatically handles Italian month names in URLs
- Files are named using the link text from the webpage for better readability
- Duplicate files (same URL) are only downloaded once
- The hierarchy is preserved even when files appear in nested subfolders
- All four interpelli archive sections are scraped in a single run
