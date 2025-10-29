# INPS Circolari e Messaggi Scraper

Scraper for INPS (Istituto Nazionale della Previdenza Sociale) official circolari, messaggi and normativa documents.

## Overview

This scraper uses Selenium WebDriver to navigate the INPS website and download all PDF documents from the "Circolari, Messaggi e Normativa" section. It uploads them directly to Google Cloud Storage with an organized folder structure based on year and document type.

## Features

- **Automated filtering**: Filters results to show only INPS documents
- **Pagination handling**: Automatically navigates through all pages (100 results per page)
- **Cookie banner handling**: Automatically dismisses cookie consent modals
- **Document classification**: Organizes files by year and type (circolari/messaggi/altri)
- **GCS integration**: Uploads directly to Google Cloud Storage
- **Robust error handling**: Retry logic for network issues and stale elements
- **Duplicate detection**: Skips already processed documents
- **Cloud Run ready**: Designed to run as a Cloud Run Job

## Source Website

- **URL**: https://www.inps.it/it/it/inps-comunica/atti/circolari-messaggi-e-normativa.html
- **Type**: Government portal with dynamic filtering and pagination
- **Content**: Official INPS circulars, messages, and regulatory documents

## Installation

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Install Chrome and ChromeDriver (required for Selenium)
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y google-chrome-stable

# ChromeDriver will be automatically managed by webdriver-manager
```

### Docker

```bash
# Build from the scraping directory
cd kb/scraping
docker build -f inps/Dockerfile -t inps-scraper .
```

## Usage

### Local Execution

```bash
# Basic usage
python run_scraper.py --bucket-name loomy-public-documents

# With custom folder and options
python run_scraper.py \
  --bucket-name loomy-public-documents \
  --base-folder inps \
  --max-pages 100 \
  --log-level DEBUG

# Run with visible browser (non-headless)
python run_scraper.py \
  --bucket-name loomy-public-documents \
  --no-headless
```

### Docker Execution

```bash
# Run locally with Docker
docker run --rm \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -v /path/to/credentials.json:/app/credentials.json \
  inps-scraper \
  --bucket-name loomy-public-documents \
  --base-folder inps \
  --max-pages 1000 \
  --log-level INFO
```

### Cloud Run Job

```bash
# Deploy to Cloud Run Jobs
gcloud run jobs create inps-scraper \
  --image gcr.io/YOUR_PROJECT/inps-scraper \
  --region europe-west1 \
  --memory 4Gi \
  --cpu 2 \
  --max-retries 2 \
  --task-timeout 7200 \
  --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json" \
  --args="--bucket-name=loomy-public-documents,--base-folder=inps,--max-pages=1000,--log-level=INFO"

# Execute the job
gcloud run jobs execute inps-scraper --region europe-west1

# Monitor execution
gcloud run jobs executions describe EXECUTION_NAME --region europe-west1
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bucket-name` | str | `loomy-public-documents` | GCS bucket name |
| `--base-folder` | str | `inps` | Base folder path in GCS bucket |
| `--headless` | flag | `True` | Run browser in headless mode |
| `--no-headless` | flag | - | Run browser with visible UI |
| `--max-pages` | int | `1000` | Maximum number of pages to scrape |
| `--log-level` | str | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Output Structure

Files are uploaded to GCS with the following structure:

```
gs://loomy-public-documents/
└── inps/
    ├── 2024/
    │   ├── circolari/
    │   │   ├── Circolare_n_45_del_2024.pdf
    │   │   └── ...
    │   └── messaggi/
    │       ├── Messaggio_n_1234_del_2024.pdf
    │       └── ...
    ├── 2023/
    │   ├── circolari/
    │   └── messaggi/
    └── sconosciuto/
        └── altri/
```

## Technical Details

### Selenium Configuration

- **Headless mode**: Runs without GUI by default for server deployments
- **Timeouts**: 120s page load timeout, 45s element wait timeout
- **Window size**: 1600x1200 (headless mode)
- **Language**: Italian (it-IT)
- **WebDriver Manager**: Automatically downloads and manages ChromeDriver

### Scraping Strategy

1. **Initial Setup**: Opens the INPS portal and dismisses cookie banners
2. **Filtering**: Applies "Ente Emanante = INPS" filter
3. **Pagination**: Sets results per page to 100 for efficiency
4. **Navigation**: Iterates through all result pages
5. **Detail Pages**: Follows "Vai al dettaglio" links for each document
6. **Download**: Downloads PDF to memory using requests with Selenium cookies
7. **Classification**: Extracts year and type from document title
8. **Upload**: Uploads to GCS using the shared `upload_to_storage` utility

### Document Classification

- **Year**: Extracted from document title using regex `(\d{4})`
- **Type**: 
  - `circolari` if "circolar" is in title
  - `messaggi` if "messagg" is in title
  - `altri` for other documents
- **Filename**: Sanitized document title with `.pdf` extension

### Error Handling

- Network errors trigger automatic retries (up to 3 attempts)
- Stale element exceptions are handled with retry logic
- Cookie banners are automatically dismissed
- Failed downloads are logged but don't stop execution
- Duplicate URLs are tracked and skipped
- Invalid PDFs (non-PDF content) are detected and skipped

### Performance

- **Concurrency**: Single-threaded (Selenium limitation)
- **Memory**: Downloads PDFs to memory before upload (efficient)
- **Speed**: ~100 documents per page, configurable page limit
- **Typical Runtime**: Depends on document count (estimated 2-4 hours for full scrape)

## Monitoring

The scraper logs comprehensive information:

- ✅ Successfully uploaded files with page number
- ❌ Download errors with URLs and error details
- 📄 Page progress and link counts
- 📊 Final statistics summary with files per year

Example output:
```
[Pagina 1] link trovati: 100
[Pagina 1] ✅ https://www.inps.it/.../dettaglio.circolari-e-messaggi/...
...
================================================================================
✅ SCRAPING COMPLETED SUCCESSFULLY
================================================================================
📄 Total pages visited: 45
📁 Total files downloaded: 4321
❌ Total errors: 12
🗄️  Storage location: gs://loomy-public-documents/inps/

📊 Files downloaded per year:
  2024: 456 files
  2023: 789 files
  2022: 654 files
  ...
================================================================================
```

## Dependencies

- **selenium**: Browser automation for navigation
- **webdriver-manager**: Automatic ChromeDriver management
- **google-cloud-storage**: GCS upload
- **google-auth**: GCS authentication
- **requests**: HTTP client for PDF downloads
- **beautifulsoup4**: HTML parsing (optional, for future enhancements)
- **Chrome**: Headless browser (installed in Docker image)

## Limitations

- Requires Chrome/ChromeDriver installation
- Single-threaded execution (Selenium limitation)
- Memory usage proportional to largest PDF size
- Execution time depends on total document count
- Requires valid GCS credentials with write permissions

## Troubleshooting

### Chrome not found
```bash
# Install Chrome on Ubuntu/Debian
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get -f install
```

### ChromeDriver version mismatch
The webdriver-manager package automatically handles ChromeDriver versions. If issues occur:
```bash
# Clear cache and reinstall
pip uninstall webdriver-manager
pip install webdriver-manager
```

### Cookie banner not dismissed
The scraper includes multiple strategies for dismissing cookie banners. If issues persist:
- Try running with `--no-headless` to see what's blocking
- Check the INPS website for UI changes
- Update the `close_cookie_banner_if_any` function

### Timeouts or stale elements
Increase timeout values in the scraper if the website is slow:
- `PAGELOAD_TIMEOUT`: Page load timeout (default: 120s)
- `WAIT_TIMEOUT`: Element wait timeout (default: 45s)

### GCS upload failures
- Verify GCS credentials are valid
- Check bucket permissions (needs `storage.objects.create`)
- Verify bucket name is correct
- Check network connectivity to GCS

### Out of memory errors
The scraper downloads PDFs to memory before upload. If you encounter OOM:
- Increase memory allocation in Cloud Run Job (use 4Gi or 8Gi)
- Check for extremely large PDF files
- Consider implementing streaming upload (future enhancement)

## Future Enhancements

- [ ] Incremental scraping (skip already uploaded files by checking GCS)
- [ ] Parallel page processing (using multiple browser instances)
- [ ] Streaming upload for large PDFs
- [ ] Better filename sanitization
- [ ] Metadata extraction and indexing
- [ ] Progress tracking with checkpoints
- [ ] Email notifications on completion/failure

## License

Internal use only - Loomy MVP
