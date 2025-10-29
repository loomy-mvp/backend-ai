# ODCEC Modena Scraper

Scraper for documents from ODCEC Modena (Ordine Dottori Commercialisti ed Esperti Contabili di Modena).

## Overview

This scraper downloads PDF documents from the ODCEC Modena document management system and uploads them to Google Cloud Storage. It uses Selenium to navigate through the paginated document listings and downloads all available PDFs.

## Features

- **Pagination handling**: Automatically navigates through all pages of results
- **SSL compatibility**: Uses legacy SSL ciphers to handle old server configurations
- **Duplicate detection**: Skips already downloaded documents
- **GCS integration**: Uploads directly to Google Cloud Storage
- **Robust error handling**: Retry logic for network issues
- **Cloud Run ready**: Designed to run as a Cloud Run Job

## Source Website

- **URL**: https://www.commercialisti.mo.it/servizi/gestionedocumentale/
- **Type**: ASP.NET WebForms application with pagination
- **Content**: Professional documents, circulars, and publications

## Installation

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Install Chrome and ChromeDriver (required for Selenium)
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y google-chrome-stable

# Download ChromeDriver matching your Chrome version
# Place it in your PATH
```

### Docker

```bash
# Build from the scraping directory
cd kb/scraping
docker build -f odcec-modena/Dockerfile -t odcec-modena-scraper .
```

## Usage

### Local Execution

```bash
# Basic usage
python run_scraper.py --bucket-name loomy-public-documents

# With custom folder
python run_scraper.py \
  --bucket-name loomy-public-documents \
  --base-folder odcec-modena \
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
  -e GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json \
  -v /path/to/credentials.json:/path/to/credentials.json \
  odcec-modena-scraper \
  --bucket-name loomy-public-documents \
  --base-folder odcec-modena \
  --log-level INFO
```

### Cloud Run Job

```bash
# Deploy to Cloud Run Jobs
gcloud run jobs create odcec-modena-scraper \
  --image gcr.io/YOUR_PROJECT/odcec-modena-scraper \
  --region europe-west1 \
  --memory 2Gi \
  --cpu 1 \
  --max-retries 2 \
  --task-timeout 3600 \
  --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json" \
  --args="--bucket-name=loomy-public-documents,--base-folder=odcec-modena,--log-level=INFO"

# Execute the job
gcloud run jobs execute odcec-modena-scraper --region europe-west1
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bucket-name` | str | `loomy-public-documents` | GCS bucket name |
| `--base-folder` | str | `odcec-modena` | Base folder path in GCS bucket |
| `--headless` | flag | `True` | Run browser in headless mode |
| `--no-headless` | flag | - | Run browser with visible UI |
| `--log-level` | str | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Output Structure

Files are uploaded to GCS with the following structure:

```
gs://loomy-public-documents/
└── odcec-modena/
    ├── documento_001.pdf
    ├── documento_002.pdf
    └── ...
```

## Technical Details

### SSL Compatibility

The scraper uses a custom SSL adapter with `SECLEVEL=1` to handle old servers with weak DH keys:

```python
CIPHERS = "DEFAULT:@SECLEVEL=1"  # Allows 1024-bit DH keys
```

### Selenium Configuration

- **Headless mode**: Runs without GUI by default
- **Timeouts**: 60s page load timeout, 30s element wait timeout
- **Cookie sharing**: Transfers cookies from Selenium to requests for downloads

### Error Handling

- Network errors trigger automatic retries
- Invalid PDFs (non-PDF content) are skipped
- Errors are logged but don't stop execution
- Duplicate URLs are tracked and skipped

## Monitoring

The scraper logs comprehensive information:

- ✔ Successfully uploaded files
- ✖ Download errors with URLs
- Page progress and link counts
- Final statistics summary

## Dependencies

- **requests**: HTTP client with retry logic
- **selenium**: Browser automation for pagination
- **google-cloud-storage**: GCS upload
- **urllib3**: SSL configuration
- **Chrome**: Headless browser
- **ChromeDriver**: Selenium driver for Chrome

## Limitations

- Requires Chrome/ChromeDriver installation
- Memory usage depends on PDF sizes
- Execution time depends on document count (typically 100+ pages)
- Requires valid GCS credentials

## Troubleshooting

### Chrome not found
```bash
# Install Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get -f install
```

### ChromeDriver version mismatch
```bash
# Check Chrome version
google-chrome --version

# Download matching ChromeDriver
# Visit: https://chromedriver.chromium.org/downloads
```

### SSL errors
The scraper includes a weak ciphers adapter. If issues persist, check:
- Network connectivity
- Firewall rules
- Certificate validity

## License

Internal use only - Loomy MVP
