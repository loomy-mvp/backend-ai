# CCIA Modena Statuto Scraper

This scraper downloads every document listed inside the "Statuto e Regolamenti" section of the Camera di Commercio di Modena "Atti generali" page, extracts the textual content from each PDF, and uploads it to Google Cloud Storage as UTF-8 text files.

## Features

- Deterministic discovery of all links contained in the Statuto/Regolamenti section
- Automatic PDF download via Plone `@@download/file` and `resolveuid` fallbacks
- Text extraction with PyPDF2 and upload through the shared `upload_to_storage` helper
- Configurable bucket/folder through CLI flags or environment variables
- Ready for Cloud Run Jobs via the included Dockerfile

## Installation

```bash
pip install -r requirements.txt
```

If you keep your GCP credentials inside a `.env` file, they will be loaded automatically.

## Required environment variables

```bash
GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}'
# Optional overrides
GCS_BUCKET_NAME="loomy-public-documents"
CCIA_MODENA_BASE_FOLDER="ccia-modena/statuto-regolamenti"
```

## Usage

Run the helper script with sensible defaults:

```bash
python run_scraper.py --bucket-name loomy-public-documents --base-folder ccia-modena/statuto-regolamenti
```

A compatibility runner (`scrape_modena_pdfs.py`) is still available; it simply delegates to the packaged scraper class.

## Docker / Cloud Run

The provided `Dockerfile` mirrors the structure used by the other scrapers. Build and run locally:

```bash
docker build -t ccia-modena-scraper .
docker run --rm -e GCP_SERVICE_ACCOUNT_CREDENTIALS="${GCP_SERVICE_ACCOUNT_CREDENTIALS}" ccia-modena-scraper
```

Override CLI flags by appending them to the `docker run` command.
