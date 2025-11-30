# NT+ Fisco (Il Sole 24 Ore) Article Scraper

Scraper for NT+ Fisco article previews from the Il Sole 24 Ore website.

## Overview

This scraper uses Selenium WebDriver to navigate the NT+ Fisco website and extract article previews from various sections **without authentication**. It extracts publicly visible content only (title, subtitle, preview, author, date) and saves them as JSON files to Google Cloud Storage, organized by section.

## Features

- **Section-based scraping**: Navigates through all configured sections
- **Pagination handling**: Clicks "Mostra altri" to load more articles
- **Article extraction**: Extracts title, subtitle, preview, author, date, URL, and related links
- **No authentication required**: Only scrapes publicly visible previews
- **Cookie banner handling**: Automatically dismisses cookie consent modals
- **GCS integration**: Uploads directly to Google Cloud Storage
- **Robust error handling**: Retry logic for network issues and stale elements
- **Duplicate detection**: Skips already processed articles
- **Cloud Run ready**: Designed to run as a Cloud Run Job

## Sections Scraped

- Adempimenti (`/sez/adempimenti`)
- Contabilità (`/sez/contabilita`)
- Controlli e liti (`/sez/controlli-e-liti`)
- Diritto (`/sez/diritto`)
- Finanza (`/sez/finanza`)
- Imposte (`/sez/imposte`)
- Professioni (`/sez/professioni`)
- Analisi (`/sez/analisi`)
- Schede (`/sez/schede`)
- L'esperto risponde (`/sez/esperto-risponde`)
- Rubriche (`/sez/rubriche`)
- Speciali (`/sez/speciali`)

## Output Format

Each article is saved as a JSON file with the following structure:

```json
{
  "title": "Transfer pricing, aggiustamenti fuori dagli errori contabili",
  "subtitle": "I temi di NT+",
  "preview": "Lo slittamento deriva componenti basati su elementi sopravvenuti...",
  "author": "Luca Gaiani",
  "date": "2025-11-27 21:44",
  "url": "https://ntplusfisco.ilsole24ore.com/art/...",
  "related": ["Related article 1", "Related article 2"],
  "section": "imposte",
  "scraped_at": "2025-11-29T12:00:00+00:00"
}
```

## Source Website

- **URL**: https://ntplusfisco.ilsole24ore.com/
- **Type**: News portal with dynamic content loading
- **Content**: Professional tax and finance articles (previews only)

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
docker build -f nt_fisco/Dockerfile -t nt-fisco-scraper .
```

## Usage

### Local Development

```bash
# Set your Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Run with default settings
python run_scraper.py

# Run specific sections only
python run_scraper.py --sections "Imposte|SEP|Finanza"

# Run with visible browser (debugging)
python run_scraper.py --no-headless

# Limit articles per section
python run_scraper.py --max-articles 10
```

### Docker

```bash
# Run the container
docker run \
  -v /path/to/credentials.json:/app/credentials.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  nt-fisco-scraper

# Run specific sections
docker run \
  -v /path/to/credentials.json:/app/credentials.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  nt-fisco-scraper \
  --sections "Imposte|SEP|Finanza"
```

## Output Structure

Articles are saved as JSON files in GCS with the following structure:

```
gs://{bucket_name}/nt_fisco/
├── adempimenti/
│   ├── article_1.json
│   ├── article_2.json
│   └── ...
├── contabilita/
│   └── ...
├── imposte/
│   └── ...
└── ...
```

Each JSON file contains:

```json
{
    "title": "Article Title",
    "subtitle": "Article Subtitle",
    "preview": "Article preview text...",
    "summary": "Article summary if available...",
    "url": "https://ntplusfisco.ilsole24ore.com/art/...",
    "section": "imposte",
    "scraped_at": "2024-01-15T10:30:00Z"
}
```

## Cloud Run Deployment

The scraper is designed to run as a Cloud Run Job. See the Dockerfile for container configuration.

```bash
# Deploy to Cloud Run
gcloud run jobs create nt-fisco-scraper \
  --image gcr.io/PROJECT_ID/nt-fisco-scraper \
  --region europe-west1 \
  --task-timeout 3600s \
  --memory 2Gi \
  --cpu 1
```

## Notes

- The website may require authentication for full content access
- Rate limiting is implemented to avoid overloading the server
- Some sections may have different HTML structures; the scraper handles multiple patterns
