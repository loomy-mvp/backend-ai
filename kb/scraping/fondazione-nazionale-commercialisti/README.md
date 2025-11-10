# Fondazione Nazionale Commercialisti Scraper

This scraper downloads documents and publications from the Fondazione Nazionale Commercialisti website and uploads them to Google Cloud Storage.

## Features

- Scrapes multiple document sections from the FNC website
- Handles pagination automatically
- Uploads PDFs directly to Google Cloud Storage
- Deduplicates downloads
- Robust error handling with retries
- Configurable delays to avoid server overload

## Sections Scraped

### Documenti e Notizie
- Ultimi pubblicati
- Documenti di Ricerca
- Strumenti di Lavoro
- Libri
- Circolari
- Informativa Periodica
- Comunicazioni
- Focus

### Archivi
- Fondazione Pacioli (Archivio)
- Fondazione Aristeia (Archivio) - Documenti
- Fondazione Aristeia (Archivio) - Studi e Ricerche
- IRDCEC (Archivio Storico) and subsections

## Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scraper
python run_scraper.py --bucket-name your-bucket-name --base-folder fnc-docs

# Run only a specific section
python run_scraper.py --bucket-name your-bucket-name --only-section "Ultimi pubblicati"

# Limit pages for testing
python run_scraper.py --bucket-name your-bucket-name --max-pages 2
```

### Cloud Run Job

Build and deploy:

```bash
# Build the Docker image (from the scraping folder)
docker build -f fondazione-nazionale-commercialisti/Dockerfile -t gcr.io/YOUR_PROJECT/fnc-scraper .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT/fnc-scraper

# Deploy as Cloud Run Job
gcloud run jobs create fnc-scraper \
  --image gcr.io/YOUR_PROJECT/fnc-scraper \
  --region europe-west1 \
  --max-retries 2 \
  --task-timeout 3600 \
  --set-env-vars GCS_BUCKET_NAME=loomy-public-documents
```

## Configuration Options

- `--bucket-name`: GCS bucket name (required)
- `--base-folder`: Base folder in GCS bucket (default: fondazione-nazionale-commercialisti)
- `--delay`: Delay in seconds between requests (default: 2.5)
- `--retries`: Number of HTTP retries (default: 6)
- `--only-section`: Process only a specific section
- `--max-pages`: Limit number of pages per section (for testing)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Output Structure

Files are organized in GCS as:
```
gs://bucket-name/fondazione-nazionale-commercialisti/
  ├── Ultimi_pubblicati/
  │   ├── document1.pdf
  │   └── document2.pdf
  ├── Documenti_di_Ricerca/
  │   └── ...
  └── ...
```

## Notes

- The scraper respects rate limits with configurable delays
- PDFs are deduplicated across the entire run
- Failed downloads are retried automatically
- Sections with "/node/" URLs (Aristeia) are handled differently (no pagination)
