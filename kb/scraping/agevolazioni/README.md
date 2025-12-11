# Agenzia Entrate Agevolazioni Scraper

This scraper visits the Agenzia delle Entrate page `https://www.agenziaentrate.gov.it/portale/cittadini/agevolazioni`,
opens each bonus/agevolazione card in the main list, extracts the textual
content, and uploads it to Google Cloud Storage as UTF-8 `.txt` files.

The crawler uses the same *stop-on-first-existing* logic adopted by the
circolari job: while iterating through the entries, it immediately halts when it
finds a blob that is already present in the chosen folder. This lets you rerun
it regularly and only ingest newly published incentives.

## Features

- Async HTTP client with configurable concurrency (default: 5)
- Slugified file names derived from the agevolazione title
- Output file includes the title, source URL, and cleaned body text
- Automatic upload via the shared `upload_to_storage` helper
- Incremental stop logic (breaks on first existing blob)
- Simple runner script for Cloud Run Jobs or local execution

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_scraper.py \
  --bucket-name "loomy-public-documents" \
  --base-folder "agevolazioni" \
  --max-concurrent 5 \
  --log-level INFO
```

## Output Structure

```
gs://<bucket>/agevolazioni/
├── agevolazione-acquisto-prima-casa.txt
├── bonus-mobili.txt
└── ...
```

Each file contains:

```
Agevolazione Name
Fonte: https://www.agenziaentrate.gov.it/.../agevolazione

<Text body scraped from the official page>
```

## Docker / Cloud Run

From `kb/scraping` you can build a container tailored for Cloud Run Jobs:

```bash
docker build -f agevolazioni/Dockerfile -t agenzia-entrate-agevolazioni-scraper .
```

Run locally:

```bash
docker run --rm \
  -e GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}' \
  agenzia-entrate-agevolazioni-scraper \
  --bucket-name "loomy-public-documents" \
  --base-folder "agevolazioni" \
  --max-concurrent 5 \
  --log-level INFO
```

Deploy as a Cloud Run Job similarly to the circolari scraper, overriding the
image/tag name as needed.
