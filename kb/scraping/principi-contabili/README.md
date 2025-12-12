# Principi Contabili Scraper

This scraper visits the Ragioneria Generale dello Stato page for accounting principles:
`https://www.rgs.mef.gov.it/VERSIONE-I/e_government/amministrazioni_pubbliche/arconet/principi_contabili/`

It downloads and converts to PDF:
1. **First document** from the "Principi contabili generali:" section
2. **All documents** from the section immediately following "Principi contabili generali:" (e.g., "Principi contabili applicati dal 2026")

The documents are originally in `.doc` format and are converted to PDF using LibreOffice before upload to Google Cloud Storage.

## Features

- Async HTTP client with configurable concurrency (default: 3)
- DOC to PDF conversion using LibreOffice (headless mode)
- Slugified file names derived from document titles
- Automatic upload via the shared `upload_to_storage` helper
- Organized output with separate subfolders for "generali" and "applicati" documents
- Simple runner script for Cloud Run Jobs or local execution

## Prerequisites

### For Local Execution
You need LibreOffice installed on your system for DOC to PDF conversion:

**Windows:**
```bash
# Download and install from https://www.libreoffice.org/download/
# Make sure soffice.exe is in your PATH
```

**macOS:**
```bash
brew install libreoffice
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libreoffice-writer libreoffice-common
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Local Execution

```bash
cd kb/scraping/principi-contabili
python run_scraper.py \
  --bucket-name "loomy-public-documents" \
  --base-folder "principi-contabili" \
  --max-concurrent 3 \
  --log-level INFO
```

### Quick Test (without GCS upload)

For testing the scraping and conversion without uploading, you can run the scraper module directly and modify the code temporarily, or use the `--log-level DEBUG` flag to see more details.

## Output Structure

```
gs://<bucket>/principi-contabili/
├── generali/
│   └── principi-contabili-generali-a-decorrere-dal-2021-aggiornati-al-dm-1-settembre-2021.pdf
└── applicati/
    ├── principio-contabile-applicato-della-programmazione-allegato-n-4-1-d-lgs-118-2011.pdf
    ├── principio-contabile-applicato-della-contabilita-finanziaria-allegato-n-4-2-d-lgs-118-2011.pdf
    ├── principio-contabile-applicato-della-contabilita-economico-patrimoniale-allegato-n-4-3-d-lgs-118-2011.pdf
    └── principio-contabile-applicato-del-bilancio-consolidato-allegato-n-4-4-d-lgs-118-2011.pdf
```

## Docker / Cloud Run

From the `kb/scraping` directory, build the container:

```bash
docker build -f principi-contabili/Dockerfile -t principi-contabili-scraper .
```

Run locally with Docker:

```bash
docker run --rm \
  -e GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}' \
  principi-contabili-scraper \
  --bucket-name "loomy-public-documents" \
  --base-folder "principi-contabili" \
  --max-concurrent 3 \
  --log-level INFO
```

## Document Selection Logic

The scraper dynamically identifies sections based on HTML structure:

1. Finds the `<h4>` tag containing "Principi contabili generali"
2. Extracts only the **first** `.doc` link after that header
3. Finds the **next** `<h4>` tag (which will be something like "Principi contabili applicati dal 2026")
4. Extracts **all** `.doc` links from that section

This approach ensures the scraper continues working even when section names change year-over-year.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GCP_SERVICE_ACCOUNT_CREDENTIALS` | JSON string with GCP service account credentials |

## Notes

- DOC conversion requires LibreOffice and may take a few seconds per file
- The Docker image is larger than typical scrapers due to LibreOffice installation (~500MB+)
- Max concurrent is set lower (3) to avoid overloading the government website
