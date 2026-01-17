# DLGS Scraper (Decreto Legislativo)

This scraper navigates the Normattiva advanced search to find DLGS (Decreto Legislativo) documents, uses AWS Bedrock to determine their relevance for accountants, and saves relevant documents to Google Cloud Storage.

## Features

- **Selenium-based navigation**: Navigates the Normattiva advanced search interface
- **AWS Bedrock LLM integration**: Uses Amazon Nova Micro to determine document relevance
- **Smart filtering**: Only processes DLGS relevant to accountants (commercialisti)
- **Incremental updates**: Stops when encountering already-processed documents
- **GCS storage**: Uploads relevant DLGS texts to Google Cloud Storage
- **Normattiva scraping**: Extracts full article text using Akoma Ntoso format when available

## Structure

The scraper performs the following steps:

1. **Navigate to Normattiva**: Opens the advanced search page
2. **Filter by document type**: Selects "DECRETO LEGISLATIVO"
3. **Execute search**: Launches the advanced search
4. **Process results**: For each DLGS found:
   - Extract title, subtitle, and URL
   - Check if already in GCS (stop condition)
   - Call Bedrock LLM to determine relevance
   - If relevant, scrape full text and upload to GCS

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
# GCP Service Account Credentials (JSON string)
export GCP_SERVICE_ACCOUNT_CREDENTIALS='{"type": "service_account", ...}'

# AWS Credentials for Bedrock
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_BEDROCK_REGION="eu-central-1"
```

## Usage

### Run the scraper

```bash
python run_scraper.py
```

### Command-line Arguments

```bash
python run_scraper.py \
  --bucket-name "loomy-public-documents" \
  --base-folder "dlgs" \
  --log-level INFO
```

### Use as a module

```python
from src.scrape_dlgs import DLGSScraper

scraper = DLGSScraper(
    bucket_name="loomy-public-documents",
    base_folder="dlgs"
)
results = scraper.scrape()
print(f"Processed {results['total_processed']} DLGS, saved {results['files_saved']} files")
```

## Output Structure

Files are stored in GCS with the following naming convention:

```
gs://loomy-public-documents/dlgs/
├── dlgs_04_dicembre_2025_n_205.txt
├── dlgs_15_novembre_2025_n_180.txt
└── ...
```

Each file contains:
- Document metadata (title, subtitle, URL)
- Full article text extracted from Normattiva

## LLM Relevance Check

The scraper uses Amazon Nova Micro to determine if a DLGS is relevant for accountants. The prompt asks the LLM to evaluate based on the DLGS subtitle whether it would be useful for consulting or accounting activities.

## Logging

Set `--log-level DEBUG` for verbose output including:
- Navigation steps
- LLM responses

## Docker Build & Deploy

### Build the image

```bash
docker build -t gcr.io/loomy-475008/dlgs-scraper:latest -f kb/scraping/dlgs/Dockerfile .
```

### Push to GCR

```bash
docker push gcr.io/loomy-475008/dlgs-scraper:latest
```

### Create Cloud Run Job

```bash
gcloud run jobs create dlgs-scraper \
  --image gcr.io/loomy-475008/dlgs-scraper:latest \
  --region europe-west8 \
  --memory 2Gi \
  --cpu 1 \
  --task-timeout 86400 \
  --max-retries 0
```

### Execute the job

```bash
gcloud run jobs execute dlgs-scraper --region europe-west8
```
