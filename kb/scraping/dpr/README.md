# DPR Scraper (Decreto del Presidente della Repubblica)

This scraper navigates the Normattiva advanced search to find DPR documents, uses AWS Bedrock to determine their relevance for accountants, and saves relevant documents to Google Cloud Storage.

## Features

- **Selenium-based navigation**: Navigates the Normattiva advanced search interface
- **AWS Bedrock LLM integration**: Uses Amazon Nova Micro to determine document relevance
- **Smart filtering**: Only processes DPRs relevant to accountants (commercialisti)
- **Incremental updates**: Stops when encountering already-processed documents
- **GCS storage**: Uploads relevant DPR texts to Google Cloud Storage
- **Normattiva scraping**: Extracts full article text using Akoma Ntoso format when available

## Structure

The scraper performs the following steps:

1. **Navigate to Normattiva**: Opens the advanced search page
2. **Filter by document type**: Selects "DECRETO DEL PRESIDENTE DELLA REPUBBLICA"
3. **Execute search**: Launches the advanced search
4. **Process results**: For each DPR found:
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
  --base-folder "dpr" \
  --log-level INFO
```

### Use as a module

```python
from src.scrape_dpr import DPRScraper

scraper = DPRScraper(
    bucket_name="loomy-public-documents",
    base_folder="dpr"
)
results = scraper.scrape()
print(f"Processed {results['total_processed']} DPRs, saved {results['files_saved']} files")
```

## Output Structure

Files are stored in GCS with the following naming convention:

```
gs://loomy-public-documents/dpr/
├── dpr_04_dicembre_2025_n_205.txt
├── dpr_15_novembre_2025_n_180.txt
└── ...
```

Each file contains:
- Document metadata (title, subtitle, URL)
- Full article text extracted from Normattiva

## LLM Relevance Check

The scraper uses Amazon Nova Micro to determine if a DPR is relevant for accountants. The prompt asks the LLM to evaluate based on the DPR subtitle whether it would be useful for consulting or accounting activities.

## Logging

Set `--log-level DEBUG` for verbose output including:
- Navigation steps
- LLM responses
- File operations
- Error details

## Docker

Build and run with Docker:

```bash
# Build from repository root
docker build -f kb/scraping/dpr/Dockerfile -t dpr-scraper .

# Run
docker run \
  -e GCP_SERVICE_ACCOUNT_CREDENTIALS="$GCP_SERVICE_ACCOUNT_CREDENTIALS" \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e AWS_BEDROCK_REGION="eu-central-1" \
  dpr-scraper
```

## Notes

- The scraper uses headless Chrome via Selenium for JavaScript-heavy Normattiva pages
- Documents marked as "NON ANCORA ESISTENTE O VIGENTE" are skipped
- The LLM call uses prompt caching for efficiency when processing multiple documents
