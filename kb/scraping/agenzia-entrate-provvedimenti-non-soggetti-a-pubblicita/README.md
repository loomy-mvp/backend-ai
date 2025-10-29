# Agenzia Entrate Scraper

A production-ready web scraper that automatically downloads and archives Italian tax documents (provvedimenti non soggetti a pubblicità) from the Agenzia Entrate website to Google Cloud Storage.

## 📋 Overview

This scraper navigates through the Agenzia Entrate archive of "provvedimenti non soggetti a pubblicità" (measures not subject to publicity), downloading PDF, ZIP, XLSX, XML, XLS, and XSD files while preserving the original hierarchical folder structure by year and month.

### Key Features

- **Async Architecture**: High-performance concurrent downloads using `aiohttp`
- **Smart Crawling**: Intelligently follows only relevant archive links
- **Hierarchical Organization**: Files stored in `year/month/filename` structure
- **Cloud Storage**: Direct upload to Google Cloud Storage (GCS)
- **Retry Logic**: Automatic retries with exponential backoff for failed downloads
- **Progress Tracking**: Real-time logging with download statistics
- **Rate Limiting**: Configurable concurrent request limits to avoid overwhelming servers
- **Duplicate Prevention**: Tracks visited URLs and downloaded files to avoid duplicates
- **Descriptive Filenames**: Uses link text from HTML as filenames for better organization
- **Docker Support**: Ready for deployment as a Google Cloud Run Job

## 🏗️ Architecture

```
┌─────────────────┐
│  Cloud Run Job  │
│   (Scheduled)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────┐
│  AgenziaEntrate │──────▶│  Agenzia Entrate │
│     Scraper     │       │     Website      │
└────────┬────────┘       └──────────────────┘
         │
         ▼
┌─────────────────┐
│ Google Cloud    │
│ Storage Bucket  │
│                 │
│ /agenzia_entrate│
│   /2024         │
│     /Gennaio    │
│     /Febbraio   │
│   /2023         │
│     /Dicembre   │
│     ...         │
└─────────────────┘
```

## 📁 Project Structure

```
agenzia-entrate-provvedimenti-non-soggetti-a-pubblicita/
├── README.md                       # This file
├── Dockerfile                      # Container configuration for Cloud Run
├── run_scraper.py                  # Entry point script
└── src/
    ├── __init__.py
    ├── config.py                   # Configuration settings
    ├── requirements.txt            # Python dependencies
    └── scrape_provvedimenti_non_soggetti_a_pubblicita.py   # Core scraper logic
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud Project with:
  - Cloud Storage bucket created
  - Service account with Storage Admin permissions
  - Credentials configured (see Authentication section)

### Installation

1. **Clone the repository** (if not already done)

2. **Install dependencies**:
   ```powershell
   cd kb\scraping\agenzia-entrate-provvedimenti-non-soggetti-a-pubblicita
   pip install -r src\requirements.txt
   ```

3. **Configure authentication** (see [Authentication](#-authentication) section)

4. **Set environment variables**:
   ```powershell
   $env:GCS_BUCKET_NAME = "your-bucket-name"
   $env:GOOGLE_APPLICATION_CREDENTIALS = "path\to\service-account-key.json"
   ```

### Running Locally

```powershell
# Basic usage
python run_scraper.py --bucket-name your-bucket-name

# With custom base folder
python run_scraper.py --bucket-name your-bucket-name --base-folder my-custom-folder

# Using environment variable for bucket name
$env:GCS_BUCKET_NAME = "your-bucket-name"
python run_scraper.py
```

### Command Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--bucket-name` | GCS bucket name for storing files | `$env:GCS_BUCKET_NAME` | Yes |
| `--base-folder` | Base folder path in GCS bucket | `agenzia_entrate` | No |

## 🔐 Authentication

### Google Cloud Authentication

The scraper needs access to Google Cloud Storage. Choose one method:

#### Method 1: Service Account Key (Recommended for local development)

1. Create a service account in Google Cloud Console
2. Grant it "Storage Object Admin" role on your bucket
3. Download the JSON key file
4. Set environment variable:
   ```powershell
   $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account-key.json"
   ```

#### Method 2: Application Default Credentials (Cloud Run)

For Cloud Run deployments, attach the service account directly to the Cloud Run job:
```bash
gcloud run jobs create scraper-job \
  --image gcr.io/PROJECT_ID/scraper \
  --service-account SERVICE_ACCOUNT_EMAIL
```

## ⚙️ Configuration

### Scraper Settings

Edit `src/config.py` to customize:

```python
# GCS Configuration
GCS_BASE_FOLDER = "agenzia_entrate"  # Base folder in bucket

# Performance tuning
MAX_CONCURRENT_REQUESTS = 10  # 5-20 recommended

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
```

### File Types

The scraper downloads files with these extensions:
- `.pdf` - PDF documents
- `.zip` - Archived files
- `.xlsx` / `.xls` - Excel spreadsheets
- `.xml` - XML documents
- `.xsd` - XML Schema definitions

### Storage Structure

Files are stored in GCS with the following hierarchy:

```
gs://your-bucket/agenzia_entrate/
├── 2024/
│   ├── Gennaio/
│   │   ├── Provvedimento_123.pdf
│   │   └── Documento_XYZ.xml
│   ├── Febbraio/
│   │   └── ...
│   └── ...
├── 2023/
│   ├── Dicembre/
│   └── ...
└── ...
```

## 🐳 Docker Deployment

### Build the Docker Image

```powershell
# From the project root directory
docker build -t agenzia-entrate-scraper -f kb\scraping\agenzia-entrate-provvedimenti-non-soggetti-a-pubblicita\Dockerfile .
```

### Run with Docker

```powershell
docker run -e GCS_BUCKET_NAME=your-bucket-name `
           -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-key.json `
           -v C:\path\to\key.json:/secrets/gcs-key.json `
           agenzia-entrate-scraper
```

### Deploy to Google Cloud Run

1. **Build and push to Google Container Registry**:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/agenzia-entrate-scraper
   ```

2. **Create a Cloud Run Job**:
   ```bash
   gcloud run jobs create agenzia-entrate-scraper \
     --image gcr.io/PROJECT_ID/agenzia-entrate-scraper \
     --service-account SERVICE_ACCOUNT_EMAIL \
     --set-env-vars GCS_BUCKET_NAME=your-bucket-name \
     --max-retries 2 \
     --task-timeout 1h \
     --memory 2Gi \
     --cpu 2
   ```

3. **Schedule with Cloud Scheduler**:
   ```bash
   gcloud scheduler jobs create http agenzia-entrate-weekly \
     --schedule="0 2 * * 0" \
     --uri="https://RUN_JOBS_API_ENDPOINT" \
     --http-method=POST \
     --oauth-service-account-email=SERVICE_ACCOUNT_EMAIL
   ```

## 📊 Output and Monitoring

### Console Output

The scraper provides detailed logging:

```
2024-10-24 10:30:15 - INFO - 🚀 Starting Agenzia Entrate scraper
2024-10-24 10:30:15 - INFO - 📦 Target bucket: your-bucket-name
2024-10-24 10:30:15 - INFO - 📁 Base folder: agenzia_entrate
2024-10-24 10:30:16 - INFO - 📄 Crawling page (1 visited): https://...
2024-10-24 10:30:17 - INFO - Found 45 files and 12 navigation links
2024-10-24 10:30:17 - INFO - ⬇️  Starting download of 45 files from this page
2024-10-24 10:30:18 - INFO - ✅ Uploaded to GCS: agenzia_entrate/2024/Gennaio/Provv_123.pdf
...
================================================================================
✅ SCRAPING COMPLETED SUCCESSFULLY
================================================================================
📄 Total pages visited: 127
📁 Total files processed: 1,234
🗄️  Storage location: gs://your-bucket-name/agenzia_entrate/

📊 Files downloaded per year:
  2024:  156 files
  2023:  342 files
  2022:  298 files
  2021:  245 files
  2020:  193 files
================================================================================
```

### Exit Codes

- `0` - Successful execution
- `1` - Error occurred during scraping

## 🔧 Troubleshooting

### Common Issues

#### "Bucket name must be provided"
**Solution**: Set the `GCS_BUCKET_NAME` environment variable or use `--bucket-name` argument.

#### "Permission denied" errors
**Solution**: Ensure your service account has "Storage Object Admin" role on the bucket.

#### "Connection timeout" errors
**Solution**: Check network connectivity or reduce `MAX_CONCURRENT_REQUESTS` in config.py.

#### "Rate limiting" errors
**Solution**: Decrease `MAX_CONCURRENT_REQUESTS` in config.py (try 5-10).

### Debug Mode

Enable detailed logging:

```powershell
# Set environment variable
$env:LOG_LEVEL = "DEBUG"

# Run scraper
python run_scraper.py --bucket-name your-bucket-name
```

## 🧪 Development

### Project Dependencies

Core libraries:
- **aiohttp** - Async HTTP client
- **beautifulsoup4** - HTML parsing
- **google-cloud-storage** - GCS integration
- **tenacity** - Retry logic
- **python-dotenv** - Environment management (dev only)

### Testing Locally

```powershell
# Install dev dependencies
pip install -r src\requirements.txt

# Run with debug logging
$env:LOG_LEVEL = "DEBUG"
python run_scraper.py --bucket-name test-bucket --base-folder test-folder
```

### Code Structure

- **`AgenziaEntrateScraper`** - Main scraper class
  - `fetch_page()` - Downloads HTML pages
  - `download_file()` - Downloads files with retry logic
  - `upload_to_gcs()` - Uploads to Google Cloud Storage
  - `extract_links()` - Parses HTML for links
  - `crawl_page()` - Recursively crawls pages
  - `should_follow_link()` - Smart link filtering
  - `generate_file_path()` - Creates hierarchical paths

## 📝 Notes

- The scraper preserves the original hierarchy (year/month) from the website
- Filenames are derived from HTML link text for better readability
- All downloads are performed asynchronously for optimal performance
- The scraper respects rate limits to avoid overwhelming the source server
- Duplicate detection prevents re-downloading existing files

## 📚 Resources

- [Agenzia Entrate Website](https://www.agenziaentrate.gov.it)
- [Google Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Google Cloud Run Jobs](https://cloud.google.com/run/docs/create-jobs)
- [aiohttp Documentation](https://docs.aiohttp.org/)

## 📄 License

This scraper is intended for archival and research purposes. Please ensure compliance with Agenzia Entrate's terms of service and robots.txt when using this tool.

---

**Last Updated**: October 2025
