"""
Simple runner script for Cloud Run Jobs
Runs the INPS scraper and exits
"""

import argparse
import logging
import os
import sys

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # dotenv not installed - will use system environment variables
    pass

# Fix for Cloud Run: Clear GOOGLE_APPLICATION_CREDENTIALS if file doesn't exist
# This allows Google Cloud SDK to use the default service account credentials
if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
    creds_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    if not os.path.exists(creds_path):
        print(f"Warning: GOOGLE_APPLICATION_CREDENTIALS points to non-existent file: {creds_path}")
        print("Clearing environment variable to use default Cloud Run service account credentials")
        del os.environ['GOOGLE_APPLICATION_CREDENTIALS']

from src.scrape_inps import INPSScraper

# Logger will be configured in main after parsing arguments
logger = logging.getLogger(__name__)


def main(bucket_name: str, base_folder: str, headless: bool, max_pages: int, log_level: str):
    """Run the scraper with configuration"""
    
    logger.info(f"🚀 Starting INPS Circolari e Messaggi scraper")
    logger.info(f"📦 Target bucket: {bucket_name}")
    logger.info(f"📁 Base folder: {base_folder}")
    logger.info(f"🖥️  Headless mode: {headless}")
    logger.info(f"📄 Max pages: {max_pages}")
    logger.info(f"📊 Log level: {log_level}")
    
    try:
        # Initialize scraper
        scraper = INPSScraper(
            bucket_name=bucket_name,
            base_folder=base_folder,
            headless=headless,
            max_pages=max_pages,
        )
        
        # Run scraper
        results = scraper.scrape()
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("✅ SCRAPING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"📄 Total pages visited: {results['pages_visited']}")
        logger.info(f"📁 Total files downloaded: {results['files_downloaded']}")
        logger.info(f"❌ Total errors: {results['files_errored']}")
        logger.info(f"🗄️  Storage location: gs://{bucket_name}/{base_folder}/")
        
        # Print files by year
        if results.get('files_by_year'):
            logger.info("\n📊 Files downloaded per year:")
            sorted_years = sorted(results['files_by_year'].items(), key=lambda x: x[0], reverse=True)
            for year, count in sorted_years:
                logger.info(f"  {year}: {count:3d} files")
        
        logger.info("="*80)
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the INPS scraper and upload files to GCS"
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="loomy-public-documents",
        help="GCS bucket name"
    )
    parser.add_argument(
        "--base-folder",
        type=str,
        default="inps",
        help="Base folder path in GCS bucket (default: inps)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)"
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run browser with visible UI"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="Maximum number of pages to scrape (default: 1000)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging with the specified level
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )
    
    # Validate bucket name
    if not args.bucket_name:
        logger.error("❌ Bucket name must be provided via --bucket-name argument or GCS_BUCKET_NAME environment variable")
        sys.exit(1)
    
    main(args.bucket_name, args.base_folder, args.headless, args.max_pages, args.log_level)
