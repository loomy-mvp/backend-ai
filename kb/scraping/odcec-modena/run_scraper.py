"""
Simple runner script for Cloud Run Jobs
Runs the ODCEC Modena scraper and exits
"""

import argparse
import logging
import sys

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # dotenv not installed - will use system environment variables
    pass

from src.scrape_odcec import ODCECModenaScraper

# Logger will be configured in main after parsing arguments
logger = logging.getLogger(__name__)


def main(
    bucket_name: str,
    base_folder: str,
    log_level: str,
    headless: bool = True
):
    """Run the scraper with configuration"""
    
    logger.info(f"🚀 Starting ODCEC Modena scraper")
    logger.info(f"📦 Target bucket: {bucket_name}")
    logger.info(f"📁 Base folder: {base_folder}")
    logger.info(f"📊 Log level: {log_level}")
    logger.info(f"🖥️  Headless mode: {headless}")
    
    try:
        # Initialize scraper
        scraper = ODCECModenaScraper(
            bucket_name=bucket_name,
            base_folder=base_folder,
            headless=headless
        )
        
        # Run scraper
        results = scraper.scrape()
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("✅ SCRAPING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"📄 Total pages processed: {results['pages_processed']}")
        logger.info(f"📄 Total files uploaded: {results['pdf_uploaded']}")
        logger.info(f"⏭️  Total files skipped: {results['pdf_skipped']}")
        logger.info(f"❌ Total errors: {results['pdf_errors']}")
        logger.info(f"🗄️  Storage location: gs://{bucket_name}/{base_folder}/")
        logger.info("="*80)
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ODCEC Modena scraper and upload files to GCS"
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
        default="odcec-modena",
        help="Base folder path in GCS bucket"
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
        logger.error("❌ Bucket name must be provided via --bucket-name argument")
        sys.exit(1)
    
    main(
        bucket_name=args.bucket_name,
        base_folder=args.base_folder,
        log_level=args.log_level,
        headless=args.headless
    )
