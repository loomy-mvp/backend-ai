"""
Simple runner script for Cloud Run Jobs
Runs the DPR (Decreto del Presidente della Repubblica) scraper and exits
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

from src.scrape_dpr import DPRScraper

# Logger will be configured in main after parsing arguments
logger = logging.getLogger(__name__)


def main(bucket_name: str, base_folder: str, log_level: str, limit: int = 0, start_page: int = 1, use_stop_condition: bool = True):
    """Run the scraper with configuration"""
    
    logger.info("🚀 Starting DPR (Decreto del Presidente della Repubblica) scraper")
    logger.info(f"📦 Target bucket: {bucket_name}")
    logger.info(f"📁 Base folder: {base_folder}")
    logger.info(f"📊 Log level: {log_level}")
    if limit > 0:
        logger.info(f"🔢 Limit: {limit} entries")
    if start_page > 1:
        logger.info(f"📄 Starting from page: {start_page}")
    if not use_stop_condition:
        logger.info("⚠️  Stop condition DISABLED - will process all entries")
    
    try:
        # Initialize scraper
        scraper = DPRScraper(
            bucket_name=bucket_name,
            base_folder=base_folder,
            limit=limit,
            start_page=start_page,
            use_stop_condition=use_stop_condition
        )
        
        # Run scraper
        results = scraper.scrape()
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("✅ SCRAPING COMPLETED")
        logger.info("="*80)
        logger.info(f"📄 Total DPRs found: {results['total_found']}")
        logger.info(f"🔍 DPRs checked for relevance: {results['total_checked']}")
        logger.info(f"✨ Relevant DPRs: {results['relevant_count']}")
        logger.info(f"📁 Files saved: {results['files_saved']}")
        logger.info(f"⏭️  Skipped (already exists): {results['skipped_existing']}")
        logger.info(f"⏭️  Skipped (not yet published): {results['skipped_not_published']}")
        logger.info(f"🗄️  Storage location: gs://{bucket_name}/{base_folder}/")
        logger.info("="*80)
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the DPR scraper and upload files to GCS"
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
        default="dpr",
        help="Base folder path in GCS bucket (default: dpr)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of entries to process (0 = no limit, for testing)"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Start from page N (for testing older published entries)"
    )
    parser.add_argument(
        "--use-stop-condition",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Stop when existing entry found in GCS (default: True)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    main(
        bucket_name=args.bucket_name,
        base_folder=args.base_folder,
        log_level=args.log_level,
        limit=args.limit,
        start_page=args.start_page,
        use_stop_condition=args.use_stop_condition
    )
