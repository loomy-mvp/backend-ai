"""
Simple runner script for Cloud Run Jobs
Runs the Agenzia Entrate scraper and exits
"""

import argparse
import asyncio
import logging
import sys

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # dotenv not installed - will use system environment variables
    pass

from src.scrape_risoluzioni import RisoluzioniScraper

# Logger will be configured in main after parsing arguments
logger = logging.getLogger(__name__)


async def main(bucket_name: str, base_folder: str, max_concurrent: int, log_level: str):
    """Run the scraper with configuration"""
    
    logger.info(f"🚀 Starting Agenzia Entrate scraper")
    logger.info(f"📦 Target bucket: {bucket_name}")
    logger.info(f"📁 Base folder: {base_folder}")
    logger.info(f"🔧 Max concurrent requests: {max_concurrent}")
    logger.info(f"📊 Log level: {log_level}")
    
    try:
        # Initialize scraper
        scraper = RisoluzioniScraper(
            bucket_name=bucket_name,
            max_concurrent=max_concurrent,
            base_folder=base_folder
        )
        
        # Run scraper
        results = await scraper.scrape()
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("✅ SCRAPING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"📄 Total pages visited: {results['pages_visited']}")
        logger.info(f"📁 Total files processed: {results['files_downloaded']}")
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
        description="Run the Agenzia Entrate scraper and upload files to GCS"
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
        default="provvedimenti_non_soggetti_a_pubblicita",
        help="Base folder path in GCS bucket (default: provvedimenti_non_soggetti_a_pubblicita)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Number of concurrent HTTP requests (default: 10, recommended: 5-20)"
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
    
    asyncio.run(main(args.bucket_name, args.base_folder, args.max_concurrent, args.log_level))
