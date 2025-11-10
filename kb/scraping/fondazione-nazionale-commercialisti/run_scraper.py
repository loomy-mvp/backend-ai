"""
Simple runner script for Cloud Run Jobs
Runs the Fondazione Nazionale Commercialisti scraper and exits

Environment Variables:
    ONLY_SECTIONS: Comma-separated list of section names to scrape (alternative to --only-sections)
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

from src.scrape_fondazione_nazionale_commercialisti import FondazioneNazionaleScraper

# Logger will be configured in main after parsing arguments
logger = logging.getLogger(__name__)


def main(
    bucket_name: str,
    base_folder: str,
    delay: float,
    retries: int,
    log_level: str,
    only_sections: list = None,
    max_pages: int = None
):
    """Run the scraper with configuration"""
    
    logger.info(f"🚀 Starting Fondazione Nazionale Commercialisti scraper")
    logger.info(f"📦 Target bucket: {bucket_name}")
    logger.info(f"📁 Base folder: {base_folder}")
    logger.info(f"⏱️  Delay between requests: {delay}s")
    logger.info(f"🔄 HTTP retries: {retries}")
    logger.info(f"📊 Log level: {log_level}")
    if only_sections:
        logger.info(f"🎯 Processing only sections: {', '.join(only_sections)}")
    if max_pages:
        logger.info(f"📄 Max pages per section: {max_pages}")
    
    try:
        # Initialize scraper
        scraper = FondazioneNazionaleScraper(
            bucket_name=bucket_name,
            base_folder=base_folder,
            delay=delay,
            retries=retries
        )
        
        # Run scraper
        results = scraper.scrape(only_sections=only_sections, max_pages=max_pages)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("✅ SCRAPING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"📂 Total sections processed: {results['sections']}")
        logger.info(f"📄 Total files uploaded: {results['pdf_uploaded']}")
        logger.info(f"⏭️  Total files skipped: {results['pdf_skipped']}")
        logger.info(f"❌ Page errors: {results['pages_errors']}")
        logger.info(f"❌ Node errors: {results['nodes_errors']}")
        logger.info(f"🗄️  Storage location: gs://{bucket_name}/{base_folder}/")
        logger.info("="*80)
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Fondazione Nazionale Commercialisti scraper and upload files to GCS"
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
        default="fondazione-nazionale-commercialisti",
        help="Base folder path in GCS bucket"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
        help="Delay (seconds) between requests (default: 2.5)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Number of HTTP retries (default: 6)"
    )
    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help="Sections separated by |SEP| (e.g., 'Section1|SEP|Section2|SEP|Section3')"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of pages per section (0-based). None=all pages"
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
    
    # Handle sections from --sections argument
    sections_list = None
    
    if args.sections:
        # Remove leading/trailing quotes if present (from Cloud Console UI)
        sections_str = args.sections.strip().strip('"').strip("'")
        
        # Parse |SEP| separated string from --sections
        sections_list = [s.strip() for s in sections_str.split('|SEP|') if s.strip()]
        logger.info(f"📋 Using sections from --sections argument: {len(sections_list)} sections")
        logger.debug(f"Sections parsed: {sections_list}")
    
    main(
        bucket_name=args.bucket_name,
        base_folder=args.base_folder,
        delay=args.delay,
        retries=args.retries,
        log_level=args.log_level,
        only_sections=sections_list,
        max_pages=args.max_pages
    )
