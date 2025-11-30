"""
Simple runner script for Cloud Run Jobs
Runs the NT+ Fisco scraper and exits

Environment Variables:
    ONLY_SECTIONS: Comma-separated list of section names to scrape
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
    pass

from src.scrape_nt_fisco import NTFiscoScraper, SECTIONS

logger = logging.getLogger(__name__)


# Hardcoded local output folder
LOCAL_OUTPUT_FOLDER = "c:/Users/leoac/Work/Companies/Loomy (personal)/loomy/kb/scraping/output"


def main(
    bucket_name: str,
    base_folder: str,
    headless: bool,
    max_articles: int,
    max_scrolls: int,
    delay: float,
    log_level: str,
    only_sections: list = None,
    save_local: bool = False
):
    """Run the scraper with configuration"""
    
    logger.info(f"🚀 Starting NT+ Fisco scraper")
    if save_local:
        logger.info(f"💾 Saving locally to: {LOCAL_OUTPUT_FOLDER}/{base_folder}/")
    else:
        logger.info(f"📦 Target bucket: {bucket_name}")
    logger.info(f"📁 Base folder: {base_folder}")
    logger.info(f"🖥️  Headless mode: {headless}")
    logger.info(f"📄 Max articles per section: {max_articles if max_articles > 0 else 'unlimited'}")
    logger.info(f"📜 Max scrolls per section: {max_scrolls}")
    logger.info(f"⏱️  Delay between actions: {delay}s")
    logger.info(f"📊 Log level: {log_level}")
    if only_sections:
        logger.info(f"🎯 Processing only sections: {', '.join(only_sections)}")
    
    try:
        # Initialize scraper
        scraper = NTFiscoScraper(
            bucket_name=bucket_name,
            base_folder=base_folder,
            headless=headless,
            max_articles=max_articles,
            max_scrolls=max_scrolls,
            delay=delay,
            save_local=save_local,
            local_output_folder=LOCAL_OUTPUT_FOLDER if save_local else None,
        )
        
        # Run scraper
        results = scraper.scrape(only_sections=only_sections)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("✅ SCRAPING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"📂 Total sections processed: {results['sections']}")
        logger.info(f"📰 Total articles scraped: {results['articles_scraped']}")
        logger.info(f"📤 Total articles uploaded: {results['articles_uploaded']}")
        logger.info(f"⏭️  Total articles skipped: {results['articles_skipped']}")
        logger.info(f"❌ Total errors: {results['errors']}")
        if save_local:
            logger.info(f"🗄️  Storage location: {LOCAL_OUTPUT_FOLDER}/{base_folder}/")
        else:
            logger.info(f"🗄️  Storage location: gs://{bucket_name}/{base_folder}/")
        logger.info("="*80)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the NT+ Fisco scraper and upload articles to GCS"
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
        default="nt_fisco",
        help="Base folder path in GCS bucket (default: nt_fisco)"
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
        "--max-articles",
        type=int,
        default=0,
        help="Maximum articles per section (0 = unlimited)"
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=100,
        help="Maximum scroll operations per section (default: 100)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between actions in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help="Sections separated by |SEP| (e.g., 'Imposte|SEP|Finanza|SEP|Diritto')"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        default=False,
        help="Save files locally to kb/scraping/output instead of GCS"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    # Validate bucket name
    if not args.bucket_name:
        logger.error("❌ Bucket name must be provided via --bucket-name argument")
        sys.exit(1)
    
    # Handle sections from --sections argument
    sections_list = None
    
    if args.sections:
        sections_str = args.sections.strip().strip('"').strip("'")
        if sections_str:
            sections_list = [s.strip() for s in sections_str.split("|SEP|") if s.strip()]
            
            # Validate section names
            valid_sections = list(SECTIONS.keys())
            for section in sections_list:
                if section not in valid_sections:
                    logger.warning(f"⚠️ Unknown section: '{section}'. Valid sections: {valid_sections}")
    
    # Also check environment variable
    if not sections_list:
        env_sections = os.environ.get("ONLY_SECTIONS", "").strip()
        if env_sections:
            sections_list = [s.strip() for s in env_sections.split(",") if s.strip()]
    
    main(
        bucket_name=args.bucket_name,
        base_folder=args.base_folder,
        headless=args.headless,
        max_articles=args.max_articles,
        max_scrolls=args.max_scrolls,
        delay=args.delay,
        log_level=args.log_level,
        only_sections=sections_list,
        save_local=args.save_local
    )
