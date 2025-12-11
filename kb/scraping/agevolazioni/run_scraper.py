"""Runner script for the Agenzia Entrate agevolazioni scraper."""

import argparse
import asyncio
import logging
import sys

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:  # pragma: no cover - optional dependency
    pass

from src.scrape_agevolazioni import AgevolazioniScraper

logger = logging.getLogger(__name__)


async def main(bucket_name: str, base_folder: str, max_concurrent: int, log_level: str) -> None:
    logger.info("🚀 Starting Agenzia Entrate Agevolazioni scraper")
    logger.info("📦 Target bucket: %s", bucket_name)
    logger.info("📁 Base folder: %s", base_folder)
    logger.info("🔧 Max concurrent requests: %d", max_concurrent)
    logger.info("📊 Log level: %s", log_level)

    try:
        scraper = AgevolazioniScraper(
            bucket_name=bucket_name,
            base_folder=base_folder,
            max_concurrent=max_concurrent,
        )
        results = await scraper.scrape()
        logger.info("\n" + "=" * 80)
        logger.info("✅ AGEVOLAZIONI SCRAPING COMPLETED")
        logger.info("=" * 80)
        logger.info("🔗 Links discovered: %d", results.get("links_discovered", 0))
        logger.info("📄 Entries processed: %d", results.get("pages_processed", 0))
        logger.info("📁 Files uploaded: %d", results.get("files_uploaded", 0))
        logger.info("🛑 Stop requested: %s", results.get("stop_requested", False))
        logger.info("🗄️  Storage location: gs://%s/%s/", bucket_name, base_folder.strip("/"))
        logger.info("=" * 80)
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.error("❌ Scraping failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Agenzia Entrate agevolazioni scraper and upload files to GCS"
    )
    parser.add_argument(
        "--bucket-name",
        default="loomy-public-documents",
        help="GCS bucket name",
    )
    parser.add_argument(
        "--base-folder",
        default="agevolazioni",
        help="Base folder path in GCS bucket (default: agevolazioni)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Number of concurrent HTTP requests (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    if not args.bucket_name:
        logger.error("❌ Bucket name must be provided via --bucket-name or GCS_BUCKET_NAME")
        sys.exit(1)

    asyncio.run(main(args.bucket_name, args.base_folder, args.max_concurrent, args.log_level))
