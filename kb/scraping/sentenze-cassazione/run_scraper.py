"""Cloud Run entrypoint for the ForoEuropeo Cassazione scraper."""

import argparse
import logging
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:  # pragma: no cover - optional dependency
    pass

from src.scrape_sentenze import ForoEuropeoScraper

logger = logging.getLogger(__name__)


def main(
    bucket_name: str,
    base_folder: str,
    limit_categories: int | None,
    request_delay: float,
    log_level: str,
) -> None:
    logger.info("🚀 Starting ForoEuropeo Cassazione scraper")
    logger.info("📦 Bucket: %s", bucket_name)
    logger.info("📁 Base folder: %s", base_folder)
    if limit_categories is not None:
        logger.info("📊 Limit categories: %s", limit_categories)
    logger.info("⏱️  Delay between requests: %.2fs", request_delay)
    logger.info("🔍 Log level: %s", log_level)

    try:
        scraper = ForoEuropeoScraper(
            bucket_name=bucket_name,
            base_folder=base_folder,
            request_delay=request_delay,
        )
        results = scraper.scrape(limit_categories=limit_categories)
    except Exception as exc:
        logger.error("❌ Scraping failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("✅ SCRAPING COMPLETED")
    logger.info("=" * 80)
    logger.info("📚 Categories processed: %d", results["categories_processed"])
    logger.info("📄 Articles discovered: %d", results["articles_discovered"])
    logger.info("📁 Files uploaded: %d", results["files_uploaded"])
    logger.info("🔁 Duplicates skipped: %d", results["duplicates_skipped"])
    logger.info("⚠️  Articles failed: %d", results["articles_failed"])
    logger.info("🗄️  Storage location: gs://%s/%s", bucket_name, base_folder)

    if results["files_by_category"]:
        logger.info("\n📊 Files uploaded per category:")
        for category, count in results["files_by_category"].items():
            logger.info("  %-40s %5d", category, count)

    logger.info("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ForoEuropeo Cassazione scraper and upload files to GCS",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="loomy-public-documents",
        help="GCS bucket name",
    )
    parser.add_argument(
        "--base-folder",
        type=str,
        default="sentenze_cassazione",
        help="Folder inside the bucket where files will be stored",
    )
    parser.add_argument(
        "--limit-categories",
        type=int,
        default=None,
        help="If provided, only process the first N categories",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.5,
        help="Seconds to wait between HTTP requests (default: 0.5)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
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
        logger.error("Bucket name is required via --bucket-name")
        sys.exit(1)

    main(
        bucket_name=args.bucket_name,
        base_folder=args.base_folder,
        limit_categories=args.limit_categories,
        request_delay=args.request_delay,
        log_level=args.log_level,
    )
