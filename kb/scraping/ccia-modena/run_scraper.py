"""Runner script for the CCIA Modena Statuto scraper."""

import argparse
import logging
import sys

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:  # pragma: no cover - optional dependency
    pass

from src.scrape_modena import ModenaStatutoScraper

logger = logging.getLogger(__name__)


def main(bucket_name: str, base_folder: str, log_level: str) -> int:
    logger.info("🚀 Starting CCIA Modena Statuto scraper")
    logger.info("📦 Target bucket: %s", bucket_name)
    logger.info("📁 Base folder: %s", base_folder)
    logger.info("📊 Log level: %s", log_level)

    try:
        scraper = ModenaStatutoScraper(bucket_name=bucket_name, base_folder=base_folder)
        results = scraper.scrape()

        logger.info("\n" + "=" * 80)
        logger.info("✅ SCRAPING COMPLETED")
        logger.info("=" * 80)
        for key, value in results.items():
            logger.info("%s: %s", key, value)
        logger.info("=" * 80)
        logger.info("🗄️  Storage location: gs://%s/%s/", bucket_name, base_folder)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ Scraping failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CCIA Modena Statuto scraper")
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="loomy-public-documents",
        help="Target GCS bucket (default: loomy-public-documents)",
    )
    parser.add_argument(
        "--base-folder",
        type=str,
        default="ccia-modena/statuto-regolamenti",
        help="Folder inside the bucket where text files will be stored",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    sys.exit(main(args.bucket_name, args.base_folder.strip("/"), args.log_level))
