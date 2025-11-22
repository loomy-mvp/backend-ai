"""Compatibility runner that delegates to the packaged scraper."""

from __future__ import annotations

import json
import logging
import os
import sys

from src.scrape_modena import ModenaStatutoScraper


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    bucket_name = os.getenv("GCS_BUCKET_NAME", "loomy-public-documents")
    base_folder = os.getenv("CCIAA_MODENA_BASE_FOLDER", "cciaa/cciaa-modena/statuto-regolamenti")
    scraper = ModenaStatutoScraper(bucket_name=bucket_name, base_folder=base_folder)
    summary = scraper.scrape()
    logging.info("Summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
