"""Scraper for Agenzia Entrate agevolazioni pages.

This scraper downloads the textual content for every agevolazione listed on the
cittadini portal page and uploads each entry as a UTF-8 ``.txt`` file to Google
Cloud Storage. The stop condition mirrors the one used in the circolari job: the
scraper halts immediately after encountering the first file that already exists
in the destination folder, allowing incremental runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from google.cloud import storage  # type: ignore
from google.oauth2 import service_account  # type: ignore

try:
    from src.utils.upload_to_storage import upload_to_storage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local execution fallback
    import sys

    UTILS_PATH = Path(__file__).resolve().parents[2] / "utils"
    if str(UTILS_PATH) not in sys.path:
        sys.path.append(str(UTILS_PATH))
    from upload_to_storage import upload_to_storage  # type: ignore

logger = logging.getLogger(__name__)


class AgevolazioniScraper:
    """Scrape textual agevolazioni content and upload it to GCS."""

    BASE_URL = "https://www.agenziaentrate.gov.it"
    START_URL = f"{BASE_URL}/portale/cittadini/agevolazioni"

    def __init__(
        self,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
        max_concurrent: int = 5,
        base_folder: str = "agevolazioni",
    ) -> None:
        self.bucket_name = bucket_name
        self.base_folder = base_folder.strip("/") or "agevolazioni"
        self.storage_client = storage_client or self._build_storage_client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.max_concurrent = max_concurrent
        self.stop_requested = False
        self.fetched_links: List[Dict[str, str]] = []
        self.generated_files: Dict[str, str] = {}
        self.pages_processed = 0
        self.files_uploaded = 0

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a page and return its HTML."""

        try:
            logger.debug("Fetching %s", url)
            async with session.get(url, timeout=60) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Error fetching %s: %s", url, exc)
            return None

    async def fetch_agevolazioni_links(self, session: aiohttp.ClientSession) -> List[Dict[str, str]]:
        """Return the ordered list of agevolazioni links from the landing page."""

        html = await self.fetch_page(session, self.START_URL)
        if not html:
            return []
        soup = BeautifulSoup(html, "lxml")
        sections = soup.find_all(
            "section", id=lambda value: value and value.startswith("portlet_PageCategoryFilter_INSTANCE")
        )
        target_section = sections[0] if sections else soup
        links: List[Dict[str, str]] = []
        for anchor in target_section.select("ul.link-list.page-filter li a[href]"):
            text = anchor.get_text(strip=True)
            href = anchor.get("href")
            if not text or not href:
                continue
            absolute_url = urljoin(self.BASE_URL, href)
            links.append({"title": text, "url": absolute_url})
        self.fetched_links = links
        logger.info("Discovered %d agevolazioni entries", len(links))
        return links

    @staticmethod
    def _slugify(title: str) -> str:
        normalized = unicodedata.normalize("NFKD", title)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-")
        return cleaned.lower() or "agevolazione"

    def _build_blob_path(self, filename: str) -> str:
        if self.base_folder:
            return f"{self.base_folder}/{filename}"
        return filename

    def _already_uploaded(self, filename: str) -> bool:
        blob_path = self._build_blob_path(filename)
        exists = self.bucket.get_blob(blob_path) is not None
        if exists:
            logger.info("🛑 Found existing blob for '%s'; stopping scraper.", blob_path)
        return exists

    @staticmethod
    def _build_storage_client() -> storage.Client:
        """Return a GCS client honoring explicit env credentials when available."""

        gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
        if gcp_credentials_info:
            logger.info("Using GCP credentials from environment variable")
            credentials_dict = json.loads(gcp_credentials_info)
            gcp_credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            return storage.Client(credentials=gcp_credentials)
        logger.info("Using default GCP credentials (Cloud Run service account)")
        return storage.Client()

    async def upload_text(self, text: str, filename: str) -> bool:
        blob_path = self._build_blob_path(filename)
        folder_path = str(Path(blob_path).parent)
        if folder_path == ".":
            folder_path = self.base_folder
        folder = folder_path or None
        pdf_obj = {
            "name": Path(filename).name,
            "bytes": text.encode("utf-8"),
            "extension": "txt",
        }
        try:
            upload_to_storage(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                pdf_obj=pdf_obj,
                folder=folder,
            )
            logger.info("✅ Uploaded '%s'", blob_path)
            return True
        except Exception as exc:  # pragma: no cover - GCS issues
            logger.error("Failed to upload %s: %s", blob_path, exc)
            return False

    @staticmethod
    def _extract_text(html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "lxml")
        main = soup.find(attrs={"role": "main"}) or soup
        content = main.select_one("div.journal-content-article") or main
        for tag in content.select("script,style,nav,header,footer,form,aside,button"):
            tag.decompose()
        raw_text = content.get_text("\n", strip=True)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return None
        return "\n".join(lines)

    @staticmethod
    def _format_document(title: str, url: str, body: str) -> str:
        header = [title.strip(), f"Fonte: {url.strip()}"]
        return "\n".join(header + ["", body, ""]).strip() + "\n"

    async def scrape(self) -> Dict[str, object]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        timeout = aiohttp.ClientTimeout(total=120)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
            links = await self.fetch_agevolazioni_links(session)
            for index, item in enumerate(links, start=1):
                if self.stop_requested:
                    break
                title = item["title"].strip()
                url = item["url"].strip()
                filename = f"{self._slugify(title)}.txt"
                if self._already_uploaded(filename):
                    self.stop_requested = True
                    break
                html = await self.fetch_page(session, url)
                if not html:
                    continue
                body = self._extract_text(html)
                if not body:
                    logger.warning("Skipping '%s' because no textual content was extracted", title)
                    continue
                document_text = self._format_document(title, url, body)
                uploaded = await self.upload_text(document_text, filename)
                self.pages_processed += 1
                if uploaded:
                    self.files_uploaded += 1
                    blob_path = self._build_blob_path(filename)
                    self.generated_files[title] = blob_path
                logger.debug("Processed %s (%d/%d)", title, index, len(links))

        logger.info(
            "Scraping finished: %d processed, %d uploaded, stop_requested=%s",
            self.pages_processed,
            self.files_uploaded,
            self.stop_requested,
        )
        return {
            "links_discovered": len(self.fetched_links),
            "pages_processed": self.pages_processed,
            "files_uploaded": self.files_uploaded,
            "stop_requested": self.stop_requested,
            "uploaded_files": self.generated_files,
        }


async def main() -> None:
    bucket_name = "loomy-public-documents"
    scraper = AgevolazioniScraper(bucket_name=bucket_name)
    await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
