"""Scraper for the CCIA Modena "Statuto e Regolamenti" documents."""

from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from google.cloud import storage
from google.oauth2 import service_account
from PyPDF2 import PdfReader

try:  # Prefer the Docker/Cloud Run layout first
    from src.utils.upload_to_storage import upload_to_storage
except ModuleNotFoundError:  # Fall back to shared utils when running locally
    import sys

    SCRAPING_ROOT = Path(__file__).resolve().parents[2]
    UTILS_DIR = SCRAPING_ROOT / "utils"
    if UTILS_DIR.exists():
        sys.path.append(str(SCRAPING_ROOT))
        from utils.upload_to_storage import upload_to_storage  # type: ignore
    else:  # pragma: no cover - only hit when repo layout changes
        raise

BASE_URL = "https://www.mo.camcom.it"
ATTI_GENERALI_PATH = "/amministrazione-trasparente/disposizioni-generali/atti-generali"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "it,en-US;q=0.7,en;q=0.3",
}

DocumentLink = Tuple[str, str]


def slugify(text: str, max_length: int = 64) -> str:
    """Convert arbitrary text into a filename-friendly slug."""

    text = text.lower().strip()
    try:
        from unidecode import unidecode  # type: ignore

        text = unidecode(text)
    except Exception:
        text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_length]


class ModenaStatutoScraper:
    """Scrapes PDF summaries and uploads extracted text to GCS."""

    def __init__(
        self,
        bucket_name: str,
        base_folder: str = "ccia-modena/statuto-regolamenti",
        storage_client: Optional[storage.Client] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.base_folder = base_folder.strip("/") or "ccia-modena"
        self.storage_client = storage_client or self._build_storage_client()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def _build_storage_client(self) -> storage.Client:
        creds_env = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
        if not creds_env:
            return storage.Client()
        credentials_info = json.loads(creds_env)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return storage.Client(credentials=credentials)

    def scrape(self) -> Dict[str, int]:
        summary: Dict[str, int] = {
            "documents_found": 0,
            "documents_processed": 0,
            "download_failures": 0,
            "empty_documents": 0,
            "documents_uploaded": 0,
            "upload_failures": 0,
        }
        with requests.Session() as session:
            session.headers.update(DEFAULT_HEADERS)
            links = self._get_statuto_links(session)
            summary["documents_found"] = len(links)
            for title, url in links:
                summary["documents_processed"] += 1
                self.logger.info("Processing '%s'", title)
                pdf_data = self._fetch_pdf(session, url)
                if not pdf_data:
                    summary["download_failures"] += 1
                    continue
                text_content = self._extract_text_from_pdf(pdf_data)
                if not text_content.strip():
                    summary["empty_documents"] += 1
                    continue
                filename = self._build_filename(title, url)
                try:
                    self._upload_text_content(text_content, filename)
                    summary["documents_uploaded"] += 1
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("Failed to upload '%s': %s", filename, exc)
                    summary["upload_failures"] += 1
        return summary

    def _get_statuto_links(self, session: requests.Session) -> List[DocumentLink]:
        url = f"{BASE_URL}{ATTI_GENERALI_PATH}"
        self.logger.info("Fetching index page: %s", url)
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        heading = soup.find(
            lambda tag: tag.name and tag.get_text(strip=True).startswith("Statuto e Regolamenti")
        )
        if not heading:
            raise RuntimeError("Could not locate the 'Statuto e Regolamenti' section on the page.")
        links: List[DocumentLink] = []
        for sibling in heading.find_all_next():
            if sibling.name and sibling.name.startswith("h"):
                break
            for anchor in sibling.find_all("a", href=True):
                href = anchor["href"].strip()
                if "/allegati/" not in href:
                    continue
                full_url = href if href.startswith("http") else f"{BASE_URL}{href}"
                title = anchor.get_text(strip=True) or slugify(Path(href).stem)
                links.append((title, full_url))
        unique_links: List[DocumentLink] = []
        seen: set[str] = set()
        for title, href in links:
            if href in seen:
                continue
            unique_links.append((title, href))
            seen.add(href)
        self.logger.info("Found %d candidate documents", len(unique_links))
        return unique_links

    def _fetch_pdf(self, session: requests.Session, page_url: str) -> Optional[bytes]:
        self.logger.debug("Accessing summary page: %s", page_url)
        try:
            page_resp = session.get(page_url, timeout=30)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Error fetching page %s: %s", page_url, exc)
            return None
        download_url = page_url.rstrip("/") + "/@@download/file"
        try:
            resp = session.get(download_url, timeout=60)
            if resp.ok and resp.headers.get("content-type", "").lower().startswith("application/pdf"):
                return resp.content
        except Exception:  # noqa: BLE001
            pass
        soup = BeautifulSoup(page_resp.text, "html.parser")
        embed = soup.find("embed", attrs={"internalid": True})
        if embed:
            internal_id = embed.get("internalid")
            if internal_id:
                resolve_url = f"{BASE_URL}/resolveuid/{internal_id}"
                try:
                    resolved = session.get(resolve_url, timeout=60)
                    if (
                        resolved.ok
                        and resolved.headers.get("content-type", "").lower().startswith("application/pdf")
                    ):
                        return resolved.content
                except Exception:  # noqa: BLE001
                    pass
        self.logger.warning("Failed to retrieve PDF content for %s", page_url)
        return None

    def _extract_text_from_pdf(self, data: bytes) -> str:
        output: List[str] = []
        with io.BytesIO(data) as pdf_io:
            reader = PdfReader(pdf_io)
            for page_index, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        output.append(page_text)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Unable to extract text from page %s: %s", page_index, exc)
        return "\n\n".join(output)

    def _build_filename(self, title: str, link: str) -> str:
        candidate: Sequence[str] = [title, Path(urlparse(link).path).stem, "document"]
        for value in candidate:
            slug = slugify(value or "")
            if slug:
                return f"{slug}.txt"
        return "document.txt"

    def _upload_text_content(self, text: str, filename: str) -> None:
        pdf_obj = {
            "name": filename,
            "bytes": text.encode("utf-8"),
            "extension": "txt",
        }
        upload_to_storage(
            storage_client=self.storage_client,
            bucket_name=self.bucket_name,
            pdf_obj=pdf_obj,
            folder=self.base_folder,
        )
