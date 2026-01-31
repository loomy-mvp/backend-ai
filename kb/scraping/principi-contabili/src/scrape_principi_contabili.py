"""Scraper for Principi Contabili from Ragioneria Generale dello Stato.

This scraper visits the RGS principi contabili page and downloads:
1. The first document from "Principi contabili generali:" section
2. All documents from the section immediately following "Principi contabili generali:"

DOC files are converted to PDF using LibreOffice and uploaded to Google Cloud Storage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup, Tag
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


class PrincipiContabiliScraper:
    """Scrape Principi Contabili DOC files, convert to PDF, and upload to GCS."""

    BASE_URL = "https://www.rgs.mef.gov.it"
    START_URL = f"{BASE_URL}/VERSIONE-I/e_government/amministrazioni_pubbliche/arconet/principi_contabili/"

    def __init__(
        self,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
        max_concurrent: int = 3,
        base_folder: str = "principi-contabili",
    ) -> None:
        self.bucket_name = bucket_name
        self.base_folder = base_folder.strip("/") or "principi-contabili"
        self.storage_client = storage_client or self._build_storage_client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.max_concurrent = max_concurrent
        self.stop_requested = False
        self.fetched_links: List[Dict[str, str]] = []
        self.generated_files: Dict[str, str] = {}
        self.files_processed = 0
        self.files_uploaded = 0

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a page and return its HTML."""
        try:
            logger.debug("Fetching %s", url)
            async with session.get(url, timeout=60) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as exc:
            logger.error("Error fetching %s: %s", url, exc)
            return None

    async def download_file(self, session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
        """Download a file and return its bytes."""
        try:
            logger.debug("Downloading %s", url)
            async with session.get(url, timeout=120) as response:
                response.raise_for_status()
                return await response.read()
        except Exception as exc:
            logger.error("Error downloading %s: %s", url, exc)
            return None

    def _parse_sections(self, html: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Parse the HTML and extract:
        1. The first document from "Principi contabili generali:" section
        2. All documents from the section immediately following it

        Returns two lists: (generali_docs, applicati_docs)
        """
        soup = BeautifulSoup(html, "lxml")
        
        generali_docs: List[Dict[str, str]] = []
        applicati_docs: List[Dict[str, str]] = []

        # Find all h3 headers (section titles)
        h3_headers = soup.find_all("h3")
        
        generali_h3 = None
        next_h3 = None
        
        for i, h3 in enumerate(h3_headers):
            header_text = h3.get_text(strip=True).lower()
            if "principi contabili generali" in header_text:
                generali_h3 = h3
                # The next h3 should be the "Principi contabili applicati dal 20XX" section
                if i + 1 < len(h3_headers):
                    next_h3 = h3_headers[i + 1]
                break

        if not generali_h3:
            logger.warning("Could not find 'Principi contabili generali' section")
            return generali_docs, applicati_docs

        # The structure is: the h3 is inside a li.divDescription, followed by more li elements
        # containing the document links (span.doc > a[href])
        
        # Find the li containing the generali h3
        generali_li = generali_h3.find_parent("li", class_="divDescription")
        
        if generali_li:
            # Iterate through subsequent siblings until we hit the next section header
            current_li = generali_li.find_next_sibling("li")
            while current_li:
                # Check if this li contains a h3 (next section)
                h3_in_li = current_li.find("h3")
                if h3_in_li:
                    # We've reached the next section, stop collecting generali docs
                    break
                
                # Look for doc links in this li
                for link in current_li.select("span.doc a[href]"):
                    href = link.get("href", "")
                    if href.endswith(".doc"):
                        title = link.get_text(strip=True)
                        if title and href:
                            absolute_url = urljoin(self.BASE_URL, href)
                            generali_docs.append({"title": title, "url": absolute_url})
                            # We only want the first document
                            break
                
                if generali_docs:
                    break
                    
                current_li = current_li.find_next_sibling("li")

        # Now find the next section (e.g., "Principi contabili applicati dal 2026")
        if next_h3:
            next_section_name = next_h3.get_text(strip=True)
            logger.info("Next section after 'Principi contabili generali': %s", next_section_name)
            
            # Find the li containing the next section h3
            next_li = next_h3.find_parent("li", class_="divDescription")
            
            if next_li:
                # Find the h3 after next_h3 to know where to stop
                third_h3 = None
                for i, h3 in enumerate(h3_headers):
                    if h3 == next_h3 and i + 1 < len(h3_headers):
                        third_h3 = h3_headers[i + 1]
                        break
                
                # Iterate through subsequent siblings until we hit the third section
                current_li = next_li.find_next_sibling("li")
                while current_li:
                    # Check if this li contains a h3 (third section)
                    h3_in_li = current_li.find("h3")
                    if h3_in_li:
                        # We've reached the next section, stop
                        break
                    
                    # Look for doc links in this li
                    for link in current_li.select("span.doc a[href]"):
                        href = link.get("href", "")
                        if href.endswith(".doc"):
                            title = link.get_text(strip=True)
                            if title and href:
                                absolute_url = urljoin(self.BASE_URL, href)
                                applicati_docs.append({"title": title, "url": absolute_url})
                    
                    current_li = current_li.find_next_sibling("li")

        logger.info("Found %d document(s) in 'Principi contabili generali' section", len(generali_docs))
        logger.info("Found %d document(s) in the following section", len(applicati_docs))

        return generali_docs, applicati_docs

    @staticmethod
    def _slugify(title: str) -> str:
        normalized = unicodedata.normalize("NFKD", title)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-")
        return cleaned.lower() or "principio-contabile"

    def _build_blob_path(self, filename: str, subfolder: str = "") -> str:
        if subfolder:
            path = f"{self.base_folder}/{subfolder}/{filename}"
        elif self.base_folder:
            path = f"{self.base_folder}/{filename}"
        else:
            path = filename
        return path

    def _already_uploaded(self, filename: str, subfolder: str = "") -> bool:
        blob_path = self._build_blob_path(filename, subfolder)
        exists = self.bucket.get_blob(blob_path) is not None
        if exists:
            logger.info("📄 File already exists: '%s'", blob_path)
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

    def _get_libreoffice_path(self) -> str:
        """Get the path to the LibreOffice executable."""
        import platform
        import shutil
        
        # First check if soffice is in PATH
        soffice_path = shutil.which("soffice")
        if soffice_path:
            return soffice_path
        
        # Check common installation paths
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            ]
        elif platform.system() == "Darwin":  # macOS
            common_paths = [
                "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            ]
        else:  # Linux
            common_paths = [
                "/usr/bin/soffice",
                "/usr/local/bin/soffice",
            ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        return "soffice"  # Fall back to hoping it's in PATH

    def _convert_doc_to_pdf(self, doc_bytes: bytes, original_filename: str) -> Optional[bytes]:
        """
        Convert a DOC file to PDF using LibreOffice.
        Returns the PDF bytes or None if conversion fails.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            doc_path = temp_path / original_filename
            
            # Write the DOC file
            doc_path.write_bytes(doc_bytes)
            
            # Get LibreOffice path
            soffice_path = self._get_libreoffice_path()
            
            # Convert using LibreOffice
            try:
                result = subprocess.run(
                    [
                        soffice_path,
                        "--headless",
                        "--convert-to", "pdf",
                        "--outdir", str(temp_path),
                        str(doc_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                
                if result.returncode != 0:
                    logger.error("LibreOffice conversion failed: %s", result.stderr)
                    return None
                
                # Find the output PDF
                pdf_filename = doc_path.stem + ".pdf"
                pdf_path = temp_path / pdf_filename
                
                if not pdf_path.exists():
                    logger.error("PDF file not found after conversion: %s", pdf_path)
                    return None
                
                return pdf_path.read_bytes()
                
            except subprocess.TimeoutExpired:
                logger.error("LibreOffice conversion timed out for %s", original_filename)
                return None
            except FileNotFoundError:
                logger.error("LibreOffice (soffice) not found at %s. Please install LibreOffice.", soffice_path)
                return None
            except Exception as exc:
                logger.error("Error converting %s to PDF: %s", original_filename, exc)
                return None

    async def upload_pdf(self, pdf_bytes: bytes, filename: str, subfolder: str = "") -> bool:
        """Upload a PDF file to GCS."""
        blob_path = self._build_blob_path(filename, subfolder)
        folder_path = str(Path(blob_path).parent)
        if folder_path == ".":
            folder_path = self.base_folder
        folder = folder_path or None
        
        pdf_obj = {
            "name": Path(filename).name,
            "bytes": pdf_bytes,
            "extension": "pdf",
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
        except Exception as exc:
            logger.error("Failed to upload %s: %s", blob_path, exc)
            return False

    async def process_document(
        self,
        session: aiohttp.ClientSession,
        doc: Dict[str, str],
        subfolder: str = "",
    ) -> bool:
        """Download a DOC file, convert to PDF, and upload to GCS."""
        title = doc["title"]
        url = doc["url"]
        
        # Generate PDF filename from title
        pdf_filename = f"{self._slugify(title)}.pdf"
        
        # Check if already uploaded - stop incrementally when existing file found
        if self._already_uploaded(pdf_filename, subfolder):
            logger.info("🛑 Found existing file; stopping scraper for incremental mode.")
            self.stop_requested = True
            return False
        
        # Download DOC file
        doc_bytes = await self.download_file(session, url)
        if not doc_bytes:
            logger.warning("Failed to download: %s", url)
            return False
        
        # Get original filename from URL for conversion
        original_filename = url.split("/")[-1]
        
        # Convert to PDF
        logger.info("🔄 Converting %s to PDF...", original_filename)
        pdf_bytes = self._convert_doc_to_pdf(doc_bytes, original_filename)
        if not pdf_bytes:
            logger.warning("Failed to convert to PDF: %s", original_filename)
            return False
        
        # Upload PDF
        uploaded = await self.upload_pdf(pdf_bytes, pdf_filename, subfolder)
        if uploaded:
            self.files_uploaded += 1
            blob_path = self._build_blob_path(pdf_filename, subfolder)
            self.generated_files[title] = blob_path
        
        return uploaded

    async def scrape(self) -> Dict[str, object]:
        """Main scraping method."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        
        async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
            # Fetch the main page
            html = await self.fetch_page(session, self.START_URL)
            if not html:
                logger.error("Failed to fetch main page")
                return {
                    "links_discovered": 0,
                    "files_processed": 0,
                    "files_uploaded": 0,
                    "stop_requested": self.stop_requested,
                    "uploaded_files": self.generated_files,
                }
            
            # Parse sections and get document links
            generali_docs, applicati_docs = self._parse_sections(html)
            
            all_docs = generali_docs + applicati_docs
            self.fetched_links = all_docs
            
            # Process "Principi contabili generali" documents (just the first one)
            for doc in generali_docs:
                if self.stop_requested:
                    break
                self.files_processed += 1
                await self.process_document(session, doc, subfolder="generali")
            
            # Process "Principi contabili applicati" documents (all from the next section)
            for doc in applicati_docs:
                if self.stop_requested:
                    break
                self.files_processed += 1
                await self.process_document(session, doc, subfolder="applicati")

        logger.info(
            "Scraping finished: %d processed, %d uploaded, stop_requested=%s",
            self.files_processed,
            self.files_uploaded,
            self.stop_requested,
        )
        return {
            "links_discovered": len(self.fetched_links),
            "files_processed": self.files_processed,
            "files_uploaded": self.files_uploaded,
            "stop_requested": self.stop_requested,
            "uploaded_files": self.generated_files,
        }


async def main() -> None:
    bucket_name = "loomy-public-documents"
    scraper = PrincipiContabiliScraper(bucket_name=bucket_name)
    await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
