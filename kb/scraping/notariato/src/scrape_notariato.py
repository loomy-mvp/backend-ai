"""
Scraper for Agenzia delle Entrate - Provvedimenti non soggetti a pubblicità
============================================================================

This script recursively navigates through the Agenzia delle Entrate website
and downloads all PDF documents from the "Provvedimenti non soggetti a pubblicità"
section, including both archive and current documents.

Downloaded content is uploaded to a Google Cloud Storage bucket using the
existing :func:`utils.upload_to_storage` utility. The folder hierarchy in
GCS mirrors the structure discovered on the website: files are organised by
year with the original filename preserved whenever possible.

The scraping process honours a maximum concurrency level to avoid
overloading the remote server. It also deduplicates navigation and
download operations via internal ``visited_urls`` and ``downloaded_files``
sets. Robustness is improved with retry logic provided by ``tenacity``
decorators.
"""

import asyncio
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, unquote

import aiohttp
from bs4 import BeautifulSoup
from google.cloud import storage  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.upload_to_storage import upload_to_storage  # type: ignore


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class NotariatoScraper:
    """Scraper for Agenzia delle Entrate Provvedimenti non soggetti a pubblicità.

    The scraper starts from multiple entry points (archive and current years),
    discovers documents, and uploads those documents to a Google Cloud Storage
    bucket. It preserves the hierarchy in the destination path and de‑duplicates
    downloads across the entire run.

    Parameters
    ----------
    bucket_name : str
        Name of the target GCS bucket.
    storage_client : Optional[storage.Client], optional
        Preconfigured Google Cloud Storage client. If not provided a new
        client will be instantiated.
    max_concurrent : int, optional
        Maximum number of concurrent HTTP requests (both for page fetches
        and file downloads). Defaults to 10.
    base_folder : str, optional
        Root folder inside the bucket where files should be uploaded.
        Defaults to ``notariato``.
    """

    # Base domain used to restrict crawling
    BASE_URL = "https://www.agenziaentrate.gov.it"
    
    # Multiple entry points for different years
    START_URLS = [
        "https://www.agenziaentrate.gov.it/portale/archivio/normativa-prassi-archivio-documentazione/provvedimenti/altri-provvedimenti-non-soggetti",
        "https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/altri-provvedimenti-non-soggetti-attuale/provvedimenti-2017",
        "https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/altri-provvedimenti-non-soggetti-attuale/provvedimenti-2018",
        "https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/altri-provvedimenti-non-soggetti-attuale/provvedimenti-2019",
        "https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/altri-provvedimenti-non-soggetti-attuale/provvedimenti-2020",
        "https://www.agenziaentrate.gov.it/portale/provvedimenti-2021",
        "https://www.agenziaentrate.gov.it/portale/archivio-2022-provvedimenti-del-direttore-non-soggetti-a-pubblicit%25c3%25a0-legale",
        "https://www.agenziaentrate.gov.it/portale/archivio-2023-provvedimenti-del-direttore-non-soggetti-a-pubblicit%25c3%25a0-legale",
        "https://www.agenziaentrate.gov.it/portale/provvedimenti-2024",
        "https://www.agenziaentrate.gov.it/portale/provvedimenti-2025-non-soggetti-a-pubblicita",
        "https://www.agenziaentrate.gov.it/portale/gennaio-2025-non-soggetti-a-pubblicita",
        "https://www.agenziaentrate.gov.it/portale/marzo-2025-provvedimenti-non-soggetti-a-pubblicita"
    ]

    # File extensions to download (lower‑case including leading dot)
    TARGET_EXTENSIONS: Set[str] = {".pdf"}

    # Maximum concurrent HTTP requests
    MAX_CONCURRENT_REQUESTS = 10

    def __init__(
        self,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        base_folder: str = "notariato",
    ) -> None:
        self.bucket_name = bucket_name
        self.storage_client = storage_client or storage.Client()
        self.max_concurrent = max_concurrent
        self.base_folder = base_folder
        self.visited_urls: Set[str] = set()
        self.downloaded_files: Set[str] = set()
        self.file_paths: Dict[str, str] = {}
        self.files_by_year: Dict[str, int] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stop_requested = False
        self.bucket = self.storage_client.bucket(self.bucket_name)

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch the HTML content of a page.

        If the request fails or returns a non‑success status code an error
        will be logged and ``None`` returned. A semaphore is used to limit
        concurrent fetches. A simple timeout prevents hanging requests.

        Parameters
        ----------
        session : aiohttp.ClientSession
            The active session used for requests.
        url : str
            The absolute URL to fetch.

        Returns
        -------
        Optional[str]
            The text of the response or ``None`` on failure.
        """
        async with self.semaphore:
            try:
                logger.debug(f"Fetching page: {url}")
                async with session.get(url, timeout=30) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as exc:
                logger.error(f"Error fetching {url}: {exc}")
                return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def download_file(
        self, session: aiohttp.ClientSession, url: str, file_path: str
    ) -> bool:
        """Download a file from ``url`` and upload it to GCS.

        The download is retried on transient errors up to three times with
        exponential backoff. If the upload succeeds, the method returns
        ``True``; otherwise ``False`` is returned.

        Parameters
        ----------
        session : aiohttp.ClientSession
            Active HTTP session.
        url : str
            Absolute URL of the file to download.
        file_path : str
            Destination path relative to the base folder in GCS.

        Returns
        -------
        bool
            ``True`` if the file was uploaded successfully, ``False`` otherwise.
        """
        if self.stop_requested:
            return False
        async with self.semaphore:
            try:
                logger.info(f"Downloading file: {url}")
                async with session.get(url, timeout=60) as response:
                    response.raise_for_status()
                    content = await response.read()
                    success = await self.upload_to_gcs(content, file_path, url)
                    if success:
                        logger.info(f"✅ Uploaded file to GCS: {file_path}")
                    return success
            except Exception as exc:
                logger.error(f"Error downloading {url}: {exc}")
                return False

    async def upload_to_gcs(self, content: bytes, file_path: str, url: str) -> bool:
        """Upload the given content to Google Cloud Storage.

        This method wraps the call to the shared ``upload_to_storage`` utility
        provided by the project. The bucket folder structure is composed
        by combining the ``base_folder`` with any directory components in
        ``file_path``. The file name and extension are extracted for
        completeness.

        Parameters
        ----------
        content : bytes
            Raw bytes of the file to upload.
        file_path : str
            Relative path (including directories and filename) inside the
            ``base_folder`` where the file will be stored.
        url : str
            Original URL of the downloaded file, used only for logging.

        Returns
        -------
        bool
            ``True`` if the upload succeeded, ``False`` otherwise.
        """
        try:
            full_folder, filename, _ = self._build_gcs_location(file_path)
            extension = Path(file_path).suffix.lstrip(".")
            pdf_obj = {
                "name": filename,
                "bytes": content,
                "extension": extension,
            }
            # Compose folder prefix in GCS
            upload_to_storage(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                pdf_obj=pdf_obj,
                folder=full_folder,
            )
            return True
        except Exception as exc:
            logger.error(f"❌ Failed to upload {file_path} to GCS: {exc}")
            return False

    def _normalize_folder(self, folder: str) -> str:
        return folder.strip("/").replace("\\", "/") if folder else ""

    def _build_gcs_location(self, file_path: str) -> Tuple[str, str, str]:
        path_obj = Path(file_path)
        filename = path_obj.name
        folder_path = str(path_obj.parent) if path_obj.parent != Path(".") else ""
        full_folder = f"{self.base_folder}/{folder_path}" if folder_path else self.base_folder
        normalized_folder = self._normalize_folder(full_folder)
        blob_path = f"{normalized_folder}/{filename}" if normalized_folder else filename
        return full_folder, filename, blob_path

    def _already_uploaded(self, file_path: str) -> bool:
        _, _, blob_path = self._build_gcs_location(file_path)
        return self.bucket.get_blob(blob_path) is not None

    def get_file_extension(self, url: str) -> Optional[str]:
        """Return the file extension if ``url`` points to a target file.

        The method checks both the end of the path and any part of the path
        for known extensions, accommodating URLs that embed the file name
        within additional path segments (e.g. hashed download URLs).

        Parameters
        ----------
        url : str
            URL to inspect.

        Returns
        -------
        Optional[str]
            The matched file extension (including leading dot) or ``None`` if
            no supported extension was found.
        """
        parsed = urlparse(url)
        path = unquote(parsed.path)
        # direct match on suffix
        ext = Path(path).suffix.lower()
        if ext in self.TARGET_EXTENSIONS:
            return ext
        # search within the path
        for target_ext in self.TARGET_EXTENSIONS:
            if target_ext in path.lower():
                return target_ext
        # Check for /documents/ pattern which typically indicates PDF
        if '/documents/' in path.lower():
            return '.pdf'
        return None

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing invalid characters."""
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', ' ', filename)
        return filename.strip()

    def extract_year_from_url(self, url: str) -> Optional[str]:
        """Extract year from URL if present.

        Parameters
        ----------
        url : str
            URL to parse.

        Returns
        -------
        Optional[str]
            The year as a string or None if not found.
        """
        # Look for 4-digit year patterns (2014-2025)
        year_match = re.search(r'(20[1-2][0-9])', url)
        if year_match:
            return year_match.group(1)
        return None

    def generate_file_path(self, url: str, link_text: str = "") -> str:
        """Construct a relative file path for a downloaded document.

        Parameters
        ----------
        url : str
            Original file URL.
        link_text : str, optional
            Visible anchor text associated with the link.

        Returns
        -------
        str
            A relative path combining year and filename.
        """
        ext = self.get_file_extension(url) or ".pdf"
        year = self.extract_year_from_url(url)
        
        if link_text:
            filename = self.sanitize_filename(link_text.strip())
            # Remove extension hints
            filename = re.sub(r'\s*[-–]\s*(pdf|zip)\s*$', '', filename, flags=re.IGNORECASE)
            # Limit length
            if len(filename) > 200:
                filename = filename[:200]
            if not filename.lower().endswith(ext.lower()):
                filename = f"{filename}{ext}"
        else:
            # Derive from URL
            parsed = urlparse(url)
            url_filename = Path(unquote(parsed.path)).name
            if url_filename and "." in url_filename:
                filename = self.sanitize_filename(url_filename)
            else:
                # Use hash to ensure uniqueness
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"document_{url_hash}{ext}"
        
        # Construct path with year if available
        if year:
            return f"{year}/{filename}"
        return filename

    def is_file_url(self, url: str) -> bool:
        """Return ``True`` if the URL points to a downloadable file."""
        return self.get_file_extension(url) is not None

    def should_follow_link(self, url: str, current_url: str) -> bool:
        """Determine whether a link should be crawled.

        Parameters
        ----------
        url : str
            Target URL being evaluated.
        current_url : str
            URL of the page on which the link was found.

        Returns
        -------
        bool
            ``True`` if the link should be followed; ``False`` otherwise.
        """
        # Skip already visited links
        if url in self.visited_urls:
            return False
        # Stay within the same domain
        if not url.startswith(self.BASE_URL):
            return False
        url_lower = url.lower()
        # Only follow pages related to provvedimenti
        if not any(kw in url_lower for kw in ['provvedimenti', 'gennaio', 'febbraio', 'marzo', 'aprile', 
                                                 'maggio', 'giugno', 'luglio', 'agosto', 'settembre', 
                                                 'ottobre', 'novembre', 'dicembre']):
            return False
        # Exclude unrelated sections
        exclude_patterns = [
            'circolari',
            'risoluzioni',
            'interpello',
            '/modelli',
            'servizi',
        ]
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False
        return True

    async def extract_links(
        self, html: str, base_url: str
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Parse a page and return navigation and file links.

        Parameters
        ----------
        html : str
            HTML content of the current page.
        base_url : str
            Absolute URL of the current page.

        Returns
        -------
        Tuple[List[str], List[Tuple[str, str]]]
            A tuple containing a list of navigation URLs and a list of
            tuples describing file links. Each file tuple contains the
            absolute URL and the visible link text.
        """
        soup = BeautifulSoup(html, 'html.parser')
        navigation_links: List[str] = []
        file_links: List[Tuple[str, str]] = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(base_url, href).split('#')[0].rstrip('/')
            
            # Identify file links
            if self.is_file_url(absolute_url):
                if absolute_url not in self.downloaded_files:
                    link_text = a.get_text(strip=True)
                    file_links.append((absolute_url, link_text))
            else:
                if self.should_follow_link(absolute_url, base_url):
                    navigation_links.append(absolute_url)
        
        return navigation_links, file_links

    async def crawl_page(self, session: aiohttp.ClientSession, url: str) -> None:
        """Recursively crawl a page and process its links."""
        if self.stop_requested:
            return
        if url in self.visited_urls:
            return
        self.visited_urls.add(url)
        logger.info(f"📄 Crawling: {url}")
        
        html = await self.fetch_page(session, url)
        if not html:
            return
        
        navigation_links, file_links = await self.extract_links(html, url)
        download_tasks: List[asyncio.Task] = []
        
        # Schedule file downloads
        for file_url, link_text in file_links:
            if self.stop_requested:
                break
            if file_url not in self.downloaded_files:
                file_path = self.generate_file_path(file_url, link_text)
                if self._already_uploaded(file_path):
                    logger.info(
                        "🛑 Found existing blob for '%s'; stopping scraper.", file_path
                    )
                    self.stop_requested = True
                    break
                self.downloaded_files.add(file_url)
                self.file_paths[file_url] = file_path
                
                # Track counts by year
                year = self.extract_year_from_url(file_url)
                if year:
                    self.files_by_year[year] = self.files_by_year.get(year, 0) + 1
                
                download_tasks.append(
                    asyncio.create_task(self.download_file(session, file_url, file_path))
                )
        
        if download_tasks:
            logger.info(f"⬇️  Starting download of {len(download_tasks)} file(s) from this page")
            await asyncio.gather(*download_tasks, return_exceptions=True)
        if self.stop_requested:
            return
        
        # Traverse child pages
        crawl_tasks: List[asyncio.Task] = []
        for nav_url in navigation_links:
            if self.stop_requested:
                break
            if nav_url not in self.visited_urls:
                crawl_tasks.append(asyncio.create_task(self.crawl_page(session, nav_url)))
        
        if crawl_tasks:
            await asyncio.gather(*crawl_tasks, return_exceptions=True)

    async def scrape(self) -> Dict[str, object]:
        """Entry point for running the scraper asynchronously.

        Returns
        -------
        Dict[str, object]
            A dictionary summarising the crawl: number of pages visited,
            number of files downloaded, a mapping of file URLs to their
            destination paths, and a count of files per year.
        """
        logger.info(f"Starting scrape from {len(self.START_URLS)} entry points")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        
        async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
            # Crawl all start URLs
            tasks = [self.crawl_page(session, url) for url in self.START_URLS]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Scraping complete!")
        logger.info(f"Pages visited: {len(self.visited_urls)}")
        logger.info(f"Files downloaded: {len(self.downloaded_files)}")
        
        if self.files_by_year:
            logger.info("\n" + "=" * 60)
            logger.info("📊 FILES DOWNLOADED PER YEAR")
            logger.info("=" * 60)
            for year, count in sorted(self.files_by_year.items(), key=lambda x: x[0], reverse=True):
                logger.info(f"  {year}: {count:3d} file(s)")
            total_files = sum(self.files_by_year.values())
            logger.info("=" * 60)
            logger.info(f"  TOTAL: {total_files} file(s)")
            logger.info("=" * 60)
        
        return {
            "pages_visited": len(self.visited_urls),
            "files_downloaded": len(self.downloaded_files),
            "file_paths": self.file_paths,
            "files_by_year": self.files_by_year,
        }


async def main() -> None:
    """Example entry point for running the scraper from the command line."""
    bucket_name = os.getenv("GCS_BUCKET_NAME", "loomy-public-documents")
    scraper = NotariatoScraper(
        bucket_name=bucket_name, 
        max_concurrent=10, 
        base_folder="notariato"
    )
    results = await scraper.scrape()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SCRAPING SUMMARY")
    print("=" * 80)
    print(f"Total pages visited: {results['pages_visited']}")
    print(f"Total files downloaded: {results['files_downloaded']}")
    print(f"Bucket: gs://{bucket_name}/{scraper.base_folder}/")
    
    if results.get("files_by_year"):
        print("\n📊 Files downloaded per year:")
        for year, count in sorted(results["files_by_year"].items(), key=lambda x: x[0], reverse=True):
            print(f"  {year}: {count:3d} file(s)")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
