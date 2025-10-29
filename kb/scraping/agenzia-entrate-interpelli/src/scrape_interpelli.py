"""
Scraper for Agenzia Entrate Interpelli archive
===============================================

This script recursively navigates through the **Interpelli** sections
of the Agenzia delle Entrate website and downloads all documents of interest
from each year and month folder. Files with the following extensions are
considered for download: PDF, ZIP, XLSX, XLS, XSD and XML. Nested
sub‑directories beneath the month pages (for example folders that group
attachments for a single interpello) are followed automatically, ensuring that
no qualifying document is missed.

The scraper handles multiple interpelli archives:
1. Archivio Interpelli - Main interpelli archive
2. Archivio istanze di interpello sui nuovi investimenti
3. Archivio principi di diritto
4. Archivio risposte alle istanze di consulenza giuridica

Downloaded content is uploaded to a Google Cloud Storage bucket using the
existing :func:`utils.upload_to_storage` utility. The folder hierarchy in
GCS mirrors the structure discovered on the website: files are organised by
year and month, with the original filename preserved whenever possible.

To execute the scraper directly, set the ``GCS_BUCKET_NAME`` environment
variable or pass the bucket name explicitly when instantiating
``InterpelliScraper`` in your own code. See the ``main`` function at the
bottom of this file for an example.

The scraping process honours a maximum concurrency level to avoid
overloading the remote server. It also deduplicates navigation and
download operations via internal ``visited_urls`` and ``downloaded_files``
sets. Robustness is improved with retry logic provided by ``tenacity``
decorators.

Note
----
This script is designed to be self contained. It intentionally avoids
hard‑coding a list of year/month pages; instead it discovers them
dynamically by following links that include interpelli-related keywords in
their path. Should the structure of the site change, updating the
``should_follow_link`` method may be sufficient to adapt the crawler.
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


class InterpelliScraper:
    """Scraper for the Agenzia Entrate interpelli archives.

    The scraper starts from multiple root archive pages, discovers year and month
    pages, identifies documents of interest based on extension, and uploads
    those documents to a Google Cloud Storage bucket. It preserves the
    hierarchy of year and month in the destination path and de‑duplicates
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
        Defaults to ``interpelli``.
    """

    # Base domain used to restrict crawling
    BASE_URL = "https://www.agenziaentrate.gov.it"
    
    # Entry points for the interpelli archives
    START_URLS = [
        "https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/risposte-agli-interpelli/interpelli/archivio-interpelli",
        "https://www.agenziaentrate.gov.it/portale/web/guest/archivio-istanze-di-interpello-sui-nuovi-investimenti",
        "https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/risposte-agli-interpelli/principi-di-diritto/archivio-principi-di-diritto",
        "https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/risposte-agli-interpelli/risposte-alle-istanze-di-consulenza-giuridica/archivio-risposte-alle-istanze-di-consulenza-giuridica"
    ]

    # File extensions to download (lower‑case including leading dot)
    TARGET_EXTENSIONS: Set[str] = {".pdf", ".zip", ".xlsx", ".xls", ".xsd", ".xml"}

    # Maximum concurrent HTTP requests
    MAX_CONCURRENT_REQUESTS = 10

    def __init__(
        self,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        base_folder: str = "interpelli",
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
            path_obj = Path(file_path)
            filename = path_obj.name
            # Directory path relative to base_folder, or empty string
            folder_path = str(path_obj.parent) if path_obj.parent != Path(".") else ""
            extension = path_obj.suffix.lstrip(".")
            pdf_obj = {
                "name": filename,
                "bytes": content,
                "extension": extension,
            }
            # Compose folder prefix in GCS
            full_folder = (
                f"{self.base_folder}/{folder_path}" if folder_path else self.base_folder
            )
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
        return None

    def extract_hierarchy_from_url(self, url: str) -> List[str]:
        """Extract year and month from the given URL if present.

        The year is detected as any four‑digit sequence starting with ``19``
        or ``20``. The month is detected using Italian month names.
        Both components are returned in a list preserving order. If a
        component is not found it will be absent from the returned list.

        Parameters
        ----------
        url : str
            URL whose path is parsed for hierarchy components.

        Returns
        -------
        List[str]
            A list containing year and optionally month.
        """
        parsed = urlparse(url)
        path = unquote(parsed.path.lower())
        # detect year anywhere in the path
        year_match = re.search(r"(19\d{2}|20\d{2})", path)
        year = year_match.group(1) if year_match else None
        # detect Italian months
        month_names = [
            "gennaio",
            "febbraio",
            "marzo",
            "aprile",
            "maggio",
            "giugno",
            "luglio",
            "agosto",
            "settembre",
            "ottobre",
            "novembre",
            "dicembre",
        ]
        month = None
        for m in month_names:
            if m in path:
                # Capitalise first letter for folder naming (e.g. "Gennaio")
                month = m.capitalize()
                break
        hierarchy: List[str] = []
        if year:
            hierarchy.append(year)
        if month:
            hierarchy.append(month)
        return hierarchy

    def generate_file_path(
        self, url: str, hierarchy: List[str], link_text: str = ""
    ) -> str:
        """Construct a relative file path for a downloaded document.

        The hierarchy list typically contains the year and month extracted
        from the page on which the link was found. The link text is used to
        derive a sensible filename: illegal characters are replaced and
        trailing extension annotations (e.g. "- pdf") are stripped. If
        ``link_text`` is blank a name derived from the URL (or a hash of the
        URL when necessary) is used instead.

        Parameters
        ----------
        url : str
            Original file URL.
        hierarchy : List[str]
            List of components representing the current year/month context.
        link_text : str, optional
            Visible anchor text associated with the link.

        Returns
        -------
        str
            A relative path combining hierarchy and filename.
        """
        ext = self.get_file_extension(url) or ".pdf"
        if link_text:
            filename = link_text.strip()
            # Remove extension hints like "- pdf" or "– pdf"
            filename = re.sub(r"\s*[-–]\s*(pdf|zip|xlsx|xls|xml|xsd)\s*$", "", filename, flags=re.IGNORECASE)
            # Replace invalid filename characters
            filename = re.sub(r"[<>:\\|/?*\n\r\t]", "_", filename)
            # Collapse multiple spaces
            filename = re.sub(r"\s+", " ", filename)
            # Limit length to avoid extremely long names
            if len(filename) > 200:
                filename = filename[:200]
            if not filename.lower().endswith(ext.lower()):
                filename = f"{filename}{ext}"
        else:
            # Derive from URL
            parsed = urlparse(url)
            url_filename = Path(unquote(parsed.path)).name
            if url_filename and "." in url_filename:
                filename = url_filename
            else:
                # Use hash to ensure uniqueness
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"document_{url_hash}{ext}"
        path_parts = list(hierarchy) if hierarchy else []
        path_parts.append(filename)
        return "/".join(path_parts)

    def is_file_url(self, url: str) -> bool:
        """Return ``True`` if URL points to a downloadable file."""
        return self.get_file_extension(url) is not None

    def should_follow_link(self, url: str, current_url: str) -> bool:
        """Determine whether a link should be crawled.

        Only same‑domain, unvisited links that appear to relate to the
        interpelli archives are considered. The current implementation allows
        navigation if the URL contains interpelli-related substrings in its
        path. Links to other sections of the site (e.g. ``risoluzioni`` or
        ``provvedimenti``) are explicitly ignored. Exclusion patterns
        prevent the crawler from drifting into unrelated areas such as
        software downloads or general services.

        Parameters
        ----------
        url : str
            Target URL being evaluated.
        current_url : str
            URL of the page on which the link was found (unused at present
            but available for context).

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
        
        # Required patterns - URL must contain at least one of these
        interpelli_patterns = [
            "interpelli",
            "interpello",
            "principi-di-diritto",
            "consulenza-giuridica",
            "nuovi-investimenti"
        ]
        
        # Check if URL matches any required interpelli pattern
        has_interpelli = any(pattern in url_lower for pattern in interpelli_patterns)
        
        if not has_interpelli:
            return False
        
        # Exclude sections unrelated to the interpelli archives
        exclude_patterns = [
            "risoluzioni",
            "circolari",
            "provvedimenti-del-direttore",
            "/provvedimenti/",
            "software",
            "specifiche-tecniche",
            "/modelli",
            "codici-tributo",
            "agenzia-comunica",
            "servizi",
            "documentazione-economica",
            "scadenzario",
            "territorio",
            "newslett",
        ]
        
        # Don't follow if URL contains excluded patterns (unless it clearly has interpelli)
        for pattern in exclude_patterns:
            if pattern in url_lower:
                # Only exclude if the interpelli pattern is not strong enough
                if not any(strong in url_lower for strong in ["archivio-interpelli", "principi-di-diritto", "consulenza-giuridica"]):
                    return False
        
        return True

    async def extract_links(
        self, html: str, base_url: str
    ) -> Tuple[List[str], List[Tuple[str, List[str], str]]]:
        """Parse a page and return navigation and file links.

        Parameters
        ----------
        html : str
            HTML content of the current page.
        base_url : str
            Absolute URL of the current page (used for resolving relative
            links and extracting hierarchy).

        Returns
        -------
        Tuple[List[str], List[Tuple[str, List[str], str]]]
            A tuple containing a list of navigation URLs and a list of
            tuples describing file links. Each file tuple contains the
            absolute URL, the hierarchy extracted from ``base_url`` (year,
            month), and the visible link text.
        """
        soup = BeautifulSoup(html, "html.parser")
        navigation_links: List[str] = []
        file_links: List[Tuple[str, List[str], str]] = []
        current_hierarchy = self.extract_hierarchy_from_url(base_url)
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            absolute_url = urljoin(base_url, href).split("#")[0].rstrip("/")
            
            # Identify file links
            if self.is_file_url(absolute_url):
                if absolute_url not in self.downloaded_files:
                    link_text = a.get_text(strip=True)
                    file_links.append((absolute_url, current_hierarchy, link_text))
            else:
                if self.should_follow_link(absolute_url, base_url):
                    navigation_links.append(absolute_url)
        
        return navigation_links, file_links

    async def crawl_page(self, session: aiohttp.ClientSession, url: str) -> None:
        """Recursively crawl a page and process its links."""
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
        for file_url, hierarchy, link_text in file_links:
            if file_url not in self.downloaded_files:
                self.downloaded_files.add(file_url)
                file_path = self.generate_file_path(file_url, hierarchy, link_text)
                self.file_paths[file_url] = file_path
                # Track counts by year
                if hierarchy and hierarchy[0]:
                    year = hierarchy[0]
                    self.files_by_year[year] = self.files_by_year.get(year, 0) + 1
                download_tasks.append(
                    asyncio.create_task(self.download_file(session, file_url, file_path))
                )
        if download_tasks:
            logger.info(f"⬇️  Starting download of {len(download_tasks)} file(s) from this page")
            await asyncio.gather(*download_tasks, return_exceptions=True)
        # Traverse child pages
        crawl_tasks: List[asyncio.Task] = []
        for nav_url in navigation_links:
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
        logger.info(f"Starting scrape from {len(self.START_URLS)} root URLs")
        for url in self.START_URLS:
            logger.info(f"  - {url}")
        
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/93.0.4577.82 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        
        async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
            # Crawl all start URLs
            crawl_tasks = [
                asyncio.create_task(self.crawl_page(session, start_url))
                for start_url in self.START_URLS
            ]
            await asyncio.gather(*crawl_tasks, return_exceptions=True)
        
        logger.info("Scraping complete!")
        # Log summary
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
    scraper = InterpelliScraper(bucket_name=bucket_name, max_concurrent=10, base_folder="interpelli")
    results = await scraper.scrape()
    # Print a short summary to stdout
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
    # Display a few sample paths
    print("\nSample file paths:")
    for i, (url, path) in enumerate(list(results["file_paths"].items())[:10], 1):
        print(f"{i}. {path}")
        print(f"   Source: {url}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
