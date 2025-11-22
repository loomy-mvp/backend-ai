"""
Scraper for Agenzia Entrate Risoluzioni website
Scrapes all PDF, ZIP, XLSX, XLS, XSD, and XML files from the risoluzioni archive
and uploads them to Google Cloud Storage preserving the full path hierarchy.
"""

import asyncio
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, unquote

import aiohttp
from bs4 import BeautifulSoup
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential

# Import upload utility - handle both direct execution and module import
# import sys
# from pathlib import Path as PathLib
# # Go up to scraping folder and add utils to path
# scraping_path = PathLib(__file__).parent.parent.parent
# utils_path = scraping_path / 'utils'
# if str(utils_path) not in sys.path:
#     sys.path.insert(0, str(utils_path))
# from upload_to_storage import upload_to_storage

from src.utils.upload_to_storage import upload_to_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RisoluzioniScraper:
    """Scraper for Agenzia Entrate risoluzioni archive"""
    
    BASE_URL = "https://www.agenziaentrate.gov.it"
    START_URL = "https://www.agenziaentrate.gov.it/portale/normativa-e-prassi/risoluzioni/archivio-risoluzioni"
    
    # File extensions to download
    TARGET_EXTENSIONS = {'.pdf', '.zip', '.xlsx', '.xml', '.xls', '.xsd'}
    
    # Maximum concurrent requests
    MAX_CONCURRENT_REQUESTS = 10
    
    def __init__(
        self, 
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        base_folder: str = "risoluzioni"
    ):
        """
        Initialize the scraper
        
        Args:
            bucket_name: GCS bucket name for storing files
            storage_client: Optional GCS client. If None, will create one
            max_concurrent: Maximum concurrent HTTP requests
            base_folder: Base folder path in GCS bucket
        """
        self.bucket_name = bucket_name
        self.storage_client = storage_client or storage.Client()
        self.max_concurrent = max_concurrent
        self.base_folder = base_folder
        self.visited_urls: Set[str] = set()
        self.downloaded_files: Set[str] = set()
        self.file_paths: Dict[str, str] = {}  # Maps file URL to its hierarchy path
        self.files_by_year: Dict[str, int] = {}  # Tracks count of files per year
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stop_requested = False
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a page with retry logic"""
        async with self.semaphore:
            try:
                logger.info(f"Fetching: {url}")
                async with session.get(url, timeout=30) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def download_file(self, session: aiohttp.ClientSession, url: str, file_path: str) -> bool:
        """Download a file with retry logic"""
        if self.stop_requested:
            return False
        async with self.semaphore:
            try:
                logger.info(f"Downloading: {url}")
                async with session.get(url, timeout=60) as response:
                    response.raise_for_status()
                    content = await response.read()
                    
                    success = await self.upload_to_gcs(content, file_path, url)
                    
                    if success:
                        logger.info(f"Successfully processed: {file_path}")
                        return True
                    return False
                    
            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")
                return False
    
    async def upload_to_gcs(self, content: bytes, file_path: str, url: str) -> bool:
        """
        Upload file to Google Cloud Storage using the existing upload_to_storage utility
        
        Args:
            content: File content as bytes
            file_path: Full path in GCS (including year/month/filename)
            url: Source URL of the file
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            full_folder, filename, _ = self._build_gcs_location(file_path)
            extension = Path(file_path).suffix.lstrip('.')
            
            # Create pdf_obj structure expected by upload_to_storage
            pdf_obj = {
                "name": filename,
                "bytes": content,
                "extension": extension
            }
            
            # Combine base_folder with the year/month path
            # Upload using the existing utility function
            upload_to_storage(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                pdf_obj=pdf_obj,
                folder=full_folder
            )
            
            logger.info(f"✅ Uploaded to GCS: {full_folder}/{filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {file_path} to GCS: {e}")
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
        """
        Extract file extension from URL
        Handles URLs where extension appears in the middle of the path
        """
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        # First try: Check if path ends with a target extension
        ext = Path(path).suffix.lower()
        if ext in self.TARGET_EXTENSIONS:
            return ext
        
        # Second try: Check if any target extension appears anywhere in the path
        for target_ext in self.TARGET_EXTENSIONS:
            if target_ext in path.lower():
                return target_ext
        
        return None
    
    def generate_file_path(self, url: str, hierarchy: List[str], link_text: str = "") -> str:
        """
        Generate a unique file path preserving hierarchy
        Format: year/month/filename (if month present) or year/filename (using link text as filename)
        
        Args:
            url: File URL
            hierarchy: List containing [year, month] (month is optional)
            link_text: Text from the HTML link element
        """
        # Get file extension
        ext = self.get_file_extension(url)
        if not ext:
            ext = '.pdf'  # Default to PDF if extension not found
        
        # Clean link text to use as filename
        if link_text:
            # Remove unwanted characters and clean up the text
            filename = link_text.strip()
            # Remove " - pdf" or similar suffixes
            filename = re.sub(r'\s*-\s*(pdf|zip|xlsx|xml|xls|xsd)\s*$', '', filename, flags=re.IGNORECASE)
            # Replace invalid filename characters
            filename = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', filename)
            # Replace multiple spaces with single space
            filename = re.sub(r'\s+', ' ', filename)
            # Limit filename length (keep first 200 chars)
            if len(filename) > 200:
                filename = filename[:200]
            # Add extension if not already present
            if not filename.lower().endswith(ext.lower()):
                filename = f"{filename}{ext}"
        else:
            # Fallback: generate from URL hash if no link text
            parsed = urlparse(url)
            url_filename = Path(unquote(parsed.path)).name
            if url_filename and '.' in url_filename:
                filename = url_filename
            else:
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"document_{url_hash}{ext}"
        
        # Build path: year/month/filename (if month exists) or year/filename
        path_parts = []
        if hierarchy:
            if hierarchy[0]:  # Year
                path_parts.append(hierarchy[0])
            if len(hierarchy) > 1 and hierarchy[1]:  # Month (if present)
                path_parts.append(hierarchy[1])
        path_parts.append(filename)
        
        return '/'.join(path_parts)
    
    def extract_hierarchy_from_url(self, url: str) -> List[str]:
        """Extract hierarchy information from URL path"""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        # Extract year if present
        year_match = re.search(r'(20\d{2})', path)
        year = year_match.group(1) if year_match else None
        
        # Extract month if present (Italian month names)
        month_pattern = r'(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)'
        month_match = re.search(month_pattern, path, re.IGNORECASE)
        month = month_match.group(1).capitalize() if month_match else None
        
        hierarchy = []
        if year:
            hierarchy.append(year)
        if month:
            hierarchy.append(month)
        
        return hierarchy
    
    def is_file_url(self, url: str) -> bool:
        """Check if URL points to a downloadable file"""
        return self.get_file_extension(url) is not None
    
    def is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain"""
        return url.startswith(self.BASE_URL)
    
    def should_follow_link(self, url: str, current_path: str) -> bool:
        """
        Determine if we should follow a link
        Only follows links related to risoluzioni
        """
        # Don't follow if already visited
        if url in self.visited_urls:
            return False
        
        # Only follow same-domain links
        if not self.is_same_domain(url):
            return False
        
        url_lower = url.lower()
        
        # Required patterns - must match at least one
        required_patterns = [
            'risoluzioni',
            'archivio-risoluzioni',
        ]
        
        # Year patterns - matches year-based archive pages
        year_patterns = [
            r'risoluzioni-20\d{2}',  # risoluzioni-2024, risoluzioni-2021, etc.
            r'20\d{2}',  # Just a year in the URL
        ]
        
        # Month patterns - matches month-based pages
        italian_months = [
            'gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno',
            'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre'
        ]
        month_pattern_found = any(month in url_lower for month in italian_months)
        
        # Check if URL matches any required pattern
        has_required = any(pattern in url_lower for pattern in required_patterns)
        has_year = any(re.search(pattern, url_lower) for pattern in year_patterns)
        
        # Accept if it has required patterns OR (has year pattern AND has risoluzioni in URL)
        if has_required:
            is_valid = True
        elif has_year and 'risoluzioni' in url_lower:
            is_valid = True
        elif month_pattern_found and 'risoluzioni' in url_lower and '20' in url_lower:
            # Month pages with year and risoluzioni keyword
            is_valid = True
        else:
            is_valid = False
        
        if not is_valid:
            return False
        
        # Exclude patterns that are definitely not what we want
        exclude_patterns = [
            'circolari',
            'provvedimenti',
            'interpello',
            'software',
            'specifiche-tecniche',
            'studi-di-settore',
            'scadenzario',
            'ufficio-studi',
            'osservatorio',
            'territorio',
        ]
        
        # Don't follow if URL contains excluded patterns
        for pattern in exclude_patterns:
            if pattern in url_lower and 'risoluzioni' not in url_lower:
                return False
        
        return True
    
    async def extract_links(self, html: str, base_url: str) -> tuple[List[str], List[tuple[str, List[str], str]]]:
        """
        Extract both navigation links and file links from HTML
        Returns: (navigation_links, file_links_with_hierarchy_and_text)
        Each file link is a tuple: (url, hierarchy, link_text)
        """
        soup = BeautifulSoup(html, 'html.parser')
        navigation_links = []
        file_links = []
        
        # Extract hierarchy from current page URL
        current_hierarchy = self.extract_hierarchy_from_url(base_url)
        
        logger.info(f"Extracting links from: {base_url}")
        logger.info(f"Current hierarchy: {current_hierarchy}")
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            # Normalize URL (remove fragments, trailing slashes)
            absolute_url = absolute_url.split('#')[0].rstrip('/')
            
            if self.is_file_url(absolute_url):
                # It's a downloadable file
                if absolute_url not in self.downloaded_files:
                    # Extract link text and clean it
                    link_text = link.get_text(strip=True)
                    file_links.append((absolute_url, current_hierarchy, link_text))
                    logger.debug(f"Found file: {absolute_url} - Text: {link_text}")
            elif self.should_follow_link(absolute_url, base_url):
                # It's a navigation link
                navigation_links.append(absolute_url)
                logger.debug(f"Found navigation link: {absolute_url}")
        
        logger.info(f"Found {len(file_links)} files and {len(navigation_links)} navigation links")
        
        return navigation_links, file_links
    
    async def crawl_page(self, session: aiohttp.ClientSession, url: str):
        """Crawl a single page and process its links"""
        if self.stop_requested:
            return
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        logger.info(f"📄 Crawling page ({len(self.visited_urls)} visited): {url}")
        
        html = await self.fetch_page(session, url)
        if not html:
            return
        
        # Extract links
        navigation_links, file_links = await self.extract_links(html, url)
        
        # Download files
        download_tasks = []
        for file_url, hierarchy, link_text in file_links:
            if self.stop_requested:
                break
            if file_url not in self.downloaded_files:
                file_path = self.generate_file_path(file_url, hierarchy, link_text)
                if self._already_uploaded(file_path):
                    logger.info(
                        "🛑 Found existing blob for '%s'; stopping scraper.", file_path
                    )
                    self.stop_requested = True
                    break
                self.downloaded_files.add(file_url)
                self.file_paths[file_url] = file_path
                
                # Track files by year
                if hierarchy and hierarchy[0]:
                    year = hierarchy[0]
                    self.files_by_year[year] = self.files_by_year.get(year, 0) + 1
                
                task = self.download_file(session, file_url, file_path)
                download_tasks.append(task)

        # Download files concurrently
        if download_tasks:
            logger.info(f"⬇️  Starting download of {len(download_tasks)} files from this page")
            await asyncio.gather(*download_tasks, return_exceptions=True)
        if self.stop_requested:
            return
        
        # Recursively crawl navigation links
        crawl_tasks = []
        for nav_url in navigation_links:
            if self.stop_requested:
                break
            if nav_url not in self.visited_urls:
                task = self.crawl_page(session, nav_url)
                crawl_tasks.append(task)
        
        # Crawl pages concurrently
        if crawl_tasks:
            logger.info(f"🔗 Following {len(crawl_tasks)} navigation links from this page")
            await asyncio.gather(*crawl_tasks, return_exceptions=True)
    
    async def scrape(self):
        """Main scraping method"""
        logger.info(f"Starting scrape from: {self.START_URL}")
        
        # Configure session with headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        
        async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
            await self.crawl_page(session, self.START_URL)
        
        # Log statistics
        logger.info(f"Scraping complete!")
        logger.info(f"Pages visited: {len(self.visited_urls)}")
        logger.info(f"Files downloaded: {len(self.downloaded_files)}")
        
        # Log files per year recap
        if self.files_by_year:
            logger.info("\n" + "="*60)
            logger.info("📊 FILES DOWNLOADED PER YEAR")
            logger.info("="*60)
            
            # Sort by year (descending)
            sorted_years = sorted(self.files_by_year.items(), key=lambda x: x[0], reverse=True)
            
            for year, count in sorted_years:
                logger.info(f"  {year}: {count:3d} files")
            
            logger.info("="*60)
            logger.info(f"  TOTAL: {len(self.downloaded_files)} files")
            logger.info("="*60)
        
        return {
            'pages_visited': len(self.visited_urls),
            'files_downloaded': len(self.downloaded_files),
            'file_paths': self.file_paths,
            'files_by_year': self.files_by_year
        }


async def main():
    """Main entry point - example usage"""
    import os
    import json
    from google.oauth2 import service_account
    
    # Configuration
    BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "loomy-public-documents")
    
    # use load-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        # dotenv not installed - will use system environment variables
        pass
    # Setup GCP credentials
    gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")

    if gcp_credentials_info:
        gcp_credentials_info = json.loads(gcp_credentials_info)
        gcp_service_account_credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)
        storage_client = storage.Client(credentials=gcp_service_account_credentials)
    else:
        storage_client = storage.Client()
    
    scraper = RisoluzioniScraper(
        bucket_name=BUCKET_NAME,
        storage_client=storage_client,
        max_concurrent=10,
        base_folder="risoluzioni"
    )
    results = await scraper.scrape()
    
    # Print summary
    print("\n" + "="*80)
    print("SCRAPING SUMMARY")
    print("="*80)
    print(f"Total pages visited: {results['pages_visited']}")
    print(f"Total files processed: {results['files_downloaded']}")
    print(f"Bucket: gs://{BUCKET_NAME}/risoluzioni/")
    
    # Print files by year
    if results.get('files_by_year'):
        print("\n📊 Files downloaded per year:")
        sorted_years = sorted(results['files_by_year'].items(), key=lambda x: x[0], reverse=True)
        for year, count in sorted_years:
            print(f"  {year}: {count:3d} files")
    
    print("\nSample file paths:")
    for i, (url, path) in enumerate(list(results['file_paths'].items())[:10], 1):
        print(f"{i}. {path}")
        print(f"   URL: {url}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
