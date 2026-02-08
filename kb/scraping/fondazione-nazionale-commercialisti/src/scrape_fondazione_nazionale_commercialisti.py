"""
Scraper for Fondazione Nazionale Commercialisti
================================================

This script scrapes documents and publications from the Fondazione Nazionale
Commercialisti website (https://www.fondazionenazionalecommercialisti.it) and
uploads them to Google Cloud Storage.

The scraper handles:
- Multiple document sections with pagination
- Direct node URLs (Aristeia archives without pagination)
- PDF downloads with retry logic
- Deduplication across runs
- Configurable delays to avoid server overload

Sections scraped include:
- Ultimi pubblicati
- Documenti di Ricerca
- Strumenti di Lavoro
- Libri
- Circolari
- Informativa Periodica
- Comunicazioni
- Focus
- Fondazione Pacioli (Archivio)
- Fondazione Aristeia (Archivio)
- IRDCEC (Archivio Storico)

Author: Generated for FNC scraping project
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from google.cloud import storage
from google.oauth2 import service_account
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from src.utils.upload_to_storage import upload_to_storage
except ModuleNotFoundError:  # Local execution fallback
    import sys
    UTILS_PATH = Path(__file__).resolve().parents[2] / "utils"
    if str(UTILS_PATH) not in sys.path:
        sys.path.append(str(UTILS_PATH))
    from upload_to_storage import upload_to_storage


logger = logging.getLogger(__name__)


class FondazioneNazionaleScraper:
    """Scraper for Fondazione Nazionale Commercialisti documents.
    
    Parameters
    ----------
    bucket_name : str
        Name of the target GCS bucket.
    base_folder : str, optional
        Root folder inside the bucket where files should be uploaded.
        Defaults to "fondazione-nazionale-commercialisti".
    delay : float, optional
        Delay in seconds between HTTP requests to avoid overloading the server.
        Defaults to 2.5.
    retries : int, optional
        Number of retry attempts for failed HTTP requests.
        Defaults to 6.
    storage_client : Optional[storage.Client], optional
        Preconfigured Google Cloud Storage client. If not provided, a new
        client will be instantiated.
    """
    
    BASE_URL = "https://www.fondazionenazionalecommercialisti.it"
    
    # Section definitions: name -> URL path
    SECTIONS = {
        "Documenti di Ricerca": "/documenti-e-notizie/documenti-di-ricerca",
        "Strumenti di Lavoro": "/documenti-e-notizie/strumenti-di-lavoro",
        "Libri": "/documenti-e-notizie/libri",
        "Circolari": "/documenti-e-notizie/circolari",
        "Informativa Periodica": "/documenti-e-notizie/informativa-periodica",
        "Comunicazioni": "/documenti-e-notizie/comunicazioni",
        "Focus": "/documenti-e-notizie/focus",
        "Fondazione Pacioli (Archivio)": "/archivi/fondazione-pacioli",
        "Fondazione Aristeia - Documenti": "/node/8",
        "Fondazione Aristeia - Studi e Ricerche": "/node/9",
        "IRDCEC (Archivio Storico)": "/archivi/irdcec",
    }
    
    def __init__(
        self,
        bucket_name: str,
        base_folder: str = "fondazione-nazionale-commercialisti",
        delay: float = 2.5,
        retries: int = 6,
        storage_client: Optional[storage.Client] = None,
    ):
        self.bucket_name = bucket_name
        self.base_folder = base_folder.strip("/") or "fondazione-nazionale-commercialisti"
        self.delay = delay
        self.retries = retries
        self.storage_client = storage_client or self._build_storage_client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        # Create session with retry logic
        self.session = self._create_session()
        
        # Tracking
        self.downloaded_pdfs: Set[str] = set()
        self.pdf_uploaded = 0
        self.pdf_skipped = 0
        self.pages_errors = 0
        self.nodes_errors = 0
        self.sections_processed = 0
        
    def _build_storage_client(self) -> storage.Client:
        """Build a GCS client using environment credentials if available."""
        gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
        if gcp_credentials_info:
            logger.info("Using GCP credentials from environment variable")
            credentials_dict = json.loads(gcp_credentials_info)
            gcp_credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            return storage.Client(credentials=gcp_credentials)
        logger.info("Using default GCP credentials (Cloud Run service account)")
        return storage.Client()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic and proper headers."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        
        return session
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a filename for safe storage."""
        # Remove or replace invalid characters
        name = re.sub(r'[<>:"/\\|?*]+', '_', name)
        # Remove leading/trailing dots and spaces
        name = name.strip('. ')
        # Limit length
        if len(name) > 200:
            name = name[:200]
        return name or "document"
    
    def _normalize_folder(self, folder: str) -> str:
        """Normalize folder path for GCS."""
        return folder.strip("/").replace("\\", "/") if folder else ""
    
    def _blob_exists(self, section_name: str, filename: str) -> bool:
        """Check if a blob already exists in GCS."""
        section_folder = self._sanitize_filename(section_name).replace(" ", "_")
        folder = f"{self.base_folder}/{section_folder}"
        normalized_folder = self._normalize_folder(folder)
        blob_path = f"{normalized_folder}/{filename}" if normalized_folder else filename
        return self.bucket.get_blob(blob_path) is not None
    
    def _upload_pdf(self, pdf_bytes: bytes, section_name: str, filename: str) -> bool:
        """Upload a PDF to GCS.
        
        Parameters
        ----------
        pdf_bytes : bytes
            The PDF file content.
        section_name : str
            The section name for organizing files.
        filename : str
            The sanitized filename.
        
        Returns
        -------
        bool
            True if upload succeeded, False otherwise.
        """
        try:
            section_folder = self._sanitize_filename(section_name).replace(" ", "_")
            folder = f"{self.base_folder}/{section_folder}"
            
            pdf_obj = {
                "name": filename,
                "bytes": pdf_bytes,
                "extension": "pdf",
            }
            
            upload_to_storage(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                pdf_obj=pdf_obj,
                folder=folder,
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to upload {filename} to GCS: {e}")
            return False
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a page and return BeautifulSoup object."""
        try:
            logger.debug(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            time.sleep(self.delay)  # Respect rate limit
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """Download a PDF and return its bytes."""
        try:
            logger.debug(f"Downloading PDF: {pdf_url}")
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            # Verify it's a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type or response.content[:4] == b'%PDF':
                time.sleep(self.delay)  # Respect rate limit
                return response.content
            else:
                logger.warning(f"URL did not return a PDF: {pdf_url}")
                return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            return None
    
    def _extract_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract PDF links from a page.
        
        Returns
        -------
        List[Dict[str, str]]
            List of dicts with 'url' and 'title' keys.
        """
        pdf_links = []
        
        # Look for direct PDF links
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Check if it's a PDF link
            if href.lower().endswith('.pdf') or '/sites/default/files/' in href:
                absolute_url = urljoin(base_url, href)
                
                # Extract title from link text or nearby elements
                title = link.get_text(strip=True)
                if not title:
                    # Try to find title in parent elements
                    parent = link.find_parent(['h2', 'h3', 'h4', 'div'])
                    if parent:
                        title = parent.get_text(strip=True)
                
                # Clean up title
                title = re.sub(r'\s+', ' ', title).strip()
                if not title:
                    # Use filename from URL as fallback
                    title = Path(urlparse(href).path).stem
                
                pdf_links.append({
                    'url': absolute_url,
                    'title': title
                })
        
        return pdf_links
    
    def _get_next_page_url(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Find the next page URL from pagination links.
        
        Returns
        -------
        Optional[str]
            The absolute URL of the next page, or None if there is no next page.
        """
        # Look for pagination links
        # Common patterns: "next", "›", "»", page numbers
        
        # Try to find a "next" link
        next_link = soup.find('a', text=re.compile(r'(successiv|next|›|»)', re.IGNORECASE))
        if next_link and next_link.get('href'):
            return urljoin(current_url, next_link['href'])
        
        # Try to find pagination with rel="next"
        next_link = soup.find('a', rel='next')
        if next_link and next_link.get('href'):
            return urljoin(current_url, next_link['href'])
        
        # Try to find pagination by class
        pager = soup.find(['nav', 'div'], class_=re.compile(r'pag', re.IGNORECASE))
        if pager:
            next_link = pager.find('a', text=re.compile(r'(successiv|next|›|»)', re.IGNORECASE))
            if next_link and next_link.get('href'):
                return urljoin(current_url, next_link['href'])
        
        return None
    
    def _scrape_section(self, section_name: str, section_path: str, max_pages: Optional[int] = None) -> Dict[str, int]:
        """Scrape a single section.
        
        Parameters
        ----------
        section_name : str
            Name of the section (used for organizing files).
        section_path : str
            URL path of the section.
        max_pages : Optional[int]
            Maximum number of pages to scrape (None = all pages).
        
        Returns
        -------
        Dict[str, int]
            Statistics: pdf_uploaded, pdf_skipped, pages_errors
        """
        logger.info(f"📂 Scraping section: {section_name}")
        
        stats = {
            'pdf_uploaded': 0,
            'pdf_skipped': 0,
            'pages_errors': 0,
        }
        
        # Handle node URLs (direct pages without pagination)
        is_node_url = '/node/' in section_path
        
        current_url = urljoin(self.BASE_URL, section_path)
        page_count = 0
        
        while current_url and (max_pages is None or page_count < max_pages):
            soup = self._fetch_page(current_url)
            
            if soup is None:
                stats['pages_errors'] += 1
                break
            
            logger.info(f"  📄 Page {page_count + 1}: {current_url}")
            
            # Extract PDF links
            pdf_links = self._extract_pdf_links(soup, current_url)
            logger.info(f"  Found {len(pdf_links)} PDF links")
            
            # Process each PDF
            for pdf_info in pdf_links:
                pdf_url = pdf_info['url']
                
                # Skip if already downloaded
                if pdf_url in self.downloaded_pdfs:
                    logger.debug(f"  ⏭️  Skipping already downloaded: {pdf_url}")
                    stats['pdf_skipped'] += 1
                    continue
                
                # Generate filename
                title = pdf_info['title']
                filename = self._sanitize_filename(title)
                if not filename.lower().endswith('.pdf'):
                    filename += '.pdf'
                
                # Check if already in GCS
                if self._blob_exists(section_name, filename):
                    logger.info(f"  ⏭️  Skipping existing file: {filename}")
                    stats['pdf_skipped'] += 1
                    self.downloaded_pdfs.add(pdf_url)
                    continue
                
                # Download PDF
                pdf_bytes = self._download_pdf(pdf_url)
                
                if pdf_bytes:
                    # Upload to GCS
                    if self._upload_pdf(pdf_bytes, section_name, filename):
                        logger.info(f"  ✅ Uploaded: {filename}")
                        stats['pdf_uploaded'] += 1
                        self.downloaded_pdfs.add(pdf_url)
                    else:
                        logger.error(f"  ❌ Failed to upload: {filename}")
                else:
                    logger.warning(f"  ❌ Failed to download: {pdf_url}")
            
            page_count += 1
            
            # For node URLs, don't look for next page
            if is_node_url:
                break
            
            # Find next page
            next_url = self._get_next_page_url(soup, current_url)
            
            if next_url and next_url != current_url:
                current_url = next_url
            else:
                break  # No more pages
        
        logger.info(f"  ✓ Completed section: {section_name} ({page_count} pages)")
        return stats
    
    def scrape(self, only_sections: Optional[List[str]] = None, max_pages: Optional[int] = None) -> Dict[str, int]:
        """Run the scraper on all or specified sections.
        
        Parameters
        ----------
        only_sections : Optional[List[str]]
            If provided, only scrape these sections (by name).
        max_pages : Optional[int]
            Maximum number of pages to scrape per section.
        
        Returns
        -------
        Dict[str, int]
            Summary statistics: sections, pdf_uploaded, pdf_skipped, 
            pages_errors, nodes_errors
        """
        logger.info("🚀 Starting Fondazione Nazionale Commercialisti scraper")
        logger.info(f"📦 Bucket: {self.bucket_name}")
        logger.info(f"📁 Base folder: {self.base_folder}")
        logger.info(f"⏱️  Delay: {self.delay}s")
        logger.info(f"🔄 Retries: {self.retries}")
        
        # Determine which sections to scrape
        if only_sections:
            sections_to_scrape = {}
            for requested_section in only_sections:
                # Try exact match first
                if requested_section in self.SECTIONS:
                    sections_to_scrape[requested_section] = self.SECTIONS[requested_section]
                else:
                    # Try partial match (case-insensitive)
                    requested_lower = requested_section.lower()
                    matched = False
                    for section_name, section_path in self.SECTIONS.items():
                        if requested_lower in section_name.lower() or section_name.lower() in requested_lower:
                            sections_to_scrape[section_name] = section_path
                            logger.info(f"  Matched '{requested_section}' → '{section_name}'")
                            matched = True
                            break
                    if not matched:
                        logger.warning(f"  No match found for: {requested_section}")
            
            if not sections_to_scrape:
                logger.warning(f"No matching sections found for: {only_sections}")
                logger.info(f"Available sections: {list(self.SECTIONS.keys())}")
        else:
            sections_to_scrape = self.SECTIONS
        
        logger.info(f"📋 Sections to scrape: {len(sections_to_scrape)}")
        
        # Scrape each section
        for section_name, section_path in sections_to_scrape.items():
            try:
                stats = self._scrape_section(section_name, section_path, max_pages)
                self.pdf_uploaded += stats['pdf_uploaded']
                self.pdf_skipped += stats['pdf_skipped']
                self.pages_errors += stats['pages_errors']
                self.sections_processed += 1
            except Exception as e:
                logger.error(f"❌ Error scraping section {section_name}: {e}", exc_info=True)
                if '/node/' in section_path:
                    self.nodes_errors += 1
                else:
                    self.pages_errors += 1
        
        # Return summary
        return {
            'sections': self.sections_processed,
            'pdf_uploaded': self.pdf_uploaded,
            'pdf_skipped': self.pdf_skipped,
            'pages_errors': self.pages_errors,
            'nodes_errors': self.nodes_errors,
        }
