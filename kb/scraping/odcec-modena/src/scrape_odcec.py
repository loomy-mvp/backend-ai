#!/usr/bin/env python
# coding: utf-8

"""
ODCEC Modena Scraper
====================

Scraper for documents from ODCEC Modena (Ordine Dottori Commercialisti ed Esperti Contabili di Modena).
Downloads PDFs from the document management system and uploads them to Google Cloud Storage.
"""

import logging
import os
import re
import time
from typing import Optional, Set
from urllib.parse import urlsplit, parse_qs

import requests
from google.cloud import storage
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import create_urllib3_context

from src.utils.upload_to_storage import upload_to_storage

logger = logging.getLogger(__name__)

START_URL = (
    "https://www.commercialisti.mo.it/servizi/gestionedocumentale/"
    "ricerca_fase02.aspx?fn=7&Campo_686=8&Campo_704=37&Campo_763=&Campo_764=&AggiornaDB=Cerca"
)

# ----------------------------
# Adapter SSL per ciphers legacy (fix DH_KEY_TOO_SMALL)
# ----------------------------
CIPHERS = "DEFAULT:@SECLEVEL=1"  # consente DH 1024-bit ecc.


class WeakCiphersAdapter(HTTPAdapter):
    """HTTP Adapter with legacy SSL ciphers for compatibility with old servers"""
    
    def __init__(self, *args, **kwargs):
        # Retry robusti per i download
        retries = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        super().__init__(max_retries=retries, *args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context(ciphers=CIPHERS)
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        ctx = create_urllib3_context(ciphers=CIPHERS)
        kwargs["ssl_context"] = ctx
        return super().proxy_manager_for(*args, **kwargs)


def make_download_session():
    """Create a requests session with SSL compatibility and retry logic"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    })
    # Monta l'adapter legacy solo su HTTPS (va bene per tutto il dominio)
    s.mount("https://", WeakCiphersAdapter())
    return s


def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe storage"""
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    return name.strip().strip("._")


class ODCECModenaScraper:
    """Main scraper class for ODCEC Modena documents"""
    
    def __init__(
        self,
        bucket_name: str,
        base_folder: str = "odcec-modena",
        storage_client: Optional[storage.Client] = None,
        headless: bool = True
    ):
        self.bucket_name = bucket_name
        self.base_folder = base_folder
        self.storage_client = storage_client or storage.Client()
        self.headless = headless
        self.downloaded: Set[str] = set()
        self.session = make_download_session()
        self.driver = None
        
    def _init_driver(self):
        """Initialize Selenium WebDriver with Chrome"""
        chrome_opts = Options()
        if self.headless:
            chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--ignore-certificate-errors")
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--disable-dev-shm-usage")
        chrome_opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
        
        self.driver = webdriver.Chrome(options=chrome_opts)
        self.driver.set_page_load_timeout(60)
        
    def _cleanup_driver(self):
        """Close and cleanup the WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")
            finally:
                self.driver = None
    
    def download_and_upload_pdf(self, url: str, cookies_from_browser: list) -> bool:
        """
        Download PDF from URL and upload to GCS
        Returns True if successful, False otherwise
        """
        # Skip if already downloaded
        if url in self.downloaded:
            logger.debug(f"Skipping duplicate URL: {url}")
            return False
        
        # Unisce le cookie di Selenium nella sessione requests
        for c in cookies_from_browser:
            # limita al dominio del sito
            self.session.cookies.set(
                c["name"], c["value"], domain=c.get("domain") or "www.commercialisti.mo.it"
            )
        
        # Prova a costruire un nome file sensato
        parsed = urlsplit(url)
        filename = os.path.basename(parsed.path) or "documento.pdf"
        
        # Se è una download.aspx con querystring nomeFile, usalo come filename
        qs = parse_qs(parsed.query)
        if "nomeFile" in qs and qs["nomeFile"]:
            filename = qs["nomeFile"][0]
        
        filename = sanitize_filename(filename)
        
        # Ensure .pdf extension
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        try:
            # Scarica con ciphers legacy (fix DH_KEY_TOO_SMALL)
            with self.session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                
                # Read content
                pdf_content = b""
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        pdf_content += chunk
                
                # Skip if not a PDF
                if not pdf_content.startswith(b"%PDF"):
                    logger.warning(f"Not a PDF file: {url}")
                    return False
                
                # Upload to GCS
                pdf_obj = {
                    "name": filename,
                    "bytes": pdf_content,
                    "extension": "pdf"
                }
                
                upload_to_storage(
                    storage_client=self.storage_client,
                    bucket_name=self.bucket_name,
                    pdf_obj=pdf_obj,
                    folder=self.base_folder
                )
                
                self.downloaded.add(url)
                logger.info(f"✔ Uploaded: {filename}")
                return True
                
        except Exception as e:
            logger.error(f"✖ Error downloading {url}: {e}")
            return False
    
    def scrape(self) -> dict:
        """
        Run the scraper and return statistics
        Returns dict with upload counts and errors
        """
        stats = {
            "pdf_uploaded": 0,
            "pdf_skipped": 0,
            "pdf_errors": 0,
            "pages_processed": 0
        }
        
        try:
            # Initialize driver
            self._init_driver()
            
            logger.info(f"🚀 Starting ODCEC Modena scraper")
            logger.info(f"📦 Target bucket: {self.bucket_name}")
            logger.info(f"📁 Base folder: {self.base_folder}")
            logger.info(f"🔗 Starting URL: {START_URL}")
            
            self.driver.get(START_URL)
            page = 1
            
            while True:
                # Attende caricamento lista risultati con almeno un link "scarica"
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_all_elements_located(
                        (By.XPATH, "//a[contains(translate(.,"
                         "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'scarica')]")
                    )
                )
                
                # Raccoglie tutti i link "scarica il documento"
                links = self.driver.find_elements(
                    By.XPATH,
                    "//a[contains(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'scarica')]"
                )
                
                logger.info(f"[Pagina {page}] Link 'scarica' trovati: {len(links)}")
                
                # Estrai HREF (nei WebForms spesso è una download.aspx con query)
                hrefs = []
                for a in links:
                    href = a.get_attribute("href")
                    if href:
                        hrefs.append(href)
                
                # Scarica ciascun PDF
                cookies_list = self.driver.get_cookies()  # cookie ASP.NET_SessionId ecc.
                for href in hrefs:
                    if href in self.downloaded:
                        stats["pdf_skipped"] += 1
                        continue
                    
                    success = self.download_and_upload_pdf(href, cookies_list)
                    if success:
                        stats["pdf_uploaded"] += 1
                    else:
                        stats["pdf_errors"] += 1
                
                stats["pages_processed"] += 1
                
                # Tenta di andare alla pagina successiva (pulsante 'Avanti' / 'Successivo' / »)
                moved = False
                for xpath in [
                    "//a[contains(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'avanti')]",
                    "//a[contains(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'successiv')]",
                    "//a[normalize-space(text())='»']",
                    "//a[span[normalize-space(text())='»']]",
                ]:
                    next_candidates = self.driver.find_elements(By.XPATH, xpath)
                    if next_candidates:
                        # Click sul primo candidato
                        self.driver.execute_script("arguments[0].click();", next_candidates[0])
                        page += 1
                        moved = True
                        # Dai il tempo al pager di ricaricare
                        WebDriverWait(self.driver, 30).until(EC.staleness_of(links[0]))
                        # Piccola attesa per stabilizzare
                        time.sleep(1.2)
                        break
                
                if not moved:
                    logger.info("Fine elenco: nessuna pagina successiva trovata.")
                    break
            
            logger.info(f"\nTotale PDF scaricati: {stats['pdf_uploaded']}")
            
        except Exception as e:
            logger.error(f"❌ Scraping failed: {e}", exc_info=True)
            raise
        finally:
            self._cleanup_driver()
        
        return stats
