"""
INPS – Circolari, Messaggi e Normativa PDF Scraper
---------------------------------------------------

This script uses Selenium WebDriver together with the official INPS
website to download all PDF documents listed under the "Circolari,
Messaggi e Normativa" section and upload them to Google Cloud Storage.

It applies the following logic:

1. Open the landing page and close any cookie banners that may block
   interaction with the page.
2. Filter results to include only documents from the "Ente Emanante"
   equal to ``INPS``.
3. Increase the number of results shown per page to 100 so that fewer
   pages need to be traversed.
4. Iterate through each page, following the "Vai al dettaglio"
   links (which contain the substring ``dettaglio.circolari-e-messaggi``)
   to reach the detail page for each document.
5. On the detail page, click the "Scarica il documento" link to
   download the PDF.
6. The PDF is uploaded to GCS with a hierarchy based on the year and type
   (``circolari`` or ``messaggi``) extracted from the document title.
7. A log of successfully processed URLs and any errors encountered
   during downloading is kept in memory so that the script can
   avoid re-downloading files.

Requirements:

    pip install selenium webdriver-manager google-cloud-storage

Author: Adapted for GCS upload by ChatGPT
"""

import io
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    NoSuchElementException,
    ElementClickInterceptedException,
)
from google.cloud import storage

from src.utils.upload_to_storage import upload_to_storage


logger = logging.getLogger(__name__)


# -------------------- CONFIG --------------------

BASE_URL = (
    "https://www.inps.it/it/it/inps-comunica/atti/circolari-messaggi-e-normativa.html"
)
# Substring present in the "Vai al dettaglio" anchor href
DETAIL_HREF_SUBSTR = "dettaglio.circolari-e-messaggi"

PAGELOAD_TIMEOUT = 120
WAIT_TIMEOUT = 45
RETRY_ATTEMPTS = 3
MAX_PAGES = 1000  # Safety cap; adjust if needed


# -------------------- UTILITY FUNCTIONS --------------------

def build_driver(headless: bool = True) -> webdriver.Chrome:
    """Create a Chrome WebDriver with appropriate options.

    When ``headless`` is set, Chrome runs without a visible window which
    is suitable for server or notebook environments.
    """
    chrome_opts = webdriver.ChromeOptions()
    if headless:
        # Use the new headless mode available in recent Chrome
        chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--window-size=1600,1200")
    # Standard stability flags
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--lang=it-IT")
    
    # Use system-installed ChromeDriver (for Docker/Cloud Run)
    # Falls back to default PATH lookup if not found at /usr/local/bin/chromedriver
    try:
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=chrome_opts)
    except Exception:
        # Fallback for local development - let Selenium find chromedriver in PATH
        driver = webdriver.Chrome(options=chrome_opts)
    
    driver.set_page_load_timeout(PAGELOAD_TIMEOUT)
    return driver


def wait_css(driver: webdriver.Chrome, selector: str, timeout: int = WAIT_TIMEOUT):
    """Return the first element matching a CSS selector within timeout."""
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
    )


def click_safely(driver: webdriver.Chrome, elem):
    """Attempt to click an element, retrying a few times if intercepted.

    In some cases elements are covered by overlays or become stale; this
    helper scrolls the element into view and tries again.  As a last
    resort it executes a JavaScript click.
    """
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
    for _ in range(3):
        try:
            elem.click()
            return
        except (ElementClickInterceptedException, StaleElementReferenceException):
            time.sleep(0.7)
    # Fallback to JavaScript click
    driver.execute_script("arguments[0].click();", elem)


def close_cookie_banner_if_any(driver: webdriver.Chrome):
    """Close common cookie banners that may block the page.

    The INPS portal displays cookie consent modals that obscure the page
    until dismissed.  This function attempts to click on buttons or
    icons that close these modals by searching for typical labels or
    aria attributes.  If none are found it silently returns.
    """

    def click_button_by_text(text: str) -> bool:
        # Attempt to click a button that contains the given text
        try:
            for b in driver.find_elements(By.TAG_NAME, "button"):
                txt = (b.get_attribute("innerText") or "").strip().lower()
                if text.lower() in txt:
                    click_safely(driver, b)
                    time.sleep(0.3)
                    return True
        except Exception:
            pass
        return False

    try:
        # Close icons (X or "chiudi")
        for x in driver.find_elements(By.CSS_SELECTOR, "button, .ot-close-icon, .btn-close, .close"):
            label = " ".join(
                [
                    x.get_attribute("aria-label") or "",
                    x.get_attribute("title") or "",
                    x.get_attribute("innerText") or "",
                ]
            ).lower()
            if any(t in label for t in ["chiudi", "close", "×", "x"]):
                try:
                    click_safely(driver, x)
                    return
                except Exception:
                    pass
        # Accept or reject cookie buttons
        if click_button_by_text("accetta tutti"):
            return
        if click_button_by_text("rifiuta i cookie non tecnici"):
            return
    except Exception:
        pass


def select_ente_inps(driver: webdriver.Chrome):
    """Select the 'INPS' option for the 'Ente Emanante' filter.

    The filter uses either a native ``select`` element or a custom
    combobox built from div/buttons.  This function detects both
    patterns and sets the value accordingly, then clicks the
    'Applica filtri' button to apply the selection.
    """
    # Try native select first
    try:
        sel = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//label[contains(.,'Ente Emanante')]/following::select[1]",
                )
            )
        )
        # Use JavaScript to set the option value and dispatch a change event
        driver.execute_script(
            """
            const el = arguments[0];
            for (const o of el.options) {
                if (o.text.trim() === 'INPS') {
                    el.value = o.value;
                    el.dispatchEvent(new Event('change'));
                    break;
                }
            }
            """,
            sel,
        )
    except TimeoutException:
        # Fallback for custom select (div/combobox)
        box = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//*[contains(.,'Ente Emanante')]/following::*[(self::div or self::button)"
                    " and (@role='combobox' or contains(@class,'select'))][1]",
                )
            )
        )
        click_safely(driver, box)
        time.sleep(0.3)
        opt = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//li[normalize-space()='INPS'] | //div[normalize-space()='INPS']",
                )
            )
        )
        click_safely(driver, opt)

    # Click the apply filters button
    apply_btn = WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Applica filtri')]"))
    )
    click_safely(driver, apply_btn)
    # Wait for the results table to appear
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.presence_of_element_located((By.XPATH, "//table | //div[contains(@class,'table')]"))
    )


def set_elements_per_page_100(driver: webdriver.Chrome):
    """Set the 'Elementi per pagina' drop-down to show 100 results per page."""
    try:
        sel = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//*[contains(.,'Elementi per pagina')]/following::*[(self::select or self::button or self::div)][1]",
                )
            )
        )
        click_safely(driver, sel)
        time.sleep(0.2)
        # Try native select first
        try:
            native = sel.find_element(By.TAG_NAME, "select")
            driver.execute_script(
                """
                const el = arguments[0];
                for (const o of el.options) {
                    if (o.text.trim() === '100') {
                        el.value = o.value;
                        el.dispatchEvent(new Event('change'));
                        break;
                    }
                }
                """,
                native,
            )
        except NoSuchElementException:
            opt100 = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//li[normalize-space()='100'] | //div[normalize-space()='100'] | //button[normalize-space()='100']",
                    )
                )
            )
            click_safely(driver, opt100)
        # Give the page time to reload results
        time.sleep(1.0)
    except Exception:
        # If we fail to locate the control just continue; default may be 20
        pass


def collect_detail_links(driver: webdriver.Chrome) -> List[str]:
    """Return a list of detail-page URLs present on the current listing page."""
    anchors = driver.find_elements(By.CSS_SELECTOR, f"a[href*='{DETAIL_HREF_SUBSTR}']")
    links: List[str] = []
    for a in anchors:
        href = a.get_attribute("href")
        if href and DETAIL_HREF_SUBSTR in href:
            links.append(href)
    return links


def get_first_link_webelement(driver: webdriver.Chrome):
    """Return the WebElement corresponding to the first detail link on the page."""
    return WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, f"a[href*='{DETAIL_HREF_SUBSTR}']")
        )
    )


def get_first_result_href(driver: webdriver.Chrome) -> Optional[str]:
    """Return the href of the first detail link on the page, or None if absent."""
    try:
        el = get_first_link_webelement(driver)
        return el.get_attribute("href")
    except TimeoutException:
        return None


def find_next_button_candidates(driver: webdriver.Chrome):
    """Return a list of candidate elements that may navigate to the next page."""
    xpaths = [
        "//*[contains(., 'Vai alla pagina successiva') and (self::button or self::a)]",
        "//*[@aria-label='Vai alla pagina successiva']",
        "//*[@title='Vai alla pagina successiva']",
        # For numeric navigation: last anchor/button in a pagination list
        "//nav//*[contains(@class,'pagination')]//*[self::a or self::button][not(@aria-disabled='true')][last()]",
    ]
    elems = []
    for xp in xpaths:
        elems.extend(driver.find_elements(By.XPATH, xp))
    # Numeric buttons: look for 2,3,4,5 etc.
    nums = driver.find_elements(
        By.XPATH,
        "//a[normalize-space()='2' or normalize-space()='3' or normalize-space()='4' or normalize-space()='5'] | "
        "//button[normalize-space()='2' or normalize-space()='3' or normalize-space()='4' or normalize-space()='5']",
    )
    elems.extend(nums)
    # Deduplicate
    seen: Set[Tuple[str, str]] = set()
    unique = []
    for e in elems:
        try:
            key = (e.tag_name, (e.get_attribute("outerHTML") or "")[:200])
        except StaleElementReferenceException:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def goto_next_page_if_any(driver: webdriver.Chrome, prev_first_el) -> bool:
    """Attempt to navigate to the next page.

    Scrolls to the bottom of the page to ensure pagination controls are
    visible, then clicks on the first available candidate found by
    ``find_next_button_candidates``.  Waits for the first result
    element to become stale or its ``href`` to change in order to
    confirm that navigation occurred.

    Returns ``True`` if a navigation was attempted and the page
    changed, ``False`` otherwise.
    """
    # Scroll to bottom to reveal pagination controls
    driver.execute_script(
        "window.scrollTo({top: document.body.scrollHeight, behavior: 'instant'});"
    )
    time.sleep(0.5)
    candidates = find_next_button_candidates(driver)
    # Filter disabled buttons
    active = []
    for el in candidates:
        try:
            disabled = (el.get_attribute("aria-disabled") == "true") or (
                "disabled" in (el.get_attribute("class") or "").lower()
            )
            if not disabled and el.is_displayed():
                active.append(el)
        except StaleElementReferenceException:
            continue
    if not active:
        return False
    # Use the first viable candidate
    next_el = active[0]
    try:
        click_safely(driver, next_el)
    except Exception:
        # Fallback JS click
        driver.execute_script("arguments[0].click();", next_el)
    # Wait for first element to become stale or href to change
    try:
        WebDriverWait(driver, WAIT_TIMEOUT).until(EC.staleness_of(prev_first_el))
    except TimeoutException:
        old_href = prev_first_el.get_attribute("href")
        try:
            WebDriverWait(driver, WAIT_TIMEOUT).until(
                lambda d: (get_first_result_href(d) or "") != (old_href or "")
            )
        except TimeoutException:
            return False
    time.sleep(0.6)
    return True


def parse_year_and_type_from_detail(driver: webdriver.Chrome) -> Tuple[str, str]:
    """Extract the publication year and type ('circolari' or 'messaggi') from the detail page."""
    try:
        h1 = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "h1, .page-title, [role='heading']")
            )
        ).get_attribute("innerText").strip()
        m = re.search(r"(\d{4})", h1)
        year = m.group(1) if m else "sconosciuto"
        tipo = (
            "circolari"
            if "circolar" in h1.lower()
            else ("messaggi" if "messagg" in h1.lower() else "altri")
        )
        return year, tipo
    except Exception:
        return "sconosciuto", "altri"


def get_main_pdf_anchor(driver: webdriver.Chrome):
    """Locate the anchor element to download the main PDF from the detail page."""
    try:
        # Primary: link with explicit text
        a = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//a[contains(., 'Scarica il documento')]")
            )
        )
        return a
    except TimeoutException:
        pass
    try:
        # Fallback: first PDF link
        a = driver.find_element(By.CSS_SELECTOR, "a[href$='.pdf']")
        return a
    except NoSuchElementException:
        return None


def download_pdf_to_memory(driver: webdriver.Chrome, pdf_url: str) -> Optional[bytes]:
    """Download a PDF from the given URL and return its bytes.
    
    Uses the driver's session cookies for authentication.
    """
    import requests
    
    try:
        # Get cookies from Selenium
        cookies = driver.get_cookies()
        session_cookies = {cookie['name']: cookie['value'] for cookie in cookies}
        
        # Download the PDF
        response = requests.get(pdf_url, cookies=session_cookies, timeout=60)
        response.raise_for_status()
        
        # Verify it's a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' in content_type or response.content[:4] == b'%PDF':
            return response.content
        else:
            logger.warning(f"URL did not return a PDF: {pdf_url}")
            return None
    except Exception as e:
        logger.error(f"Error downloading PDF from {pdf_url}: {e}")
        return None


class INPSScraper:
    """Scraper for INPS Circolari and Messaggi documents.
    
    Parameters
    ----------
    bucket_name : str
        Name of the target GCS bucket.
    storage_client : Optional[storage.Client], optional
        Preconfigured Google Cloud Storage client.  If not provided a new
        client will be instantiated.
    base_folder : str, optional
        Root folder inside the bucket where files should be uploaded.
        Defaults to ``inps``.
    headless : bool, optional
        Whether to run Chrome in headless mode. Defaults to True.
    max_pages : int, optional
        Maximum number of pages to scrape. Defaults to MAX_PAGES constant.
    """
    
    def __init__(
        self,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
        base_folder: str = "inps",
        headless: bool = True,
        max_pages: int = MAX_PAGES,
    ):
        self.bucket_name = bucket_name
        self.storage_client = storage_client or storage.Client()
        self.base_folder = base_folder
        self.headless = headless
        self.max_pages = max_pages
        self.downloaded_urls: Set[str] = set()
        self.error_urls: Set[str] = set()
        self.files_by_year: dict = {}
        self.total_downloaded = 0
        self.total_errors = 0
        self.stop_requested = False
        self.bucket = self.storage_client.bucket(self.bucket_name)
    
    def upload_pdf_to_gcs(self, pdf_bytes: bytes, year: str, tipo: str, filename: str) -> bool:
        """Upload a PDF to GCS in the appropriate folder structure.
        
        Parameters
        ----------
        pdf_bytes : bytes
            The PDF file content.
        year : str
            The year folder (e.g., "2024").
        tipo : str
            The type folder ("circolari", "messaggi", or "altri").
        filename : str
            The original filename.
        
        Returns
        -------
        bool
            True if upload succeeded, False otherwise.
        """
        try:
            # Sanitize filename
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            pdf_obj = {
                "name": safe_filename,
                "bytes": pdf_bytes,
                "extension": "pdf",
            }
            
            # Folder structure: base_folder/year/tipo
            folder = f"{self.base_folder}/{year}/{tipo}"
            
            upload_to_storage(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                pdf_obj=pdf_obj,
                folder=folder,
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            return False

    def _normalize_folder(self, folder: str) -> str:
        return folder.strip("/").replace("\\", "/") if folder else ""

    def _blob_exists(self, year: str, tipo: str, filename: str) -> bool:
        folder = f"{self.base_folder}/{year}/{tipo}"
        normalized_folder = self._normalize_folder(folder)
        blob_path = f"{normalized_folder}/{filename}" if normalized_folder else filename
        return self.bucket.get_blob(blob_path) is not None
    
    def download_page_links(
        self,
        driver: webdriver.Chrome,
        links: List[str],
        page_idx: int,
    ):
        """Visit each detail link on the page and download the PDF if not already processed."""
        if self.stop_requested:
            return
        for detail_url in links:
            if self.stop_requested:
                break
            # Skip if previously downloaded or errored
            if detail_url in self.downloaded_urls or detail_url in self.error_urls:
                continue
            
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    driver.get(detail_url)
                    close_cookie_banner_if_any(driver)
                    year, tipo = parse_year_and_type_from_detail(driver)
                    
                    a = get_main_pdf_anchor(driver)
                    if a is None:
                        raise RuntimeError("PDF non trovato su pagina dettaglio")
                    
                    pdf_url = a.get_attribute("href")
                    if not pdf_url:
                        raise RuntimeError("PDF URL not found")
                    
                    # Extract filename from title or URL
                    try:
                        h1 = driver.find_element(By.CSS_SELECTOR, "h1, .page-title").get_attribute("innerText").strip()
                        filename = re.sub(r'[<>:"/\\|?*]', '_', h1)[:200] + ".pdf"
                    except:
                        filename = Path(pdf_url).name or f"document_{len(self.downloaded_urls)}.pdf"
                    if self._blob_exists(year, tipo, filename):
                        logger.info(
                            "🛑 Existing document detected at %s/%s/%s; stopping scraper.",
                            year,
                            tipo,
                            filename,
                        )
                        self.stop_requested = True
                        return
                    
                    # Download PDF to memory
                    pdf_bytes = download_pdf_to_memory(driver, pdf_url)
                    if pdf_bytes is None:
                        raise RuntimeError("Failed to download PDF")
                    
                    # Upload to GCS
                    success = self.upload_pdf_to_gcs(pdf_bytes, year, tipo, filename)
                    
                    if success:
                        self.downloaded_urls.add(detail_url)
                        self.total_downloaded += 1
                        
                        # Track by year
                        if year not in self.files_by_year:
                            self.files_by_year[year] = 0
                        self.files_by_year[year] += 1
                        
                        logger.info(f"[Pagina {page_idx}] ✅ {detail_url}")
                    else:
                        raise RuntimeError("Upload to GCS failed")
                    
                    break
                    
                except Exception as e:
                    if attempt == RETRY_ATTEMPTS - 1:
                        self.error_urls.add(detail_url)
                        self.total_errors += 1
                        logger.error(f"[Pagina {page_idx}] ❌ {detail_url} | {type(e).__name__}: {e}")
                    else:
                        time.sleep(2)
    
    def scrape(self) -> dict:
        """Main scraping method.
        
        Returns
        -------
        dict
            Statistics about the scraping run.
        """
        driver = build_driver(headless=self.headless)
        try:
            driver.get(BASE_URL)
            close_cookie_banner_if_any(driver)
            
            # Apply the INPS filter
            select_ente_inps(driver)
            set_elements_per_page_100(driver)
            time.sleep(1.0)
            
            page_idx = 1
            pages_without_change = 0
            
            while page_idx <= self.max_pages:
                if self.stop_requested:
                    break
                first_el = get_first_link_webelement(driver)
                page_links = collect_detail_links(driver)
                logger.info(f"[Pagina {page_idx}] link trovati: {len(page_links)}")
                
                before = len(self.downloaded_urls)
                self.download_page_links(driver, page_links, page_idx)
                if self.stop_requested:
                    logger.info("🛑 Existing document found in GCS; stopping scrape loop.")
                    break
                after = len(self.downloaded_urls)
                
                if after == before:
                    pages_without_change += 1
                else:
                    pages_without_change = 0
                
                # Stop if we see no new links for two pages
                if pages_without_change >= 2:
                    logger.info("[Stop] Nessun nuovo link in 2 pagine consecutive. Fine.")
                    break
                
                if self.stop_requested:
                    break
                # Navigate to next page
                moved = goto_next_page_if_any(driver, prev_first_el=first_el)
                if not moved:
                    # Attempt to reset results per page and try again
                    set_elements_per_page_100(driver)
                    time.sleep(0.5)
                    moved = goto_next_page_if_any(driver, prev_first_el=first_el)
                
                if not moved:
                    logger.info("[Stop] Nessuna pagina successiva disponibile. Fine.")
                    break
                
                page_idx += 1
            
            logger.info("Scraping completato.")
            
        finally:
            try:
                driver.quit()
            except Exception:
                pass
        
        return {
            "pages_visited": page_idx,
            "files_downloaded": self.total_downloaded,
            "files_errored": self.total_errors,
            "files_by_year": self.files_by_year,
        }
