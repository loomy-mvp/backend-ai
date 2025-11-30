"""
NT+ Fisco (Il Sole 24 Ore) Article Scraper
==========================================

This script uses Selenium WebDriver to navigate the NT+ Fisco website
and extract article previews from various sections. It saves the extracted
content (title, subtitle, preview, summary) as JSON files to Google Cloud Storage.

The scraper:
1. Opens the main NT+ Fisco page
2. Navigates to each section from the navigation menu
3. Extracts article previews visible on the page
4. Clicks "Mostra altri" to load more articles
5. Saves each article as a JSON file to GCS

Sections scraped:
- Adempimenti
- Contabilità
- Controlli e liti
- Diritto
- Finanza
- Imposte
- Professioni
- Analisi
- Schede
- L'esperto risponde
- Rubriche
- Speciali

Requirements:
    pip install selenium google-cloud-storage beautifulsoup4

Author: Loomy AI Scraper
"""

import json
import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

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
    WebDriverException,
)
from google.cloud import storage
from bs4 import BeautifulSoup

import sys
import os

# Handle imports for both local development and Cloud Run
try:
    # Cloud Run: utils is in src/utils/
    from src.utils.upload_to_storage import upload_to_storage
except ImportError:
    # Local development: utils is in parent directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.upload_to_storage import upload_to_storage


logger = logging.getLogger(__name__)


# -------------------- CONFIG --------------------

BASE_URL = "https://ntplusfisco.ilsole24ore.com/"

# Section configuration: display name -> URL path (under /sez/)
SECTIONS = {
    "Adempimenti": "adempimenti",
    "Contabilità": "contabilita",
    "Controlli e liti": "controlli-e-liti",
    "Diritto": "diritto",
    "Finanza": "finanza",
    "Imposte": "imposte",
    "Professioni": "professioni",
    "Analisi": "analisi",
    "Schede": "schede",
    "L'esperto risponde": "esperto-risponde",
    "Rubriche": "rubriche",
    "Speciali": "speciali",
}

PAGELOAD_TIMEOUT = 60
WAIT_TIMEOUT = 30
RETRY_ATTEMPTS = 3
DEFAULT_DELAY = 2.0  # Delay between actions


# -------------------- UTILITY FUNCTIONS --------------------

def build_driver(headless: bool = True) -> webdriver.Chrome:
    """Create a Chrome WebDriver with appropriate options.

    When ``headless`` is set, Chrome runs without a visible window which
    is suitable for server or notebook environments.
    """
    chrome_opts = webdriver.ChromeOptions()
    if headless:
        chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--window-size=1920,1080")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--lang=it-IT")
    chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
    chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_opts.add_experimental_option("useAutomationExtension", False)
    
    # User agent to appear more like a regular browser
    chrome_opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    try:
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=chrome_opts)
    except Exception:
        driver = webdriver.Chrome(options=chrome_opts)
    
    driver.set_page_load_timeout(PAGELOAD_TIMEOUT)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """
    })
    return driver


def wait_css(driver: webdriver.Chrome, selector: str, timeout: int = WAIT_TIMEOUT):
    """Return the first element matching a CSS selector within timeout."""
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
    )


def click_safely(driver: webdriver.Chrome, elem):
    """Attempt to click an element, retrying a few times if intercepted."""
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
    time.sleep(0.3)
    for _ in range(3):
        try:
            elem.click()
            return
        except (ElementClickInterceptedException, StaleElementReferenceException):
            time.sleep(0.5)
    driver.execute_script("arguments[0].click();", elem)


def close_cookie_banner_if_any(driver: webdriver.Chrome):
    """Close common cookie banners that may block the page."""
    try:
        # Try multiple common cookie button patterns
        patterns = [
            "//button[contains(., 'Accetta') or contains(., 'accetta')]",
            "//button[contains(., 'Accept') or contains(., 'accept')]",
            "//button[contains(., 'OK') or contains(., 'Ok')]",
            "//button[contains(., 'Continua') or contains(., 'continua')]",
            "//*[contains(@class, 'cookie')]//button",
            "//*[contains(@id, 'cookie')]//button",
            "//button[contains(@class, 'accept')]",
            "//button[contains(@class, 'consent')]",
        ]
        
        for pattern in patterns:
            try:
                buttons = driver.find_elements(By.XPATH, pattern)
                for btn in buttons:
                    if btn.is_displayed():
                        click_safely(driver, btn)
                        time.sleep(0.5)
                        return
            except Exception:
                continue
        
        # Also try to close any generic modal/overlay close buttons
        close_patterns = [
            "//button[contains(@aria-label, 'close') or contains(@aria-label, 'chiudi')]",
            "//*[contains(@class, 'close')]/button",
            "//button[contains(@class, 'btn-close')]",
        ]
        
        for pattern in close_patterns:
            try:
                buttons = driver.find_elements(By.XPATH, pattern)
                for btn in buttons:
                    if btn.is_displayed():
                        click_safely(driver, btn)
                        time.sleep(0.3)
            except Exception:
                continue
                
    except Exception as e:
        logger.debug(f"Cookie banner handling: {e}")


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Create a safe filename from text."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*\r\n\t]', '_', text)
    safe = re.sub(r'\s+', '_', safe)
    safe = re.sub(r'_+', '_', safe)
    safe = safe.strip('_')
    
    # Truncate if needed
    if len(safe) > max_length:
        safe = safe[:max_length]
    
    return safe or "untitled"


def generate_article_id(title: str, url: str = "") -> str:
    """Generate a unique ID for an article based on title and URL."""
    content = f"{title}|{url}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


# -------------------- ARTICLE EXTRACTION --------------------

def extract_article_details_from_url(driver: webdriver.Chrome, url: str) -> Optional[Dict]:
    """Visit an article URL and extract full title, subtitle/preview.
    
    The article page contains:
    - h1: Full article title
    - h2.asummary: Full subtitle/preview (longer than list preview)
    - .subsection: Section category
    
    Note: The full article body (.abody) is behind a paywall.
    
    Parameters
    ----------
    driver : webdriver.Chrome
        The Selenium WebDriver instance.
    url : str
        The article URL to visit.
    
    Returns
    -------
    Optional[Dict]
        Article data with title, subtitle, preview, or None if extraction failed.
    """
    try:
        driver.get(url)
        time.sleep(2)  # Wait for page to load
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Check for 404 page
        h1 = soup.select_one('h1')
        if h1 and 'non trovata' in h1.get_text().lower():
            logger.warning(f"Article not found (404): {url}")
            return None
        
        # Extract title
        title = h1.get_text(strip=True) if h1 else None
        if not title:
            logger.warning(f"No title found for article: {url}")
            return None
        
        # Extract preview text from p.atabs-txt (this contains the full preview/summary)
        preview_elem = soup.select_one('p.atabs-txt')
        preview = preview_elem.get_text(strip=True) if preview_elem else None
        
        # Fallback: try h2.asummary if no preview found
        if not preview:
            subtitle_elem = soup.select_one('h2.asummary, .asummary')
            preview = subtitle_elem.get_text(strip=True) if subtitle_elem else None
        
        # Extract section/category
        section_elem = soup.select_one('.meta-part.subsection, .subsection')
        section_name = section_elem.get_text(strip=True) if section_elem else None
        
        # Extract author
        author_elem = soup.select_one('.auth, .author, .byline')
        author = None
        if author_elem:
            author = author_elem.get_text(strip=True)
            author = re.sub(r'^di\s*', '', author)  # Remove "di " prefix
            # Fix "Name1eName2" -> "Name1 e Name2" (missing space around 'e')
            author = re.sub(r'(\w)e([A-Z])', r'\1 e \2', author)
            # Fix "contributo diName" -> "contributo di Name"
            author = re.sub(r'(contributo di)([A-Z])', r'\1 \2', author)
        
        # Extract date
        date_elem = soup.select_one('time.time, time')
        date = None
        if date_elem:
            date = date_elem.get('datetime') or date_elem.get_text(strip=True)
        
        return {
            "title": title,
            "preview": preview,  # The full preview/summary text from p.atabs-txt
            "section_name": section_name,
            "author": author,
            "date": date,
            "url": url,
        }
        
    except Exception as e:
        logger.error(f"Error extracting article details from {url}: {e}")
        return None


def extract_article_urls_from_page(driver: webdriver.Chrome) -> List[str]:
    """Extract all article URLs from the current section page.
    
    Returns a list of unique article URLs found on the page.
    """
    urls = []
    seen = set()
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Find all article links
    for link in soup.select('a[href*="/art/"]'):
        href = link.get('href', '')
        if href:
            if href.startswith('/'):
                href = f"https://ntplusfisco.ilsole24ore.com{href}"
            if href not in seen and '/art/' in href:
                seen.add(href)
                urls.append(href)
    
    logger.debug(f"Found {len(urls)} article URLs on page")
    return urls


def extract_articles_from_page(driver: webdriver.Chrome) -> List[Dict]:
    """Extract all article previews from the current page.
    
    Uses specific selectors for the NT+ Fisco website structure.
    """
    articles = []
    
    # Wait for content to load
    time.sleep(2)
    
    # Get page source for BeautifulSoup parsing
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Find all article containers - specific to NT+ Fisco structure
    # .aprev is the standard article preview container
    # .archcard is used in "speciali" section for archive cards
    article_containers = soup.select('.aprev, .archcard')
    
    found_articles = set()  # Track unique articles by title hash
    
    for container in article_containers:
        article = extract_nt_fisco_article(container)
        if article and article.get('title'):
            title_hash = hashlib.md5(article['title'].encode()).hexdigest()
            if title_hash not in found_articles:
                found_articles.add(title_hash)
                articles.append(article)
    
    logger.debug(f"Extracted {len(articles)} articles from page")
    return articles


def extract_nt_fisco_article(container) -> Optional[Dict]:
    """Extract article data from an NT+ Fisco article container (.aprev or .archcard)."""
    try:
        # Check if this is an archcard (speciali section) or aprev (regular)
        is_archcard = 'archcard' in container.get('class', [])
        
        # Title extraction
        title = None
        if is_archcard:
            # archcard uses h3 directly or a link inside
            title_elem = container.select_one('h3, .archcard-title')
            if title_elem:
                link = title_elem.select_one('a')
                if link:
                    title = link.get_text(strip=True)
                else:
                    title = title_elem.get_text(strip=True)
        else:
            # aprev uses h3.aprev-title
            title_elem = container.select_one('h3.aprev-title, .aprev-title')
            if title_elem:
                link = title_elem.select_one('a')
                if link:
                    title = link.get_text(strip=True)
                else:
                    title = title_elem.get_text(strip=True)
        
        if not title or len(title) < 10:
            return None
        
        # URL extraction
        url = None
        link = container.select_one('a[href*="/art/"], a[href*="/speciale/"]')
        if not link:
            link = container.select_one('a[href]')
        if link:
            href = link.get("href", "")
            if href.startswith("/"):
                url = f"https://ntplusfisco.ilsole24ore.com{href}"
            elif href.startswith("http"):
                url = href
        
        # Subtitle/kicker extraction (usually the subsection like "I temi di NT+")
        subtitle = None
        subsection_elem = container.select_one('.subsection, .meta-part.subsection, .archcard-kicker')
        if subsection_elem:
            subtitle = subsection_elem.get_text(strip=True)
        
        # Preview/excerpt extraction
        preview = None
        if is_archcard:
            excerpt_elem = container.select_one('.archcard-excerpt, p')
            if excerpt_elem:
                preview = excerpt_elem.get_text(strip=True)
        else:
            excerpt_elem = container.select_one('.aprev-excerpt, p.aprev-excerpt')
            if excerpt_elem:
                preview = excerpt_elem.get_text(strip=True)
        
        # Author extraction
        author = None
        auth_elem = container.select_one('.auth, p.auth')
        if auth_elem:
            author = auth_elem.get_text(strip=True)
            # Clean up "di " prefix and normalize spacing
            author = re.sub(r'^di\s*', '', author)
            author = re.sub(r'\s+e\s+', ', ', author)  # "X e Y" -> "X, Y"
            author = re.sub(r'\s+', ' ', author)  # Normalize spaces
        
        # Date extraction
        date = None
        time_elem = container.select_one('time.time, time')
        if time_elem:
            date = time_elem.get_text(strip=True)
            # Also get the datetime attribute if available
            datetime_attr = time_elem.get('datetime')
            if datetime_attr:
                date = datetime_attr
        
        # Related articles/links extraction (summary)
        related = []
        related_items = container.select('.aprev-rel-item a, .aprev-rel-link')
        for rel_item in related_items:
            rel_text = rel_item.get_text(strip=True)
            if rel_text and len(rel_text) > 5:
                related.append(rel_text)
        
        return {
            "title": title,
            "subtitle": subtitle,
            "preview": preview,
            "author": author,
            "date": date,
            "url": url,
            "related": related if related else None,
        }
        
    except Exception as e:
        logger.debug(f"Error extracting article data: {e}")
        return None


def click_mostra_altri(driver: webdriver.Chrome) -> bool:
    """Click the 'Mostra altri' button once to enable infinite scroll.
    
    Returns True if button was found and clicked, False otherwise.
    """
    try:
        patterns = [
            "//button[contains(., 'Mostra altri')]",
            "//button[contains(., 'mostra altri')]",
            "//a[contains(., 'Mostra altri')]",
            "//*[contains(@class, 'load-more')]//button",
            "//*[contains(@class, 'loadMore')]//button",
            "//button[contains(@class, 'load-more')]",
            "//button[contains(@class, 'loadMore')]",
        ]
        
        for pattern in patterns:
            try:
                buttons = driver.find_elements(By.XPATH, pattern)
                for btn in buttons:
                    if btn.is_displayed() and btn.is_enabled():
                        # Scroll to button
                        driver.execute_script(
                            "arguments[0].scrollIntoView({block:'center'});", btn
                        )
                        time.sleep(0.5)
                        
                        click_safely(driver, btn)
                        logger.info("Clicked 'Mostra altri' button")
                        time.sleep(2)
                        return True
                        
            except Exception as e:
                logger.debug(f"Mostra altri pattern failed: {e}")
                continue
        
        logger.debug("'Mostra altri' button not found")
        return False
        
    except Exception as e:
        logger.debug(f"Error clicking Mostra altri: {e}")
        return False


def scroll_to_load_all(driver: webdriver.Chrome, max_scrolls: int = 100, scroll_pause: float = 1.5) -> int:
    """Scroll down the page to load all articles via infinite scroll.
    
    After clicking 'Mostra altri', the page uses infinite scroll to load more articles.
    This function scrolls down until no more content loads.
    
    Parameters
    ----------
    driver : webdriver.Chrome
        The Selenium WebDriver instance.
    max_scrolls : int
        Maximum number of scroll attempts to prevent infinite loops.
    scroll_pause : float
        Time to wait after each scroll for content to load.
    
    Returns
    -------
    int
        Number of scroll operations performed.
    """
    scrolls = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    no_change_count = 0
    
    while scrolls < max_scrolls:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        scrolls += 1
        
        # Wait for new content to load
        time.sleep(scroll_pause)
        
        # Calculate new scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            no_change_count += 1
            # If height hasn't changed for 3 consecutive scrolls, we've reached the end
            if no_change_count >= 3:
                logger.info(f"Reached end of page after {scrolls} scrolls")
                break
        else:
            no_change_count = 0
            logger.debug(f"Scroll {scrolls}: page height {last_height} -> {new_height}")
        
        last_height = new_height
    
    if scrolls >= max_scrolls:
        logger.warning(f"Reached max scrolls limit ({max_scrolls})")
    
    return scrolls


def navigate_to_section(driver: webdriver.Chrome, section_name: str, section_slug: str) -> bool:
    """Navigate to a specific section by clicking on the navigation menu.
    
    Returns True if navigation succeeded, False otherwise.
    """
    try:
        # Direct URL navigation with /sez/ prefix
        section_url = f"{BASE_URL}sez/{section_slug}"
        logger.info(f"Navigating to section: {section_name} ({section_url})")
        
        driver.get(section_url)
        time.sleep(3)
        
        # Close any cookie banners
        close_cookie_banner_if_any(driver)
        
        # Check if we got a 404 or error page
        if "non trovata" in driver.page_source.lower() or "404" in driver.page_source:
            logger.warning(f"Section {section_name} returned 404, trying navigation menu")
            
            # Go back to homepage and try menu navigation
            driver.get(BASE_URL)
            time.sleep(2)
            close_cookie_banner_if_any(driver)
            
            # Try to find and click the section in the navigation
            nav_patterns = [
                f"//a[contains(., '{section_name}')]",
                f"//nav//a[contains(., '{section_name}')]",
                f"//*[contains(@class, 'nav')]//a[contains(., '{section_name}')]",
                f"//a[contains(@href, '{section_slug}')]",
            ]
            
            for pattern in nav_patterns:
                try:
                    links = driver.find_elements(By.XPATH, pattern)
                    for link in links:
                        if link.is_displayed():
                            click_safely(driver, link)
                            time.sleep(2)
                            return True
                except Exception:
                    continue
            
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error navigating to section {section_name}: {e}")
        return False


# -------------------- MAIN SCRAPER CLASS --------------------

class NTFiscoScraper:
    """Scraper for NT+ Fisco articles.
    
    Parameters
    ----------
    bucket_name : str
        Name of the target GCS bucket.
    storage_client : Optional[storage.Client], optional
        Preconfigured Google Cloud Storage client.
    base_folder : str, optional
        Root folder inside the bucket. Defaults to 'nt_fisco'.
    headless : bool, optional
        Whether to run Chrome in headless mode. Defaults to True.
    max_articles : int, optional
        Maximum number of articles to scrape per section. 0 = unlimited.
    max_scrolls : int, optional
        Maximum number of scroll operations per section. Defaults to 100.
    delay : float, optional
        Delay between actions in seconds.
    save_local : bool, optional
        If True, save files locally instead of GCS.
    local_output_folder : str, optional
        Local folder path when save_local is True.
    """
    
    def __init__(
        self,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
        base_folder: str = "nt_fisco",
        headless: bool = True,
        max_articles: int = 0,
        max_scrolls: int = 100,
        delay: float = DEFAULT_DELAY,
        save_local: bool = False,
        local_output_folder: Optional[str] = None,
    ):
        self.bucket_name = bucket_name
        self.base_folder = base_folder
        self.headless = headless
        self.max_articles = max_articles
        self.max_scrolls = max_scrolls
        self.delay = delay
        self.save_local = save_local
        self.local_output_folder = local_output_folder
        self.processed_articles: Set[str] = set()
        self.stats = {
            "sections": 0,
            "articles_scraped": 0,
            "articles_uploaded": 0,
            "articles_skipped": 0,
            "errors": 0,
        }
        
        # Only initialize GCS if not saving locally
        if not save_local:
            self.storage_client = storage_client or storage.Client()
            self.bucket = self.storage_client.bucket(self.bucket_name)
        else:
            self.storage_client = None
            self.bucket = None
    
    def scrape(self, only_sections: Optional[List[str]] = None) -> Dict:
        """Run the scraper for all sections or specified sections.
        
        Parameters
        ----------
        only_sections : Optional[List[str]]
            If provided, only scrape these sections (by display name).
        
        Returns
        -------
        Dict
            Statistics about the scraping run.
        """
        driver = None
        
        try:
            logger.info("🚀 Starting NT+ Fisco scraper")
            driver = build_driver(headless=self.headless)
            
            # Open the homepage first
            logger.info(f"Opening homepage: {BASE_URL}")
            driver.get(BASE_URL)
            time.sleep(3)
            close_cookie_banner_if_any(driver)
            
            # Determine which sections to scrape
            sections_to_scrape = SECTIONS
            if only_sections:
                sections_to_scrape = {
                    k: v for k, v in SECTIONS.items() 
                    if k in only_sections
                }
                logger.info(f"Scraping only sections: {list(sections_to_scrape.keys())}")
            
            # Scrape each section
            for section_name, section_slug in sections_to_scrape.items():
                try:
                    self._scrape_section(driver, section_name, section_slug)
                    self.stats["sections"] += 1
                    time.sleep(self.delay)
                except Exception as e:
                    logger.error(f"Error scraping section {section_name}: {e}")
                    self.stats["errors"] += 1
            
            logger.info(f"✅ Scraping completed. Stats: {self.stats}")
            
        except Exception as e:
            logger.error(f"❌ Fatal scraping error: {e}", exc_info=True)
            self.stats["errors"] += 1
            
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
        
        return self.stats
    
    def _scrape_section(self, driver: webdriver.Chrome, section_name: str, section_slug: str):
        """Scrape all articles from a single section.
        
        Process:
        1. Navigate to section page
        2. Click 'Mostra altri' button once (if present)
        3. Scroll down to load all articles via infinite scroll
        4. Collect all article URLs from the fully loaded page
        5. Visit each article URL to extract full title, subtitle and preview
        """
        logger.info(f"📂 Starting section: {section_name}")
        
        if not navigate_to_section(driver, section_name, section_slug):
            logger.warning(f"⚠️ Could not navigate to section: {section_name}")
            return
        
        # Step 1: Click 'Mostra altri' button once to enable infinite scroll
        click_mostra_altri(driver)
        
        # Step 2: Scroll down until no more content loads
        scroll_count = scroll_to_load_all(driver, max_scrolls=self.max_scrolls, scroll_pause=1.5)
        logger.info(f"Scrolled {scroll_count} times to load all content")
        
        # Step 3: Collect all article URLs from the fully loaded page
        article_urls = extract_article_urls_from_page(driver)
        logger.info(f"Found {len(article_urls)} article URLs in section {section_name}")
        
        section_articles = 0
        
        # Step 4: Visit each article URL to extract full details
        for url in article_urls:
            if self.max_articles > 0 and section_articles >= self.max_articles:
                logger.info(f"Reached max articles limit ({self.max_articles}) for section")
                break
            
            # Check if already processed
            url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
            if url_hash in self.processed_articles:
                continue
            
            # Visit article page and extract details
            article = extract_article_details_from_url(driver, url)
            
            if not article:
                self.stats["errors"] += 1
                continue
            
            self.processed_articles.add(url_hash)
            self.stats["articles_scraped"] += 1
            section_articles += 1
            
            # Add metadata
            article['section'] = section_slug
            article['scraped_at'] = datetime.now(timezone.utc).isoformat()
            
            # Upload to GCS
            if self._upload_article(article, section_slug):
                self.stats["articles_uploaded"] += 1
                logger.info(f"  📰 [{section_articles}] {article['title'][:50]}...")
            else:
                self.stats["articles_skipped"] += 1
            
            # Delay between article requests
            time.sleep(self.delay)
        
        logger.info(f"✅ Completed section {section_name}: {section_articles} articles uploaded")
    
    def _upload_article(self, article: Dict, section_slug: str) -> bool:
        """Upload an article as JSON to GCS or save locally.
        
        Returns True if uploaded successfully, False if skipped or failed.
        """
        try:
            # Generate filename from title
            title = article.get('title', 'untitled')
            article_id = generate_article_id(title, article.get('url', ''))
            filename = f"{sanitize_filename(title)}_{article_id}.json"
            
            # Create JSON content
            json_content = json.dumps(article, ensure_ascii=False, indent=2)
            
            # Folder structure: base_folder/section_slug
            folder = f"{self.base_folder}/{section_slug}"
            
            if self.save_local:
                # Save locally
                local_folder = os.path.join(self.local_output_folder, folder)
                os.makedirs(local_folder, exist_ok=True)
                file_path = os.path.join(local_folder, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_content)
                
                logger.debug(f"Saved article locally: {title[:50]}...")
            else:
                # Upload to GCS
                json_bytes = json_content.encode('utf-8')
                
                # Create the file object for upload
                file_obj = {
                    "name": filename,
                    "bytes": json_bytes,
                    "extension": "json",
                }
                
                upload_to_storage(
                    storage_client=self.storage_client,
                    bucket_name=self.bucket_name,
                    pdf_obj=file_obj,
                    folder=folder,
                )
                
                logger.debug(f"Uploaded article: {title[:50]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading article: {e}")
            self.stats["errors"] += 1
            return False
    
    def _already_uploaded(self, section_slug: str, filename: str) -> bool:
        """Check if an article already exists in GCS or locally."""
        try:
            folder = f"{self.base_folder}/{section_slug}"
            
            if self.save_local:
                # Check locally
                local_path = os.path.join(self.local_output_folder, folder, filename)
                return os.path.exists(local_path)
            else:
                # Check in GCS
                blob_path = f"{folder}/{filename}"
                return self.bucket.get_blob(blob_path) is not None
        except Exception:
            return False
