"""
DLGS Scraper for Normattiva
===========================

This script navigates the Normattiva advanced search to find DLGS 
(Decreto Legislativo) documents, uses AWS Bedrock to determine their 
relevance for accountants (commercialisti), and saves relevant documents 
to Google Cloud Storage.

The scraper:
1. Uses Selenium to navigate Normattiva's JavaScript-heavy interface
2. Filters for DLGS document types
3. Extracts metadata (title, subtitle, URL) for each result
4. Checks if document already exists in GCS (stop condition)
5. Uses AWS Bedrock Nova Lite to determine relevance
6. Scrapes full text for relevant documents
7. Uploads to GCS with standardized naming
"""

import json
import logging
import os
import re
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import boto3
import requests
from botocore.config import Config
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from google.cloud import storage as gcs
from google.oauth2 import service_account
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from .prompts import RELEVANCE_PROMPT

# Suppress XML parsed as HTML warning (expected for Akoma Ntoso)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

# Bedrock defaults
DEFAULT_MODEL_ID = "eu.amazon.nova-lite-v1:0"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 5000
DEFAULT_TOP_P = 0.9

# Italian month names for filename formatting
ITALIAN_MONTHS = {
    "01": "gennaio", "02": "febbraio", "03": "marzo", "04": "aprile",
    "05": "maggio", "06": "giugno", "07": "luglio", "08": "agosto",
    "09": "settembre", "10": "ottobre", "11": "novembre", "12": "dicembre"
}

# Request settings for scraping
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


class DLGSScraper:
    """Scraper for DLGS (Decreto Legislativo) documents from Normattiva.
    
    This scraper navigates the Normattiva advanced search interface using
    Selenium, filters for DLGS documents, evaluates their relevance using
    AWS Bedrock, and saves relevant documents to Google Cloud Storage.
    
    Parameters
    ----------
    bucket_name : str
        Name of the target GCS bucket.
    base_folder : str
        Root folder inside the bucket where files should be uploaded.
        Defaults to "dlgs".
    """
    
    BASE_URL = "https://www.normattiva.it"
    SEARCH_URL = "https://www.normattiva.it/ricerca/avanzata"
    
    def __init__(
        self,
        bucket_name: str = "loomy-public-documents",
        base_folder: str = "dlgs",
        limit: int = 0,
        start_page: int = 1,
        use_stop_condition: bool = True,
    ) -> None:
        self.bucket_name = bucket_name
        self.base_folder = base_folder
        self.limit = limit  # 0 = no limit
        self.start_page = start_page  # For testing older entries
        self.use_stop_condition = use_stop_condition  # Stop when existing entry found
        self.gcs_client = self._get_gcs_client()
        self.bedrock_client = self._get_bedrock_client()
        self.bucket = self.gcs_client.bucket(self.bucket_name)
        self.driver: Optional[webdriver.Chrome] = None
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        
        # Statistics
        self.stats = {
            "total_found": 0,
            "total_checked": 0,
            "relevant_count": 0,
            "files_saved": 0,
            "skipped_existing": 0,
            "skipped_not_published": 0,
        }
    
    def _get_gcs_client(self) -> gcs.Client:
        """Create and return a Google Cloud Storage client."""
        gcp_credentials_info = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIALS")
        if gcp_credentials_info:
            gcp_credentials_info = json.loads(gcp_credentials_info)
            gcp_service_account_credentials = service_account.Credentials.from_service_account_info(
                gcp_credentials_info
            )
            return gcs.Client(credentials=gcp_service_account_credentials)
        else:
            return gcs.Client()
    
    def _get_bedrock_client(self):
        """Create and return a Bedrock Runtime client."""
        config = Config(
            region_name=os.getenv("AWS_BEDROCK_REGION", "eu-central-1"),
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        return boto3.client("bedrock-runtime", config=config)
    
    def _init_selenium(self) -> webdriver.Chrome:
        """Initialize Selenium WebDriver with Chrome."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        )
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    def _close_selenium(self) -> None:
        """Close Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _check_exists_in_gcs(self, blob_name: str) -> bool:
        """Check if a blob exists in GCS."""
        blob_path = f"{self.base_folder}/{blob_name}"
        blob = self.bucket.blob(blob_path)
        return blob.exists()
    
    def _list_existing_urls(self) -> set:
        """List all existing DLGS URLs from GCS metadata files."""
        existing_urls = set()
        try:
            prefix = f"{self.base_folder}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                if blob.name.endswith(".txt"):
                    try:
                        content = blob.download_as_text(encoding="utf-8")
                        # Extract URL from the metadata at the beginning of the file
                        if "url:" in content.lower():
                            lines = content.split("\n")
                            for line in lines[:10]:  # Check first 10 lines
                                if line.lower().startswith("url:"):
                                    url = line.split(":", 1)[1].strip().strip('"').strip("'")
                                    existing_urls.add(url)
                                    break
                    except Exception as e:
                        logger.debug(f"Could not parse {blob.name}: {e}")
        except Exception as e:
            logger.info(f"GCS folder check: {e} (may not exist yet)")
        return existing_urls
    
    def _save_to_gcs(self, filename: str, content: str) -> bool:
        """Save content to GCS."""
        try:
            blob_path = f"{self.base_folder}/{filename}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                content.encode("utf-8"),
                content_type="text/plain; charset=utf-8"
            )
            logger.info(f"✅ Saved {blob_path} to GCS")
            return True
        except Exception as e:
            logger.error(f"Failed to save to GCS: {e}")
            return False
    
    def _call_bedrock(self, subtitle: str) -> dict:
        """Call AWS Bedrock to determine relevance.
        
        Returns dict with 'relevant' boolean key.
        """
        prompt = RELEVANCE_PROMPT.format(dlgs_subtitle=subtitle)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                ],
            }
        ]
        
        try:
            response = self.bedrock_client.converse(
                modelId=DEFAULT_MODEL_ID,
                messages=messages,
                inferenceConfig={
                    "temperature": DEFAULT_TEMPERATURE,
                    "maxTokens": DEFAULT_MAX_TOKENS,
                    "topP": DEFAULT_TOP_P,
                },
            )
            
            # Extract response text
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            
            response_text = ""
            for block in content_blocks:
                if "text" in block:
                    response_text += block["text"]
            
            logger.debug(f"Bedrock response: {response_text}")
            
            # Parse the response - handle various formats
            return self._parse_relevance_response(response_text)
            
        except Exception as e:
            logger.error(f"Bedrock API call failed: {e}")
            # Default to not relevant on error
            return {"relevant": False}
    
    def _parse_relevance_response(self, response_text: str) -> dict:
        """Parse LLM response to extract relevance boolean.
        
        Handles various response formats including:
        - {"relevant": true}
        - {"relevant": false}
        - {"relevant": True}
        - {"relevant": False}
        - Malformed JSON with true/false/True/False
        """
        response_text = response_text.strip()
        
        # Try direct JSON parse first
        try:
            result = json.loads(response_text)
            if "relevant" in result:
                return {"relevant": bool(result["relevant"])}
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON-like pattern
        json_pattern = r'\{[^}]*"relevant"\s*:\s*(true|false|True|False)[^}]*\}'
        match = re.search(json_pattern, response_text, re.IGNORECASE)
        if match:
            try:
                # Normalize boolean values for JSON parsing
                json_str = match.group(0)
                json_str = re.sub(r'\bTrue\b', 'true', json_str)
                json_str = re.sub(r'\bFalse\b', 'false', json_str)
                result = json.loads(json_str)
                return {"relevant": bool(result.get("relevant", False))}
            except json.JSONDecodeError:
                pass
        
        # Fallback: look for true/false anywhere in response
        response_lower = response_text.lower()
        if '"relevant": true' in response_lower or '"relevant":true' in response_lower:
            return {"relevant": True}
        if '"relevant": false' in response_lower or '"relevant":false' in response_lower:
            return {"relevant": False}
        
        # Last resort: check if the word "true" or "false" appears
        if "true" in response_lower and "false" not in response_lower:
            return {"relevant": True}
        
        logger.warning(f"Could not parse relevance from: {response_text}")
        return {"relevant": False}
    
    def _generate_filename(self, title: str) -> Optional[str]:
        """Generate filename from DLGS title.
        
        Expected title format: "DECRETO LEGISLATIVO 4 dicembre 2025, n. 205"
        Output format: "dlgs_04_dicembre_2025_n_205.txt"
        """
        # Pattern to match date and number
        pattern = r"(\d{1,2})\s+(\w+)\s+(\d{4}),?\s*n\.?\s*(\d+)"
        match = re.search(pattern, title, re.IGNORECASE)
        
        if match:
            day = match.group(1).zfill(2)
            month = match.group(2).lower()
            year = match.group(3)
            number = match.group(4)
            
            return f"dlgs_{day}_{month}_{year}_n_{number}.txt"
        
        # Fallback: sanitize title
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", title)
        sanitized = re.sub(r"__+", "_", sanitized).strip("_")[:100]
        return f"dlgs_{sanitized}.txt" if sanitized else None
    
    def _fetch(self, session: requests.Session, url: str, **kwargs) -> requests.Response:
        """Fetch a URL with retries."""
        for attempt in range(MAX_RETRIES):
            try:
                response = session.get(
                    url, headers=self.headers, timeout=REQUEST_TIMEOUT, **kwargs
                )
                response.raise_for_status()
                return response
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = (attempt + 1) * 2
                logger.debug(f"Request to {url} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")
    
    def _extract_articles_akn(self, xml_text: str) -> List[Tuple[str, str]]:
        """Extract articles from Akoma Ntoso XML."""
        import xml.etree.ElementTree as ET
        
        articles: List[Tuple[str, str]] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:
            logger.debug(f"Failed to parse AKN XML: {exc}")
            return articles
        
        for art in root.findall('.//{*}article'):
            num_el = art.find('.//{*}num')
            article_num = num_el.text.strip() if num_el is not None and num_el.text else 'Articolo'
            texts: List[str] = []
            for elem in art.iter():
                if elem is num_el:
                    continue
                if elem.text:
                    texts.append(elem.text.strip())
            article_text = ' '.join(t for t in texts if t)
            articles.append((article_num, article_text))
        
        return articles
    
    def _extract_articles_html(self, html: str) -> List[Tuple[str, str]]:
        """Extract articles from HTML page."""
        soup = BeautifulSoup(html, "lxml")
        body = soup.find(class_="bodyTesto")
        if not body:
            return []
        
        h2 = body.find("h2", class_="article-num-akn")
        if h2 and h2.get_text(strip=True):
            article_num = h2.get_text(strip=True)
        else:
            h_generic = body.find(re.compile(r'^(h[1-6]|strong)$'))
            article_num = h_generic.get_text(strip=True) if h_generic else 'Articolo'
        
        lines: List[str] = []
        comma_divs = body.find_all('div', class_='art-comma-div-akn')
        if comma_divs:
            for comma_div in comma_divs:
                num_span = comma_div.find('span', class_='comma-num-akn')
                txt_span = comma_div.find('span', class_='art_text_in_comma')
                if num_span or txt_span:
                    num = num_span.get_text(strip=True) if num_span else ''
                    txt = txt_span.get_text(strip=True) if txt_span else comma_div.get_text(strip=True)
                    combined = (num + ' ' + txt).strip()
                    lines.append(combined)
                else:
                    lines.append(comma_div.get_text(strip=True))
        
        if not lines:
            raw = body.get_text(separator='\n', strip=True)
            raw = re.sub(r'\n+', '\n', raw)
            lines = [raw]
        
        article_text = '\n'.join(lines)
        return [(article_num, article_text)]
    
    def _parse_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Parse document title from page."""
        selectors = [
            ("span", {"class": "legTitle"}),
            ("h3", {"class": "doc-title"}),
            ("div", {"class": "leggeTitle"}),
            ("title", {}),
        ]
        for tag, attrs in selectors:
            found = soup.find(tag, attrs=attrs) if attrs else soup.find(tag)
            if found and found.get_text(strip=True):
                return found.get_text(strip=True)
        return None
    
    def _check_not_published(self, html: str) -> bool:
        """Check if DLGS is not yet published."""
        return "NON ANCORA ESISTENTE O VIGENTE" in html
    
    def _scrape_dlgs_text(self, dlgs_url: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
        """Scrape full DLGS text from Normattiva URL.
        
        Returns tuple of (title, list of (article_num, article_text))
        """
        session = requests.Session()
        articles: List[Tuple[str, str]] = []
        
        try:
            logger.info(f"  Loading DLGS page: {dlgs_url}")
            resp = self._fetch(session, dlgs_url)
        except Exception as exc:
            logger.error(f"  Failed to load DLGS page {dlgs_url}: {exc}")
            return None, articles
        
        # Check if not published
        if self._check_not_published(resp.text):
            logger.info("  DLGS not yet published, skipping")
            return None, []
        
        soup = BeautifulSoup(resp.text, "lxml")
        law_title = self._parse_title(soup)
        
        # Try Akoma Ntoso link first
        akn_link = None
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'caricaAKN' in href:
                akn_link = urljoin(dlgs_url, href)
                break
        
        if akn_link:
            logger.debug(f"    Found Akoma Ntoso link: {akn_link}")
            try:
                akn_resp = self._fetch(session, akn_link)
                arts = self._extract_articles_akn(akn_resp.text)
                if arts:
                    logger.info(f"    Extracted {len(arts)} articles from AKN")
                    return law_title, arts
                logger.debug("    No articles found in AKN; falling back to HTML")
            except Exception as exc:
                logger.debug(f"    Failed to download or parse AKN: {exc}")
        
        # Fallback to HTML scraping
        page_source = resp.text
        article_paths = re.findall(r"showArticle\('(/atto/caricaArticolo[^']+)'", page_source)
        
        if article_paths:
            logger.info(f"    Found {len(article_paths)} article link(s) in page source")
            articles_map: OrderedDict[str, List[str]] = OrderedDict()
            seen_paths: set = set()
            duplicates_count = 0
            
            for path in article_paths:
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                full_url = urljoin(dlgs_url, path)
                
                try:
                    article_resp = self._fetch(session, full_url)
                except Exception as exc:
                    logger.debug(f"      Failed to fetch article {full_url}: {exc}")
                    continue
                
                art_tuples = self._extract_articles_html(article_resp.text)
                if not art_tuples:
                    continue
                
                art_num, art_text = art_tuples[0]
                segs = articles_map.setdefault(art_num, [])
                if art_text in segs:
                    duplicates_count += 1
                    if duplicates_count > 10:
                        logger.debug("      Too many duplicate article segments; stopping.")
                        break
                    continue
                
                duplicates_count = 0
                segs.append(art_text)
                logger.debug(f"      Fetched {art_num} (segment {len(segs)})")
                time.sleep(0.5)
            
            if articles_map:
                for num, segs in articles_map.items():
                    joined = '\n\n'.join(segs)
                    articles.append((num, joined))
                return law_title, articles
        
        # Fallback to initial article on page
        logger.debug("    Falling back to initial article on page")
        first_article = self._extract_articles_html(resp.text)
        return law_title, first_article
    
    def _save_screenshot(self, name: str) -> None:
        """Save a screenshot for debugging."""
        if self.driver:
            try:
                screenshot_path = f"debug_{name}.png"
                self.driver.save_screenshot(screenshot_path)
                logger.debug(f"Screenshot saved: {screenshot_path}")
            except Exception as e:
                logger.debug(f"Could not save screenshot: {e}")
    
    def _navigate_and_search(self) -> List[Dict]:
        """Navigate Normattiva and execute DLGS search.
        
        Returns list of dicts with keys: url, title, subtitle
        """
        self.driver = self._init_selenium()
        entries = []
        
        try:
            logger.info("Navigating to Normattiva advanced search...")
            self.driver.get(self.SEARCH_URL)
            wait = WebDriverWait(self.driver, 20)
            
            # Wait for page to load
            time.sleep(3)
            self._save_screenshot("01_page_loaded")
            
            # Click on "Denominazione atto" dropdown
            logger.info("Clicking on 'Denominazione atto' filter...")
            dropdown_xpath = '//*[@id="ricercaAvanzataBean"]/fieldset[2]/div[2]/div/div/button'
            dropdown = wait.until(EC.element_to_be_clickable((By.XPATH, dropdown_xpath)))
            dropdown.click()
            time.sleep(2)  # Wait longer for dropdown to open
            self._save_screenshot("02_dropdown_opened")
            
            # Select "DECRETO LEGISLATIVO"
            logger.info("Selecting 'DECRETO LEGISLATIVO'...")
            
            # First, log all available options in the dropdown for debugging
            js_list_options = """
            var options = [];
            var selectors = ['li label', 'li a', 'li span', '.dropdown-menu li', '.multiselect-container li label'];
            for (var s = 0; s < selectors.length; s++) {
                var items = document.querySelectorAll(selectors[s]);
                for (var i = 0; i < items.length; i++) {
                    var text = (items[i].textContent || items[i].innerText || '').trim();
                    if (text && text.length > 3 && text.length < 100) {
                        options.push(text.substring(0, 80));
                    }
                }
            }
            return options.slice(0, 30);  // Return first 30 options
            """
            available_options = self.driver.execute_script(js_list_options)
            logger.info(f"Available dropdown options: {available_options}")
            
            # Now try to find and click the DLGS option
            # IMPORTANT: Must match EXACTLY "DECRETO LEGISLATIVO"
            js_click_script = """
            var searchText = 'DECRETO LEGISLATIVO';
            var selectors = ['.multiselect-container li label', 'li label', 'li a', 'li span', 'option', '.dropdown-menu li'];
            
            for (var s = 0; s < selectors.length; s++) {
                var items = document.querySelectorAll(selectors[s]);
                for (var i = 0; i < items.length; i++) {
                    var text = (items[i].textContent || items[i].innerText || '').toUpperCase().trim();
                    // Must match exactly "DECRETO LEGISLATIVO"
                    // Exclude other types that might partially match
                    if (text === searchText || text.startsWith(searchText + ' ') || text.endsWith(' ' + searchText)) {
                        // Try clicking the item or its checkbox/input
                        var input = items[i].querySelector('input[type="checkbox"]');
                        if (input) {
                            input.click();
                            return 'clicked checkbox in ' + selectors[s] + ': ' + text.substring(0, 60);
                        }
                        items[i].click();
                        return 'clicked ' + selectors[s] + ': ' + text.substring(0, 60);
                    }
                }
            }
            return 'not found';
            """
            
            js_result = self.driver.execute_script(js_click_script)
            logger.info(f"JavaScript click result: {js_result}")
            
            # If JavaScript didn't find it, the filter is NOT applied - we must abort
            if js_result == 'not found':
                self._save_screenshot("03_dropdown_error")
                # Log the page source for debugging
                with open("debug_dropdown.html", "w", encoding="utf-8") as f:
                    try:
                        dropdown_elem = self.driver.find_element(By.XPATH, '//*[@id="ricercaAvanzataBean"]/fieldset[2]/div[2]')
                        f.write(dropdown_elem.get_attribute('outerHTML'))
                    except Exception:
                        f.write(self.driver.page_source)
                logger.error("Saved dropdown HTML to debug_dropdown.html")
                raise Exception(
                    "CRITICAL: Could not find 'DECRETO LEGISLATIVO' option in dropdown. "
                    "Aborting to prevent unfiltered search. Check debug_dropdown.html for available options."
                )
            
            time.sleep(1)
            self._save_screenshot("03_option_selected")
            
            # Click search button
            logger.info("Launching advanced search...")
            search_button_xpath = '//*[@id="ricercaAvanzataBean"]/fieldset[6]/div/div/button'
            search_button = wait.until(EC.element_to_be_clickable((By.XPATH, search_button_xpath)))
            search_button.click()
            
            # Wait for results page to load - the URL changes after search
            time.sleep(5)
            self._save_screenshot("04_search_results")
            
            # Log current URL for debugging
            logger.info(f"Current URL after search: {self.driver.current_url}")
            
            # Extract and log total results count - this verifies the filter is working
            try:
                # Look for "Sono stati trovati: N atti" text
                results_text = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Sono stati trovati')]")
                total_count_text = results_text.text.strip()
                logger.info(f"📊 TOTAL RESULTS: {total_count_text}")
            except Exception as e:
                logger.warning(f"Could not find total results count: {e}")
                # Try alternative - search in page source
                try:
                    import re
                    match = re.search(r'Sono stati trovati[^0-9]*([0-9.,]+)\s*atti', self.driver.page_source)
                    if match:
                        logger.info(f"📊 TOTAL RESULTS: Sono stati trovati: {match.group(1)} atti")
                except:
                    pass
            
            # Extract entries
            logger.info("Extracting search results...")
            page_num = 1
            
            # Skip to start_page if specified
            if self.start_page > 1:
                logger.info(f"Skipping to page {self.start_page}...")
                for skip_page in range(1, self.start_page):
                    try:
                        next_selectors = [
                            "//a[contains(@class, 'page-link') and contains(text(), '›')]",
                            "//a[contains(@class, 'next')]",
                            "//li[contains(@class, 'next')]/a",
                            "//a[contains(text(), 'Successiv')]",
                        ]
                        next_button = None
                        for next_sel in next_selectors:
                            try:
                                next_button = self.driver.find_element(By.XPATH, next_sel)
                                if next_button and next_button.is_displayed():
                                    break
                            except:
                                continue
                        if next_button:
                            next_button.click()
                            time.sleep(2)
                            page_num += 1
                    except Exception as e:
                        logger.warning(f"Failed to skip to page {skip_page + 1}: {e}")
                        break
                logger.info(f"Now on page {page_num}")
            
            while True:
                logger.info(f"Processing results page {page_num}...")
                self._save_screenshot(f"05_results_page_{page_num}")
                
                # Try multiple strategies to find results
                found_entries = []
                
                # Strategy 1: Look for Normattiva result items (heading_N starts from 1, not 0)
                n = 1  # Start from 1, not 0
                max_attempts = 100  # Limit to prevent infinite loops
                while n <= max_attempts:
                    try:
                        # Try to find entry with index n
                        # The structure is: div#heading_N > p > a (with title and link)
                        heading_xpath = f'//*[@id="heading_{n}"]'
                        heading_elem = self.driver.find_element(By.XPATH, heading_xpath)
                        
                        # Get the link inside the heading - this contains the title
                        try:
                            link_elem = heading_elem.find_element(By.XPATH, ".//a[@title='Dettaglio atto']")
                            title = link_elem.text.strip()
                            url = link_elem.get_attribute("href")
                        except:
                            # Try any anchor
                            try:
                                link_elem = heading_elem.find_element(By.XPATH, ".//a")
                                title = link_elem.text.strip()
                                url = link_elem.get_attribute("href")
                            except:
                                title = heading_elem.text.strip()
                                url = None
                        
                        # Get subtitle from the second <p> tag in heading (contains the law description)
                        subtitle = ""
                        try:
                            # The subtitle is in the second <p> inside heading_N (first <p> contains the title link)
                            p_tags = heading_elem.find_elements(By.XPATH, ".//p")
                            if len(p_tags) >= 2:
                                # Second <p> tag contains the subtitle in brackets [...]
                                text = self.driver.execute_script(
                                    "return arguments[0].textContent || arguments[0].innerText || '';", 
                                    p_tags[1]
                                )
                                text = text.strip() if text else ""
                                # Remove brackets if present
                                if text.startswith("[") and text.endswith("]"):
                                    text = text[1:-1].strip()
                                if text and len(text) > 10:
                                    subtitle = text[:500]
                                    logger.debug(f"Found subtitle from second <p>: {subtitle[:100]}...")
                            
                            # Fallback: try any text after the first <p>
                            if not subtitle:
                                all_text = self.driver.execute_script(
                                    "return arguments[0].textContent || arguments[0].innerText || '';", 
                                    heading_elem
                                )
                                # Try to extract text between [...] 
                                import re
                                bracket_match = re.search(r'\[([^\]]+)\]', all_text)
                                if bracket_match:
                                    text = bracket_match.group(1).strip()
                                    if len(text) > 10:
                                        subtitle = text[:500]
                                        logger.debug(f"Found subtitle from bracket extraction: {subtitle[:100]}...")
                        except Exception as e:
                            logger.debug(f"Subtitle extraction failed: {e}")
                        
                        if title and "DECRETO" in title.upper():
                            found_entries.append({
                                "url": url or "",
                                "title": title,
                                "subtitle": subtitle
                            })
                            logger.debug(f"  Found entry {n}: {title[:60]}...")
                        
                        n += 1
                        
                    except Exception:
                        # No more entries with this pattern
                        break
                
                # Strategy 2: If no results found, try alternative selectors
                if not found_entries:
                    logger.debug("Trying alternative selectors...")
                    # Try finding result cards/panels
                    alt_selectors = [
                        "//div[contains(@class, 'card')]//h5",
                        "//div[contains(@class, 'result')]//a",
                        "//div[contains(@class, 'panel')]//a",
                        "//table//tr/td/a",
                        "//li[contains(@class, 'list-group-item')]//a",
                    ]
                    
                    for selector in alt_selectors:
                        try:
                            elements = self.driver.find_elements(By.XPATH, selector)
                            if elements:
                                logger.debug(f"Found {len(elements)} elements with {selector}")
                                for elem in elements:
                                    text = elem.text.strip()
                                    href = elem.get_attribute("href") if elem.tag_name == "a" else None
                                    if text and "DECRETO" in text.upper():
                                        found_entries.append({
                                            "url": href or "",
                                            "title": text,
                                            "subtitle": ""
                                        })
                        except Exception as e:
                            logger.debug(f"Selector {selector} failed: {e}")
                
                # Add found entries to main list
                entries.extend(found_entries)
                logger.info(f"  Found {len(found_entries)} entries on page {page_num}")
                
                # Check limit
                if self.limit > 0 and len(entries) >= self.limit:
                    logger.info(f"Reached limit of {self.limit} entries, stopping pagination")
                    entries = entries[:self.limit]
                    break
                
                if not found_entries:
                    # No results on this page, stop
                    break
                
                # Try to go to next page
                try:
                    next_selectors = [
                        "//a[contains(@class, 'page-link') and contains(text(), '›')]",
                        "//a[contains(@class, 'next')]",
                        "//li[contains(@class, 'next')]/a",
                        "//a[contains(text(), 'Successiv')]",
                    ]
                    
                    next_button = None
                    for next_sel in next_selectors:
                        try:
                            next_button = self.driver.find_element(By.XPATH, next_sel)
                            if next_button and next_button.is_displayed():
                                break
                        except:
                            continue
                    
                    if next_button and next_button.is_enabled():
                        next_button.click()
                        time.sleep(3)
                        page_num += 1
                    else:
                        break
                except Exception:
                    # No more pages
                    break
            
            logger.info(f"Total entries found: {len(entries)}")
            return entries
            
        finally:
            self._close_selenium()
    
    def scrape(self) -> Dict:
        """Run the complete scraping process.
        
        Returns dict with statistics.
        """
        logger.info("Starting DLGS scraper...")
        
        # Get existing URLs from GCS to check for stop condition
        existing_urls = self._list_existing_urls()
        logger.info(f"Found {len(existing_urls)} existing DLGS in GCS")
        
        # Navigate and get search results
        entries = self._navigate_and_search()
        self.stats["total_found"] = len(entries)
        
        if not entries:
            logger.warning("No DLGS entries found")
            return self.stats
        
        # Process each entry
        for i, entry in enumerate(entries, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing entry {i}/{len(entries)}")
            logger.info(f"Title: {entry['title'][:80]}...")
            logger.info(f"Subtitle: {entry['subtitle'][:120]}...")
            logger.info(f"URL: {entry['url']}")
            
            # Check if already exists (stop condition)
            if self.use_stop_condition and entry["url"] in existing_urls:
                logger.info("⏹️  Entry already exists in GCS - stopping (incremental update complete)")
                self.stats["skipped_existing"] += 1
                break
            
            # Generate filename to check if file exists
            filename = self._generate_filename(entry["title"])
            if self.use_stop_condition and filename and self._check_exists_in_gcs(filename):
                logger.info(f"⏹️  File {filename} already exists - stopping")
                self.stats["skipped_existing"] += 1
                break
            
            self.stats["total_checked"] += 1
            
            # Check relevance with Bedrock
            logger.info(f"Checking relevance for: {entry['subtitle'][:100]}...")
            relevance = self._call_bedrock(entry["subtitle"])
            
            if not relevance.get("relevant", False):
                logger.info("❌ Not relevant for commercialisti, skipping")
                continue
            
            logger.info("✅ Relevant! Scraping full text...")
            self.stats["relevant_count"] += 1
            
            # Scrape full text
            if not entry["url"]:
                logger.warning("No URL available, skipping")
                continue
            
            title, articles = self._scrape_dlgs_text(entry["url"])
            
            if articles is None or len(articles) == 0:
                # Check if it was because DLGS is not published
                if title is None:
                    self.stats["skipped_not_published"] += 1
                continue
            
            # Generate filename
            if not filename:
                logger.warning("Could not generate filename, skipping")
                continue
            
            # Build content
            content_parts = [
                f"url: {entry['url']}",
                f"title: {entry['title']}",
                f"subtitle: {entry['subtitle']}",
                "",
                "---",
                "",
            ]
            
            if title:
                content_parts.append(title)
                content_parts.append("")
            
            for art_num, art_text in articles:
                content_parts.append(f"{art_num}")
                content_parts.append(art_text.strip())
                content_parts.append("")
            
            content = "\n".join(content_parts)
            
            # Save to GCS
            if self._save_to_gcs(filename, content):
                self.stats["files_saved"] += 1
            
            # Small delay between entries
            time.sleep(1)
        
        logger.info(f"\n{'='*60}")
        logger.info("Scraping complete!")
        return self.stats
