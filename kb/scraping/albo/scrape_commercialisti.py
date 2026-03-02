"""
Scraper for the Italian Commercialisti (Dottori Commercialisti ed Esperti Contabili)
public register — https://ricerca.commercialisti.it/RicercaIscritti

Usage:
    python scrape_commercialisti.py                          # all CAPs
    python scrape_commercialisti.py --max-caps 3            # test: first 3 CAPs
    python scrape_commercialisti.py --resume 20121          # continue from a CAP
    python scrape_commercialisti.py --csv path/to/file.csv  # custom CSV
"""

import csv
import time
import re
import json
import logging
import sys
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
load_dotenv()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    NoSuchElementException,
    ElementClickInterceptedException,
)
from bs4 import BeautifulSoup

# ─── GCS ─────────────────────────────────────────────────────────────────────
from google.cloud import storage
from google.oauth2 import service_account

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("commercialisti.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
SEARCH_URL = (
    "https://ricerca.commercialisti.it/RicercaIscritti"
    "?search=True&ente=00000000-0000-0000-0000-000000000000"
    "&cap={cap}&sezione=0"
)

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


# Timing constants (can be overridden via environment variables)
WAIT_TIMEOUT      = _env_float("COMM_WAIT_TIMEOUT", 15.0)   # seconds for explicit waits
PAGE_LOAD_DELAY   = _env_float("COMM_PAGE_LOAD_DELAY", 0.25)  # seconds after page navigation
CLICK_DELAY       = _env_float("COMM_CLICK_DELAY", 0.5)  # seconds after clicking a container
BACK_DELAY        = _env_float("COMM_BACK_DELAY", 0.5)  # seconds after history.back()
INTER_CAP_DELAY   = _env_float("COMM_INTER_CAP_DELAY", 0.5)  # seconds between different CAP searches

# ─── GCS config ──────────────────────────────────────────────────────────────
GCS_BUCKET        = "loomy-marketing"
GCS_CAP_CSV_PATH  = "albo/cap/cap_comuni.csv"
GCS_OUTPUT_JSON   = "albo/commercialisti_data.json"
GCS_START_FROM    = "albo/cap/start_from_cap.txt"

# CSS selectors (Kendo UI)
SEL_LIST            = "#listIscritti > div"
SEL_PAGER           = "#pagerIscritti"
SEL_PAGER_SELECT    = "#pagerIscritti select"             # hidden <select> for page size
SEL_PAGER_DROPDOWN  = "#pagerIscritti .k-pager-sizes .k-select"  # Kendo trigger
SEL_PAGER_NAV       = "#pagerIscritti a.k-pager-nav"     # first/prev/next/last buttons
SEL_PAGER_INFO      = "#pagerIscritti .k-pager-info"     # "1 - 10 di 42" info text


class CommercialisitiScraper:
    """Selenium scraper for the Italian Commercialisti national register."""

    # Fields to extract from the detail page via regex on the plain text
    FIELD_PATTERNS: Dict[str, str] = {
        "nome_completo":        r"Iscritto\s*:\s*([A-ZÀÈÉÌÒÙÁÍÓÚ][A-ZÀÈÉÌÒÙÁÍÓÚ\s\'-]+?)(?:\s*Nato|\n|$)",
        "luogo_nascita":        r"Nato\s+a\s*:\s*([^,\n]+?)(?=\s*il\s*:|\n|$)",
        "data_nascita":         r"il\s*:\s*(\d{2}/\d{2}/\d{4})",
        "ordine":               r"Ordine\s+di\s*:\s*([^\n]+)",
        "data_anzianita":       r"Data\s+anzianit[àa]\s*:\s*(\d{2}/\d{2}/\d{4})",
        "titolo_professionale": r"Titolo\s+professionale\s*:\s*([^\n]+)",
        "revisore_contabile":   r"Revisore\s+contabile\s*:\s*(\w+)",
        "data_iscrizione":      r"Iscritto\s+all['\u2019]?\s*Albo\s*:\s*(\d{2}/\d{2}/\d{4})",
        "sezione":              r"Sezione\s*:\s*([A-Z])",
        "altro_titolo":         r"Altro\s+titolo\s+professionale\s*:\s*([^\n]+)",
        "sede_studio":          r"Sede\s+studio\s*:\s*([^\n]+)",
        "data_modifica":        r"Data\s+ultima\s+modifica\s*:\s*(\d{2}/\d{2}/\d{4})",
    }

    def __init__(self, output_file: str = "commercialisti_data.json"):
        self.output_file = output_file
        self.driver: Optional[webdriver.Chrome] = None
        self._wait: Optional[WebDriverWait] = None
        self.results: List[Dict] = []   # only holds the current CAP's records in-flight
        self.failed_caps: List[tuple] = []
        self.storage_client = None
        self.total_records: int = 0     # grand total flushed to disk so far

    # ─── Driver ──────────────────────────────────────────────────────────────

    def setup_driver(self) -> bool:
        try:
            opts = webdriver.ChromeOptions()
            opts.add_argument("--headless=new")
            opts.add_argument("--window-size=1920,1080")
            opts.add_argument("--disable-blink-features=AutomationControlled")
            opts.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
            opts.add_experimental_option("excludeSwitches", ["enable-automation"])
            opts.add_experimental_option("useAutomationExtension", False)
            # Critical stability flags for long-running headless sessions
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")   # avoids /dev/shm OOM crashes
            opts.add_argument("--disable-gpu")
            opts.add_argument("--disable-extensions")
            opts.add_argument("--disable-infobars")
            opts.add_argument("--disable-notifications")
            opts.add_argument("--disable-background-networking")
            opts.add_argument("--disable-default-apps")
            opts.add_argument("--disable-sync")
            opts.add_argument("--disable-translate")
            opts.add_argument("--no-first-run")
            opts.add_argument("--safebrowsing-disable-auto-update")
            self.driver = webdriver.Chrome(options=opts)
            # Raise page-load and JS timeouts well above the 120s WebDriver default
            self.driver.set_page_load_timeout(90)
            self.driver.set_script_timeout(60)
            self._wait = WebDriverWait(self.driver, WAIT_TIMEOUT)
            logger.info("[OK] WebDriver initialised (headless)")
            return True
        except Exception as exc:
            logger.error(f"[ERROR] WebDriver init failed: {exc}")
            return False


    def _driver_is_alive(self) -> bool:
        """Return True if the WebDriver session is still responsive."""
        try:
            _ = self.driver.current_url   # simple liveness ping
            return True
        except Exception:
            return False

    def close_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass

    # ─── GCS helpers ─────────────────────────────────────────────────────────

    def _init_gcs(self):
        """Initialise the GCS client using explicit credentials or ADC."""
        try:
            gcp_credentials_info = os.environ.get("GCP_SERVICE_ACCOUNT_CREDENTIALS")
            if gcp_credentials_info:
                logger.info("Using GCP credentials from environment variable")
                gcp_credentials_info = json.loads(gcp_credentials_info)
                gcp_credentials = service_account.Credentials.from_service_account_info(
                    gcp_credentials_info
                )
                self.storage_client = storage.Client(credentials=gcp_credentials)
            else:
                logger.info("Using default GCP credentials (Cloud Run service account)")
                self.storage_client = storage.Client()
            logger.info("[OK] GCS client initialised")
        except Exception as exc:
            logger.error(f"[GCS] Could not initialise storage client: {exc}")
            self.storage_client = None

    def _gcs_download_csv(self, local_path: str) -> bool:
        """Download the CAP CSV from GCS to *local_path*. Returns True on success."""
        if not self.storage_client:
            return False
        try:
            bucket = self.storage_client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_CAP_CSV_PATH)
            blob.download_to_filename(local_path)
            logger.info(f"[GCS] Downloaded gs://{GCS_BUCKET}/{GCS_CAP_CSV_PATH} → {local_path}")
            return True
        except Exception as exc:
            logger.error(f"[GCS] Download failed: {exc}")
            return False

    def _gcs_upload(self, local_path: str, gcs_path: str, content_type: str = "application/octet-stream"):
        """Upload *local_path* to GCS."""
        if not self.storage_client:
            return
        try:
            bucket = self.storage_client.bucket(GCS_BUCKET)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path, content_type=content_type)
            logger.info(f"[GCS] Uploaded {local_path} → gs://{GCS_BUCKET}/{gcs_path}")
        except Exception as exc:
            logger.error(f"[GCS] Upload failed for {gcs_path}: {exc}")

    def _gcs_download_output(self, local_path: str) -> bool:
        """Download the existing output JSONL from GCS to *local_path* so that
        appends in a fresh container don't overwrite previous runs' data.
        Returns True if the file was restored, False if it didn't exist or failed."""
        if not self.storage_client:
            return False
        try:
            bucket = self.storage_client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_OUTPUT_JSON)
            if not blob.exists():
                logger.info(f"[GCS] No existing output at gs://{GCS_BUCKET}/{GCS_OUTPUT_JSON} — starting fresh")
                return False
            blob.download_to_filename(local_path)
            # Count pre-existing records so total_records is accurate
            with open(local_path, encoding="utf-8") as f:
                self.total_records = sum(1 for line in f if line.strip())
            logger.info(
                f"[GCS] Restored existing output → {local_path} "
                f"({self.total_records} existing records)"
            )
            return True
        except Exception as exc:
            logger.error(f"[GCS] Could not restore output file: {exc}")
            return False

    def _gcs_read_start_cap(self) -> Optional[str]:
        """Read the resume CAP from gs://GCS_BUCKET/GCS_START_FROM.
        Returns the CAP string (e.g. '22079') or None if the file does not
        exist or its content is not a valid 5-digit CAP code.

        NOTE: This method reads *only* from GCS_START_FROM.  It never falls
        back to GCS_OUTPUT_JSON, so deleting start_from_cap.txt always
        causes the job to restart from the very first CAP.
        """
        if not self.storage_client:
            return None
        try:
            # list_blobs() only enumerates *live* objects — soft-deleted /
            # noncurrent versions are invisible, unlike blob.exists() which
            # can still resolve them in some GCS soft-delete configurations.
            live_blobs = list(
                self.storage_client.list_blobs(
                    GCS_BUCKET, prefix=GCS_START_FROM, max_results=1
                )
            )
            if not live_blobs:
                logger.info(
                    f"[GCS] {GCS_START_FROM} has no live version — starting from the first CAP"
                )
                return None
            bucket = self.storage_client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_START_FROM)
            cap = blob.download_as_text().strip()
            # Accept only a bare 5-digit string so a stale JSON payload or any
            # other corrupt content is silently ignored.
            if cap and cap.isdigit() and len(cap) == 5:
                logger.info(f"[GCS] Resume CAP read from {GCS_START_FROM}: {cap}")
                return cap
            logger.warning(
                f"[GCS] {GCS_START_FROM} contains unexpected value '{cap[:40]}' "
                "(not a 5-digit CAP) — starting from the first CAP"
            )
        except Exception as exc:
            logger.info(f"[GCS] Could not read {GCS_START_FROM} (starting from scratch): {exc}")
        return None

    def _gcs_write_current_cap(self, cap: str):
        """Overwrite gs://GCS_BUCKET/GCS_START_FROM with *cap* so the job
        can resume automatically after a timeout/crash."""
        if not self.storage_client:
            return
        try:
            bucket = self.storage_client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_START_FROM)
            logger.debug(f"[GCS] Writing checkpoint CAP to {GCS_START_FROM}: {cap}")
            blob.upload_from_string(cap, content_type="text/plain")
            logger.debug(f"[GCS] Checkpoint CAP written: {cap}")
        except Exception as exc:
            logger.error(f"[GCS] Failed to write checkpoint CAP: {exc}")

    # ─── CAP list ────────────────────────────────────────────────────────────

    @staticmethod
    def get_unique_caps(csv_file: str) -> List[str]:
        """Return sorted list of unique *numeric* 5-digit CAP codes from the CSV."""
        caps: set = set()
        try:
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    code = row.get("cap_code", "").strip()
                    # Skip wildcard entries like "001xx", "091xx", etc.
                    if code and code.isdigit() and len(code) == 5:
                        caps.add(code)
            result = sorted(caps)
            logger.info(f"[OK] {len(result)} unique numeric CAP codes loaded")
            return result
        except FileNotFoundError:
            logger.error(f"[ERROR] CSV not found: {csv_file}")
            return []

    # ─── Waiting helpers ─────────────────────────────────────────────────────

    def _wait_for_list(self, timeout: int = WAIT_TIMEOUT) -> bool:
        """Return True when #listIscritti contains at least one <div> child."""
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SEL_LIST))
            )
            return True
        except TimeoutException:
            return False

    def _wait_for_list_gone(self, timeout: int = 8) -> bool:
        """Return True when the list disappears (navigated to detail page)."""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, SEL_LIST)) == 0
            )
            return True
        except TimeoutException:
            return False

    # ─── Pager helpers ───────────────────────────────────────────────────────

    def _set_page_size_20(self):
        """
        Set the Kendo pager page-size dropdown to 20 items per page.
        Tries two strategies:
          1. Use the hidden <select> element directly (fastest).
          2. Click the visible Kendo dropdown widget and pick the 20-option.
        """
        try:
            # Strategy 1: hidden <select>
            selects = self.driver.find_elements(By.CSS_SELECTOR, SEL_PAGER_SELECT)
            if selects:
                sel = Select(selects[0])
                # Find and pick the option whose value == "20"
                for opt in sel.options:
                    if opt.get_attribute("value") == "20":
                        sel.select_by_value("20")
                        logger.info("   [PAGER] Page size → 20 (hidden <select>)")
                        time.sleep(1.5)
                        self._wait_for_list()
                        return
                # If no "20" option, try selecting the last (largest) one
                if sel.options:
                    sel.select_by_index(len(sel.options) - 1)
                    logger.info(
                        f"   [PAGER] Page size → {sel.options[-1].text} (largest option)"
                    )
                    time.sleep(1.5)
                    self._wait_for_list()
                return
        except Exception as e:
            logger.debug(f"   [PAGER/select] {str(e)[:80]}")

        try:
            # Strategy 2: Kendo dropdown click + item selection
            trigger = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SEL_PAGER_DROPDOWN))
            )
            self.driver.execute_script("arguments[0].click();", trigger)
            time.sleep(0.6)

            # Items appear in a popup list outside the pager
            items = self.driver.find_elements(
                By.CSS_SELECTOR,
                ".k-popup .k-item, .k-list .k-item, ul.k-list li",
            )
            for item in items:
                if item.text.strip() == "20":
                    self.driver.execute_script("arguments[0].click();", item)
                    logger.info("   [PAGER] Page size → 20 (Kendo popup)")
                    time.sleep(1.5)
                    self._wait_for_list()
                    return

            logger.debug("   [PAGER] Option '20' not found in dropdown")
        except Exception as e:
            logger.debug(f"   [PAGER/kendo] {str(e)[:80]}")

    def _get_total_items(self) -> Optional[int]:
        """
        Parse the Kendo pager info text like "1 - 10 di 42" to get the total.
        Returns None if not parseable.
        """
        try:
            info_el = self.driver.find_element(By.CSS_SELECTOR, SEL_PAGER_INFO)
            text = info_el.text.strip()
            # Italian: "1 - 10 di 42"  or  "elementi 1 - 10 di 42"
            m = re.search(r"di\s+(\d+)", text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    def _click_next_page(self) -> bool:
        """
        Click the Kendo pager 'Next page' button.
        Returns False if the button is absent or disabled (last page).
        """
        try:
            nav_btns = self.driver.find_elements(By.CSS_SELECTOR, SEL_PAGER_NAV)
            # Kendo renders 4 nav buttons in order: first, prev, next, last.
            # Find the "next" one by title / aria-label.
            next_btn = None
            for btn in nav_btns:
                title = (btn.get_attribute("title") or "").lower()
                aria  = (btn.get_attribute("aria-label") or "").lower()
                if any(kw in title + aria for kw in ("next", "success", "prossim", "avanti")):
                    next_btn = btn
                    break
            # Fallback: buttons are [first, prev, next, last] → index 2
            if next_btn is None and len(nav_btns) >= 3:
                next_btn = nav_btns[2]

            if next_btn is None:
                return False

            cls = (
                (next_btn.get_attribute("class") or "") +
                (next_btn.find_element(By.XPATH, "..").get_attribute("class") or "")
            )
            if "k-state-disabled" in cls or "k-disabled" in cls:
                logger.info("   [PAGER] Next button disabled — this is the last page")
                return False

            self.driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(PAGE_LOAD_DELAY)
            self._wait_for_list()
            return True

        except Exception as e:
            logger.debug(f"   [NEXT_PAGE] {str(e)[:80]}")
            return False

    def _navigate_to_page(self, page_number: int):
        """
        Navigate to a specific page number by clicking 'Next' repeatedly.
        Used to restore pagination position after a hard re-navigation.
        """
        for _ in range(page_number - 1):
            if not self._click_next_page():
                break

    # ─── Back navigation ─────────────────────────────────────────────────────

    def _go_back_to_list(self, cap: str, current_page: int = 1) -> bool:
        """
        Return to the list page (from a detail page).

        Strategy 1: driver.back() — uses the browser history stack.
        Strategy 2: hard re-navigate to the search URL and restore pagination.
        """
        # Attempt 1: browser back via Selenium
        try:
            self.driver.back()
            # Wait for the Kendo list to reappear AND have content.
            # driver.back() triggers client-side re-render; the list container
            # may appear briefly empty before Kendo populates it.
            def _list_has_items(d):
                els = d.find_elements(By.CSS_SELECTOR, SEL_LIST)
                return len(els) > 0

            WebDriverWait(self.driver, BACK_DELAY + 7).until(_list_has_items)
            time.sleep(0.8)  # let Kendo finish rendering
            containers = self.driver.find_elements(By.CSS_SELECTOR, SEL_LIST)
            if containers:
                logger.debug("   [BACK] driver.back() succeeded")
                return True
        except Exception as e:
            logger.debug(f"   [BACK/driver.back] {str(e)[:60]}")

        # Attempt 2: hard re-navigate to the search URL
        logger.debug(f"   [BACK] driver.back failed — hard re-navigating to CAP {cap}")
        try:
            self.driver.get(SEARCH_URL.format(cap=cap))
            time.sleep(PAGE_LOAD_DELAY)
            if not self._wait_for_list(timeout=10):
                return False
            self._set_page_size_20()
            # Restore pagination position
            if current_page > 1:
                self._navigate_to_page(current_page)
            return self._wait_for_list(timeout=8)
        except Exception as e:
            logger.error(f"   [BACK/renavigating] {str(e)[:80]}")
            return False

    # ─── Data extraction ─────────────────────────────────────────────────────

    def _extract_record(self, soup: BeautifulSoup, cap: str) -> Dict:
        """
        Extract a commercialista record from the detail page HTML using
        the known CSS structure of the page.

        Layout inside #main-content:
          h2 > i                            → "Iscritto: {nome}"
          div.box-avvisi.row.row-evento
            div:nth-child(1)               → first info block (vari campi)
            div:nth-child(2) / col-md-4    → 4 <p> elements (e.g. luogo nascita, data, ...)
            div:nth-child(3) / col-md-8    → 3 <p> elements (ordine, data iscrizione, sezione…)
            div:nth-child(4)               → 1 <p> (e.g. sede studio / altro titolo)
          p (last)                         → "Data ultima modifica: …"
        """
        record: Dict = {field: "" for field in self.FIELD_PATTERNS}
        record["telefono"] = ""
        record["cap_code"] = cap
        record["scraped_at"] = datetime.now().isoformat(timespec="seconds")

        main = soup.select_one("#main-content")
        if not main:
            # Fallback: parse the whole page with regex
            full_text = soup.get_text("\n")
            for field, pattern in self.FIELD_PATTERNS.items():
                try:
                    m = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
                    if m:
                        record[field] = m.group(1).strip()[:200]
                except Exception:
                    pass
            return record

        # ── Name from header ──────────────────────────────────────────────────
        # Real HTML: <h2><i class="fa fa-user"></i> Iscritto  NOME COGNOME</h2>
        # The name is a text node inside h2, NOT inside the <i> tag.
        h2_el = main.select_one("h2")
        if h2_el:
            raw = h2_el.get_text(" ", strip=True)
            # Strip icon text artifacts and "Iscritto" prefix (no colon on page)
            cleaned = re.sub(r"^[Ii]scritto\s*:?\s*", "", raw).strip()
            if cleaned:
                record["nome_completo"] = cleaned[:200]

        # ── Info box ──────────────────────────────────────────────────────────
        box = main.select_one("div.box-avvisi.row.row-evento")
        if box:
            children = box.find_all("div", recursive=False)

            def _paragraphs(div_el):
                """Return list of stripped paragraph texts inside a div."""
                return [p.get_text(strip=True) for p in div_el.find_all("p")]

            # ── div:nth-child(1) — miscellaneous first block ─────────────────
            if len(children) >= 1:
                for p_text in _paragraphs(children[0]):
                    self._assign_by_regex(record, p_text)

            # ── div:nth-child(2) / col-md-4 — 4 items ────────────────────────
            # Expected: Nato a, il (data nascita), titolo professionale, revisore
            if len(children) >= 2:
                paras = _paragraphs(children[1])
                field_map_2 = [
                    ("luogo_nascita",        r"Nato\s+a\s*:\s*(.+?)(?=\s*il\s*:|\n|$)"),
                    ("data_nascita",         r"il\s*:\s*(\d{2}/\d{2}/\d{4})"),
                    ("titolo_professionale", r"Titolo\s+professionale\s*:\s*(.+)"),
                    ("revisore_contabile",   r"Revisore\s+contabile\s*:\s*(.+)"),
                ]
                for p_text in paras:
                    for field, pat in field_map_2:
                        if not record[field]:
                            m = re.search(pat, p_text, re.IGNORECASE)
                            if m:
                                record[field] = m.group(1).strip()[:200]
                    # Also run generic fallback
                    self._assign_by_regex(record, p_text)

            # ── div:nth-child(3) / col-md-8 — 3 items ────────────────────────
            # Expected: ordine, data_anzianita / data_iscrizione, sezione
            if len(children) >= 3:
                paras = _paragraphs(children[2])
                field_map_3 = [
                    ("ordine",         r"Ordine\s+di\s*:\s*(.+)"),
                    ("data_anzianita", r"Data\s+anzianit[àa]\s*:\s*(\d{2}/\d{2}/\d{4})"),
                    ("data_iscrizione",r"Iscritto\s+all['\u2019]?\s*Albo\s*:\s*(\d{2}/\d{2}/\d{4})"),
                    ("sezione",        r"Sezione\s*:\s*([A-Z0-9]+)"),
                ]
                for p_text in paras:
                    for field, pat in field_map_3:
                        if not record[field]:
                            m = re.search(pat, p_text, re.IGNORECASE)
                            if m:
                                record[field] = m.group(1).strip()[:200]
                    self._assign_by_regex(record, p_text)

            # ── div:nth-child(4) — sede studio + phone (icon-based) ────────────
            if len(children) >= 4:
                div4 = children[3]

                # Real HTML inside the <strong>:
                #   <i class="fa fa-map-marker"></i> VIA ROMA 17 ... (RM)
                #   <i class="fa fa-phone"></i> 069551605 <br/>
                # Both icons are siblings inside the same <strong>.
                # Walk children: collect addr text before fa-phone, phone text after.
                for strong in div4.find_all("strong"):
                    phone_icon = strong.find("i", class_="fa-phone")
                    if not phone_icon:
                        continue
                    addr_parts, phone_parts = [], []
                    past_phone = False
                    for child in strong.children:
                        is_tag = getattr(child, "name", None) is not None  # True only for Tag nodes
                        if is_tag and "fa-phone" in (child.get("class") or []):
                            past_phone = True
                            continue
                        if is_tag and "fa-" in " ".join(child.get("class") or []):
                            # other icon (e.g. fa-map-marker) — skip the tag itself
                            continue
                        text = str(child).strip()
                        if not text or text == "<br/>":
                            continue
                        if past_phone:
                            phone_parts.append(text)
                        else:
                            addr_parts.append(text)
                    addr = re.sub(r"\s+", " ", " ".join(addr_parts)).strip()
                    phone = re.sub(r"\D", "", " ".join(phone_parts))
                    if addr and not record["sede_studio"]:
                        record["sede_studio"] = addr[:200]
                    if phone and not record["telefono"]:
                        record["telefono"] = phone
                    break

                # Fallback: regex on paragraph text for sede_studio / altro_titolo
                if not record["sede_studio"] or not record["altro_titolo"]:
                    paras = _paragraphs(div4)
                    field_map_4 = [
                        ("sede_studio",  r"Sede\s+studio\s*:\s*(.+)"),
                        ("altro_titolo", r"Altro\s+titolo\s+professionale\s*:\s*(.+)"),
                    ]
                    for p_text in paras:
                        for field, pat in field_map_4:
                            if not record[field]:
                                m = re.search(pat, p_text, re.IGNORECASE)
                                if m:
                                    record[field] = m.group(1).strip()[:200]
                        self._assign_by_regex(record, p_text)

        # ── Update date (last <p> in #main-content) ───────────────────────────
        all_p = main.find_all("p")
        for p in reversed(all_p):
            text = p.get_text(strip=True)
            m = re.search(
                r"Data\s+ultima\s+modifica\s*:\s*(\d{2}/\d{2}/\d{4})",
                text, re.IGNORECASE,
            )
            if m:
                record["data_modifica"] = m.group(1).strip()
                break

        return record

    def _assign_by_regex(self, record: Dict, text: str):
        """Apply all FIELD_PATTERNS to *text* and fill in any empty fields."""
        for field, pattern in self.FIELD_PATTERNS.items():
            if record.get(field):   # already filled — skip
                continue
            try:
                m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if m:
                    record[field] = m.group(1).strip()[:200]
            except Exception:
                pass

    # ─── Per-CAP append ────────────────────────────────────────────────────

    def _append_cap_results(self, records: List[Dict]):
        """
        Append *records* to the output file as newline-delimited JSON (JSONL),
        upload the updated file to GCS, then let the caller clear the list
        to free memory.
        """
        if not records:
            return
        try:
            with open(self.output_file, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.total_records += len(records)
            logger.debug(
                f"[SAVE] Appended {len(records)} records → {self.output_file} "
                f"(total on disk: {self.total_records})"
            )
            self._gcs_upload(self.output_file, GCS_OUTPUT_JSON, "application/json")
        except Exception as exc:
            logger.error(f"[SAVE_ERROR] Append failed: {exc}")

    # ─── Core: scrape one CAP ────────────────────────────────────────────────

    def scrape_cap(self, cap: str) -> List[Dict]:
        """Navigate to the search page for *cap* and extract all records."""
        records: List[Dict] = []

        url = SEARCH_URL.format(cap=cap)
        logger.info(f"   → {url}")
        self.driver.get(url)
        time.sleep(PAGE_LOAD_DELAY)

        # No results for this CAP → return early
        if not self._wait_for_list(timeout=10):
            logger.info(f"   [EMPTY] No results for CAP {cap}")
            return records

        # Increase page size to 20 so we paginate less
        self._set_page_size_20()

        total_items = self._get_total_items()
        if total_items is not None:
            logger.info(f"   [INFO] Total entries for CAP {cap}: {total_items}")

        page_num = 1
        processed_on_page = 0   # items processed on the current page

        while True:
            logger.info(f"   [PAGE {page_num}] Scanning…")

            # Collect initial count of containers on this page
            containers = self.driver.find_elements(By.CSS_SELECTOR, SEL_LIST)
            total_on_page = len(containers)

            if total_on_page == 0:
                logger.info(f"   [PAGE {page_num}] No containers — stopping")
                break

            logger.info(f"   [PAGE {page_num}] {total_on_page} containers visible")

            for idx in range(total_on_page):
                try:
                    # Always re-query the container by nth-child position (1-based)
                    # to avoid StaleElementReferenceException after navigating back.
                    nth = idx + 1
                    css_nth = f"#listIscritti > div:nth-child({nth})"
                    try:
                        container = WebDriverWait(self.driver, 8).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, css_nth))
                        )
                    except TimeoutException:
                        logger.debug(f"   [{nth}] Container not found — skipping")
                        continue

                    # Get display name before clicking
                    try:
                        label = container.find_element(By.TAG_NAME, "h2").text.strip()[:60]
                    except Exception:
                        try:
                            label = container.text.strip().split("\n")[0][:60]
                        except Exception:
                            label = f"Item {nth}"

                    logger.info(f"   [{nth}/{total_on_page}] Clicking: {label}")

                    # Remember the current URL so we can detect navigation
                    url_before = self.driver.current_url

                    # Prefer clicking the <a> link inside the container (if present)
                    # so the browser records a proper history entry for driver.back()
                    click_target = container
                    try:
                        link = container.find_element(By.TAG_NAME, "a")
                        if link:
                            click_target = link
                    except NoSuchElementException:
                        pass

                    # Scroll into view, then click via JS to avoid interception
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", click_target
                    )
                    time.sleep(0.3)
                    self.driver.execute_script("arguments[0].click();", click_target)

                    # Wait for navigation: URL must change to the detail page
                    navigated = False
                    try:
                        WebDriverWait(self.driver, 10).until(
                            lambda d: d.current_url != url_before
                        )
                        navigated = True
                    except TimeoutException:
                        pass

                    if not navigated:
                        # Fallback: wait for list to disappear (SPA behaviour)
                        self._wait_for_list_gone(timeout=6)

                    # Give the detail page time to fully render
                    time.sleep(CLICK_DELAY)

                    # ── Wait for the detail-page header to appear ──────────────
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "#main-content > h2 > i")
                            )
                        )
                    except TimeoutException:
                        # Not on a detail page — log and go back
                        logger.warning(
                            f"   [{nth}] Detail header not found after click; "
                            f"URL: {self.driver.current_url[:80]}"
                        )
                        self._go_back_to_list(cap, current_page=page_num)
                        continue

                    # ── Parse the detail page ──────────────────────────────────
                    soup = BeautifulSoup(self.driver.page_source, "html.parser")
                    record = self._extract_record(soup, cap)
                    records.append(record)
                    processed_on_page += 1

                    logger.info(
                        f"   [OK] {record.get('nome_completo', '?')[:50]} | "
                        f"Ordine: {record.get('ordine', '?')[:30]}"
                    )

                    # ── Go back to the search list ─────────────────────────────
                    if not self._go_back_to_list(cap, current_page=page_num):
                        logger.warning(
                            f"   [WARN] Could not return to list — aborting CAP {cap}"
                        )
                        return records

                    # After going back, re-check visible containers count
                    new_containers = self.driver.find_elements(By.CSS_SELECTOR, SEL_LIST)
                    total_on_page = len(new_containers)

                except StaleElementReferenceException:
                    logger.debug(f"   [{idx+1}] Stale element — skipping")
                    self._go_back_to_list(cap, current_page=page_num)
                    continue

                except Exception as exc:
                    logger.error(f"   [{idx+1}] Unexpected error: {str(exc)[:120]}")
                    try:
                        self._go_back_to_list(cap, current_page=page_num)
                    except Exception:
                        pass
                    continue

            # Try to advance to the next page
            processed_on_page = 0
            if self._click_next_page():
                page_num += 1
                # After pagination, re-apply page-size (Kendo sometimes resets it)
                # Only if the dropdown is visible; _set_page_size_20 handles this gracefully
                self._set_page_size_20()
            else:
                logger.info(f"   [DONE] No more pages for CAP {cap}")
                break

        logger.info(f"   [DONE] CAP {cap} → {len(records)} records extracted")
        return records

    # ─── Main scrape loop ────────────────────────────────────────────────────

    def scrape_all(
        self,
        cap_list: List[str],
        max_caps: Optional[int] = None,
        delay: float = INTER_CAP_DELAY,
        resume_from: Optional[str] = None,
    ):
        """Scrape all CAPs, with optional limit and resume support."""
        # Initialise GCS first (needed for checkpoint uploads)
        self._init_gcs()

        # Restore any previously accumulated output so appends don't clobber it
        self._gcs_download_output(self.output_file)

        if not self.setup_driver():
            logger.error("[FATAL] Cannot initialise WebDriver — exiting")
            return

        if max_caps:
            cap_list = cap_list[:max_caps]

        if resume_from:
            try:
                start_idx = cap_list.index(resume_from)
                cap_list = cap_list[start_idx:]
                logger.info(f"[RESUME] Starting from CAP {resume_from}")
            except ValueError:
                logger.warning(
                    f"[RESUME] CAP {resume_from} not found — starting from the beginning"
                )

        total = len(cap_list)
        logger.info(f"\n{'='*60}")
        logger.info(f"[START] {total} CAP codes to scrape")
        logger.info(f"{'='*60}\n")

        try:
            for i, cap in enumerate(cap_list, 1):
                logger.info(f"\n[{i}/{total}] Processing CAP: {cap}")

                # ── Overwrite GCS checkpoint so a restart picks up here ───────
                self._gcs_write_current_cap(cap)

                # ── Driver health-check: restart if session is dead ───────────
                if not self._driver_is_alive():
                    logger.warning("[RECOVER] WebDriver session lost — restarting…")
                    self.close_driver()
                    if not self.setup_driver():
                        logger.error("[FATAL] Could not restart WebDriver — exiting")
                        break

                try:
                    recs = self.scrape_cap(cap)
                    self._append_cap_results(recs)   # flush to disk immediately
                    self.results.clear()              # free memory
                    if recs:
                        logger.info(
                            f"   +{len(recs)} records | Running total: {self.total_records}"
                        )
                    else:
                        logger.info(f"   No records for CAP {cap}")
                except Exception as exc:
                    err_str = str(exc)
                    logger.error(f"   [CAP_ERROR] {cap}: {err_str[:100]}")
                    self.failed_caps.append((cap, err_str[:80]))

                    # If this looks like a WebDriver/connection error, restart now
                    if any(kw in err_str.lower() for kw in (
                        "read timed out", "connectionreset", "connectionrefused",
                        "max retries", "remotedisconnected", "webdriver", "session"
                    )):
                        logger.warning("[RECOVER] Connection error detected — restarting driver")
                        self.close_driver()
                        if not self.setup_driver():
                            logger.error("[FATAL] Could not restart WebDriver — exiting")
                            break


                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("\n[INTERRUPTED] Saving progress and exiting…")
        finally:
            self.close_driver()
            self._print_summary(total)

    def _print_summary(self, total_caps: int):
        logger.info(f"\n{'='*60}")
        logger.info("Scraping complete!")
        logger.info(f"  CAPs processed : {total_caps}")
        logger.info(f"  Total records  : {self.total_records}")
        logger.info(f"  Failed CAPs    : {len(self.failed_caps)}")
        logger.info(f"  Output JSON    : {self.output_file}")
        if self.failed_caps:
            logger.warning("  Failed CAPs (first 20):")
            for cap, reason in self.failed_caps[:20]:
                logger.warning(f"    - {cap}: {reason}")
            if len(self.failed_caps) > 20:
                logger.warning(f"    … and {len(self.failed_caps) - 20} more")
        logger.info(f"{'='*60}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Scrape the Italian Commercialisti national register"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Path to a local CAP CSV file. "
            "If omitted the file is downloaded from "
            f"gs://{GCS_BUCKET}/{GCS_CAP_CSV_PATH}"
        ),
    )
    parser.add_argument(
        "--output",
        default="commercialisti_data.json",
        help="Local output JSONL file",
    )
    parser.add_argument(
        "--max-caps",
        type=int,
        default=None,
        help="Process only the first N CAPs (useful for testing)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CAP",
        help="Resume scraping starting from this CAP code",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=INTER_CAP_DELAY,
        help=f"Seconds to wait between CAPs (default: {INTER_CAP_DELAY})",
    )
    args = parser.parse_args()

    scraper = CommercialisitiScraper(output_file=args.output)

    # ── Resolve the CAP CSV ───────────────────────────────────────────────────
    csv_path = args.csv
    _tmp_csv = None   # track temp file so we can clean it up

    if csv_path is None:
        # Attempt to download from GCS
        scraper._init_gcs()
        if scraper.storage_client:
            _tmp_csv = tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, prefix="cap_comuni_"
            )
            _tmp_csv.close()
            csv_path = _tmp_csv.name
            if not scraper._gcs_download_csv(csv_path):
                logger.error("[ERROR] Could not download CAP CSV from GCS — exiting")
                return
        else:
            logger.error(
                "[ERROR] No --csv supplied and GCS client unavailable — exiting"
            )
            return
    # ─────────────────────────────────────────────────────────────────────────

    caps = CommercialisitiScraper.get_unique_caps(csv_path)

    # Clean up temp file after loading
    if _tmp_csv is not None:
        try:
            os.unlink(csv_path)
        except Exception:
            pass

    if not caps:
        logger.error("[ERROR] No valid CAP codes found — exiting")
        return

    logger.info(f"[INFO] First CAP: {caps[0]}  |  Last CAP: {caps[-1]}")

    # ── Resolve resume CAP: CLI flag > GCS file > start from beginning ──────
    resume_cap = args.resume
    if resume_cap is None:
        resume_cap = scraper._gcs_read_start_cap()
        if resume_cap:
            logger.info(f"[INFO] Auto-resuming from GCS checkpoint CAP: {resume_cap}")
        else:
            logger.info("[INFO] No resume CAP found — starting from the beginning")

    scraper.scrape_all(
        cap_list=caps,
        max_caps=args.max_caps,
        delay=args.delay,
        resume_from=resume_cap,
    )


if __name__ == "__main__":
    main()
