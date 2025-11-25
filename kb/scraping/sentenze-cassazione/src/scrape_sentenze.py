"""Scraper for the ForoEuropeo Cassazione civil summaries archive.

The scraper mirrors the conventions used by the other kb scrapers:
- pulls configuration via CLI/Cloud Run arguments
- pushes every downloaded asset directly to Google Cloud Storage
- returns a structured summary for logging/monitoring
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from google.cloud import storage

from src.utils.upload_to_storage import upload_to_storage

BASE_URL = (
    "https://www.foroeuropeo.it/rubriche-servizi-codici-2024/"
    "raccolta-di-massime-civili-della-cassazione-classificate-per-argomento-materia.html"
)

CATEGORY_PATTERN = re.compile(
    r"^/rubriche-servizi-codici-2024/"
    r"raccolta-di-massime-civili-della-cassazione-classificate-per-argomento-materia/"
    r"(\d+)-[\w\d\-]+\.html$"
)

DEFAULT_DELAY_SECONDS = 0.5


@dataclass
class ForoEuropeoScraper:
    """Scrape civil massime from ForoEuropeo and upload them to GCS."""

    bucket_name: str
    base_folder: str = "sentenze_cassazione"
    base_url: str = BASE_URL
    request_delay: float = DEFAULT_DELAY_SECONDS
    storage_client: Optional[storage.Client] = None
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage_client = self.storage_client or storage.Client()
        self.base_folder = self._normalize_folder(self.base_folder)
        self.processed_articles: Set[str] = set()
        self.files_by_category: Dict[str, int] = {}
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64; rv:117.0) "
                    "Gecko/20100101 Firefox/117.0"
                ),
                "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )

    # ---------------------------------------------------------------------
    # Parsing helpers
    # ---------------------------------------------------------------------
    def fetch(self, url: str) -> requests.Response:
        self.logger.debug("Fetching %s", url)
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        return resp

    def parse_categories(self) -> List[str]:
        resp = self.fetch(self.base_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        categories: Set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if CATEGORY_PATTERN.match(href):
                categories.add(urljoin(self.base_url, href))
        sorted_categories = sorted(categories)
        self.logger.info("Discovered %d categories", len(sorted_categories))
        return sorted_categories

    def parse_article_links(self, category_url: str) -> List[str]:
        resp = self.fetch(category_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        article_urls: Set[str] = set()
        parsed_category = urlparse(category_url)
        category_path = parsed_category.path[:-5] if parsed_category.path.endswith(".html") else parsed_category.path
        if not category_path.endswith("/"):
            category_path += "/"
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href.endswith(".html"):
                continue
            if not href.startswith("/"):
                continue
            if href.startswith(category_path):
                full_url = urljoin(category_url, href)
                if full_url.rstrip("/") == category_url.rstrip("/"):
                    continue
                article_urls.add(full_url)
        sorted_articles = sorted(article_urls)
        self.logger.info("Found %d articles in %s", len(sorted_articles), category_url)
        return sorted_articles

    def parse_article(self, article_url: str) -> Optional[Tuple[str, str, str]]:
        slug = self._article_slug(article_url)
        if slug in self.processed_articles:
            return None
        resp = self.fetch(article_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        heading = soup.find("h1", class_=re.compile("uk-h[0-9]"))
        if not heading:
            return None
        title = heading.get_text(strip=True)
        content_div = heading.find_next("div", class_=re.compile("uk-panel"))
        if not content_div:
            return None
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in content_div.find_all("p")
            if p.get_text(strip=True)
        ]
        if not paragraphs:
            return None
        body = "\n\n".join(paragraphs)
        self.processed_articles.add(slug)
        return slug, title, body

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------
    def save_article(
        self,
        category_slug: str,
        slug: str,
        title: str,
        body: str,
        article_url: str,
    ) -> Optional[str]:
        safe_category = self._safe_slug(category_slug)
        safe_slug = self._safe_slug(slug)
        folder = "/".join(part for part in [self.base_folder, safe_category] if part)
        filename = f"{safe_slug}.txt"
        payload = f"{title}\n\n{body}\n\nFonte: {article_url}\n"
        pdf_obj = {
            "name": filename,
            "bytes": payload.encode("utf-8"),
            "extension": "txt",
        }
        try:
            upload_to_storage(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                pdf_obj=pdf_obj,
                folder=folder,
            )
            return f"{folder}/{filename}" if folder else filename
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to upload %s: %s", filename, exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scrape(self, limit_categories: Optional[int] = None) -> Dict[str, int]:
        categories = self.parse_categories()
        if limit_categories is not None:
            categories = categories[:limit_categories]
        stats = {
            "categories_selected": len(categories),
            "categories_processed": 0,
            "categories_failed": 0,
            "articles_discovered": 0,
            "files_uploaded": 0,
            "duplicates_skipped": 0,
            "articles_failed": 0,
            "files_by_category": {},
        }
        for index, category_url in enumerate(categories, start=1):
            category_slug = self._category_slug(category_url)
            self.logger.info("[%d/%d] Processing category %s", index, len(categories), category_slug)
            try:
                article_urls = self.parse_article_links(category_url)
            except Exception as exc:
                stats["categories_failed"] += 1
                self.logger.error("Failed to fetch category %s: %s", category_url, exc)
                continue
            stats["categories_processed"] += 1
            stats["articles_discovered"] += len(article_urls)
            for article_url in article_urls:
                slug = self._article_slug(article_url)
                if slug in self.processed_articles:
                    stats["duplicates_skipped"] += 1
                    continue
                parsed = None
                try:
                    parsed = self.parse_article(article_url)
                except Exception as exc:  # pragma: no cover - logging only
                    self.logger.error("Error parsing %s: %s", article_url, exc)
                if not parsed:
                    stats["articles_failed"] += 1
                    continue
                slug, title, body = parsed
                uploaded_path = self.save_article(category_slug, slug, title, body, article_url)
                if not uploaded_path:
                    stats["articles_failed"] += 1
                    continue
                stats["files_uploaded"] += 1
                self.files_by_category[category_slug] = self.files_by_category.get(category_slug, 0) + 1
        stats["files_by_category"] = dict(sorted(self.files_by_category.items(), key=lambda item: item[0]))
        return stats

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _article_slug(article_url: str) -> str:
        return article_url.rstrip("/").split("/")[-1].replace(".html", "")

    @staticmethod
    def _category_slug(category_url: str) -> str:
        return category_url.rstrip("/").split("/")[-1].replace(".html", "")

    @staticmethod
    def _safe_slug(raw: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9\-]+", "_", raw).strip("_")
        return sanitized or "item"

    @staticmethod
    def _normalize_folder(folder: str) -> str:
        trimmed = folder.strip().strip("/")
        return trimmed.replace("\\", "/") if trimmed else ""


def run_standalone(bucket_name: str, base_folder: str = "sentenze_cassazione") -> Dict[str, int]:
    """Helper for local/manual execution."""
    scraper = ForoEuropeoScraper(bucket_name=bucket_name, base_folder=base_folder)
    return scraper.scrape()
