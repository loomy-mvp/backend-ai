import os
import re
import time
from collections import OrderedDict
from typing import List, Tuple, Optional

import requests
from bs4 import BeautifulSoup

###### EDIT THIS ######
TARGET_LAWS = [
    {"slug": "Codice civile", "url": "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:codice.civile:1942-03-16;262!vig="},
]

# ! Run it from inside the parlamento folder
OUTPUT_DIR = os.path.join(os.getcwd(), "selected_leggi_output")
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    sanitized = re.sub(r"__+", "_", sanitized)
    return sanitized.strip("_") or "untitled"


class NormattivaScraper:
    def __init__(self) -> None:
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        }

    def fetch(self, session: requests.Session, url: str, **kwargs) -> requests.Response:
        for attempt in range(MAX_RETRIES):
            try:
                response = session.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT, **kwargs)
                response.raise_for_status()
                return response
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = (attempt + 1) * 2
                print(f"    Request to {url} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def extract_articles_akn(self, xml_text: str) -> List[Tuple[str, str]]:
        import xml.etree.ElementTree as ET

        articles: List[Tuple[str, str]] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:
            print(f"      Failed to parse AKN XML: {exc}")
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

    def extract_articles_html(self, html: str) -> List[Tuple[str, str]]:
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

    def parse_title(self, soup: BeautifulSoup) -> Optional[str]:
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

    def scrape_law(self, norm_url: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
        session = requests.Session()
        articles: List[Tuple[str, str]] = []
        try:
            print(f"  Loading Normattiva page: {norm_url}")
            resp = self.fetch(session, norm_url)
        except Exception as exc:
            print(f"  Failed to load Normattiva page {norm_url}: {exc}")
            return None, articles
        soup = BeautifulSoup(resp.text, "lxml")
        law_title = self.parse_title(soup)
        akn_link = None
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'caricaAKN' in href:
                akn_link = requests.compat.urljoin(norm_url, href)
                break
        if akn_link:
            print(f"    Found Akoma Ntoso link: {akn_link}")
            try:
                akn_resp = self.fetch(session, akn_link)
                arts = self.extract_articles_akn(akn_resp.text)
                if arts:
                    print(f"    Extracted {len(arts)} articles from AKN")
                    return law_title, arts
                print("    No articles found in AKN; falling back to HTML")
            except Exception as exc:
                print(f"    Failed to download or parse AKN: {exc}")
        page_source = resp.text
        article_paths = re.findall(r"showArticle\('(/atto/caricaArticolo[^']+)'", page_source)
        if article_paths:
            print(f"    Found {len(article_paths)} article link(s) in page source")
            articles_map: OrderedDict[str, List[str]] = OrderedDict()
            seen_paths: set[str] = set()
            duplicates_count = 0
            for path in article_paths:
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                full_url = requests.compat.urljoin(norm_url, path)
                try:
                    article_resp = self.fetch(session, full_url)
                except Exception as exc:
                    print(f"      Failed to fetch article {full_url}: {exc}")
                    continue
                art_tuples = self.extract_articles_html(article_resp.text)
                if not art_tuples:
                    continue
                art_num, art_text = art_tuples[0]
                segs = articles_map.setdefault(art_num, [])
                if art_text in segs:
                    duplicates_count += 1
                    if duplicates_count > 10:
                        print("      Too many duplicate article segments; stopping.")
                        break
                    continue
                duplicates_count = 0
                segs.append(art_text)
                print(f"      Fetched {art_num} (segment {len(segs)})")
                time.sleep(0.5)
            if articles_map:
                for num, segs in articles_map.items():
                    joined = '\n\n'.join(segs)
                    articles.append((num, joined))
                return law_title, articles
        print("    Falling back to initial article on page")
        first_article = self.extract_articles_html(resp.text)
        return law_title, first_article


def save_articles(filename: str, articles: List[Tuple[str, str]], header: Optional[str]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as handle:
        if header:
            handle.write(header.strip() + '\n\n')
        for art_num, art_text in articles:
            handle.write(f"{art_num}\n")
            handle.write(art_text.strip() + '\n\n')
    print(f"  Saved {len(articles)} article(s) to {filepath}")


def main() -> None:
    scraper = NormattivaScraper()
    for law in TARGET_LAWS:
        slug = sanitize_filename(law['slug'])
        print(f"\n=== Processing {slug} ===")
        title, articles = scraper.scrape_law(law['url'])
        if not articles:
            print("  No articles extracted; skipping file write")
            continue
        filename = f"{slug}.txt"
        save_articles(filename, articles, title)


if __name__ == "__main__":
    main()
