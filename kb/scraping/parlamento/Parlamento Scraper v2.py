#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python3
"""
scrape_leggi.py

This script crawls the Italian Parliament website and the Normattiva portal
to fetch the full text of laws for a set of thematic areas.  For each area
it enumerates every law listed under that theme, follows the link to the
Normattiva entry and downloads all of the articles belonging to that law.

Usage:
    python scrape_leggi.py

On completion the script creates one directory per thematic area in the
current working directory.  Inside each area directory there will be a
plain‑text file for each law containing the full text of all its
articles.  Progress and diagnostic information is printed to the console
to aid debugging if anything goes wrong.

Notes:

* The script uses only the standard library plus requests and
  BeautifulSoup4, which are widely available.  No external browser
  automation is required – HTTP requests are used directly.  A custom
  User‑Agent and Accept‑Language header are supplied to mimic a real
  browser.  If the Normattiva site returns a 403 status for a given
  request, you may need to adjust headers or your network settings.

* Normattiva pages rely heavily on JavaScript to load articles on
  demand.  The script first looks for an “Akoma Ntoso” export link
  (``caricaAKN``).  If found, that XML document is parsed to extract
  every article.  When no AKN export is available or the download
  fails, the code falls back to parsing the list of article links in
  the page source (``showArticle('/atto/caricaArticolo?...')``).  Each
  of those article endpoints is then requested via the same session to
  preserve cookies.  The retrieved HTML is parsed to extract the
  article number and body.  Finally, all articles are written to a
  single text file.

* The set of thematic areas is defined in the ``areas`` dictionary
  below.  Each key is a sanitised folder name and each value is the
  numeric ``area_tematica`` identifier observed in the site’s
  navigation.  If new areas need to be added or an ID is wrong, edit
  this dictionary accordingly.
"""

import os
import re
import time
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# Mapping of thematic areas to their numerical IDs on parlamento.it.
# You can edit or extend this dictionary if new areas are needed.
areas = {
    "Assistenza_previdenza_assicurazioni": 15,
    "Banche_credito_moneta": 10,
    "Bilancio_dello_Stato_e_manovra_finanziaria": 33,
    "Borsa_e_attivita_finanziarie": 3,
    "Commercio_con_l_estero": 9,
    "Commercio_e_servizi": 11,
    "Diritto_commerciale_e_delle_societa": 18,
    "Diritto_e_giustizia": 19,
    "Finanza_locale_e_regionale": 22,
    "Finanze_e_fisco": 23,
    "Industria_e_artigianato": 24,
    "Lavori_pubblici_edilizia_e_politica_abitativa": 2,
    "Occupazione_lavoro_e_professioni": 26,
    "Politica_economica_e_privatizzazioni": 28,
    "Pubblica_amministrazione_pubblico_impiego_e_servizi_pubblici": 29,
    "Regioni_e_autonomie_locali": 4,
    "Tutela_dei_lavoratori_sindacati_e_sicurezza_nel_lavoro": 5,
}


def sanitize_filename(name: str) -> str:
    """Return a filesystem‑safe version of *name*.

    Replaces any character that is not alphanumeric, hyphen or underscore
    with an underscore and collapses consecutive underscores.  This
    avoids problems with special characters in file names.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    sanitized = re.sub(r"__+", "_", sanitized)
    return sanitized.strip("_") or "untitled"


class LawScraper:
    """Encapsulates scraping logic for Parliament and Normattiva."""

    def __init__(self):
        # Use a common headers dict to look like a modern browser.
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        }

    def fetch(self, session: requests.Session, url: str, **kwargs) -> requests.Response:
        """Send a GET request with retry and headers.

        If a request fails due to a transient network error, it will be
        retried a few times before giving up.  Raises on non‑200
        responses.  The caller should catch exceptions to handle
        failures gracefully.
        """
        tries = 3
        for attempt in range(tries):
            try:
                response = session.get(url, headers=self.headers, timeout=30, **kwargs)
                response.raise_for_status()
                return response
            except Exception as exc:
                if attempt < tries - 1:
                    wait = (attempt + 1) * 2
                    print(f"    Request to {url} failed ({exc}); retrying in {wait} seconds…")
                    time.sleep(wait)
                    continue
                raise

    def list_laws(self, area_id: int) -> list[tuple[str, str]]:
        """Return a list of (title, normattiva_url) tuples for the given area.

        Crawls the paginated list of laws for *area_id* on
        parlamento.it.  Follows "next" links until there are no more
        pages.  All discovered Normattiva URLs are returned along with
        the displayed title.  If no laws are found an empty list is
        returned.
        """
        session = requests.Session()
        laws: list[tuple[str, str]] = []
        # Start from the first page.
        next_url = f"https://www.parlamento.it/Parlamento/523?area_tematica={area_id}"
        page_count = 1
        while next_url:
            print(f"    Fetching list page {page_count}: {next_url}")
            resp = self.fetch(session, next_url)
            soup = BeautifulSoup(resp.text, "lxml")
            # Extract all law entries in this page.
            for dt in soup.select("dl.leggi_dl dt"):
                # Each dt contains a span.normaattiva with the Normattiva link.
                span = dt.find("span", class_="normaattiva")
                if not span:
                    continue
                a = span.find("a")
                if not a or not a.get("href"):
                    continue
                title = a.get_text(strip=True)
                norm_url = a['href'].strip()
                # Normalise relative links.
                norm_url = urljoin(next_url, norm_url)
                laws.append((title, norm_url))
            print(f"      Found {len(laws)} laws so far…")
            # Find the next page link.
            next_link = soup.find("li", class_="next")
            if next_link and next_link.find("a"):
                href = next_link.find("a").get("href")
                next_url = urljoin(next_url, href)
                page_count += 1
                # Small pause to avoid hammering the server.
                time.sleep(1)
            else:
                next_url = None
        return laws

    def extract_articles_akn(self, xml_text: str) -> list[tuple[str, str]]:
        """Parse an Akoma Ntoso XML document and return a list of articles.

        Each returned tuple contains the article number (e.g. "Art. 1")
        and the concatenated text of all paragraphs/commi within that
        article.  If parsing fails or no articles are found an empty
        list is returned.  The function gracefully ignores XML
        namespace prefixes.
        """
        import xml.etree.ElementTree as ET
        articles: list[tuple[str, str]] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:
            print(f"      Failed to parse AKN XML: {exc}")
            return articles
        # Determine namespace mapping.  Akoma Ntoso documents usually
        # declare a default namespace which we need to include when
        # searching for elements.  If none is declared we search
        # without a namespace.
        ns = {'akn': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
        # Find all article elements.  They may be nested under
        # <body>→<section> etc., so we perform a deep search.
        for art in root.findall('.//{*}article'):
            num_el = art.find('.//{*}num')
            article_num = num_el.text.strip() if num_el is not None else 'Articolo'
            # Extract all text content inside the article.  We join
            # paragraphs and commi with spaces to make a readable
            # continuous string.  Filtering None prevents crashes on
            # elements with no text.
            texts: list[str] = []
            for elem in art.iter():
                if elem.text and elem is not num_el:
                    texts.append(elem.text.strip())
            article_text = ' '.join(t for t in texts if t)
            articles.append((article_num, article_text))
        return articles

    def extract_articles_html(self, html: str) -> list[tuple[str, str]]:
        """Extract article text from an HTML fragment returned by caricaArticolo.

        This parser attempts to preserve the numbering of the comma (paragraph)
        structure used by Normattiva.  Each article is typically composed of
        several ``div`` elements with class ``art-comma-div-akn``.  Inside
        each comma div there is a ``span.comma-num-akn`` containing the
        paragraph number (e.g. ``1.``) and a ``span.art_text_in_comma``
        containing the paragraph text.  When available, we interleave these
        numbers and texts into separate lines.  If those spans are not
        present we fall back to the raw text.

        Returns a list with a single tuple (article_num, article_text),
        where article_num is the heading (e.g. ``Art. 5``) and
        article_text is the joined comma texts separated by newlines.
        """
        soup = BeautifulSoup(html, "lxml")
        body = soup.find(class_="bodyTesto")
        if not body:
            return []
        # Article number is usually in h2.article-num-akn
        h2 = body.find("h2", class_="article-num-akn")
        if h2:
            article_num = h2.get_text(strip=True)
        else:
            # Fallback: look for any heading inside body
            h_generic = body.find(re.compile(r'^(h[1-6]|strong)$'))
            article_num = h_generic.get_text(strip=True) if h_generic else 'Articolo'
        lines: list[str] = []
        # Normattiva groups each comma in a dedicated div
        comma_divs = body.find_all('div', class_='art-comma-div-akn')
        if comma_divs:
            for comma_div in comma_divs:
                num_span = comma_div.find('span', class_='comma-num-akn')
                txt_span = comma_div.find('span', class_='art_text_in_comma')
                if num_span or txt_span:
                    num = num_span.get_text(strip=True) if num_span else ''
                    txt = txt_span.get_text(strip=True) if txt_span else comma_div.get_text(strip=True)
                    # Combine number and text; strip to avoid double spaces
                    combined = (num + ' ' + txt).strip()
                    lines.append(combined)
                else:
                    lines.append(comma_div.get_text(strip=True))
        # If no comma-divs were found, fall back to the body text as one piece
        if not lines:
            # Attempt to preserve simple numbering by splitting on \n
            raw = body.get_text(separator='\n', strip=True)
            # Replace multiple newlines with single newline
            raw = re.sub(r'\n+', '\n', raw)
            lines = [raw]
        article_text = '\n'.join(lines)
        return [(article_num, article_text)]

    def scrape_law(self, norm_url: str) -> list[tuple[str, str]]:
        """Retrieve all articles for a single law from Normattiva.

        Given a Normattiva base URL for a law, this method attempts
        multiple strategies to extract all articles:
        1. Look for an Akoma Ntoso export link and parse it.
        2. Parse the list of article loaders (caricaArticolo) and
           request each one via the same session.
        3. As a last resort, parse the initial page’s loaded article
           (which is usually Art. 1).

        Returns a list of (article_num, article_text) tuples.  If no
        articles are found the list will be empty.
        """
        session = requests.Session()
        articles: list[tuple[str, str]] = []
        try:
            print(f"        Loading Normattiva page: {norm_url}")
            resp = self.fetch(session, norm_url)
        except Exception as exc:
            print(f"        Failed to load Normattiva page {norm_url}: {exc}")
            return articles
        soup = BeautifulSoup(resp.text, "lxml")
        # Attempt 1: find AKN export link.
        akn_link = None
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'caricaAKN' in href:
                akn_link = urljoin(norm_url, href)
                break
        if akn_link:
            print(f"        Found Akoma Ntoso link: {akn_link}")
            try:
                akn_resp = self.fetch(session, akn_link)
                # If the response is XML, decode as text; otherwise it's a file to download.
                xml_text = akn_resp.text
                arts = self.extract_articles_akn(xml_text)
                if arts:
                    print(f"        Extracted {len(arts)} articles from AKN")
                    return arts
                else:
                    print("        No articles found in AKN; falling back to HTML")
            except Exception as exc:
                print(f"        Failed to download or parse AKN: {exc}")
        # Attempt 2: parse caricaArticolo links from the page source.
        page_source = resp.text
        article_paths = re.findall(r"showArticle\('(/atto/caricaArticolo[^']+)'", page_source)
        if article_paths:
            print(f"        Found {len(article_paths)} article link(s) in page source")
            # Use an ordered map to preserve the order of appearance of each article number.
            from collections import OrderedDict
            articles_map: "OrderedDict[str, list[str]]" = OrderedDict()
            duplicates_count = 0
            seen_paths: set[str] = set()
            for path in article_paths:
                # Avoid requesting the same endpoint twice
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                full_url = urljoin(norm_url, path)
                try:
                    article_resp = self.fetch(session, full_url)
                except Exception as exc:
                    print(f"          Failed to fetch article {full_url}: {exc}")
                    continue
                art_tuples = self.extract_articles_html(article_resp.text)
                if not art_tuples:
                    continue
                art_num, art_text = art_tuples[0]
                # Append text segments for the same article number.
                segs = articles_map.setdefault(art_num, [])
                # Deduplicate identical segments within the same article number.
                if art_text in segs:
                    duplicates_count += 1
                    # If too many duplicates in a row, abort to avoid infinite loop.
                    if duplicates_count > 10:
                        print("          Too many duplicate article segments encountered; aborting further article fetches.")
                        break
                    continue
                duplicates_count = 0
                segs.append(art_text)
                print(f"          Fetched {art_num} (segment {len(segs)})")
                # Polite pause between requests
                time.sleep(0.5)
            # Assemble final list by concatenating segments per article number
            if articles_map:
                for num, segs in articles_map.items():
                    joined = '\n\n'.join(segs)
                    articles.append((num, joined))
                return articles
        # Attempt 3: fall back to the first article loaded in the HTML.
        print("        Falling back to initial article on page")
        first_article = self.extract_articles_html(resp.text)
        if first_article:
            return first_article
        return []

    def run(self):
        """Run the entire scraping process over all configured areas."""
        base_output = os.path.join(os.getcwd(), 'leggi_output')
        os.makedirs(base_output, exist_ok=True)
        for area_name, area_id in areas.items():
            print(f"\n=== Processing thematic area: {area_name} (ID {area_id}) ===")
            area_dir = os.path.join(base_output, area_name)
            os.makedirs(area_dir, exist_ok=True)
            try:
                law_list = self.list_laws(area_id)
            except Exception as exc:
                print(f"  Failed to list laws for area {area_name}: {exc}")
                continue
            print(f"  Total laws found: {len(law_list)}")
            for idx, (title, norm_url) in enumerate(law_list, 1):
                print(f"\n  [{idx}/{len(law_list)}] Processing law: {title}")
                articles = self.scrape_law(norm_url)
                if not articles:
                    print("    No articles extracted; skipping saving")
                    continue
                filename = sanitize_filename(title) + '.txt'
                filepath = os.path.join(area_dir, filename)
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        for art_num, art_text in articles:
                            # Write article heading and text separated by blank lines
                            f.write(f"{art_num}\n")
                            f.write(art_text.strip() + '\n\n')
                    print(f"    Saved {len(articles)} unique article(s) to {filepath}")
                except Exception as exc:
                    print(f"    Failed to write file {filepath}: {exc}")


if __name__ == '__main__':
    scraper = LawScraper()
    scraper.run()


# In[1]:




