import random
import re
import requests
from bs4 import BeautifulSoup
from typing import Optional, List, Dict
from urllib.parse import urljoin

from scrappers.pubMed_scrapper.url_resolver import (
    find_citedby_section,
    is_pubmed_article_url,
    is_see_all_link
)
from doc_recoverers.interfaces.scrapper import Scraper
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"
]

HEADERS_TEMPLATE = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Pragma": "no-cache",
    "TE": "trailers"
}

TIMEOUT = 20


def get_random_headers():
    headers = HEADERS_TEMPLATE.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)
    return headers


class PubMedScraper(Scraper):
    def get_metadata(self, url: str) -> Dict[str, object]:
        headers = get_random_headers()
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        title = soup.select_one("h1.heading-title")
        title_text = title.get_text(strip=True) if title else ""
        abstract_parts = soup.select("div.abstract-content > p")
        abstract = " ".join(p.get_text(strip=True) for p in abstract_parts).strip()
        authors = list(dict.fromkeys([
            re.sub(r'[^a-zA-Z\s\-\.]', '', a.get_text(strip=True).rstrip(",")).strip()
            for a in soup.select("div.authors-list span.authors-list-item")
        ]))
        return {"title": title_text, "abstract": abstract, "authors": authors}

    def get_full_text(self, url: str) -> Optional[str]:
        return None

    def get_similar_urls(self, url: str) -> List[str]:
        headers = get_random_headers()
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        section = soup.select_one("section.similar-articles, div.similar-articles")
        if not section:
            return []
        urls = []
        for a in section.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/") and href.count("/") >= 2:
                urls.append(urljoin("https://pubmed.ncbi.nlm.nih.gov", href))
        return urls

    def get_cited_by_urls(self, url: str) -> List[str]:
        headers = get_random_headers()
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        cited_section = find_citedby_section(soup)
        if not cited_section:
            return []
        urls = []
        entries = cited_section.select('.citation-item, .docsum, li.citation, div.citation')
        if entries:
            for entry in entries:
                link = entry.find('a', href=True)
                if link:
                    href = link['href'].strip()
                    if is_pubmed_article_url(href):
                        urls.append(urljoin("https://pubmed.ncbi.nlm.nih.gov", href))
        else:
            for a in cited_section.find_all('a', href=True):
                href = a['href'].strip()
                if is_pubmed_article_url(href) and not is_see_all_link(a):
                    urls.append(urljoin("https://pubmed.ncbi.nlm.nih.gov", href))
        return urls

    def get_reference_urls(self, url: str) -> List[str]:
        similar = set(self.get_similar_urls(url))
        cited = set(self.get_cited_by_urls(url))
        return list(similar.union(cited))