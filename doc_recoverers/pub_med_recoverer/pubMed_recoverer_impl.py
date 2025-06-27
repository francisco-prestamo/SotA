import random
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Set, Dict, Any
import requests
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from controller.interfaces.doc_recoverer import DocRecoverer
from entities.document import Document
from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner


class PubMedRecoverer(DocRecoverer):
    PUBMED_BACKOFF_MIN = 1.0
    PUBMED_BACKOFF_MAX = 2.0
    TIMEOUT = 15
    MAX_RETRIES = 100
    PDF_THREADS = 20
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    @property
    def name(self) -> str:
        return "PubMed Recoverer"

    @property
    def description(self) -> str:
        return "Retrieves documents from PubMed using a text query and fetches PDFs via DOI negotiation."

    def recover(self, query: str, k: int) -> Set[Document]:
        """Retrieve documents from PubMed based on a query.

        Args:
            query: Search query string
            k: Number of documents to retrieve

        Returns:
            Set of recovered Document objects
        """
        ids = self._search_pubmed(query, k)
        documents = set()
        futures = []

        metas = []
        for pmid in ids:
            meta = self._fetch_metadata(pmid)
            metas.append((pmid, meta))

        with ThreadPoolExecutor(max_workers=self.PDF_THREADS) as executor:
            for pmid, meta in metas:
                if meta.get("doi"):
                    futures.append(executor.submit(
                        self._fetch_via_doi,
                        meta["doi"],
                        identifier=pmid,
                        title=meta["title"],
                        abstract=meta["abstract"],
                        authors=meta["authors"]
                    ))

            for future in as_completed(futures):
                doc = future.result()
                if doc:
                    documents.add(doc)

        return documents

    def _search_pubmed(self, query: str, k: int) -> list[str]:
        """Search PubMed for document IDs matching the query.

        Args:
            query: Search query string
            k: Number of documents to retrieve

        Returns:
            List of PubMed IDs
        """
        params = {"db": "pubmed", "term": query, "retmax": k, "retmode": "json"}

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(self.ESEARCH_URL, params=params, timeout=self.TIMEOUT)
                if resp.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(resp, self.PUBMED_BACKOFF_MIN, self.PUBMED_BACKOFF_MAX)
                    time.sleep(sleep_time)
                    continue
                resp.raise_for_status()
                return resp.json().get("esearchresult", {}).get("idlist", [])
            except requests.HTTPError as e:
                if e.response.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(e.response, self.PUBMED_BACKOFF_MIN, self.PUBMED_BACKOFF_MAX)
                    time.sleep(sleep_time)
                else:
                    return []
            except requests.RequestException:
                time.sleep(random.uniform(self.PUBMED_BACKOFF_MIN, self.PUBMED_BACKOFF_MAX))
        return []

    def _fetch_metadata(self, pmid: str) -> Dict[str, Any]:
        """Fetch metadata for a PubMed document.

        Args:
            pmid: PubMed ID

        Returns:
            Dictionary containing title, abstract, authors, and DOI
        """
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(self.EFETCH_URL, params=params, timeout=self.TIMEOUT)
                if resp.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(resp, self.PUBMED_BACKOFF_MIN, self.PUBMED_BACKOFF_MAX)
                    time.sleep(sleep_time)
                    continue
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                article = root.find(".//Article")

                if article is None:
                    return {"title": "", "abstract": "", "authors": [], "doi": None}

                title = article.find("ArticleTitle").text or ""
                abstract = " ".join(e.text or "" for e in article.findall(".//AbstractText")).strip()
                authors = [
                    f"\n{a.findtext('ForeName', '')} {a.findtext('LastName', '')}".strip()
                    for a in article.findall(".//Author")
                ]

                doi = None
                for a in root.findall(".//ArticleId"):
                    if a.attrib.get("IdType") == "doi":
                        doi = a.text.strip()
                        break
                    else:
                        print(f"{title} {a.attrib.get("IdType")} {a.attrib.get("Url")}\n")

                print(f"Doi: {doi} , Title: {title}")

                return {"title": title, "abstract": abstract, "authors": authors, "doi": doi}
            except requests.HTTPError as e:
                if e.response.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(e.response, self.PUBMED_BACKOFF_MIN, self.PUBMED_BACKOFF_MAX)
                    time.sleep(sleep_time)
                else:
                    return {"title": "", "abstract": "", "authors": [], "doi": None}
            except requests.RequestException:
                time.sleep(random.uniform(self.PUBMED_BACKOFF_MIN, self.PUBMED_BACKOFF_MAX))
        return {"title": "", "abstract": "", "authors": [], "doi": None}

    def _fetch_via_doi(
            self, doi: str, *, identifier: str, title: str, abstract: str, authors: list[str]
    ) -> Document | None:
        """Retrieve PDF via DOI and create a Document object.

        Args:
            doi: Document DOI
            identifier: PubMed ID
            title: Document title
            abstract: Document abstract
            authors: List of authors

        Returns:
            Document object or None
        """
        try:
            head = requests.head(
                f"https://doi.org/{doi}",
                headers={"Accept": "application/pdf"},
                allow_redirects=True,
                timeout=self.TIMEOUT
            )
            head.raise_for_status()
            pdf_resp = requests.get(head.url, timeout=self.TIMEOUT)
            print(f"{head.url}\n\n")
            pdf_resp.raise_for_status()



            reader = PdfReader(BytesIO(pdf_resp.content))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
            content = DocumentContentCleaner.clean_document(raw)

            if not content:
                return None

            return Document(
                id=identifier,
                title=title,
                abstract=abstract,
                authors=authors,
                content=content
            )
        except Exception:
            return None

    def _get_sleep_time(self, response, min_backoff, max_backoff) -> float:
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after) + random.uniform(0.1, 0.5)
            except ValueError:
                pass
        return random.uniform(min_backoff, max_backoff)