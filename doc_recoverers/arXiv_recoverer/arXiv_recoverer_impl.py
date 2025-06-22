import random
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Set, Dict, Any
import requests
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from controller.interfaces.doc_recoverer import DocRecoverer
from entities.document import Document
from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner


class ArXivRecoverer(DocRecoverer):
    BACKOFF_MIN = 3.0
    BACKOFF_MAX = 5.0
    TIMEOUT = 15
    MAX_RETRIES = 100
    BASE_SEARCH_URL = "http://export.arxiv.org/api/query"
    PDF_THREADS = 20

    @property
    def name(self) -> str:
        return "arXiv Recoverer"

    @property
    def description(self) -> str:
        return "Retrieves documents from arXiv using text queries with inline PDF download and cleaning."

    def recover(self, query: str, k: int) -> Set[Document]:
        """Retrieve documents from arXiv based on a query.

        Args:
            query: Search query string
            k: Number of documents to retrieve

        Returns:
            Set of recovered Document objects
        """
        entries = self._search_arxiv(query, k)
        documents = set()
        futures = []

        with ThreadPoolExecutor(max_workers=self.PDF_THREADS) as executor:
            for entry in entries:
                futures.append(executor.submit(self._process_entry, entry))

            for future in as_completed(futures):
                doc = future.result()
                if doc:
                    documents.add(doc)

        return documents

    def _process_entry(self, entry: Dict[str, Any]) -> Document | None:
        try:
            resp = requests.get(entry["pdf_url"], timeout=self.TIMEOUT)
            resp.raise_for_status()
            reader = PdfReader(BytesIO(resp.content))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
            content = DocumentContentCleaner.clean_document(raw)

            if not content:
                return None

            return Document(
                id=entry["id"],
                title=entry["title"],
                abstract=entry["summary"],
                authors=entry["authors"],
                content=content
            )
        except Exception:
            return None

    def _search_arxiv(self, query: str, k: int) -> list[Dict[str, Any]]:
        """Search arXiv for documents matching the query.

        Args:
            query: Search query string
            k: Number of documents to retrieve

        Returns:
            List of document metadata dictionaries
        """
        params = {
            "search_query": query,
            "start": 0,
            "max_results": k,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(self.BASE_SEARCH_URL, params=params, timeout=self.TIMEOUT)
                if resp.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(resp)
                    time.sleep(sleep_time)
                    continue
                resp.raise_for_status()
                return self._parse_response(resp.content)
            except requests.HTTPError as e:
                if e.response.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(e.response)
                    time.sleep(sleep_time)
                else:
                    return []
            except requests.RequestException:
                time.sleep(random.uniform(self.BACKOFF_MIN, self.BACKOFF_MAX))
        return []

    def _parse_response(self, xml_content: bytes) -> list[Dict[str, Any]]:
        """Parse arXiv API response XML into document metadata.

        Args:
            xml_content: Raw XML response from arXiv API

        Returns:
            List of document metadata dictionaries
        """
        root = ET.fromstring(xml_content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = []

        for ent in root.findall("atom:entry", ns):
            raw_id = ent.find("atom:id", ns).text
            paper_id = self._extract_arxiv_id(raw_id)
            title = ent.find("atom:title", ns).text.strip()
            summary = ent.find("atom:summary", ns).text.strip()
            authors = [author.find("atom:name", ns).text for author in ent.findall("atom:author", ns)]

            entries.append({
                "id": paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf"
            })

        return entries

    def _get_sleep_time(self, response) -> float:
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after) + random.uniform(0.1, 0.5)
            except ValueError:
                pass
        return random.uniform(self.BACKOFF_MIN, self.BACKOFF_MAX)

    @staticmethod
    def _extract_arxiv_id(url: str) -> str:
        """Extract arXiv ID from URL.

        Args:
            url: arXiv document URL

        Returns:
            Clean arXiv ID string
        """
        match = re.search(r"abs/([^v]+)", url)
        return match.group(1) if match else url.rstrip("/").split("/")[-1]