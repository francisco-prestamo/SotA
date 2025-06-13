import time
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Set, Dict, Any
import requests
from PyPDF2 import PdfReader
from controller.interfaces.doc_recoverer import DocRecoverer
from entities.document import Document
from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner


class ArXivRecoverer(DocRecoverer):
    """Retrieve documents from arXiv based on text queries with PDF processing."""
    BASE_SEARCH_URL = "http://export.arxiv.org/api/query"
    MAX_RESULTS = 10
    MAX_RETRIES = 3
    TIMEOUT = 15

    @property
    def name(self) -> str:
        return "arXiv Recoverer"

    @property
    def description(self) -> str:
        return "Retrieves documents from arXiv using text queries with inline PDF download and cleaning."

    def recover(self, query: str) -> Set[Document]:
        """Retrieve documents from arXiv based on a query.

        Args:
            query: Search query string

        Returns:
            Set of recovered Document objects
        """
        entries = self._search_arxiv(query)
        documents = set()

        for entry in entries:
            try:
                resp = requests.get(entry["pdf_url"], timeout=self.TIMEOUT)
                resp.raise_for_status()
                reader = PdfReader(BytesIO(resp.content))
                raw = "\n".join(page.extract_text() or "" for page in reader.pages)
                content = DocumentContentCleaner.clean_document(raw)

                if content:
                    documents.add(Document(
                        id=entry["id"],
                        title=entry["title"],
                        abstract=entry["summary"],
                        authors=entry["authors"],
                        content=content
                    ))
            except Exception:
                continue

        return documents

    def _search_arxiv(self, query: str) -> list[Dict[str, Any]]:
        """Search arXiv for documents matching the query.

        Args:
            query: Search query string

        Returns:
            List of document metadata dictionaries
        """
        params = {
            "search_query": query,
            "start": 0,
            "max_results": self.MAX_RESULTS,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(self.BASE_SEARCH_URL, params=params, timeout=self.TIMEOUT)
                if resp.status_code == 200:
                    return self._parse_response(resp.content)
                if resp.status_code in {429, 500, 503}:
                    time.sleep(2 ** attempt)
                else:
                    resp.raise_for_status()
            except requests.RequestException:
                if attempt == self.MAX_RETRIES - 1:
                    return []
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