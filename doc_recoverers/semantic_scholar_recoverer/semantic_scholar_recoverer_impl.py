import time
from typing import Set
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from PyPDF2 import PdfReader
from controller.interfaces.doc_recoverer import DocRecoverer
from entities.document import Document
from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner


class SemanticScholarRecoverer(DocRecoverer):
    """Recover documents from Semantic Scholar with arXiv PDF fallback."""
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    MAX_RETRIES = 5

    @property
    def name(self) -> str:
        return "Semantic Scholar Recoverer"

    @property
    def description(self) -> str:
        return "Retrieves documents from Semantic Scholar using a text query, with arXiv fallback."

    def recover(self, query: str) -> Set[Document]:
        """Retrieve documents from Semantic Scholar based on a query.

        Args:
            query: Search query string

        Returns:
            Set of recovered Document objects
        """
        params = {"query": query, "fields": "title,abstract,authors,openAccessPdf", "limit": 10}
        resp = None

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=10)
                if resp.status_code == 200:
                    break
                if resp.status_code in {403, 429}:
                    time.sleep(2 ** attempt)
                else:
                    resp.raise_for_status()
            except requests.RequestException:
                if attempt == self.MAX_RETRIES - 1:
                    raise

        if not resp or resp.status_code != 200:
            raise Exception(f"Failed Semantic Scholar query after {self.MAX_RETRIES} attempts")

        papers = resp.json().get("data", [])
        documents = set()

        for paper in papers:
            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "") or ""
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            pdf_info = paper.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url")
            doc = None

            if pdf_url:
                doc = self._fetch_and_clean(pdf_url, paper_id=paper.get("paperId"))

            if not doc and title:
                arxiv_docs = self._recover_from_arxiv(title)
                if arxiv_docs:
                    doc = next(iter(arxiv_docs))

            if doc:
                documents.add(
                    Document(
                        id=doc.id,
                        title=title or doc.title,
                        abstract=abstract or doc.abstract,
                        authors=authors or doc.authors,
                        content=doc.content
                    )
                )

        return documents

    @staticmethod
    def _fetch_and_clean(url: str, paper_id: str) -> Document | None:
        """Fetch PDF content from URL and clean it.

        Args:
            url: PDF download URL
            paper_id: Semantic Scholar paper ID

        Returns:
            Document object or None if processing fails
        """
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            reader = PdfReader(BytesIO(r.content))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
            content = DocumentContentCleaner.clean_document(raw)

            if not content:
                return None

            return Document(id=paper_id, title="", abstract="", authors=[], content=content)
        except Exception:
            return None

    @staticmethod
    def _recover_from_arxiv(query: str) -> Set[Document]:
        """Fallback to arXiv when Semantic Scholar PDF is unavailable.

        Args:
            query: Title or arXiv ID to search for

        Returns:
            Set containing single Document or empty set
        """
        if ":" in query or "arxiv.org" in query:
            aid = query.split("/")[-1]
            search_q = f"id:{aid}"
        else:
            search_q = f"all:{query}"

        try:
            resp = requests.get(f"http://export.arxiv.org/api/query?search_query={search_q}&start=0&max_results=1",
                                timeout=10)
            resp.raise_for_status()
        except Exception:
            return set()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return set()

        arxiv_id = entry.find("atom:id", ns).text.rsplit("/", 1)[-1]
        title = entry.find("atom:title", ns).text.strip()
        abstract = entry.find("atom:summary", ns).text.strip()
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        pdf_url = next(
            (link.attrib["href"] for link in entry.findall("atom:link", ns)
             if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf"),
            f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        )

        try:
            r = requests.get(pdf_url, timeout=15)
            r.raise_for_status()
            reader = PdfReader(BytesIO(r.content))
            raw = "\n".join(p.extract_text() or "" for p in reader.pages)
            content = DocumentContentCleaner.clean_document(raw)

            if not content:
                return set()

            return {
                Document(
                    id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    content=content
                )
            }
        except Exception:
            return set()