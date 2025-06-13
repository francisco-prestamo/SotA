import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Set, Dict, Any
import requests
from PyPDF2 import PdfReader
from controller.interfaces.doc_recoverer import DocRecoverer
from entities.document import Document
from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner


class PubMedRecoverer(DocRecoverer):
    """Recover documents from PubMed with DOI PDF extraction and arXiv fallback."""
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ARXIV_API = "http://export.arxiv.org/api/query"
    MAX_RESULTS = 10
    TIMEOUT = 15

    @property
    def name(self) -> str:
        return "PubMed Recoverer"

    @property
    def description(self) -> str:
        return "Retrieves documents from PubMed using a text query, fetches metadata, attempts PDF via DOI negotiation, then falls back to arXiv by title."

    def recover(self, query: str) -> Set[Document]:
        """Retrieve documents from PubMed based on a query.

        Args:
            query: Search query string

        Returns:
            Set of recovered Document objects
        """
        ids = self._search_pubmed(query)
        documents = set()

        for pmid in ids:
            meta = self._fetch_metadata(pmid)
            title = meta.get("title", "")
            abstract = meta.get("abstract", "")
            authors = meta.get("authors", [])
            doi = meta.get("doi")
            doc = None

            if doi:
                doc = self._fetch_via_doi(
                    doi,
                    identifier=pmid,
                    title=title,
                    abstract=abstract,
                    authors=authors
                )

            if not doc and title:
                arxiv_docs = self._recover_from_arxiv(title)
                if arxiv_docs:
                    doc = next(iter(arxiv_docs))

            if doc:
                documents.add(doc)

        return documents

    def _search_pubmed(self, query: str) -> list[str]:
        """Search PubMed for document IDs matching the query.

        Args:
            query: Search query string

        Returns:
            List of PubMed IDs
        """
        params = {"db": "pubmed", "term": query, "retmax": self.MAX_RESULTS, "retmode": "json"}
        resp = requests.get(self.ESEARCH_URL, params=params, timeout=self.TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])

    def _fetch_metadata(self, pmid: str) -> Dict[str, Any]:
        """Fetch metadata for a PubMed document.

        Args:
            pmid: PubMed ID

        Returns:
            Dictionary containing title, abstract, authors, and DOI
        """
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        resp = requests.get(self.EFETCH_URL, params=params, timeout=self.TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        article = root.find(".//Article")

        if article is None:
            return {"title": "", "abstract": "", "authors": [], "doi": None}

        title = article.find("ArticleTitle").text or ""
        abstract = " ".join(e.text or "" for e in article.findall(".//AbstractText")).strip()
        authors = [
            f"{a.findtext('ForeName', '')} {a.findtext('LastName', '')}".strip()
            for a in article.findall(".//Author")
        ]

        # Fixed the DOI extraction
        doi = None
        for a in root.findall(".//ArticleId"):
            if a.attrib.get("IdType") == "doi":
                doi = a.text.strip()
                break

        return {"title": title, "abstract": abstract, "authors": authors, "doi": doi}

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

    def _recover_from_arxiv(self, query: str) -> Set[Document]:
        """Fallback to arXiv when PubMed PDF is unavailable.

        Args:
            query: Title or arXiv ID to search for

        Returns:
            Set containing single Document or empty set
        """
        search_q = f"id:{query.split('/')[-1]}" if (":" in query or "arxiv.org" in query) else f"all:{query}"

        try:
            resp = requests.get(
                f"{self.ARXIV_API}?search_query={search_q}&start=0&max_results=1",
                timeout=self.TIMEOUT
            )
            resp.raise_for_status()
        except Exception:
            return set()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        if (entry := root.find("atom:entry", ns)) is None:
            return set()

        arxiv_id = entry.find("atom:id", ns).text.rsplit("/", 1)[-1]
        title = entry.find("atom:title", ns).text.strip()
        abstract = entry.find("atom:summary", ns).text.strip()
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        pdf_url = next(
            (link.attrib["href"] for link in entry.findall("atom:link", ns)
             if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf"
             ), f"https://arxiv.org/pdf/{arxiv_id}.pdf")

        try:
            r = requests.get(pdf_url, timeout=self.TIMEOUT)
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