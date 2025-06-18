from typing import Set
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from PyPDF2 import PdfReader
from controller.interfaces.doc_recoverer import DocRecoverer
from entities.document import Document
from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner

CROSSREF_API = "https://api.crossref.org/works/"


class DOIRecoverer(DocRecoverer):
    """Retrieve documents by DOI with Crossref metadata and arXiv fallback."""

    @property
    def name(self) -> str:
        return "DOI Recoverer"

    @property
    def description(self) -> str:
        return "Fetches papers by DOI via Crossref and DOI negotiation with arXiv fallback."

    def recover(self, query: str) -> Set[Document]:
        """Retrieve a document by its DOI.

        Args:
            query: DOI identifier string

        Returns:
            Set containing single Document or empty set
        """
        try:
            cr = requests.get(CROSSREF_API + query, timeout=10)
            cr.raise_for_status()
            msg = cr.json().get("message", {})

            title = (msg.get("title") or [""])[0]
            abstract = self._clean_abstract(msg.get("abstract", ""))
            authors = self._extract_authors(msg.get("author", []))

            head = requests.head(
                f"https://doi.org/{query}",
                headers={"Accept": "application/pdf"},
                allow_redirects=True,
                timeout=10,
            )
            head.raise_for_status()

            pdf_resp = requests.get(head.url, timeout=15)
            pdf_resp.raise_for_status()

            reader = PdfReader(BytesIO(pdf_resp.content))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
            content = DocumentContentCleaner.clean_document(raw)

            if not content:
                raise ValueError("No text after cleaning PDF.")

            return {
                Document(
                    id=query,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    content=content,
                )
            }
        except Exception:
            return self._recover_from_arxiv(title if "title" in locals() else query)

    @staticmethod
    def _clean_abstract(abstract: str) -> str:
        """Remove HTML tags from abstract if present."""
        if abstract.startswith("<"):
            import re
            return re.sub(r"<[^>]+>", "", abstract)
        return abstract

    @staticmethod
    def _extract_authors(authors_list: list) -> list[str]:
        """Format author names from Crossref metadata."""
        return [
            f"{a.get('given', '').strip()} {a.get('family', '').strip()}".strip()
            for a in authors_list
            if a.get("given") or a.get("family")
        ]

    @staticmethod
    def _recover_from_arxiv(query: str) -> Set[Document]:
        """Fallback to arXiv when DOI resolution fails.

        Args:
            query: Title or arXiv ID to search for

        Returns:
            Set containing single Document or empty set
        """
        search_q = f"id:{query.split('/')[-1]}" if (":" in query or "arxiv.org" in query) else f"all:{query}"

        try:
            resp = requests.get(
                f"http://export.arxiv.org/api/query?search_query={search_q}&start=0&max_results=1",
                timeout=10
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
             if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf"),
            f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        )

        try:
            pdf_resp = requests.get(pdf_url, timeout=15)
            pdf_resp.raise_for_status()
            reader = PdfReader(BytesIO(pdf_resp.content))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
            content = DocumentContentCleaner.clean_document(raw)

            if not content:
                return set()

            return {
                Document(
                    id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    content=content,
                )
            }
        except Exception:
            return set()