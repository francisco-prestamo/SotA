import random
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Set
import requests
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from controller.interfaces.doc_recoverer import DocRecoverer
from entities.document import Document
from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner


class SemanticScholarRecoverer(DocRecoverer):
    SEMANTIC_BACKOFF_MIN = 1.0
    SEMANTIC_BACKOFF_MAX = 3.0
    ARXIV_BACKOFF_MIN = 3.0
    ARXIV_BACKOFF_MAX = 5.0
    TIMEOUT = 15
    MAX_RETRIES = 100
    PDF_THREADS = 20
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    @property
    def name(self) -> str:
        return "Semantic Scholar Recoverer"

    @property
    def description(self) -> str:
        return "Retrieves documents from Semantic Scholar using a text query, with arXiv fallback."

    def recover(self, query: str, k: int) -> Set[Document]:
        """Retrieve documents from Semantic Scholar based on a query.

        Args:
            query: Search query string
            k: Number of documents to retrieve

        Returns:
            Set of recovered Document objects
        """
        params = {"query": query, "fields": "title,abstract,authors,openAccessPdf", "limit": k}
        resp = None



        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=self.TIMEOUT)
                if resp.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(resp, self.SEMANTIC_BACKOFF_MIN, self.SEMANTIC_BACKOFF_MAX)
                    time.sleep(sleep_time)
                    continue
                if resp.status_code == 200:
                    break
                else:
                    break
            except requests.RequestException:
                time.sleep(random.uniform(self.SEMANTIC_BACKOFF_MIN, self.SEMANTIC_BACKOFF_MAX))



        if not resp or resp.status_code != 200:
            return set()

        papers = resp.json().get("data", [])
        documents = set()

        # Use thread pool to process all papers concurrently
        with ThreadPoolExecutor(max_workers=self.PDF_THREADS) as executor:
            # Submit all papers for processing
            futures = []
            for paper in papers:
                futures.append(executor.submit(self._process_paper, paper))

            # Collect results as they complete
            for future in as_completed(futures):
                doc = future.result()
                if doc:
                    documents.add(doc)

        return documents

    def _process_paper(self, paper: dict) -> Document | None:
        """Process a single paper from Semantic Scholar response.

        Args:
            paper: Paper dictionary from Semantic Scholar API

        Returns:
            Document object or None if processing fails
        """

        paper_id = paper.get("paperId")
        title = paper.get("title", "").strip()
        abstract = paper.get("abstract", "") or ""
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        print(f"start => {paper_id}")
        start_time = time.time()


        # First try to get PDF content if available
        if paper.get("openAccessPdf", {}).get("url"):
            pdf_doc = self._fetch_and_clean(paper["openAccessPdf"]["url"], paper_id)
            if pdf_doc:
                end_time = time.time()
                print(f"end => {paper_id}")
                print(f"Time elapsed: {end_time - start_time:.2f} seconds")
                # Update with metadata from Semantic Scholar
                return Document(
                    id=paper_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    content=pdf_doc.content
                )

        # Fallback to arXiv if no PDF available and title exists
        if title:
            arxiv_docs = self._recover_from_arxiv(title)
            if arxiv_docs:
                doc = next(iter(arxiv_docs))
                end_time = time.time()
                print(f"end => {paper_id}")
                print(f"Time elapsed: {end_time - start_time:.2f} seconds")
                return Document(
                    id=doc.id,
                    title=title or doc.title,
                    abstract=abstract or doc.abstract,
                    authors=authors or doc.authors,
                    content=doc.content
                )

        # If no content available, return document with metadata only (optional)
        # You might want to skip these entirely by returning None
        if title:
            end_time = time.time()
            print(f"end => {paper_id}")
            print(f"Time elapsed: {end_time - start_time:.2f} seconds")
            # Only return if we have at least a title
            return Document(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                content=""  # Empty content
            )
        end_time = time.time()
        print(f"end => {paper_id}")
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        return None

    def _fetch_and_clean(self, url: str, paper_id: str) -> Document | None:
        """Fetch PDF content from URL and clean it.

        Args:
            url: PDF download URL
            paper_id: Semantic Scholar paper ID

        Returns:
            Document object or None if processing fails
        """
        try:
            r = requests.get(url, timeout=self.TIMEOUT)
            r.raise_for_status()
            reader = PdfReader(BytesIO(r.content))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
            content = DocumentContentCleaner.clean_document(raw)

            if not content:
                return None

            return Document(id=paper_id, title="", abstract="", authors=[], content=content)
        except Exception:
            return None

    def _recover_from_arxiv(self, query: str) -> Set[Document]:
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

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(
                    f"http://export.arxiv.org/api/query?search_query={search_q}&start=0&max_results=1",
                    timeout=self.TIMEOUT
                )
                if resp.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(resp, self.ARXIV_BACKOFF_MIN, self.ARXIV_BACKOFF_MAX)
                    time.sleep(sleep_time)
                    continue
                resp.raise_for_status()
                break
            except requests.HTTPError as e:
                if e.response.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(e.response, self.ARXIV_BACKOFF_MIN, self.ARXIV_BACKOFF_MAX)
                    time.sleep(sleep_time)
                else:
                    return set()
            except requests.RequestException:
                time.sleep(random.uniform(self.ARXIV_BACKOFF_MIN, self.ARXIV_BACKOFF_MAX))
        else:
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

    def _get_sleep_time(self, response, min_backoff, max_backoff) -> float:
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after) + random.uniform(0.1, 0.5)
            except ValueError:
                pass
        return random.uniform(min_backoff, max_backoff)