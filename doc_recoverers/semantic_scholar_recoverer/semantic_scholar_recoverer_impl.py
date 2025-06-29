import random
import time
from io import BytesIO
from typing import Set, Optional, Tuple
import requests
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from controller.interfaces.doc_recoverer import DocRecoverer
from entities.document import Document
from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner

class SemanticScholarRecoverer(DocRecoverer):
    SEMANTIC_BACKOFF_MIN = 1.0
    SEMANTIC_BACKOFF_MAX = 3.0
    TIMEOUT = 15
    MAX_RETRIES = 100
    PDF_THREADS = 20
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    @property
    def name(self) -> str:
        return "Semantic Scholar Recoverer"

    @property
    def description(self) -> str:
        return "Retrieves documents from Semantic Scholar using a text query."

    def recover(self, query: str, k: int, date_filter: Optional[Tuple[str, str]] = None) -> Set[Document]:
        params = {
            "query": query,
            "fields": "title,abstract,authors,openAccessPdf",
            "limit": min(100, max(10, k * 2))
        }
        if date_filter:
            start_date, end_date = date_filter
            start_year = start_date.split('-')[0]
            end_year = end_date.split('-')[0]
            params["year"] = f"{start_year}-{end_year}"

        resp = None

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=self.TIMEOUT)
                if resp.status_code in {429, 503, 301}:
                    sleep_time = self._get_sleep_time(resp)
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
        papers_with_pdf = [p for p in papers if p.get("openAccessPdf") and p["openAccessPdf"].get("url")]
        selected_papers = papers_with_pdf[:min(len(papers_with_pdf), k)]

        documents = set()
        if not selected_papers:
            return documents

        with ThreadPoolExecutor(max_workers=self.PDF_THREADS) as executor:
            futures = []
            for paper in selected_papers:
                futures.append(executor.submit(self._process_paper, paper))

            for future in as_completed(futures):
                doc = future.result()
                if doc:
                    documents.add(doc)

        return documents

    def _process_paper(self, paper: dict) -> Document | None:
        paper_id = paper.get("paperId")
        title = paper.get("title", "").strip()
        abstract = paper.get("abstract", "") or ""
        authors = [a.get("name", "") for a in paper.get("authors", [])]

        pdf_url = paper["openAccessPdf"]["url"]
        content = self._fetch_pdf_content(pdf_url)

        if not title or not content:
            return None

        return Document(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            content=content
        )

    def _fetch_pdf_content(self, url: str) -> str:
        try:
            r = requests.get(url, timeout=self.TIMEOUT)
            r.raise_for_status()
            reader = PdfReader(BytesIO(r.content))
            raw = "\n".join(page.extract_text() or "" for page in reader.pages)
            return DocumentContentCleaner.clean_document(raw) or ""
        except Exception:
            return ""

    def _get_sleep_time(self, response) -> float:
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after) + random.uniform(0.1, 0.5)
            except ValueError:
                pass
        return random.uniform(self.SEMANTIC_BACKOFF_MIN, self.SEMANTIC_BACKOFF_MAX)