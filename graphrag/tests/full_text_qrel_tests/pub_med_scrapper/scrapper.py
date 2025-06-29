import os
from pathlib import Path
from typing import List, Optional, Tuple
import re

from PyPDF2 import PdfReader


from doc_recoverers.doc_utils.doc_cleaner import DocumentContentCleaner
from .models.author_spec import AuthorSpec
from ..nfcorpus_test_generator.interfaces import (
    PubMedRecoverer,
    RecovererResponse,
)
from .metadata_getter import fetch_article_metadata
from .spoof import download_pmid_documents
from .models.metadata_result import MetadataResult


URL_RE = re.compile(r"^http://www\.ncbi\.nlm\.nih\.gov/pubmed/([0-9]+)(?:/.*)?$")

# for doc in dataset.docs_iter():
#     match = regex.match(doc.url)
#     if match:
#         answ += [match[1]]

AuthorsAndDoi = Tuple[List[AuthorSpec], str]


class PubMedScrapper(PubMedRecoverer):
    def __init__(self, data_location_root: Path) -> None:
        self.data_location_root = data_location_root
        if not os.path.isdir(self.data_location_root):
            os.makedirs(data_location_root)

    def get_documents(self, urls: List[str]) -> List[Optional[RecovererResponse]]:
        pmids = self._extract_pmids(urls)
        metadatas = self._get_metadatas(pmids)
        authors_and_dois = self._get_authors_and_dois(metadatas)
        dois = self._get_dois_by_index(pmids, authors_and_dois)
        pdf_paths = self._get_pdfs(pmids, dois)

        return self._merge_results(authors_and_dois, pdf_paths)

    def _merge_results(self, authors_and_dois: List[Optional[AuthorsAndDoi]], pdf_paths: List[Optional[str]]) -> List[Optional[RecovererResponse]]:
        answ = []
        for author_and_doi, pdf_path in zip(authors_and_dois, pdf_paths):
            if author_and_doi and pdf_path:
                authors, _ = author_and_doi
                author_names = [author.name for author in authors]
                content = self._read_content(pdf_path)
                if content == None:
                    answ.append(None)
                    continue

                answ.append(RecovererResponse(
                    authors=author_names,
                    content=content
                ))

            else:
                answ.append(None)

        return answ

    def _read_content(self, pdf_path: str) -> Optional[str]:
        reader = PdfReader(pdf_path)
        try:
            raw = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            content = DocumentContentCleaner.clean_document(raw)
        except:
            return None

        return content

    def _get_dois_by_index(
        self,
        pmids: List[Optional[str]],
        authors_and_dois: List[Optional[AuthorsAndDoi]],
    ) -> List[Optional[str]]:
        answ = []
        for pmid, author_and_doi in zip(pmids, authors_and_dois):
            if pmid and author_and_doi:
                _, doi = author_and_doi
                answ.append(doi)
            else:
                answ.append(None)

        return answ

    def _get_pdfs(
        self, pmids: List[Optional[str]], dois: List[Optional[str]]
    ) -> List[Optional[str]]:
        pmid_to_doi = {}
        for pmid, doi in zip(pmids, dois):
            if pmid and doi:
                pmid_to_doi[pmid] = doi

        pdf_paths = download_pmid_documents(pmid_to_doi, self.data_location_root)

        answ = []
        for pmid in pmids:
            if pmid and pmid in pdf_paths and pdf_paths[pmid] != None:
                answ.append(pdf_paths[pmid])
            else:
                answ.append(None)

        return answ

    def _extract_pmids(self, urls: List[str]) -> List[Optional[str]]:
        answ = []
        for url in urls:
            match = URL_RE.match(url)
            if match:
                answ.append(match[1])
            else:
                answ.append(None)

        return answ

    def _get_authors_and_dois(
        self, metadatas: List[Optional[MetadataResult]]
    ) -> List[Optional[AuthorsAndDoi]]:
        answ = []
        for m in metadatas:
            if m == None:
                answ.append(None)
                continue

            doi_found = False
            for idspec in m.article_ids:
                if idspec.idtype == "doi":
                    answ.append((m.authors, idspec.value))
                    doi_found = True
                    break

            if not doi_found:
                answ.append(None)

        return answ

    def _get_metadatas(
        self, pmids: List[Optional[str]]
    ) -> List[Optional[MetadataResult]]:
        valid_ids = [id for id in pmids if id]

        metadatas = fetch_article_metadata(valid_ids, self.data_location_root)
        metadatas = {k: v for k, v in metadatas.items() if v != None}

        answ = []
        for id in pmids:
            if id == None or id not in metadatas:
                answ.append(None)
                continue

            answ.append(metadatas[id])

        return answ
