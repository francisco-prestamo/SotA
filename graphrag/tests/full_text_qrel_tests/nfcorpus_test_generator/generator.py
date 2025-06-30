from ast import Tuple
import ir_datasets
from typing import Dict, List, Set

from pydantic import BaseModel
import random

from entities import Document
from .interfaces import PubMedRecoverer, RecovererResponse
from .. import TestCaseRepository, TestCase, DocumentRepository


class DatasetTestCase(BaseModel):
    query_id: str
    query: str
    document_relevances: Dict[str, int]


class DatasetDocument(BaseModel):
    id: str
    url: str
    abstract: str
    title: str


class TestCaseGenerator:
    def __init__(
        self,
        recoverer: PubMedRecoverer,
        test_case_repository: TestCaseRepository,
        document_repository: DocumentRepository,
    ) -> None:
        self.recoverer = recoverer
        self.test_case_repo = test_case_repository
        self.document_repository = document_repository
        self.cached_documents: Dict[str, Document] = {
            doc.id: doc for doc in self.document_repository.get_documents()
        }

    def generate_test_cases(self, n: int) -> List[TestCase]:
        dataset = ir_datasets.load("nfcorpus/test")
        all_docs = self._get_all_dataset_docs(dataset)
        all_docs_set = set([id for id in all_docs])
        tcs = []

        if n == None or n <= 0:
            raise ValueError(f"cannot generate {n} test cases")

        i = 0
        for query_id, test_case in self._get_dataset_test_cases(dataset).items():

            relevances = test_case.document_relevances

            relevants = set([id for id, rel in relevances.items() if rel == 2])
            kinda_relevants = set([id for id, rel in relevances.items() if rel == 1])
            non_relevants = all_docs_set.difference(relevants.union(kinda_relevants))

            if self._filtered_out(test_case, relevants, kinda_relevants):
                continue

            random.seed(query_id)

            relevants_to_get = min(5, len(relevants))
            relevants = self._recover_from(
                self._assign_documents(relevants, all_docs), relevants_to_get
            )

            kinda_relevants_to_get = 2 * relevants_to_get
            kinda_relevants = self._recover_from(
                self._assign_documents(kinda_relevants, all_docs),
                kinda_relevants_to_get,
            )

            non_relevants_to_get = 2 * kinda_relevants_to_get
            non_relevants = self._recover_from(
                self._assign_documents(non_relevants, all_docs), non_relevants_to_get
            )

            tc = self._build_test_case(
                query_id, test_case.query, relevants, kinda_relevants, non_relevants
            )

            self.test_case_repo.store_test_case(tc)

            tcs.append(tc)

            i += 1
            print(f"[INFO | TEST CASE GENERATOR] Generated test case {i}/{n}")
            if i > n:
                break

        return tcs

    def _get_all_dataset_docs(self, dataset) -> Dict[str, DatasetDocument]:
        answ = {}
        for doc in dataset.docs_iter():
            answ[doc.doc_id] = DatasetDocument(
                id=doc.doc_id,
                url=doc.url,
                title=doc.title,
                abstract=doc.abstract,
            )

        return answ

    def _build_test_case(
        self,
        id: str,
        query: str,
        relevants: Dict[str, Document],
        kinda_relevants: Dict[str, Document],
        non_relevants: Dict[str, Document],
    ) -> TestCase:
        documents = (
            list(relevants.values())
            + list(kinda_relevants.values())
            + list(non_relevants.values())
        )
        relevances = {
            doc.id: 2 if doc.id in relevants else 1 if doc.id in kinda_relevants else 0
            for doc in documents
        }

        return TestCase(id=id, query=query, documents=documents, relevance=relevances)

    def _assign_documents(
        self, ids: Set[str], documents: Dict[str, DatasetDocument]
    ) -> Dict[str, DatasetDocument]:
        return {id: documents[id] for id in ids}

    def _recover_from(
        self, docs: Dict[str, DatasetDocument], k: int
    ) -> Dict[str, Document]:
        ids = [id for id, _ in docs.items()]
        recovered: Dict[str, Document] = self._get_from_cache(docs, k)

        while len(recovered) < k and len(ids) > 0:
            to_attempt_recover = self._extract_random_ids_to_attempt_to_recover(
                ids, min(k - len(recovered), len(ids))
            )
            attempt_recovered_docs = self._attempt_recover(
                [docs[id] for id in to_attempt_recover]
            )

            for doc in attempt_recovered_docs:
                recovered[doc.id] = doc

        return recovered

    def _get_from_cache(
        self, docs: Dict[str, DatasetDocument], k: int
    ) -> Dict[str, Document]:
        answ = {}
        for id in docs:
            if id in self.cached_documents:
                answ[id] = self.cached_documents[id]

        if len(answ) > k:
            ids = sorted([id for id in answ])
            ids = random.sample(ids, k)
            answ = {id: answ[id] for id in ids}

        return answ

    # will output a list with <= as many documents as the input, in no particular order
    def _attempt_recover(self, documents: List[DatasetDocument]) -> List[Document]:
        answ = []

        urls = [doc.url for doc in documents]
        recovered_docs = self.recoverer.get_documents(urls)
        for i, doc_to_recover in enumerate(recovered_docs):
            if doc_to_recover == None:
                continue
            dataset_doc = documents[i]
            document = self._build_document(dataset_doc, doc_to_recover)
            self.document_repository.store_document(document)
            answ.append(document)

        return answ

    def _extract_random_ids_to_attempt_to_recover(
        self, ids: List[str], k: int
    ) -> List[str]:
        attempt = random.sample(sorted(ids), k=k)
        for id in attempt:
            ids.remove(id)

        return attempt

    def _build_document(
        self, dataset_document: DatasetDocument, recoverer_response: RecovererResponse
    ) -> Document:
        return Document(
            id=dataset_document.id,
            title=dataset_document.title,
            abstract=dataset_document.abstract,
            content=recoverer_response.content,
            authors=recoverer_response.authors,
        )

    def _filtered_out(
        self, test_case: DatasetTestCase, relevants: Set[str], kinda_relevants: Set[str]
    ):
        # return not (3 <= len(relevants) <= 10 and len(kinda_relevants) > 0)
        not_enough_docs = not (len(relevants) >= 3 and len(kinda_relevants) > 0)
        already_cached = self.test_case_repo.test_case_exists(test_case.query_id)

        return not_enough_docs or already_cached

    def _get_dataset_test_cases(self, dataset) -> Dict[str, DatasetTestCase]:
        test_cases: Dict[str, DatasetTestCase] = {}
        queries_by_id = self._get_queries(dataset)

        for qrel in dataset.qrels_iter():
            query_id: str = qrel.query_id
            doc_id: str = qrel.doc_id
            relevance: int = qrel.relevance

            if query_id not in test_cases:
                test_cases[query_id] = DatasetTestCase(
                    query_id=query_id,
                    query=queries_by_id[query_id],
                    document_relevances={},
                )
            test_cases[query_id].document_relevances[doc_id] = relevance

        return test_cases

    def _get_queries(self, dataset) -> Dict[str, str]:
        queries: Dict[str, str] = {}
        for q in dataset.queries_iter():
            queries[q.query_id] = q.title

        return queries
