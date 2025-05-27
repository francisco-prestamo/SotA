from controller.interfaces.doc_ranker_response import DocRankerResponse
from entities.document import Document
from abc import ABC, abstractmethod
from typing import List

class DocRanker(ABC):
    @abstractmethod
    def rank_docs(self, docs: List[Document]) -> DocRankerResponse:
        """
        :param docs: List of Documents to rank
        :returns: DocRankerResponse: which contains a list of ordered scored documents
        """
        pass

class GraphRagRanker(DocRanker):
    @abstractmethod
    def rank_docs(self, docs: List[Document]) -> DocRankerResponse:
        """
        :param docs: List of Documents to rank
        :returns: DocRankerResponse: which contains a list of ordered scored documents
        """
        pass