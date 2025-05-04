from abc import ABC, abstractmethod
from typing import Set

from entities.document import Document

class DocsetBuilder:

    @abstractmethod
    def build_doc_set(self, query: str) -> Set[Document]:
        """
        Builds a set of documents that are relevant for a State of the Art
        of a paper described by the query

        :param query: Query describing the paper
        :returns: A set of relevant documents
        """
