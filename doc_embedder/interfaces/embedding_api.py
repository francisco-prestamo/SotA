from abc import ABC, abstractmethod
from typing import List
from entities.document import Document
from entities.embedding import Embedding

class EmbeddingAPI(ABC):

    @abstractmethod
    def generate_embeddings(self, docs: List[Document]) -> List[Embedding]:
        """
        Send a query to the LLM and get back a set of embedding vectors.
        :param docs: List of the contents of documents to embed
        :returns: A list e where e[i] is the embedding of docs[i]
        """

