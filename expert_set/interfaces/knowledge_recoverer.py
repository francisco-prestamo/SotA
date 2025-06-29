from abc import ABC, abstractmethod
from typing import List

from entities import Document


class KnowledgeRecoverer(ABC):
    @abstractmethod
    def recover_docs(self, query: str, k: int) -> List[Document]:
        pass


    @abstractmethod
    def get_survey_docs(self, query: str, k: int = 3) -> List[Document]:
        pass
