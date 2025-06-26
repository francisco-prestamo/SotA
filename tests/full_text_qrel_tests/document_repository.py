from abc import ABC, abstractmethod
from typing import List, Optional

from entities import Document


class DocumentRepository(ABC):
    @abstractmethod
    def store_document(self, document: Document):
        pass

    @abstractmethod
    def get_document(self, id: str) -> Optional[Document]:
        pass

    @abstractmethod
    def get_documents(self) -> List[Document]:
        pass

    @abstractmethod
    def document_exists(self, id: str) -> bool:
        pass

