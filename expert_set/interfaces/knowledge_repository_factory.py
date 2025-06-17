from abc import ABC, abstractmethod

from .knowledge_repository import KnowledgeRepository


class KnowledgeRepositoryFactory(ABC):
    @abstractmethod
    def create_knowledge_repository(self) -> KnowledgeRepository:
        pass

