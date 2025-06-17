from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic, List

T = TypeVar("T")


class KnowledgeRepository(ABC, Generic[T]):
    @abstractmethod
    def store_document(self, document: T, get_content: Callable[[T], str]) -> None:
        pass

    @abstractmethod
    def query_knowledge(self, query: str, k: int) -> List[T]:
        pass
