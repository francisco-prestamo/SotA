from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel


class RecovererResponse(BaseModel):
    authors: List[str]
    content: str


class PubMedRecoverer(ABC):
    @abstractmethod
    def get_documents(self, urls: List[str]) -> List[Optional[RecovererResponse]]:
        pass
