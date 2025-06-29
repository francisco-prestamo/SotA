from typing import Dict, List
from pydantic import BaseModel
from entities import Document


class TestCase(BaseModel):
    id: str
    query: str
    documents: List[Document]
    relevance: Dict[str, int]
