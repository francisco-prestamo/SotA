from pydantic import BaseModel
from typing import List

class DomainExtractionRequest(BaseModel):
    """Request model for domain extraction"""
    title: str
    authors: List[str]
    abstract: str
