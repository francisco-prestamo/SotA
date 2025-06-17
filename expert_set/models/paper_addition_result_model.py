from pydantic import BaseModel
from typing import List

class PaperAdditionResult(BaseModel):
    """Result model for paper addition process"""
    papers_added: List[str]
    summary: str
