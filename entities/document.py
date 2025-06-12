from abc import ABC, abstractmethod
from typing import Optional
from entities.embedding import Embedding
from pydantic import BaseModel, Field

class Document(BaseModel):
    id: str
    title: str
    abstract: str
    content: str
