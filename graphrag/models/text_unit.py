from typing import Optional
from entities.embedding import Embedding
from abc import ABC


class TextUnit:
    """
    Interface for representing a text unit.
    """
    def __init__(self, document_id: str, text: str, unit_id: Optional[str], position: Optional[int], number_tokens: int, embedding: Embedding):
        self._document_id = document_id
        self._text = text
        self._unit_id = unit_id
        self._position = position
        self._number_tokens = number_tokens
        self._embedding = embedding

    @property
    def document_id(self) -> str:
        return self._document_id

    @property
    def text(self) -> str:
        return self._text
    
    @text.setter
    def text(self, value: str):
        self._text = value

    @property
    def unit_id(self) -> Optional[str]:
        return self._unit_id

    @property
    def position(self) -> Optional[int]:
        return self._position


    @property
    def number_tokens(self) -> int:
        return self._number_tokens
    
    @number_tokens.setter
    def number_tokens(self, value: int):
        self._number_tokens = value

    @property
    def embedding(self) -> Embedding:
        return self._embedding
