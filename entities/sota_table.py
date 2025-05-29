from entities.document import Document
from typing import List, Dict, Any
from pydantic import BaseModel


class PaperFeaturesModel(BaseModel):
    authors: List[str]
    title: str
    year: int
    domain: str
    features: Dict[str, Dict[str, Any]]

class SotaTable:
    def __init__(self):
        """
        Initializes the SotaTable with columns, rows, documents, features, and paper_features.
        :param columns: List of column names.
        :param rows: List of rows, each row is a list of strings.
        :param documents: List of Document objects associated with the table.
        :param features: List of feature names (str).
        :param paper_features: List of PaperFeaturesModel, one for each document.
        """
        self.features: list[str] = []
        self.documents_features : list[tuple[Document, PaperFeaturesModel]] = []

    def __str__(self) -> str:
        """
        Returns a string representation of the SOTA table (columns, rows, and features).
        """
        return ""