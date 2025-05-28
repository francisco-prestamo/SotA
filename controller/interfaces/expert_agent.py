from abc import ABC, abstractmethod
from entities.document import Document
from typing import Any

class ExpertAgent(ABC):
    """
    Interface for an expert agent that processes a document and extracts specific information/features.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def extract_features(self, document: Document) -> Any:
        """
        Extracts features or information from a document relevant to the expert's specialty.
        Args:
            document (Document): The document to process.
        Returns:
            Any: The extracted features or information (could be a dict, list, DataFrame, etc.).
        """
        pass
