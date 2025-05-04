from typing import List, Tuple
from entities.document import Document

class ScoredDocument:
    """
    A Document and its score, as calculated by a ranker
    """
    def __init__(self, doc: Document, score: float):
        self.doc = doc
        self.score = score

class DocRankerResponse:
    """
    The expected response from the document ranker
    """
    def __init__(self, ranked_documents: List[ScoredDocument]) -> None:
        """
        :param ranked_documents: A list of scored documents, sorted from greater to smaller score
        """
        self.ranked_documents = ranked_documents
