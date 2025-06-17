from typing import List
from board.interfaces.json_generator import JsonGenerator
from entities.sota_table import SotaTable
from pydantic import BaseModel, Field
from graphrag.graphrag import GraphRag


class ThesisKnowledgeModel(BaseModel):
    description: str = Field(default="")
    thoughts: List[str] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)

    def record_version(self) -> None:
        """Capture current description in history before modification."""
        self.history.append(self.description)


class Board:
    """Central knowledge repository."""

    def __init__(
            self,
            json_generator: JsonGenerator,
            graph_rag: GraphRag,
            initial_thesis_description: str = ""
    ):
        self.json_generator: JsonGenerator = json_generator
        self.graph_rag = graph_rag
        self.knowledge_graph = graph_rag.build_knowledge_graph([])
        self.sota_table: SotaTable = SotaTable()
        self.thesis_knowledge = ThesisKnowledgeModel(
            description=initial_thesis_description,
            history=[initial_thesis_description] if initial_thesis_description else []
        )

    def update_thesis_description(self, new_description: str) -> None:
        """
        Update thesis description and maintain version history.

        Args:
            new_description: New description text
        """
        self.thesis_knowledge.record_version()
        self.thesis_knowledge.description = new_description