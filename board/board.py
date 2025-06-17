from board.interfaces.json_generator import JsonGenerator
from entities.sota_table import SotaTable
from pydantic import BaseModel, Field
from graphrag.graphrag import GraphRag


class ThesisKnowledgeModel(BaseModel):
    thoughts: list[str] = Field(default=[])
    description: str = Field(default="")

class Board:
    def __init__(
        self,
        json_generator: JsonGenerator,
        graph_rag: GraphRag
    ):
        self.json_generator: JsonGenerator = json_generator
        self.graph_rag = graph_rag
        self.knowledge_graph = graph_rag.build_knowledge_graph([])
        self.sota_table: SotaTable = SotaTable.model_construct()
        self.thesis_knowledge: ThesisKnowledgeModel = ThesisKnowledgeModel.model_construct()

