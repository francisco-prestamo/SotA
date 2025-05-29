from board.interfaces.json_generator import JsonGenerator
from entities.sota_table import SotaTable
from graphrag.knowledge_graph import KnowledgeGraph
from graphrag.graphrag import GraphRAGBuilder
from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum
import asyncio


class ThesisKnowledgeModel(BaseModel):
    thoughts: list[str] = []
    description: str

class ItemStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskType(str, Enum):
    RESEARCH_PAPERS = "research_papers"
    EXPERTS = "experts"
    COMMUNICATION = "communication"

class Task(BaseModel):
    title: str
    description: str
    type: TaskType
    priority: Priority
    is_async: bool = False
    response: Optional[BaseModel] = None


class Board:
    def __init__(
        self,
        json_generator: JsonGenerator,
    ):

        self.json_generator: JsonGenerator = json_generator
        self.graphrag_builder: GraphRAGBuilder = GraphRAGBuilder(json_generator)
        self.knowledge_graph: KnowledgeGraph = self.graphrag_builder.build_knowledge_graph([])
        self.sota_table: SotaTable = None
        self.thesis_knowledge: ThesisKnowledgeModel = ThesisKnowledgeModel(thoughts=[], description="")
        self.k: List[str] = []
        self.kanban: Dict[str, list[Task]] = {
            ItemStatus.TODO: [],
            ItemStatus.IN_PROGRESS: [],
            ItemStatus.DONE: []
        }

    def add_(self, item: Task):
        self.kanban[ItemStatus.TODO].append(item)
        

    async def add_item_async(self, item: Task):
        if(item.is_async):
            item.status = ItemStatus.IN_PROGRESS
            self.kanban[ItemStatus.TODO].append(item)

            while item.status != ItemStatus.DONE:
                await asyncio.sleep(0.5)

            return item.response
        else:
            raise ValueError("Item is not async")


