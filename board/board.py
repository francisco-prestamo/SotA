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
    id: str
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
        self.kanban: Dict[str, list[Task]] = {
            ItemStatus.TODO: [],
            ItemStatus.IN_PROGRESS: [],
            ItemStatus.DONE: []
        }

    def add_task(self, task: Task):
        self.kanban[ItemStatus.TODO].append(task)

    async def add_task_async(self, task: Task):
        if task.is_async:
            self.kanban[ItemStatus.TODO].append(task)

            timeout = 600
            elapsed = 0
            while task.id not in [t.id for t in self.kanban[ItemStatus.DONE]]:
                await asyncio.sleep(3)
                elapsed += 3
                if elapsed >= timeout:
                    raise TimeoutError(f"Task '{task.title}' did not complete within {timeout} seconds.")

            return task.response
        else:
            raise ValueError("Task is not async")


