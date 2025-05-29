from board.interfaces.json_generator import JsonGenerator
from entities.sota_table import SotaTable
from graphrag.knowledge_graph import KnowledgeGraph
from graphrag.graphrag import GraphRAGBuilder
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from enum import Enum
import asyncio
import threading


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

class TaskSkill(str, Enum):
    RESEARCH_PAPERS = "research_papers"
    EXPERTS = "experts"
    COMMUNICATION = "communication"

class Task(BaseModel):
    id: str
    title: str
    description: str
    skill: TaskSkill
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
        self._kanban_lock = threading.Lock()

    def get_task_by_skill(self, skill: Any) -> Optional[Task]:
        with self._kanban_lock:
            for task in self.kanban[ItemStatus.TODO]:
                if task.skill == skill:
                    return task
        return None

    def add_task(self, task: Task):
        with self._kanban_lock:
            self.kanban[ItemStatus.TODO].append(task)

    async def add_task_async(self, task: Task):
        if task.is_async:
            with self._kanban_lock:
                self.kanban[ItemStatus.TODO].append(task)

            timeout = 600
            elapsed = 0
            interval = 3
            while True:
                with self._kanban_lock:
                    done_ids = [t.id for t in self.kanban[ItemStatus.DONE]]
                if task.id in done_ids:
                    break
                await asyncio.sleep(interval)
                elapsed += interval
                if elapsed >= timeout:
                    raise TimeoutError(f"Task '{task.title}' did not complete within {timeout} seconds.")

            return task.response
        else:
            raise ValueError("Task is not async")

    def remove_task_from_kanban(self, task_id: str, status: ItemStatus):
        """
        Remove a task by id from a specific kanban status list.
        """
        with self._kanban_lock:
            self.kanban[status] = [t for t in self.kanban[status] if t.id != task_id]

    def move_task_between_statuses(self, task_id: str, from_status: ItemStatus, to_status: ItemStatus):
        """
        Move a task by id from one kanban status to another.
        """
        with self._kanban_lock:
            task = next((t for t in self.kanban[from_status] if t.id == task_id), None)
            if task:
                self.kanban[from_status].remove(task)
                self.kanban[to_status].append(task)

    def edit_task(self, task_id: str, status: ItemStatus, new_task: Task):
        """
        Edit a task by id in a specific kanban status list. Updates are passed as keyword arguments.
        """
        with self._kanban_lock:
            self.remove_task_from_kanban(task_id, status)
            self.kanban[status].append(new_task)