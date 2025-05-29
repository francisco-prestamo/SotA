from typing import List, Callable, Optional, Any
from enum import Enum
from board.board import TaskSkill, Task, Board, ItemStatus
from board.interfaces.json_generator import JsonGenerator
import time

class Agent:
    def __init__(self,name:str,description:str, skills: List[TaskSkill],json_generator: JsonGenerator):
        """
        Initializes an Agent with a name, description, and a list of skills.
        :param name: The name of the agent.
        :param description: A brief description of the agent's role.
        :param skills: A list of skills that the agent possesses.
        """
        self.name = name
        self.description = description
        self.skills = skills
        self.json_generator = json_generator
        self.task_in_progress: Optional[Task] = None

    def run(self, board: Board, skill: Any):
        """
        Continuously process tasks: if a task is in progress, work on it; otherwise, get a new task by skill from the board.
        """
        while True:
            if self.task_in_progress:
                task_solved = self.work_in_task(self.task_in_progress)
                board.edit_task(self.task_in_progress.id,self.task_in_progress.status, task_solved)
                board.move_task_between_statuses(self.task_in_progress.id, ItemStatus.IN_PROGRESS, ItemStatus.DONE)
                self.task_in_progress = None
            else:
                task = board.get_task_by_skill(skill)
                if task:
                    board.move_task_between_statuses(task.id, ItemStatus.TODO, ItemStatus.IN_PROGRESS)
                    self.task_in_progress = task
                    print(f"Picked up new task: {task.title}")
                else:
                    print("No tasks available for the given skill. Waiting...")
                    time.sleep(2)

    def work_in_task(self, task: Task) -> Task:
        """
        Returns the current task in progress.
        """


        return self.task_in_progress