from typing import List
from expert_set.models import BuildExpertCommand


class ReceptionistAgentMock:
    def __init__(self) -> None:
        pass

    def interact(self) -> List[BuildExpertCommand]:
        c = []
        for _ in range(3):
            c.append(BuildExpertCommand(
                name="John Doe",
                description="A mock",
                query="What is the meaning of life"
            ))


        return c

