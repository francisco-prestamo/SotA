from expert_set.interfaces import KnowledgeRecoverer as ExpSetKnowledgeRec
from receptionist_agent.interfaces import KnowledgeRecoverer as ReceptKnowledgeRec
from entities import Document

from typing import List

class RecovererAgentMock(ReceptKnowledgeRec, ExpSetKnowledgeRec):
    def __init__(self):
        self.id = 0

    def recover_docs(self, query: str, k: int) -> List[Document]:
        answ = []
        for _ in range(k):
            id = self._generate_id()
            d = Document(
                id=id,
                title="Mock title",
                abstract="Mock abstract",
                authors=["Mock Author 1", "Mock Author 2"],
                content="Mock content"
            )
            answ.append(d)


        return answ

    def get_survey_docs(self, query: str, k: int = 3) -> List[Document]:
        return self.recover_docs(query, k)

    def _generate_id(self) -> str:
        self.id += 1
        return str(self.id)

