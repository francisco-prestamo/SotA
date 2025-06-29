from typing import List
from entities import Document

from .interfaces import KnowledgeRepositoryFactory, KnowledgeRepository, KnowledgeRecoverer
from .utils import chunk_document
from .models import BuildExpertCommand, Expert


class ExpertBuilder:
    def __init__(
        self,
        document_recoverer: KnowledgeRecoverer,
        knowledge_repository_factory: KnowledgeRepositoryFactory,
    ):
        self.document_recoverer = document_recoverer
        self.knowledge_repository_factory = knowledge_repository_factory

    def build_experts(
        self, expert_build_commands: List[BuildExpertCommand]
    ) -> List[Expert]:
        from concurrent.futures import ThreadPoolExecutor

        # Use ThreadPoolExecutor to parallelize expert building
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self._build_expert, expert_build_commands))
        return results

    def _build_expert(self, expert_build_command: BuildExpertCommand) -> Expert:
        query = expert_build_command.query
        docs = self.document_recoverer.get_survey_docs(query)
        knowledge = self._initialize_expert_knowledge(docs)

        return Expert(
            name=expert_build_command.name,
            description=expert_build_command.description,
            knowledge=knowledge,
        )

    def _initialize_expert_knowledge(
        self, survey_docs: List[Document]
    ) -> KnowledgeRepository:
        knowledge = self.knowledge_repository_factory.create_knowledge_repository()
        for doc in survey_docs:
            self._chunk_and_store(doc, knowledge)

        return knowledge

    def _chunk_and_store(self, doc: Document, knowledge: KnowledgeRepository):
        chunks = chunk_document(doc)
        for chunk in chunks:
            knowledge.store_document(chunk, lambda c: c.chunk)
