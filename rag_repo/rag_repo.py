from typing import Callable, Generic, List, TypeVar
from expert_set.interfaces import KnowledgeRepository, KnowledgeRepositoryFactory
from rag_repo.interfaces.vectorial_db import VectorialDBFactory
from .interfaces import RagRepoTextEmbedder, VectorialDB


T = TypeVar("T")

class RagRepo(Generic[T], KnowledgeRepository[T]):
    def __init__(self, text_embedder: RagRepoTextEmbedder, vector_repo: VectorialDB):
        self.text_embedder = text_embedder
        self.vector_db = vector_repo
        self.docs = {}

    def store_document(self, document: T, get_content: Callable[[T], str]) -> None:
        content = get_content(document)
        embedding = self.text_embedder.embed(content)
        id = self.vector_db.store(embedding.vector)
        self.docs[id] = document

    def query_knowledge(self, query: str, k: int) -> List[T]:
        vector = self.text_embedder.embed(query).vector
        ids = self.vector_db.get_closest(vector, k)

        return [self.docs[id] for id in ids]


class RagRepoFactory(KnowledgeRepositoryFactory):
    def __init__(self, text_embedder: RagRepoTextEmbedder, vector_repo_factory: VectorialDBFactory):
        self.embedder = text_embedder
        self.vec_repo_factory = vector_repo_factory


    def create_knowledge_repository(self) -> KnowledgeRepository:
        return RagRepo(self.embedder, self.vec_repo_factory.create_vectorial_db())
