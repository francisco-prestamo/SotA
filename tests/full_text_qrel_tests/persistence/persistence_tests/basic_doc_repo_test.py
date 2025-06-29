from entities.document import Document
from ..sqlite_document_repo import SQLiteDocumentRepository
from .helpers import deep_equal_models


def create_and_get_doc(repo: SQLiteDocumentRepository):
    doc = Document(id="1", title="2", abstract="3", authors=["4"], content="5")

    repo.store_document(doc)

    docs = repo.get_documents()

    assert len(docs) == 1
    assert deep_equal_models(docs[0], doc)
