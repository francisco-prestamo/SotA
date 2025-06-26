import os
from typing import Callable, List

from deepdiff.diff import DeepDiff
from pydantic import BaseModel

from entities import Document
from ... import TestCase
from .. import SQLiteTestCaseRepo
from ..sqlite_document_repo import SQLiteDocumentRepository

def get_mock_document(id: str) -> Document:
    return Document(
        id=id,
        title="title " + id,
        abstract="abst " + id,
        authors=["Alice", id],
        content="content of doc with id " + id
    )

def generate_test_case(id: str, query: str, docs: List[Document], relevances: List[int]):
    assert len(docs) == len(relevances)
    return TestCase(
        id=id,
        query=query,
        documents=docs,
        relevance={doc.id: relevances[i] for i, doc in enumerate(docs)}
    )

def create_doc_repo() -> SQLiteDocumentRepository:
    doc_repo = SQLiteDocumentRepository("doc_db")
    return doc_repo


def delete_doc_repo():
    os.remove("doc_db")


def create_tc_repo() -> SQLiteTestCaseRepo:
    doc_repo = create_doc_repo()
    repo = SQLiteTestCaseRepo("db", doc_repo)
    return repo


def delete_tc_repo():
    delete_doc_repo()
    os.remove("db")

def run_doc_repo_test(test_func: Callable[[SQLiteDocumentRepository], None]):
    repo = create_doc_repo()
    try:
        test_func(repo)
    except Exception as e:
        delete_doc_repo()
        raise

    delete_doc_repo()




def run_test_case_repo_test(test_func: Callable[[SQLiteTestCaseRepo], None]):
    repo = create_tc_repo()
    try:
        test_func(repo)
    except Exception as e:
        delete_tc_repo()
        raise

    delete_tc_repo()

def deep_equal_models(m1: BaseModel, m2: BaseModel, ignore_order=False) -> bool:
    if bool(DeepDiff(m1.model_dump(), m2.model_dump(), ignore_order=ignore_order)):
        print(m1.model_dump())
        print(m2.model_dump())
        return False

    return True
