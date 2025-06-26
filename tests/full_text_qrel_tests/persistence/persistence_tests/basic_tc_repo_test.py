from .. import SQLiteTestCaseRepo

from .helpers import get_mock_document, generate_test_case, deep_equal_models

def create_and_get_test_case(repo: SQLiteTestCaseRepo):

    doc1 = get_mock_document("1")
    doc2 = get_mock_document("2")
    doc3 = get_mock_document("3")

    tc1 = generate_test_case("1", "why", [doc1, doc2, doc3], [0, 1, 2])

    repo.store_test_case(tc1)

    assert repo.document_repository.get_documents().__len__() == 3
    tcs = repo.get_test_cases()
    assert len(tcs) == 1
    assert deep_equal_models(tcs[0], tc1, ignore_order=True)




