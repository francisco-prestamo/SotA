
import os

from tests.full_text_qrel_tests.persistence import SQLiteDocumentRepository, SQLiteTestCaseRepo

data_path = ".test_case_data"
doc_db_path = os.path.join(data_path, "doc-db")
tc_db_path = os.path.join(data_path, "tc-db")

if not os.path.isdir(doc_db_path) or not os.path.isdir(tc_db_path):
    print(
        "ERROR: test case databases not found, generate them with the test case generator or copy them to\n"
        f"{data_path}/doc-db\n"
        f"{data_path}/tc-db\n"
    )
    exit()


doc_repo = SQLiteDocumentRepository(doc_db_path)
tc_repo = SQLiteTestCaseRepo(tc_db_path, doc_repo)

print("\n".join([
    f"ID: {tc.id} | Query: {tc.query} | Total Docs: {len(tc.documents)}"
    for tc in tc_repo.get_test_cases()
]))

