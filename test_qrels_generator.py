from random import random
from time import sleep
from typing import List, Optional
from tests.full_text_qrel_tests.persistence.persistence_tests.basic_tc_repo_test import *
from tests.full_text_qrel_tests.persistence.persistence_tests.basic_doc_repo_test import *
from tests.full_text_qrel_tests.persistence.persistence_tests.helpers import (
    run_test_case_repo_test,
    run_doc_repo_test,
)
from tests.full_text_qrel_tests.persistence import SQLiteTestCaseRepo, SQLiteDocumentRepository

from tests.full_text_qrel_tests.nfcorpus_test_generator import TestCaseGenerator
from tests.full_text_qrel_tests.nfcorpus_test_generator.interfaces import PubMedRecoverer, RecovererResponse

class PubMedMockRec(PubMedRecoverer):
    def get_documents(self, urls: List[str]) -> List[Optional[RecovererResponse]]:
        answ = []
        for url in urls:
            print("Getting url...")
            # sleep(.5)
            if random() < .75:
                print("Miss")
                answ.append(None)
            else:
                answ.append(RecovererResponse(
                    authors=["Mock1", "Mock2"],
                    content="MockContent"
                ))

        return answ



run_doc_repo_test(create_and_get_doc)
run_test_case_repo_test(create_and_get_test_case)

doc_repo = SQLiteDocumentRepository("doc-db")

tc_repo = SQLiteTestCaseRepo("tc-db", doc_repo)


gen = TestCaseGenerator(PubMedMockRec(), tc_repo, doc_repo)
gen.generate_test_cases()

print([(tc.id, tc.query, len(tc.documents)) for tc in tc_repo.get_test_cases()])
