import os
from pathlib import Path

from tests.full_text_qrel_tests.persistence.persistence_tests.basic_tc_repo_test import *
from tests.full_text_qrel_tests.persistence.persistence_tests.basic_doc_repo_test import *
from tests.full_text_qrel_tests.persistence import (
    SQLiteTestCaseRepo,
    SQLiteDocumentRepository,
)
from scrappers.pub_med_scrapper_2 import PubMedScrapper
from tests.full_text_qrel_tests.nfcorpus_test_generator import TestCaseGenerator


data_path = ".scrapper_data"
scrapper = PubMedScrapper(Path(data_path))
doc_db_path = os.path.join(data_path, "doc-db")
tc_db_path = os.path.join(data_path, "tc-db")

doc_repo = SQLiteDocumentRepository(doc_db_path)

tc_repo = SQLiteTestCaseRepo(tc_db_path, doc_repo)


gen = TestCaseGenerator(scrapper, tc_repo, doc_repo)
gen.generate_test_cases()

print([(tc.id, tc.query, len(tc.documents)) for tc in tc_repo.get_test_cases()])
