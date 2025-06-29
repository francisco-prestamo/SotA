import sqlite3
from typing import List
from .. import DocumentRepository
from .. import TestCase, TestCaseRepository

class SQLiteTestCaseRepo(TestCaseRepository):
    def __init__(self, db_path: str, document_repository: DocumentRepository):
        """Initialize the SQLite TestCase repository."""
        self.db_path = db_path
        self.document_repository = document_repository
        self._create_tables()

    def _create_tables(self):
        """Create all necessary tables for storing test cases and their relationships."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create the table for storing test case-document relationships
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_cases (
                    test_case_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    relevance INTEGER NOT NULL,
                    PRIMARY KEY(test_case_id, document_id)
                ) STRICT
            ''')

            # Create the table for storing the query string related to each test case
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_cases_queries (
                    test_case_id TEXT NOT NULL,
                    query_str TEXT NOT NULL,
                    FOREIGN KEY (test_case_id) REFERENCES test_cases (test_case_id)
                ) STRICT
            ''')

    def store_test_case(self, tc: TestCase):
        """Store a test case, its associated documents, and query string."""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Store the query string for this test case
            cursor.execute('''
                INSERT INTO test_cases_queries (test_case_id, query_str)
                VALUES (?, ?)
            ''', (tc.id, tc.query))

            cursor.execute('''
                DELETE FROM test_cases WHERE test_case_id = ?
            ''', (tc.id,))

            # For each document in the test case
            for doc in tc.documents:
                # Ensure the document exists in the document repository
                if not self.document_repository.document_exists(doc.id):
                    self.document_repository.store_document(doc)

                # Store the test case-document relationship
                cursor.execute('''
                    INSERT OR REPLACE INTO test_cases (test_case_id, document_id, relevance)
                    VALUES (?, ?, ?)
                ''', (tc.id, doc.id, tc.relevance.get(doc.id, 0)))
            conn.commit()

    def get_test_cases(self) -> List[TestCase]:
        """Retrieve all test cases with their queries and associated documents."""
        test_cases = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT test_case_id, document_id, relevance FROM test_cases')
            rows = cursor.fetchall()

            # Group by test_case_id
            for row in rows:
                test_case_id, document_id, relevance = row
                if test_case_id not in test_cases:
                    test_cases[test_case_id] = {
                        'documents': [],
                        'relevance': {}
                    }
                test_cases[test_case_id]['documents'].append(document_id)
                test_cases[test_case_id]['relevance'][document_id] = relevance

            # Fetch query string for each test case
            cursor.execute('SELECT test_case_id, query_str FROM test_cases_queries')
            query_rows = cursor.fetchall()
            query_map = {row[0]: row[1] for row in query_rows}

        # Fetch the full documents from the document repository
        all_documents = self.document_repository.get_documents()
        document_map = {doc.id: doc for doc in all_documents}

        # Now merge the results into TestCase objects
        return [
            TestCase(
                id=test_case_id,
                query=query_map[test_case_id],
                documents=[document_map[doc_id] for doc_id in test_case_data['documents']],
                relevance=test_case_data['relevance']
            )
            for test_case_id, test_case_data in test_cases.items()
        ]
