import sqlite3
from typing import List, Optional
from entities import Document
import json
from .. import DocumentRepository


class SQLiteDocumentRepository(DocumentRepository):
    def __init__(self, db_path: str):
        """Initialize SQLite document repository."""
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """Create the documents table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    content TEXT NOT NULL
                ) STRICT
            """
            )
            conn.commit()

    def store_document(self, document: Document):
        """Store a document in the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Prepare the authors as a JSON string
            authors_json = json.dumps(document.authors)
            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (id, title, abstract, authors, content)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    document.id,
                    document.title,
                    document.abstract,
                    authors_json,
                    document.content,
                ),
            )
            conn.commit()

    def get_document(self, id: str) -> Optional[Document]:
        """Get a document by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, abstract, authors, content FROM documents WHERE id = ?",
                (id,),
            )
            row = cursor.fetchone()
            if row:
                # Deserialize authors from JSON
                authors = json.loads(row[3])
                return Document(
                    id=row[0],
                    title=row[1],
                    abstract=row[2],
                    authors=authors,
                    content=row[4],
                )
            return None

    def get_documents(self) -> List[Document]:
        """Get all documents."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, abstract, authors, content FROM documents"
            )
            rows = cursor.fetchall()
            documents = []
            for row in rows:
                authors = json.loads(row[3])
                documents.append(
                    Document(
                        id=row[0],
                        title=row[1],
                        abstract=row[2],
                        authors=authors,
                        content=row[4],
                    )
                )
            return documents

    def document_exists(self, id: str) -> bool:
        """Check if a document with the given ID exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM documents WHERE id = ?", (id,))
            return cursor.fetchone() is not None
