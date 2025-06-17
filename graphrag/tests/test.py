import ir_datasets
import time
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
import numpy as np
from PIL.ImageChops import offset
from pydantic import BaseModel, Field
from llm_models.text_embedders.nomic_ai import NomicAIEmbedder
from llm_models.json_generators.gemini import GeminiJsonGenerator
from graphrag.graphrag import GraphRag
import random
import json


class Document(BaseModel):
    """Represents a recoverable document."""

    id: str
    """Unique identifier of the document."""

    title: str
    """Title of the document."""

    abstract: str
    """Abstract of the document."""

    authors: list[str]
    """List of authors of the document."""

    content: str
    """Contents of the document."""

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


@dataclass
class IRTestResult:
    """Results from IR dataset testing"""
    query_id: str
    query_text: str
    direct_docs: List[str]
    response_based_docs: List[str]
    direct_time: float
    response_time: float
    find_docs_time: float
    total_response_based_time: float
    relevant_doc_ids: List[str]  # High-scored documents
    direct_precision: float
    response_precision: float
    direct_recall: float
    response_recall: float
    # Additional metrics for scored documents
    direct_ndcg: float
    response_ndcg: float
    direct_avg_score: float
    response_avg_score: float


class GraphRagMSMARCOTester:
    """Test GraphRag on MS MARCO dataset with scored documents"""

    def __init__(self, graph_rag: 'GraphRag', dataset_name: str = 'msmarco-document/train'):
        self.graph_rag = graph_rag
        self.dataset_name = dataset_name
        self.dataset = ir_datasets.load(dataset_name)
        self.doc_store = self.dataset.docs_store()

    def get_test_queries_with_scores(self, max_queries: int = 5, max_docs_per_query: int = 100) -> Dict[str, Dict]:
        """Get queries and their scored documents from MS MARCO"""
        test_queries = {}
        query_count = 0

        print(f"Collecting first {max_queries} queries and their scored documents...")

        # Get queries
        queries_list = []
        offset = random.randint(10, 10000)
        i=0
        for query in self.dataset.queries_iter():
            i+=1
            if i<offset:
                continue

            if query_count >= max_queries:
                break
            queries_list.append(query)
            query_count += 1

        # For each query, get its scored documents
        for query in queries_list:
            query_id = query.query_id
            print(f"Processing query {query_id}: {query.text}")

            docs_scores = {}
            doc_count = 0

            # Get scored documents for this query
            for scored_doc in self.dataset.scoreddocs_iter():
                if scored_doc.query_id == query_id:
                    docs_scores[scored_doc.doc_id] = scored_doc.score
                    doc_count += 1
                    if doc_count >= max_docs_per_query:
                        break

            if docs_scores:  # Only include queries that have scored documents
                test_queries[query_id] = {
                    'text': query.text,
                    'scored_docs': docs_scores
                }
                print(f"  Found {len(docs_scores)} scored documents")
            else:
                print(f"  No scored documents found, skipping query")

        print(f"Final: {len(test_queries)} queries with scored documents")
        return test_queries

    def prepare_documents_with_scores(self, test_queries: Dict[str, Dict],
                                      top_scored_per_query: int = 25,
                                      random_docs_per_query: int = 25) -> List[Document]:
        """Prepare documents: top scored docs + random docs for each query"""

        all_scored_doc_ids = set()
        query_doc_mapping = {}

        # Process each query's scored documents
        for query_id, query_data in test_queries.items():
            docs_scores = query_data['scored_docs']

            # Convert negative scores to positive by adding absolute minimum
            scores_list = list(docs_scores.values())
            min_score = min(scores_list)
            print(f"MinScore {min_score}")
            query_doc_mapping[query_id] = min_score
            if min_score < 0:
                # Convert all scores to positive
                adjusted_scores = {doc_id: score + abs(min_score) for doc_id, score in docs_scores.items()}
            else:
                adjusted_scores = docs_scores.copy()

            # Sort by adjusted score (descending) and take top N
            sorted_docs = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
            top_docs = sorted_docs[:top_scored_per_query]

            query_doc_mapping[query_id] = {
                'scored_docs': dict(top_docs),
                'original_scores': docs_scores
            }

            # Add to global set
            all_scored_doc_ids.update([doc_id for doc_id, _ in top_docs])

            print(f"Query {query_id}: Selected {len(top_docs)} top scored documents")

        # Add random documents that don't appear in any scored documents
        print("Adding random non-scored documents...")

        # Get a sample of all available document IDs
        all_available_doc_ids = []
        doc_count = 0
        max_scan = 10000  # Limit scanning to avoid memory issues

        for doc in self.dataset.docs_iter():
            all_available_doc_ids.append(doc.doc_id)
            doc_count += 1
            if doc_count >= max_scan:
                break

        # Get all scored docs that were not selected as top docs
        all_scored_but_not_top_docs = []
        for query_id, query_data in test_queries.items():
            # Get all scored doc IDs for this query
            all_query_scored_docs = set(query_data['scored_docs'].keys())
            # Get the top docs that were selected for this query
            top_docs_for_query = set(query_doc_mapping[query_id]['scored_docs'].keys())
            # Add docs that have scores but weren't in the top selection
            all_scored_but_not_top_docs.extend(list(all_query_scored_docs - top_docs_for_query))
        
        # Randomly select from scored docs that weren't in the top selection
        total_random_needed = random_docs_per_query * len(test_queries)
        if len(all_scored_but_not_top_docs) >= total_random_needed:
            random_doc_ids = set(random.sample(all_scored_but_not_top_docs, total_random_needed))
        else:
            random_doc_ids = set(all_scored_but_not_top_docs)
            print(f"Warning: Only {len(all_scored_but_not_top_docs)} scored but not top docs available")

        # Combine all document IDs to load
        target_doc_ids = all_scored_doc_ids.union(random_doc_ids)

        print(f"Loading {len(target_doc_ids)} documents:")
        print(f"  Scored documents: {len(all_scored_doc_ids)}")
        print(f"  Random documents: {len(random_doc_ids)}")

        # Load documents using doc_store
        documents = []
        loaded_count = 0

        for doc_id in target_doc_ids:
            try:
                doc = self.doc_store.get(doc_id)
                if doc:
                    documents.append(Document(
                        id=doc.doc_id,
                        title=doc.title if hasattr(doc, 'title') and doc.title else f"Document {doc.doc_id}",
                        abstract=doc.body[:200] if hasattr(doc, 'body') else "",
                        authors=[],
                        content=doc.body if hasattr(doc, 'body') else doc.text if hasattr(doc, 'text') else ""
                    ))
                    loaded_count += 1

                    if loaded_count % 50 == 0:
                        print(f"  Loaded {loaded_count} documents...")

            except Exception as e:
                print(f"Error loading document {doc_id}: {e}")
                continue

        print(f"Successfully loaded {len(documents)} documents")

        # Store the query-document mapping for evaluation
        self.query_doc_mapping = query_doc_mapping

        return documents

    def get_relevant_docs_for_query(self, query_id: str) -> List[str]:
        """Get high-scored (relevant) documents for a query"""
        if query_id in self.query_doc_mapping:
            return list(self.query_doc_mapping[query_id]['scored_docs'].keys())
        return []

    def get_doc_score(self, query_id: str, doc_id: str) -> float:
        """Get the original score for a document given a query"""
        if query_id in self.query_doc_mapping:
            return self.query_doc_mapping[query_id]['original_scores'].get(doc_id, 0.0)
        return 0.0

    def calculate_metrics(self, retrieved_docs: List[str], relevant_docs: List[str]) -> tuple:
        """Calculate precision and recall"""
        if not retrieved_docs:
            return 0.0, 0.0

        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        intersect = retrieved_set.intersection(relevant_set)

        precision = len(intersect) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(intersect) / len(relevant_set) if relevant_set else 0.0

        return precision, recall

    def calculate_ndcg(self, retrieved_docs: List[str], query_id: str, k: int = None) -> float:
        """Calculate NDCG score"""
        if not retrieved_docs:
            return 0.0

        if k is None:
            k = len(retrieved_docs)

        # Get relevance scores (use adjusted positive scores)
        relevance_scores = []
        for doc_id in retrieved_docs[:k]:
            score = self.get_doc_score(query_id, doc_id)
            # Convert to positive if needed
            if score < 0:
                score = 0.0  # Treat negative scores as 0 relevance
            relevance_scores.append(score)

        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate IDCG (ideal DCG)
        all_scores = []
        if query_id in self.query_doc_mapping:
            all_scores = list(self.query_doc_mapping[query_id]['scored_docs'].values())

        ideal_scores = sorted(all_scores, reverse=True)[:k]
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_avg_score(self, retrieved_docs: List[str], query_id: str) -> float:
        """Calculate average score of retrieved documents"""
        if not retrieved_docs:
            return 0.0

        scores = [self.get_doc_score(query_id, doc_id) for doc_id in retrieved_docs]
        return np.mean(scores) if scores else 0.0

    def test_single_query(self, query_id: str, query_text: str, relevant_docs: List[str],
                          kg: 'KnowledgeGraph', k: int = 10) -> IRTestResult:
        """Test a single query with both approaches"""

        print(f"\nTesting query: {query_text[:100]}...")
        print(f"High-scored documents: {len(relevant_docs)}")

        # Method 1: Direct document finding
        print("Method 1: Direct document finding")
        start_time = time.time()
        direct_docs = self.graph_rag.find_documents(query_text, kg, k)
        direct_time = time.time() - start_time

        direct_doc_ids = [doc.id for doc in direct_docs]
        direct_precision, direct_recall = self.calculate_metrics(direct_doc_ids, relevant_docs)
        direct_ndcg = self.calculate_ndcg(direct_doc_ids, query_id, k)
        direct_avg_score = self.calculate_avg_score(direct_doc_ids, query_id)

        print(f"  Found {len(direct_docs)} documents in {direct_time:.3f}s")
        print(f"  Precision: {direct_precision:.3f}, Recall: {direct_recall:.3f}")
        print(f"  NDCG: {direct_ndcg:.3f}, Avg Score: {direct_avg_score:.3f}")

        # Method 2: Response-based approach
        print("Method 2: Generate response then find documents")

        # Step 1: Generate response
        start_response = time.time()
        response = self.graph_rag.respond(query_text, kg, c=3)
        response_time = time.time() - start_response

        print(f"  Generated response in {response_time:.3f}s")
        print(f"  Response preview: {response[:150]}...")

        # Step 2: Find documents based on response
        start_find = time.time()
        response_docs = self.graph_rag.find_documents(response, kg, k)
        find_docs_time = time.time() - start_find

        total_response_based_time = response_time + find_docs_time

        response_doc_ids = [doc.id for doc in response_docs]
        response_precision, response_recall = self.calculate_metrics(response_doc_ids, relevant_docs)
        response_ndcg = self.calculate_ndcg(response_doc_ids, query_id, k)
        response_avg_score = self.calculate_avg_score(response_doc_ids, query_id)

        print(f"  Found {len(response_docs)} documents in {find_docs_time:.3f}s")
        print(f"  Total time: {total_response_based_time:.3f}s")
        print(f"  Precision: {response_precision:.3f}, Recall: {response_recall:.3f}")
        print(f"  NDCG: {response_ndcg:.3f}, Avg Score: {response_avg_score:.3f}")

        return IRTestResult(
            query_id=query_id,
            query_text=query_text,
            direct_docs=direct_doc_ids,
            response_based_docs=response_doc_ids,
            direct_time=direct_time,
            response_time=response_time,
            find_docs_time=find_docs_time,
            total_response_based_time=total_response_based_time,
            relevant_doc_ids=relevant_docs,
            direct_precision=direct_precision,
            response_precision=response_precision,
            direct_recall=direct_recall,
            response_recall=response_recall,
            direct_ndcg=direct_ndcg,
            response_ndcg=response_ndcg,
            direct_avg_score=direct_avg_score,
            response_avg_score=response_avg_score
        )

    def run_test_suite(self, max_queries: int = 5, k: int = 10, rebuild_kg: bool = True,
                       top_scored_per_query: int = 25, random_docs_per_query: int = 25) -> List[IRTestResult]:
        """Run complete test suite with MS MARCO scored documents"""
        print("=" * 80)
        print("GRAPHRAG MS MARCO DATASET TEST SUITE - SCORED DOCUMENTS")
        print("=" * 80)

        # Get test queries with their scored documents
        test_queries = self.get_test_queries_with_scores(max_queries)

        if not test_queries:
            print("No queries with scored documents found!")
            return []

        # Load documents (scored + random)
        documents = self.prepare_documents_with_scores(
            test_queries,
            top_scored_per_query,
            random_docs_per_query
        )

        if not documents:
            print("No documents loaded!")
            return []

        if rebuild_kg:
            print("\nBuilding knowledge graph...")
            start_kg = time.time()
            kg = self.graph_rag.build_knowledge_graph(documents)
            kg_time = time.time() - start_kg
            print(f"Knowledge graph built in {kg_time:.2f}s")
        else:
            print("Using existing knowledge graph...")
            kg = None

        # Run tests
        results = []
        print(f"\nTesting {len(test_queries)} queries...")
        print("-" * 80)

        for query_id, query_data in test_queries.items():
            relevant_docs = self.get_relevant_docs_for_query(query_id)
            result = self.test_single_query(
                query_id,
                query_data['text'],
                relevant_docs,
                kg,
                k
            )
            results.append(result)
            print(f"Completed query {query_id}")

        # Print summary
        self.print_summary(results)
        return results

    def print_summary(self, results: List[IRTestResult]):
        """Print test summary statistics"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        if not results:
            print("No results to summarize")
            return

        # Calculate averages
        avg_direct_time = np.mean([r.direct_time for r in results])
        avg_response_time = np.mean([r.total_response_based_time for r in results])
        avg_direct_precision = np.mean([r.direct_precision for r in results])
        avg_response_precision = np.mean([r.response_precision for r in results])
        avg_direct_recall = np.mean([r.direct_recall for r in results])
        avg_response_recall = np.mean([r.response_recall for r in results])
        avg_direct_ndcg = np.mean([r.direct_ndcg for r in results])
        avg_response_ndcg = np.mean([r.response_ndcg for r in results])
        avg_direct_score = np.mean([r.direct_avg_score for r in results])
        avg_response_score = np.mean([r.response_avg_score for r in results])

        print(f"Tested {len(results)} queries")
        print(f"Dataset: {self.dataset_name}")
        print()

        print("PERFORMANCE COMPARISON:")
        print(f"Direct Method:")
        print(f"  Average Time: {avg_direct_time:.3f}s")
        print(f"  Average Precision: {avg_direct_precision:.3f}")
        print(f"  Average Recall: {avg_direct_recall:.3f}")
        print(f"  Average NDCG: {avg_direct_ndcg:.3f}")
        print(f"  Average Score: {avg_direct_score:.3f}")
        print()

        print(f"Response-Based Method:")
        print(f"  Average Time: {avg_response_time:.3f}s")
        print(f"  Average Precision: {avg_response_precision:.3f}")
        print(f"  Average Recall: {avg_response_recall:.3f}")
        print(f"  Average NDCG: {avg_response_ndcg:.3f}")
        print(f"  Average Score: {avg_response_score:.3f}")
        print()

        # Comparisons
        if avg_response_time > avg_direct_time:
            slowdown = avg_response_time / avg_direct_time
            print(f"Response-based method is {slowdown:.2f}x slower")
        else:
            speedup = avg_direct_time / avg_response_time
            print(f"Response-based method is {speedup:.2f}x faster")

        precision_diff = avg_response_precision - avg_direct_precision
        recall_diff = avg_response_recall - avg_direct_recall
        ndcg_diff = avg_response_ndcg - avg_direct_ndcg
        score_diff = avg_response_score - avg_direct_score

        print(f"Precision difference: {precision_diff:+.3f}")
        print(f"Recall difference: {recall_diff:+.3f}")
        print(f"NDCG difference: {ndcg_diff:+.3f}")
        print(f"Average Score difference: {score_diff:+.3f}")

        # Individual query results
        print(f"\nINDIVIDUAL QUERY RESULTS:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Query: {result.query_text[:80]}...")
            print(
                f"   Direct: P={result.direct_precision:.3f}, R={result.direct_recall:.3f}, NDCG={result.direct_ndcg:.3f}")
            print(
                f"   Response: P={result.response_precision:.3f}, R={result.response_recall:.3f}, NDCG={result.response_ndcg:.3f}")


def main():
    """Main test function with MS MARCO scored documents"""

    # Initialize GraphRag instance
    graph_rag = GraphRag(
        text_embedder=NomicAIEmbedder(dimensions=128),
        json_generator=GeminiJsonGenerator(),
        max_tokens=1000,
        low_consume=False,
    )

    # Initialize tester
    tester = GraphRagMSMARCOTester(graph_rag, dataset_name="msmarco-document/train")

    # Run test suite
    results = tester.run_test_suite(
        max_queries=1,  # Test with 3 queries
        k=3,  # Retrieve top 10 documents
        rebuild_kg=True,
        top_scored_per_query=3,  # Top 25 scored docs per query
        random_docs_per_query=7,  # Plus 25 random docs per query
    )

    # Save results
    if results:
        print("\nSaving results...")
        with open('graphrag_msmarco_test_results2.json', 'w') as f:
            json.dump([vars(r) for r in results], f, indent=2)


if __name__ == "__main__":
    main()