import os
import json
import time
from typing import Dict, List, Set
from pydantic import BaseModel

from graphrag.tests.full_text_qrel_tests.persistence import SQLiteDocumentRepository, SQLiteTestCaseRepo
from graphrag.tests.full_text_qrel_tests.test_case import TestCase
from graphrag.graphrag import GraphRag
from graphrag.knowledge_graph import KnowledgeGraph
from llm_models.json_generators.gemini import GeminiJsonGenerator
from llm_models import FireworksApi
from llm_models.text_embedders.gemini import GeminiEmbedder
from entities.document import Document


class ReturnedDocuments(BaseModel):
    relevants_returned: List[str] = []
    medium_relevants_returned: List[str] = []
    non_relevants_returned: List[str] = []


class TestResult(BaseModel):
    rag: ReturnedDocuments
    graphrag: ReturnedDocuments
    graphrag_response: str
    graphrag_index_time_ms: float
    graphrag_response_time_ms: float
    rag_find_documents_time_ms: float
    graphrag_find_documents_time_ms: float
    relevants_ids: List[str]
    medium_relevants_ids: List[str]
    non_relevants_ids: List[str]





def main():
    """
    Main function that orchestrates the test execution process.
    """
    # Setup paths
    data_path = ".test_case_data"
    results_file = os.path.join("graphrag", "tests", "full_text_qrel_tests", "test_results.json")
    
    try:
        # Initialize repositories
        log("Initializing repositories...")
        doc_repo, tc_repo = initialize_repositories(data_path)
        
        # Initialize GraphRAG
        log("Initializing GraphRAG...")
        graph_rag = initialize_graphrag()
        
        # Load existing results
        log("Loading existing results...")
        results = load_existing_results(results_file)
        log(f"Loaded {len(results)} existing test results")
        
        # Get all test cases
        log("Loading test cases...")
        test_cases = tc_repo.get_test_cases()
        log(f"Total test cases: {len(test_cases)}")
        
        # Process each test case
        for i, test_case in enumerate(test_cases):
            log(f"Processing test case {i+1}/{len(test_cases)}: {test_case.id}")
            
            # Skip if already processed
            if test_case.id in results:
                log(f"  Skipping {test_case.id} (already processed)")
                continue
            
            try:
                log(f"  Filtering test case")
                test_case = filter_test_case(test_case)

                # Run the test case
                log(f"  Running test case")
                test_result = run_test_case(test_case, graph_rag)
                
                # Store results
                results[test_case.id] = test_result
                
                # Save results after each test case
                save_results(results, results_file)
                log(f"  Results saved for {test_case.id}")
                
            except Exception as e:
                log(f"  Error processing test case {test_case.id}: {e}")
                continue
        
        log(f"Test run completed. Results saved to {results_file}")
        log(f"Processed {len(results)} test cases")
        
    except Exception as e:
        log(f"Fatal error: {e}")
        return 1
    
    return 0


def run_test_case(
    test_case: TestCase, 
    graph_rag: GraphRag
) -> TestResult:
    """
    Run a single test case with both RAG and GraphRAG methods.
    The test_case may be filtered or unfiltered; this function does not care.
    """
    try:
        # Build knowledge graph and measure index time
        log(f"    Building knowledge graph with {len(test_case.documents)} documents")
        start_time = time.time()
        knowledge_graph = graph_rag.build_knowledge_graph(test_case.documents)
        graphrag_index_time_ms = (time.time() - start_time) * 1000
        
        # Run RAG method (direct document retrieval) and measure time
        start_time = time.time()
        retrieved_docs_rag = graph_rag.find_documents(test_case.query, knowledge_graph, k=10)
        rag_find_documents_time_ms = (time.time() - start_time) * 1000
        rag_results = categorize_documents_by_relevance(retrieved_docs_rag, test_case.relevance)
        
        # Run GraphRAG method (response generation) and measure time
        start_time = time.time()
        graphrag_response = graph_rag.respond(test_case.query, knowledge_graph, 10)
        graphrag_response_time_ms = (time.time() - start_time) * 1000
        
        # Run GraphRAG method (response -> document retrieval) and measure time
        start_time = time.time()
        retrieved_docs_graphrag = graph_rag.find_documents(graphrag_response, knowledge_graph, k=10)
        graphrag_find_documents_time_ms = (time.time() - start_time) * 1000
        graphrag_results = categorize_documents_by_relevance(retrieved_docs_graphrag, test_case.relevance)
        
        # Extract ids from the test case
        relevants_ids = [doc.id for doc in test_case.documents if test_case.relevance.get(doc.id, 0) == 2]
        medium_relevants_ids = [doc.id for doc in test_case.documents if test_case.relevance.get(doc.id, 0) == 1]
        non_relevants_ids = [doc.id for doc in test_case.documents if test_case.relevance.get(doc.id, 0) == 0]
        
        return TestResult(
            rag=rag_results,
            graphrag=graphrag_results,
            graphrag_response=graphrag_response,
            graphrag_index_time_ms=graphrag_index_time_ms,
            graphrag_response_time_ms=graphrag_response_time_ms,
            rag_find_documents_time_ms=rag_find_documents_time_ms,
            graphrag_find_documents_time_ms=graphrag_find_documents_time_ms,
            relevants_ids=relevants_ids,
            medium_relevants_ids=medium_relevants_ids,
            non_relevants_ids=non_relevants_ids
        )
    except Exception as e:
        raise Exception(f"Failed to run test case {test_case.id}: {e}")


def initialize_repositories(data_path: str) -> tuple[SQLiteDocumentRepository, SQLiteTestCaseRepo]:
    """
    Initialize document and test case repositories.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        Tuple of (document_repository, test_case_repository)
        
    Raises:
        FileNotFoundError: If database files don't exist
    """
    doc_db_path = os.path.join(data_path, "doc-db")
    tc_db_path = os.path.join(data_path, "tc-db")
    
    if not os.path.isfile(doc_db_path):
        raise FileNotFoundError(f"Document database not found at {doc_db_path}")
    if not os.path.isfile(tc_db_path):
        raise FileNotFoundError(f"Test case database not found at {tc_db_path}")
    
    doc_repo = SQLiteDocumentRepository(doc_db_path)
    tc_repo = SQLiteTestCaseRepo(tc_db_path, doc_repo)
    
    return doc_repo, tc_repo


def initialize_graphrag() -> GraphRag:
    """
    Initialize GraphRAG with required components.
    
    Returns:
        Initialized GraphRag instance
        
    Raises:
        Exception: If initialization fails
    """
    try:
        # json_gen = GeminiJsonGenerator()
        json_gen = FireworksApi()
        embedder = GeminiEmbedder(dimensions=128)
        graph_rag = GraphRag(
            text_embedder=embedder, 
            json_generator=json_gen, 
            low_consume=False, 
            max_tokens=1000
        )
        return graph_rag
    except Exception as e:
        raise Exception(f"Failed to initialize GraphRAG: {e}")

def filter_test_case(test_case: TestCase) -> TestCase:
    """
    Returns a new TestCase with only relevant and medium relevant documents (removes non-relevants).
    """
    filtered_docs = [doc for doc in test_case.documents if test_case.relevance.get(doc.id, 0) > 0]
    filtered_relevance = {doc_id: rel for doc_id, rel in test_case.relevance.items() if rel > 0}
    return TestCase(
        id=test_case.id,
        query=test_case.query,
        documents=filtered_docs,
        relevance=filtered_relevance
    )

# def filter_test_case(test_case: TestCase) -> TestCase:
#     """
#     Mock version: returns a TestCase with only the first document (if any).
#     """
#     if not test_case.documents:
#         return TestCase(id=test_case.id, query=test_case.query, documents=[], relevance={})
#     doc = test_case.documents[0]
#     rel = test_case.relevance.get(doc.id, 0)
#     return TestCase(
#         id=test_case.id,
#         query=test_case.query,
#         documents=[doc],
#         relevance={doc.id: rel}
#     )

def load_existing_results(results_file: str) -> Dict[str, TestResult]:
    """
    Load existing test results from JSON file.
    
    Args:
        results_file: Path to the JSON results file
        
    Returns:
        Dictionary mapping test case IDs to TestResult objects
        
    Raises:
        Exception: If loading fails
    """
    if not os.path.exists(results_file):
        return {}
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
            return {
                test_id: TestResult.model_validate(result_data) 
                for test_id, result_data in data.items()
            }
    except Exception as e:
        raise Exception(f"Failed to load existing results from {results_file}: {e}")


def save_results(results: Dict[str, TestResult], results_file: str):
    """
    Save test results to JSON file.
    
    Args:
        results: Dictionary mapping test case IDs to TestResult objects
        results_file: Path to the JSON results file
        
    Raises:
        Exception: If saving fails
    """
    try:
        # Convert TestResult objects to dictionaries using model_dump
        serializable_results = {
            test_id: result.model_dump()
            for test_id, result in results.items()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    except Exception as e:
        raise Exception(f"Failed to save results to {results_file}: {e}")


def categorize_documents_by_relevance(
    retrieved_docs: List[Document], 
    relevance_map: Dict[str, int]
) -> ReturnedDocuments:
    """
    Categorize retrieved documents by their relevance scores.
    
    Args:
        retrieved_docs: List of retrieved documents
        relevance_map: Dictionary mapping document IDs to relevance scores (2=relevant, 1=medium, 0=not relevant)
        
    Returns:
        ReturnedDocuments object with categorized document IDs
    """
    result = ReturnedDocuments()
    
    for doc in retrieved_docs:
        relevance = relevance_map.get(doc.id, 0)
        if relevance == 2:
            result.relevants_returned.append(doc.id)
        elif relevance == 1:
            result.medium_relevants_returned.append(doc.id)
        else:  # relevance == 0
            result.non_relevants_returned.append(doc.id)
    
    return result


def log(message: str):
    print("[TEST CASE RUNNER] " + message)


if __name__ == "__main__":
    exit(main()) 
