from pubMed_recoverer_impl import PubMedRecoverer

# Create an instance of the PubMed recoverer
pubmed_recoverer = PubMedRecoverer()

# Define a search query and number of documents to retrieve
search_query = "machine learning in cancer diagnosis"
num_documents = 5

# Retrieve documents
recovered_docs = pubmed_recoverer.recover(query=search_query, k=num_documents)

# Process the results
print(f"Recoverer: {pubmed_recoverer.name}")
print(f"Description: {pubmed_recoverer.description}")
print(f"\nRecovered {len(recovered_docs)} documents for query: '{search_query}'")

for i, doc in enumerate(recovered_docs, 1):
    print(f"\nDocument {i}:")
    print(f"  ID: {doc.id}")
    print(f"  Title: {doc.title[:80] + '...' if len(doc.title) > 80 else doc.title}")
    print(f"  Authors: {', '.join(doc.authors[:3])}{'...' if len(doc.authors) > 3 else ''}")
    print(f"  Abstract: {doc.abstract[:100] + '...' if len(doc.abstract) > 100 else doc.abstract}")
    print(f"  Content length: {len(doc.content)} characters")
    print(f"  Content sample: {doc.content[:100] + '...' if len(doc.content) > 100 else doc.content}")

# Example output when no documents are found
if not recovered_docs:
    print("\nNo documents found. This could be because:")
    print("- The query didn't match any PubMed articles")
    print("- Found articles didn't have accessible PDFs via DOI")
    print("- There were network/timeout issues during retrieval")