from typing import List

def build_search_query_synthesis_prompt(queries: List[str]) -> str:
    """Build prompt for synthesizing multiple search queries into one"""
    queries_list = "\n".join([f"- {query}" for query in queries])
    
    return f"""Synthesize the following search queries from different experts into a single, comprehensive search query.

EXPERT QUERIES:
{queries_list}

INSTRUCTIONS:
1. Combine the key concepts from all queries
2. Maintain the specificity of each expert's focus
3. Create a query that maximizes coverage of all expert interests
4. Keep the query concise but comprehensive
5. Use relevant keywords and technical terms

Provide only the synthesized search query as a string response."""
