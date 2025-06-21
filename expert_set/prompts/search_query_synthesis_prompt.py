from typing import List


def build_search_query_synthesis_prompt(queries: List[str]) -> str:
    """Build prompt for detecting the most important research focus and creating a targeted research description"""
    queries_list = "\n".join([f"- {query}" for query in queries])

    return f"""Analyze the following search queries and identify the MOST IMPORTANT research topic that needs to be searched.

EXPERT QUERIES:
{queries_list}

INSTRUCTIONS:
1. Identify the core research problem or topic that appears most critical across the queries
2. Determine what specific information would be most valuable to retrieve
3. Consider what would be most relevant for academic databases like Semantic Scholar
4. Create a research description that:
   - Clearly explains what type of research papers are needed
   - Describes the key concepts and topics to focus on
   - Identifies the specific research domain or field
   - Explains why this particular research area is the priority

TASK: 
Return a clear, descriptive explanation of what research topic should be searched for. This should be a comprehensive description that explains:
- What specific research area or problem needs to be investigated
- What kind of studies or papers would be most valuable
- What key concepts, methods, or findings should be prioritized
- Why this particular focus was chosen as the most important

Provide a detailed research description (2-3 sentences) that clearly explains what needs to be searched for and why, rather than just a search query string."""