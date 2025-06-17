from typing import List

def build_domain_extraction_prompt(
    title: str,
    authors: List[str],
    abstract: str
) -> str:
    """Build prompt for extracting the domain/field of a paper"""
    authors_str = ", ".join(authors)
    
    return f"""Analyze the following paper and determine its primary research domain/field.

PAPER INFORMATION:
- Title: {title}
- Authors: {authors_str}
- Abstract: {abstract}

INSTRUCTIONS:
1. Identify the primary research domain based on the title and abstract
2. Use standard domain names (e.g., "Computer Vision", "Natural Language Processing", "Machine Learning", etc.)
3. Be specific but not overly narrow
4. If the paper spans multiple domains, choose the most prominent one

Respond with only the domain name as a string."""
