from typing import List

def build_domain_extraction_prompt(
    title: str,
    authors: List[str],
    abstract: str
) -> str:
    """Build prompt for extracting the domain/field of a paper"""
    authors_str = ", ".join(authors)
    
    return f"""Analyze the following scientific paper and determine its primary research domain/field with high precision.

PAPER INFORMATION:
- Title: {title}
- Authors: {authors_str}
- Abstract: {abstract}

TASK:
You are a domain classification expert. Your task is to identify the most appropriate primary research domain/field for this academic paper.

INSTRUCTIONS:
1. Carefully analyze the terminology, methodologies, and research questions presented in the title and abstract
2. Use standard, widely recognized domain names (e.g., "Computer Vision", "Natural Language Processing", "Reinforcement Learning", "Bioinformatics", "Quantum Computing", etc.)
3. Be specific enough to be meaningful but not so narrow that it becomes a niche subfield
4. If the paper spans multiple domains:
   - Identify which domain contains the paper's primary contribution
   - Consider which research community would most likely be interested in this work
   - Choose the domain that best represents the paper's core focus
5. Taxonomy guidelines:
   - Use parent fields for interdisciplinary work (e.g., "Machine Learning" for work spanning multiple ML subfields)
   - Use specific subfields when the paper clearly focuses on that area (e.g., "Computer Vision" instead of just "Artificial Intelligence")
   - For novel combinations, use the most relevant established domain

Respond with ONLY the domain name as a string. Do not include any explanations, punctuation, or additional text."""
