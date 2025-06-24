from typing import TypeVar, List

from pydantic import BaseModel
    
T = TypeVar("T", bound=BaseModel)

def is_necessary_search_prompt(query: str, relevant_text_units: List[str]) -> str:
    text_units_str = "\n---\n".join(relevant_text_units)
    if len(relevant_text_units) == 0:
        text_units_str = "No relevant text units found."
    return f"""
Given a user query and a set of the most relevant text units from your knowledge base, determine if these text units are sufficient to answer the query. If not, indicate that more information should be searched for.

Query: {query}
Relevant Text Units:\n{text_units_str}

Are these text units sufficient to answer the query, or is it necessary to search for more information? Reason and then Answer true or false.
"""
