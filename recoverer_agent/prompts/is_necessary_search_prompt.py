from typing import Type, TypeVar

from pydantic import BaseModel
from recoverer_agent.models.bool_answer_model import BoolAnswerModel
    
T = TypeVar("T", bound=BaseModel)

def is_necessary_search_prompt(query: str, response:str, result_schema: Type[T]) -> str:

    return f"""
Given a user query and a possible response, determine if the response is a good answer to the query. Answer only 'yes' or 'no'.

Query: {query}
Response: {response}

Is the response a good answer to the query or it is necessary to search for more information? Reason and then Answer true or false.
{result_schema.model_json_schema()}
"""
