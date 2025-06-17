import json
from typing import Dict, Optional, Type

from pydantic import BaseModel, create_model
from ..models import ExpertDescription


class ExpertAnswerModel(BaseModel):
    reasoning: str
    more_context_needed: bool
    rag_query: Optional[str] = None


def create_answers_model(experts: Dict[str, ExpertDescription]) -> Type[BaseModel]:

    """
    Dynamically create a Pydantic model with one field per expert.
    Each field is named after the expert key and has type ExpertAnswerModel.
    """
    # Prepare fields for create_model
    model_fields = {
        expert_key: (ExpertAnswerModel, ...) for expert_key in experts.keys()
    }

    # Create dynamic model class
    AnswersModel = create_model("AnswersModel", **model_fields, __base__=BaseModel)
    return AnswersModel


def rag_queries_prompt(
    sota_markdown: str,
    # thesis: ThesisKnowledgeModel,
    thesis_desc: str,
    thoughts_on_thesis: str,
    experts_model: Dict[str, ExpertDescription],
    answer_model: Type[BaseModel],
) -> str:
    """
    Build a prompt for expert agents to update the State of the Art (SOTA) table.
    """
    answer_model_json = answer_model.model_json_schema()
    answer_model_str = json.dumps(answer_model_json, indent=2)

    prompt = f"""
This is a conversation of a team of domain experts to update a State of the Art (SOTA) table for a research thesis.

Current State of the Art Table (in Markdown):
{sota_markdown}


Thesis Context:
Description:
{thesis_desc}

Key Thoughts:
{thoughts_on_thesis}

Expert Profiles:
{experts_model}

Each expert above will provide their reasoning of whether or not they need to recover more context from their internal survey paper repository.
The output must be a JSON object mapping each expert key to an object of the form {{"reasoning":"...","requires_more_context":true|false,"rag_query":"..."}}.

Be concise and explicit:
- "reasoning": the step-by-step internal rationale of each expert.
- "requires_more_context": true if the expert needs additional documents to decide what to do next, else false.
- "rag_query": a short, focused query string to retrieve context if needed; null otherwise.

Return only the JSON object. No additional text.

Expected output schema:
{answer_model_str}
"""
    return prompt

