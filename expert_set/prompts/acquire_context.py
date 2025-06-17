import json
from typing import Dict, Optional, Type

from pydantic import BaseModel, create_model

from board.board import ThesisKnowledgeModel
from entities.sota_table import SotaTable, sota_table_to_markdown
from expert_set.models.expert import ExpertModel

class ExpertAnswerModel(BaseModel):
    reasoning: str
    more_context_needed: bool
    rag_query: Optional[str] = None

def create_answers_model(experts: Dict[str, ExpertModel]) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model with one field per expert.
    Each field is named after the expert key and has type ExpertAnswerModel.
    """
    # Prepare fields for create_model
    model_fields = {
        expert_key: (ExpertAnswerModel, ...) for expert_key in experts.keys()
    }

    # Create dynamic model class
    AnswersModel = create_model(
        'AnswersModel',
        **model_fields,
        __base__=BaseModel
    )
    return AnswersModel

def rag_queries_prompt(
    sota: SotaTable, 
    thesis: ThesisKnowledgeModel, 
    experts: Dict[str, ExpertModel]
) -> str:
    """
    Build a prompt for expert agents to update the State of the Art (SOTA) table.
    """
    # Render the SOTA table as markdown
    sota_md = sota_table_to_markdown(sota)

    # Render thesis knowledge as bullets
    thesis_desc = thesis.description.strip()
    thesis_bullets = "\n".join(f"- {thought}" for thought in thesis.thoughts)

    # Build expert definitions and pretty-print
    experts_json = {
        key: {"name": exp.name, "description": exp.description}
        for key, exp in experts.items()
    }
    experts_pretty = json.dumps(experts_json, indent=2)

    # Example output schema and pretty-print
    answer_example_json = {
        key: {"reasoning": "str", "requires_more_context": "bool", "rag_query": "str or null"}
        for key in experts
    }
    schema_pretty = json.dumps(answer_example_json, indent=2)

    prompt = f"""
This is a conversation of a team of domain experts to update a State of the Art (SOTA) table for a research thesis.

Current State of the Art Table (in Markdown):
{sota_md}

Thesis Context:
Description:
{thesis_desc}

Key Thoughts:
{thesis_bullets}

Expert Profiles:
{experts_pretty}

Each expert above will provide their reasoning of whether or not they need to recover more context from their internal survey paper repository.
The output must be a JSON object mapping each expert key to an object of the form {{"reasoning":"...","requires_more_context":true|false,"rag_query":"..."}}.

Be concise and explicit:
- "reasoning": the step-by-step internal rationale of each expert.
- "requires_more_context": true if the expert needs additional documents to decide what to do next, else false.
- "rag_query": a short, focused query string to retrieve context if needed; null otherwise.

Return only the JSON object. No additional text.

Expected output schema:
{schema_pretty}
"""
    return prompt

