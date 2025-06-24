from typing import List
from pydantic import BaseModel, Field
import json

from ..models import ExpertDescription
from .pick_action import ExpertPresentation


class DocumentRemovalIntervention(BaseModel):
    """
    Models what the intervention of an expert in the process of making a decision about which
    documents to remove from the state of the art table would look like
    """

    reasoning: str = Field(description="The rationale of the expert to choose the given documents to delete")

    documents_to_delete: List[int] = Field(
        description="List of document indices to remove (0-3 documents)",
        min_length=0,
        max_length=3
    )

class SummaryAnswerModel(BaseModel):
    summary: str


def create_removal_answers_model(experts: dict[str, ExpertDescription]) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model with one field per expert.
    Each field is named after the expert key and has type DocumentRemovalIntervention.
    """
    # Prepare fields for create_model
    model_fields = {
        expert_key: (DocumentRemovalIntervention, ...) for expert_key in experts.keys()
    }

    # Create dynamic model class
    RemovalInterventionsModel = create_model(
        "RemovalInterventionsModel", **model_fields, __base__=BaseModel
    )
    return RemovalInterventionsModel


def remove_document_prompt(
    presentations: dict[str, ExpertPresentation],
    sota_table_md: str,
    thesis_desc: str,
    thesis_thoughts: str,
) -> str:
    expert_presentation_model_str = json.dumps(
        [{"expert_id": ExpertPresentation.model_json_schema()}], indent=2
    )

    expert_strs = json.dumps(
        {id: pres.model_dump() for id, pres in presentations.items()}, indent=2
    )

    return f"""
A set of experts gather to decide which documents should be removed from the state of the art table for a given research paper.
The goal is to identify documents that are no longer relevant given the current understanding of the research paper and the experts' thoughts.

Of the research paper, a description is provided:
{thesis_desc}

The experts have made their thoughts known about the current state of the investigation:
{thesis_thoughts}

The current state of the art table is as follows:
{sota_table_md}

This is a description of the experts taking part in the investigation, in the form {expert_presentation_model_str},
notice the expert ids, they will be used later:
{expert_strs}

Each expert should analyze the current state of the art table and identify which documents (if any) should be removed because they are:
1. Redundant with other documents in the table
2. Not relevant to the research paper's focus
3. Outdated or superseded by more recent work
4. Not contributing to the understanding of the state of the art

Experts can choose to remove 0-3 documents. They should provide clear reasoning for their choices.

Given this information, each expert will now intervene in the process of making a decision, following a similar schema
as when they were described, their answers will be of the following form, consider that the expert ids must match with the id in their descriptions above
"""


def remove_document_summary_prompt(remove_document_prompt: str, answer: BaseModel, documents_removed: List[int]) -> str:
    return f"""
As the moderator in a discussion between experts about removing documents from the state of the art table, your role is to summarize
the process the experts went through to make their decisions. You are expected to provide a concise summary of the interaction,
the documents chosen for removal, and the general rationale as to why they were chosen, based on the proceedings.

Output a json object containing a summary with the following schema:
{json.dumps(SummaryAnswerModel.model_json_schema(), indent=2)}

The process of decision making is as follows:

--- Role: System ---

{remove_document_prompt}

--- Role: Experts ---

{json.dumps(answer.model_dump(), indent=2)}

--- Role: System ---

After reviewing the experts' decisions, the following documents were chosen for removal: {documents_removed}
"""

