import json
from typing import Dict, List, Type
from pydantic import BaseModel, Field, create_model

from expert_set.models.round_action import RoundAction

from ..models import ExpertDescription


class ExpertPresentation(BaseModel):
    """
    Models what the presence of the expert in the action choice would look like, i.e. what
    extra information they brought to the table, as well as their general description

    """

    expert_description: ExpertDescription = Field(
        description="Information about the expert"
    )
    extra_context: List[str] | None = Field(
        description="Recovered excerpts of research paper surveys to inform the decision of what to do next in the process of building the state of the art table"
    )


class ExpertIntervention(BaseModel):
    """
    Models what the intervention of an expert in the process of making a decision about what to
    do next would look like
    """

    reasoning: str
    action_choice: RoundAction


def create_answers_model(experts: Dict[str, ExpertDescription]) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model with one field per expert.
    Each field is named after the expert key and has type ExpertIntervention.
    """
    # Prepare fields for create_model
    model_fields = {
        expert_key: (ExpertIntervention, ...) for expert_key in experts.keys()
    }

    # Create dynamic model class
    InterventionsModel = create_model(
        "InterventionsModel", **model_fields, __base__=BaseModel
    )
    return InterventionsModel


def pick_action_prompt(
    presentations: Dict[str, ExpertPresentation],
    sota_table_md: str,
    thesis_desc: str,
    thesis_thoughts: str,
    answer_model: Type[BaseModel],
) -> str:
    expert_presentation_model_str = json.dumps(
        [{"expert_id": ExpertPresentation.model_json_schema()}], indent=2
    )
    answer_model_str = json.dumps(answer_model.model_json_schema(), indent=2)

    expert_strs = json.dumps(
        {id: pres.model_dump() for id, pres in presentations.items()}, indent=2
    )

    return f"""
A set of experts gather to build a state of the art table for a given research paper, such a table is a summary of
features of similar recent papers, that exemplify the current practices in the given field, or of foundational
papers which in turn set a lasting standard of how research and problem solving is conducted in the field.

Of the research paper, a description is provided:
{thesis_desc}

The experts have made their thoughts known about the current state of the investigation:
{thesis_thoughts}

The current state of the art table is as follows:
{sota_table_md}

This is a description of the experts taking part in the investigation, in the form {expert_presentation_model_str},
notice the expert ids, they will be used later:
{expert_strs}

The decision to be made is to either:

{RoundAction.AddDocument.value}: Add a new document to the state of the art table, this choice is to be made when there are important parts of the 
research paper such that none of the recovered papers document the state of the art in those specific fields.

{RoundAction.RemoveDocument.value}: Remove a document from the state of the art table, this choice is to be made when some document is redundant or
not relevant given the provided description

{RoundAction.AskUser.value}: Ask the user, this choice is to be made when clarification is needed for some part of the description of the paper,
the user commanded this entire operation, and it is they who help build the description of the paper

Given this information, each expert will now intervene in the process of making a decision, following a similar schema
as when they were described, their answers will be of the following form, consider that the expert ids must match with the id in their descriptions above

Output only an answer in the following schema, no extra text
{answer_model_str}
"""

class SummaryAnswerModel(BaseModel):
    summary: str

def pick_action_summary_prompt(pick_action_prompt: str, answer: BaseModel, chosen_action: RoundAction) -> str:
    return f"""

As the moderator in a discussion between experts in order to build a state of the art table for a research paper, your role at the moment
is to summarize the process the experts went through to make a decision on what to do next, you are expected to provide a concise summary
of the interaction, the chosen action and the general rationale as to why it was chosen, based on the proceedings, output
a json object containing a summary with the following schema:

{json.dumps(SummaryAnswerModel.model_json_schema(), indent=2)}

The process of decision making is as follows

--- Role: System ---

{pick_action_prompt}

--- Role: Experts ---

{json.dumps(answer.model_dump(), indent=2)}

--- Role: System ---

After a count of the votes, the following action was chosen: {chosen_action}

    """















