import json
from typing import Dict, Type
from pydantic import BaseModel, create_model

from ..models import ExpertDescription
from .pick_action import ExpertPresentation


class ExpertQuestion(BaseModel):
    """
    Models what a question from an expert to the user would look like
    """

    reasoning: str
    question: str



def create_answer_model(experts: Dict[str, ExpertDescription]) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model with one field per expert.
    Each field is named after the expert key and has type ExpertQuestion.
    """
    # Prepare fields for create_model
    interventions = {expert_key: (ExpertQuestion, ...) for expert_key in experts.keys()}

    # Create dynamic model class
    InterventionsModel = create_model(
        "InterventionsModel", **interventions, __base__=BaseModel
    )

    AnswerModel = create_model(
        "AnswerModel",
        expert_interventions=(InterventionsModel, ...),
        questions_summary=(str, ...),
        __base__=BaseModel,
    )

    return AnswerModel


def questions_prompt(
    presentations: Dict[str, ExpertPresentation],
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
A set of experts gather to ask questions to the user about a research paper they are analyzing to build a state of the
art table for a research paper, the state of the art section is the section of a research paper that explores similar
investigations on the subject by other researchers, and compares them with their own work
The goal is to clarify aspects of the paper that are not clear or need more information to properly assess its features and contributions.

Of the research paper, a description is provided:
{thesis_desc}

The experts have made their thoughts known about the current state of the investigation:
{thesis_thoughts}

The current state of the art table is as follows:
{sota_table_md}

This is a description of the experts taking part in the investigation, in the form {expert_presentation_model_str},
notice the expert ids, they will be used later:
{expert_strs}

Each expert should formulate a clear and specific question to the user about aspects of the paper that need clarification.
The questions should be focused on:
1. Technical details that are not clear from the description
2. Specific features or contributions that need more context
3. Assumptions or requirements that need to be verified
4. Any other aspect that would help better understand the paper's position in the state of the art

Given this information, each expert will now formulate their question, following a similar schema
as when they were described, their answers will be of the following form, consider that the expert ids must match with the id in their descriptions above

Output a json object of the provided schema and only that.
"""


# class QuestionsSummary(BaseModel):
#     summary: str
#     questions: List[str]


# def questions_summary_prompt(expert_questions: BaseModel) -> str:
#     return f"""
# As the moderator in a discussion between experts about questions to ask the user regarding a research paper,
# your role is to summarize the questions and their rationale. You are expected to provide a concise summary
# of the interaction and a set of questions , based on the proceedings.

# The process of question formulation is as follows:

# --- Role: System ---

# {questions_prompt}

# --- Role: Experts ---

# {json.dumps(answer.model_dump(), indent=2)}

# --- Role: System ---

# Output a json object containing a summary and a list of questions with the following schema:
# {json.dumps(QuestionsSummary.model_json_schema(), indent=2)}
# """
