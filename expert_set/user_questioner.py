from typing import Dict, List
from pydantic import BaseModel

from board.board import Board
from entities.sota_table import sota_table_to_markdown, SotaTable

from .action_picker import PickActionResult
from .prompts.pick_action import ExpertPresentation
from .prompts.ask_questions import (
    AnswerModel,
    ExpertQuestion,
    create_answer_model,
    questions_prompt,
    questions_summary_prompt,
    QuestionsSummary,
    BaseModel,
)
from .models import Expert, ExpertDescription
from .interfaces import JsonGenerator, UserQuerier


class UserQuestioner:
    def __init__(
        self,
        json_generator: JsonGenerator,
        user_querier: UserQuerier,
        board: Board,
        pick_action_result: PickActionResult,
    ):
        self.json_generator = json_generator
        self.user_querier = user_querier
        self.sota_table_md = sota_table_to_markdown(board.sota_table)
        self.thesis_thoughts = self._display_thoughts(board.thesis_knowledge.thoughts)
        self.thesis_description = board.thesis_knowledge.description
        self.expert_presentations = pick_action_result.expert_presentations
        self.board = board

    def ask_questions(self, experts: List[Expert]):
        id_to_expert = self._generate_expert_id_dict(experts)
        expert_descriptions = self._extract_descriptions_from_id_dict(id_to_expert)

        answer_model = create_answer_model(expert_descriptions)
        prompt = questions_prompt(
            self.expert_presentations,
            self.sota_table_md,
            self.thesis_description,
            self.thesis_thoughts,
            answer_model,
        )

        questions_answer = self.json_generator.generate_json(prompt, answer_model)
        questions_to_ask = self._parse_answer_and_extract_questions(questions_answer)


    def _parse_answer_and_extract_questions(self, answer: BaseModel) -> str:
        answer = AnswerModel.model_validate(answer)
        return answer.questions_summary


    def _display_thoughts(self, thoughts: List[str]) -> str:
        thesis_thoughts = ""
        for thought in thoughts:
            thesis_thoughts += "- " + thought + "\n"

        return thesis_thoughts

    def _generate_expert_id_dict(self, experts: List[Expert]) -> Dict[str, Expert]:
        answ = {}
        for i, e in enumerate(experts):
            answ["expert_" + str(i)] = e

        return answ

    def _extract_descriptions_from_id_dict(
        self, expert_id_dict: Dict[str, Expert]
    ) -> Dict[str, ExpertDescription]:
        return {id: e.expert_model for id, e in expert_id_dict.items()} 
