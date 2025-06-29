from typing import Dict, List, Tuple
from pydantic import BaseModel

from board.board import Board
from entities.sota_table import sota_table_to_markdown

from .action_picker import PickActionResult
from .prompts.ask_questions import (
    ExpertQuestion,
    create_answer_model,
    questions_prompt
)
from .models import Expert, ExpertDescription
from .interfaces import JsonGenerator, UserQuerier


class UserQuestioner:
    """Coordinates expert-driven questioning to clarify research aspects."""

    def __init__(
            self,
            json_generator: JsonGenerator,
            user_querier: UserQuerier,
            board: Board,
            pick_action_result: PickActionResult,
    ):
        """
        Initialize the UserQuestioner with necessary dependencies.

        Args:
            json_generator: For structured LLM responses
            user_querier: To interact with the user
            board: Current research state
            pick_action_result: Result from action picking phase
        """
        self.json_generator = json_generator
        self.user_querier = user_querier
        self.sota_table_md = sota_table_to_markdown(board.sota_table)
        self.thesis_thoughts = self._display_thoughts(board.thesis_knowledge.thoughts)
        self.thesis_description = board.thesis_knowledge.description
        self.expert_presentations = pick_action_result.expert_presentations
        self.board = board

    def ask_questions(self, experts: List[Expert]) -> Tuple[str, str]:
        """
        Collect and present expert-formulated questions to the user.

        Args:
            experts: Current expert team members

        Returns:
            Tuple containing:
            - Consolidated questions summary
            - User's responses to the questions
        """
        id_to_expert = self._generate_expert_id_dict(experts)
        expert_descriptions = self._extract_descriptions_from_id_dict(id_to_expert)

        answer_model = create_answer_model(expert_descriptions)
        prompt = questions_prompt(
            self.expert_presentations,
            self.sota_table_md,
            self.thesis_description,
            self.thesis_thoughts,
        )

        questions_answer = self.json_generator.generate_json(prompt, answer_model)
        questions_summary = self._parse_answer_and_extract_questions(questions_answer)
        user_answers = self.user_querier.expert_set_query_user(questions_summary)

        return questions_summary, user_answers

    def _parse_answer_and_extract_questions(self, answer: BaseModel) -> str:
        """
        Extract the consolidated questions summary from the LLM response.

        Args:
            answer: Raw response from LLM

        Returns:
            String containing consolidated questions for the user
        """
        name = "expert_interventions"
        assert hasattr(answer, name)
        qs = []
        for _, intervention in getattr(answer, name).model_dump().items():
            intervention = ExpertQuestion.model_validate(intervention)
            qs.append(intervention.question)

        qs = [f"{i + 1}. {q}" for i, q in enumerate(qs)]
        return "\n".join(["Questions: "] + qs)

    def _display_thoughts(self, thoughts: List[str]) -> str:
        """
        Format expert thoughts for inclusion in prompts.

        Args:
            thoughts: List of expert thoughts

        Returns:
            Formatted string of thoughts
        """
        return "\n".join(f"- {thought}" for thought in thoughts)

    def _generate_expert_id_dict(self, experts: List[Expert]) -> Dict[str, Expert]:
        """
        Create a mapping of expert IDs to Expert objects.

        Args:
            experts: List of experts

        Returns:
            Dictionary mapping expert IDs to Expert instances
        """
        return {f"expert_{i}": e for i, e in enumerate(experts)}

    def _extract_descriptions_from_id_dict(
            self, expert_id_dict: Dict[str, Expert]
    ) -> Dict[str, ExpertDescription]:
        """
        Extract expert descriptions from expert objects.

        Args:
            expert_id_dict: Mapping of expert IDs to Expert objects

        Returns:
            Dictionary mapping expert IDs to their descriptions
        """
        return {id: e.expert_model for id, e in expert_id_dict.items()}
