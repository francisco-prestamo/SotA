from typing import Dict, List
from pydantic import BaseModel

from board.board import Board
from entities.sota_table import sota_table_to_markdown

from .prompts.acquire_context import (
    ExpertAnswerModel,
    create_answers_model as create_rag_queries_prompt_answer_model,
    rag_queries_prompt,
)
from .prompts.pick_action import (
    ExpertIntervention,
    pick_action_prompt,
    ExpertPresentation,
    create_answers_model as create_pick_action_prompt_answer_model,
    pick_action_summary_prompt,
    SummaryAnswerModel,
)
from .models import Expert, DocumentChunk, ExpertDescription, RoundAction
from .interfaces import JsonGenerator, KnowledgeRecoverer


EXTRA_CONTEXT_AMOUNT_OF_PAPERS = 2


class PickActionResult(BaseModel):
    action: RoundAction
    summary: str
    expert_presentations: Dict[str, ExpertPresentation]


class ActionPicker:
    def __init__(
        self,
        json_generator: JsonGenerator,
        board: Board,
        knowledge_recoverer: KnowledgeRecoverer,
    ):
        self.json_generator = json_generator
        self.sota_table_md = sota_table_to_markdown(board.sota_table)
        self.thesis_thoughts = self._display_thoughts(board.thesis_knowledge.thoughts)
        self.thesis_description = board.thesis_knowledge.description
        self.document_recoverer = knowledge_recoverer

    def pick_action(self, experts: List[Expert]) -> PickActionResult:
        id_to_expert = self._generate_expert_id_dict(experts)
        expert_descriptions = self._extract_descriptions_from_id_dict(id_to_expert)

        id_to_presentation = self._generate_expert_presentations(
            id_to_expert, expert_descriptions
        )

        answer_model = create_pick_action_prompt_answer_model(expert_descriptions)
        prompt = pick_action_prompt(
            id_to_presentation,
            self.sota_table_md,
            self.thesis_description,
            self.thesis_thoughts,
            answer_model,
        )

        id_to_intervention = self.json_generator.generate_json(prompt, answer_model)

        chosen_action = self._parse_answer_and_count_votes(id_to_intervention)

        summary = self._summarize_action_picking_process(
            prompt, id_to_intervention, chosen_action
        )

        return PickActionResult(summary=summary, action=chosen_action, expert_presentations=id_to_presentation)

    def _summarize_action_picking_process(
        self,
        pick_action_prompt: str,
        pick_action_answer: BaseModel,
        chosen_action: RoundAction,
    ) -> str:

        summary_prompt = pick_action_summary_prompt(
            pick_action_prompt, pick_action_answer, chosen_action
        )

        summary = self.json_generator.generate_json(summary_prompt, SummaryAnswerModel)

        return summary.summary

    def _parse_answer_and_count_votes(
        self, id_to_intervention: BaseModel
    ) -> RoundAction:

        votes = {action.value: 0 for action in RoundAction}
        for _, intervention in id_to_intervention.model_dump().items():
            intervention = ExpertIntervention.model_validate(intervention)
            # the action choice is actually returned as a string
            votes[intervention.action_choice] += 1

        votes = list([(action, vote_count) for action, vote_count in votes.items()])

        votes = sorted(votes, key=lambda x: -x[1])

        return RoundAction(votes[0][0])

    def _generate_expert_presentations(
        self,
        id_to_expert: Dict[str, Expert],
        expert_descriptions: Dict[str, ExpertDescription],
    ) -> Dict[str, ExpertPresentation]:
        extra_context = self._acquire_context_if_necessary(
            id_to_expert, expert_descriptions
        )

        answ = {
            id: ExpertPresentation(
                expert_description=expert_descriptions[id],
                extra_context=(
                    [self._display_chunk(chunk) for chunk in chunks]
                    if len(chunks) > 0
                    else None
                ),
            )
            for id, chunks in extra_context.items()
        }

        return answ

    def _display_chunk(self, chunk: DocumentChunk) -> str:
        return "Excerpt from '" + chunk.document_title + "': \n" + chunk.chunk

    def _acquire_context_if_necessary(
        self,
        experts: Dict[str, Expert],
        expert_descriptions: Dict[str, ExpertDescription],
    ) -> Dict[str, List[DocumentChunk]]:
        """
        Returns a mapping of expert id -> List[DocumentChunk],
        showing which expert recovered what excerpt
        """

        answer_model = create_rag_queries_prompt_answer_model(expert_descriptions)
        prompt = rag_queries_prompt(
            self.sota_table_md,
            self.thesis_description,
            self.thesis_thoughts,
            expert_descriptions,
            answer_model,
        )

        answer = self.json_generator.generate_json(prompt, answer_model)

        return self._execute_expert_context_search_commands(answer, experts)

    def _execute_expert_context_search_commands(
        self, answer: BaseModel, experts: Dict[str, Expert]
    ) -> Dict[str, List[DocumentChunk]]:

        answ = {}
        for expert_id, expert_answer in answer.model_dump().items():
            expert = experts[expert_id]

            validated_answer = ExpertAnswerModel.model_validate(expert_answer)
            query = validated_answer.rag_query
            if not query:
                continue

            chunks = expert.knowledge.query_knowledge(
                query, EXTRA_CONTEXT_AMOUNT_OF_PAPERS
            )
            answ[expert_id] = chunks

        return answ

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
