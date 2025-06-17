from typing import List
from board.board import Board
from entities.sota_table import SotaTable
from recoverer_agent import RecovererAgent
from .interfaces import KnowledgeRepositoryFactory, JsonGenerator, UserQuerier
from .models import RoundAction, BuildExpertCommand
from .expert_builder import ExpertBuilder
from .action_picker import ActionPicker, PickActionResult
from .document_remover import DocumentRemover, DocumentRemovalResult
from .user_questioner import UserQuestioner
from .prompts.update_description import update_description_prompt, DescriptionUpdate
from .prompts.update_expert_set import update_expert_set_prompt, ExpertSetUpdate

MAX_ROUNDS = 10

class ExpertSet:
    """Orchestrates expert set to build State-of-the-Art tables through iterative refinement."""

    def __init__(
            self,
            json_generator: JsonGenerator,
            expert_build_commands: List[BuildExpertCommand],
            document_recoverer: RecovererAgent,
            knowledge_repository_factory: KnowledgeRepositoryFactory,
            board: Board,
            user_querier: UserQuerier,
    ):
        self.json_generator = json_generator
        self.document_recoverer = document_recoverer
        self.board = board
        self.user_querier = user_querier
        self.knowledge_repository_factory = knowledge_repository_factory

        self.expert_builder = ExpertBuilder(
            document_recoverer, knowledge_repository_factory
        )
        self.experts = self.expert_builder.build_experts(expert_build_commands)

    def build_sota(self) -> SotaTable:
        """
        Construct the State-of-the-Art table through iterative expert rounds.

        Returns:
            Completed SOTA table
        """
        for round_num in range(MAX_ROUNDS):
            action_result = self._run_expert_round()

            if self._should_terminate(action_result.action):
                break

        return self.board.sota_table

    def _run_expert_round(self) -> PickActionResult:
        """
        Execute one full expert processing round.

        Returns:
            Result of the expert decision process
        """
        action_result = self._make_experts_choose_action()

        if action_result.action == RoundAction.RemoveDocument:
            self._handle_remove_documents(action_result)
        elif action_result.action == RoundAction.AskUser:
            self._handle_user_questions(action_result)
        elif action_result.action == RoundAction.AddDocument:
            self._handle_add_documents()

        return action_result

    def _make_experts_choose_action(self) -> PickActionResult:
        """
        Coordinate expert decision-making for next round action.

        Returns:
            Structured result of action selection
        """
        picker = ActionPicker(
            self.json_generator,
            self.board,
            self.document_recoverer
        )
        return picker.pick_action(self.experts)

    def _handle_user_questions(self, action_result: PickActionResult) -> None:
        """Process user questioning workflow."""
        questioner = UserQuestioner(
            self.json_generator,
            self.user_querier,
            self.board,
            action_result
        )

        questions_summary, user_answers = questioner.ask_questions(self.experts)
        self._update_thesis_description(questions_summary, user_answers)
        self._update_expert_set(questions_summary, user_answers)

    def _update_thesis_description(self, questions: str, answers: str) -> None:
        """
        Update research paper description based on user answers.

        Args:
            questions: Questions posed to the user
            answers: User's responses
        """
        prompt = update_description_prompt(
            current_description=self.board.thesis_knowledge.description,
            questions_asked=questions,
            user_answers=answers
        )
        update = self.json_generator.generate_json(prompt, DescriptionUpdate)

        self.board.update_thesis_description(update.updated_description)

        self.board.thesis_knowledge.thoughts.append(
            f"Updated description: {update.reasoning}"
        )

    def _update_expert_set(self, questions: str, answers: str) -> None:
        """
        Improve expert set composition based on new understanding.

        Args:
            questions: Questions posed to the user
            answers: User's responses
        """
        current_description = self.board.thesis_knowledge.description

        if self.board.thesis_knowledge.history:
            old_description = self.board.thesis_knowledge.history[-1]
        else:
            old_description = current_description

        current_experts = [e.expert_model.description for e in self.experts]

        prompt = update_expert_set_prompt(
            current_description=current_description,
            old_description=old_description,
            current_experts=current_experts,
            questions_asked=questions,
            user_answers=answers
        )
        update = self.json_generator.generate_json(prompt, ExpertSetUpdate)

        self._apply_expert_updates(update)

    def _apply_expert_updates(self, update: ExpertSetUpdate) -> None:
        """
        Modify expert set based on update instructions.

        Args:
            update: Structured update instructions
        """
        if update.to_remove:
            self.experts = [
                e for e in self.experts
                if e.expert_model.name not in update.to_remove
            ]

        if update.to_add:
            new_commands = [
                BuildExpertCommand(
                    name=f"NewExpert_{i}",
                    description=desc,
                    query=self._generate_query_from_description(desc)
                )
                for i, desc in enumerate(update.to_add)
            ]
            new_experts = self.expert_builder.build_experts(new_commands)
            self.experts.extend(new_experts)

        self.board.thesis_knowledge.thoughts.append(
            f"Expert set updated: {update.whether_to_remove_reasoning} "
            f"{update.whether_to_add_reasoning}"
        )

    def _generate_query_from_description(self, description: str) -> str:
        """
        Generate search query from expert description.

        Args:
            description: Expert's domain description

        Returns:
            Search query for domain surveys
        """
        return f"Recent surveys about {description.split('.')[0]}"

    def _handle_remove_documents(self, action_result: PickActionResult) -> None:
        """Execute document removal workflow."""
        remover = DocumentRemover(
            self.json_generator,
            self.board,
            action_result
        )
        removal_result = remover.remove_documents(self.experts)
        self.board.thesis_knowledge.thoughts.append(removal_result.summary)

    def _handle_add_documents(self) -> None:
        """Placeholder for document addition workflow."""
        # TODO: Implement document addition logic

    def _should_terminate(self, action: RoundAction) -> bool:
        """
        Determine if processing should terminate.

        Args:
            action: Selected round action

        Returns:
            True if processing should terminate, False otherwise
        """
        return action not in {
            RoundAction.AddDocument,
            RoundAction.RemoveDocument,
            RoundAction.AskUser
        }