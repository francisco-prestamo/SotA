from typing import List

from board.board import Board
from entities.sota_table import SotaTable
from recoverer_agent import RecovererAgent

from .interfaces import KnowledgeRepositoryFactory, JsonGenerator, UserQuerier
from .models import BuildExpertCommand, RoundAction
from .expert_builder import ExpertBuilder
from .action_picker import ActionPicker


class ExpertSet:
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

        self.expert_builder = ExpertBuilder(
            document_recoverer, board, knowledge_repository_factory
        )
        self.experts = self.expert_builder.build_experts(expert_build_commands)

    def build_sota(self) -> SotaTable:
        action = self._make_experts_choose_action()

        # ...

    def _make_experts_choose_action(self) -> RoundAction:
        picker = ActionPicker(self.json_generator, self.board, self.document_recoverer)
        return picker.pick_action(self.experts)
