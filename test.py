from board.board import Board

from entities.sota_table import sota_table_to_markdown
from expert_set import ExpertSet
from expert_set.interfaces import user_querier
from graphrag import GraphRag
from llm_models import GeminiJsonGenerator, JsonGeneratorInspectionWrapper, GeminiEmbedder
from receptionist_agent import ReceptionistAgent
from recoverer_agent import RecovererAgent
from vectorial_db import FaissVecDBFactory
from rag_repo import RagRepoFactory
from config import _parse_args
from doc_recoverers import *
from console_user_api import ConsoleUserApi
from mocks import RecovererAgentMock, ReceptionistAgentMock

from mocks.user_agent.user_agent_configs import (
    UNSURE_CANCER_RESEARCHER,
    VAGUE_AI_RESEARCHER,
    DIRECT_CS_STUDENT,

)

_parse_args()

json_gen = GeminiJsonGenerator()
json_gen = JsonGeneratorInspectionWrapper(json_gen)

embedder = GeminiEmbedder()

graph_rag = GraphRag(embedder, json_gen, json_gen)
board = Board(json_gen, graph_rag)
scrappers = [
    SemanticScholarRecoverer(),
    ArXivRecoverer(),
    PubMedRecoverer(),
    DOIRecoverer(),
]
# recoverer = RecovererAgent(json_gen, graph_rag, scrappers, board.knowledge_graph)
recoverer = RecovererAgentMock()

researcher_config = UNSURE_CANCER_RESEARCHER
# researcher_config = VAGUE_AI_RESEARCHER
# researcher_config = DIRECT_CS_STUDENT

# user_querier = UserAgent(
#     researcher_config.paper_description,
#     researcher_config.personality_description,
#     json_gen,
# )
user_querier = ConsoleUserApi()
vector_repo_factory = FaissVecDBFactory(embedder.dim)
knowledge_repo_fatory = RagRepoFactory(embedder, vector_repo_factory)

# receptionist = ReceptionistAgent(json_gen, board, recoverer, user_querier)
receptionist = ReceptionistAgentMock()

expert_build_commands = receptionist.interact()
expert_set = ExpertSet(
    json_gen,
    expert_build_commands,
    recoverer,
    knowledge_repo_fatory,
    board,
    user_querier,
)

sota = expert_set.build_sota()

print(sota_table_to_markdown(sota))
