from board.board import Board
from console_user_api import ConsoleUserApi
from expert_set import ExpertSet
from graphrag import GraphRag
from llm_models import GeminiJsonGenerator, NomicAIEmbedder, JsonGeneratorInspectionWrapper
from receptionist_agent import ReceptionistAgent
from recoverer_agent import RecovererAgent
from vectorial_db import FaissVecDBFactory
from rag_repo import RagRepoFactory
from config import _parse_args
from doc_recoverers import *

_parse_args()

json_gen = GeminiJsonGenerator()
json_gen = JsonGeneratorInspectionWrapper(json_gen)

embedder = NomicAIEmbedder()
graph_rag = GraphRag(embedder, json_gen, json_gen)
board = Board(json_gen, graph_rag)
scrappers = [
    SemanticScholarRecoverer(),
    ArXivRecoverer(),
    PubMedRecoverer(),
    DOIRecoverer(),
]
recoverer = RecovererAgent(json_gen, graph_rag, scrappers)
user_querier = ConsoleUserApi()
vector_repo_factory = FaissVecDBFactory(embedder.dim)
knowledge_repo_fatory = RagRepoFactory(embedder, vector_repo_factory)

receptionist = ReceptionistAgent(json_gen, board, recoverer, user_querier)

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
