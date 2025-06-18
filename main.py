from board.board import Board
from console_user_api import ConsoleUserApi
from expert_set import ExpertSet
from graphrag import GraphRag
from llm_apis import FireworksApi
from receptionist_agent import ReceptionistAgent
from recoverer_agent import RecovererAgent
from vectorial_db import FaissVecDBFactory
from rag_repo import RagRepoFactory
from config import _parse_args
from doc_recoverers import *

_parse_args()

llm_api = FireworksApi()
graph_rag = GraphRag(llm_api, llm_api, llm_api)
board = Board(llm_api, graph_rag)
scrappers = [
    SemanticScholarRecoverer(),
    ArXivRecoverer(),
    PubMedRecoverer(),
    DOIRecoverer(),
]
recoverer = RecovererAgent(llm_api, graph_rag, scrappers)
user_querier = ConsoleUserApi()
vector_repo_factory = FaissVecDBFactory(llm_api.dim)
knowledge_repo_fatory = RagRepoFactory(llm_api, vector_repo_factory)

receptionist = ReceptionistAgent(llm_api, board, recoverer, user_querier)

expert_build_commands = receptionist.interact()
expert_set = ExpertSet(
    llm_api,
    expert_build_commands,
    recoverer,
    knowledge_repo_fatory,
    board,
    user_querier,
)

sota = expert_set.build_sota()
