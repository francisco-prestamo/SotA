from board.board import Board

from console_user_api import ConsoleUserApi
from entities.sota_table import sota_table_to_markdown
from expert_set import ExpertSet
from graphrag import GraphRag
from llm_models import NomicAIEmbedder
from llm_models.json_generators.gemini import GeminiJsonGenerator
from llm_models.text_embedders.gemini import GeminiEmbedder
from receptionist_agent import ReceptionistAgent
from recoverer_agent import RecovererAgent
from vectorial_db import FaissVecDBFactory
from rag_repo import RagRepoFactory
from config import _parse_args
from doc_recoverers import *


_parse_args()

json_gen = GeminiJsonGenerator()

embedder = GeminiEmbedder(dimensions=128)
graph_rag = GraphRag(text_embedder=embedder, json_generator=json_gen,low_consume=False,max_tokens=1800)
board = Board(json_gen, graph_rag)
scrappers = [
    SemanticScholarRecoverer(),
    ArXivRecoverer()
]
recoverer = RecovererAgent(json_gen, graph_rag, scrappers)
vector_repo_factory = FaissVecDBFactory(embedder.dim)
knowledge_repo_fatory = RagRepoFactory(embedder, vector_repo_factory)

user_querier = ConsoleUserApi()
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

print(sota_table_to_markdown(sota))

