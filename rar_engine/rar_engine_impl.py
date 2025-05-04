from controller.interfaces.docset_builder import DocsetBuilder

from rar_engine.interfaces.doc_recoverer import DocRecoverer
from rar_engine.interfaces.doc_repository import RelevantDocRepository
from rar_engine.interfaces.text_text_llm import TextTextLLMQuerier

class RAREngine(DocsetBuilder):
    def __init__(self, doc_recoverer: DocRecoverer, doc_repository: RelevantDocRepository, text_text_llm_querier: TextTextLLMQuerier) -> None:
        self.doc_recoverer = doc_recoverer
        self.doc_repository = doc_repository
        self.llm_querier = text_text_llm_querier

        raise NotImplementedError()


