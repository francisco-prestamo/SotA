from controller.interfaces.docset_builder import DocsetBuilder

from rar_engine.interfaces.doc_recoverer import DocRecoverer
from rar_engine.interfaces.doc_repository import RelevantDocRepository
from rar_engine.interfaces.text_text_llm import TextGenerator

class RAREngine(DocsetBuilder):
    def __init__(self, doc_recoverer: DocRecoverer, doc_repository: RelevantDocRepository, text_generator: TextGenerator) -> None:
        self.doc_recoverer = doc_recoverer
        self.doc_repository = doc_repository
        self.text_generator = text_generator

        raise NotImplementedError()


