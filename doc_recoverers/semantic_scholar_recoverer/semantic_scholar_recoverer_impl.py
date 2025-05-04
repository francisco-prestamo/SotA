from rar_engine.interfaces.doc_recoverer import DocRecoverer
from doc_recoverers.interfaces.doc_embedder import DocEmbedder

class SemanticScholarRecoverer(DocRecoverer):
    def __init__(self, embedder: DocEmbedder) -> None:
        self.embedder = embedder
        raise NotImplementedError()
