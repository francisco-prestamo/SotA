from doc_recoverers.interfaces.doc_embedder import DocEmbedder
from doc_embedder.interfaces.embedding_api import EmbeddingAPI

class DocEmbedderImpl(DocEmbedder):
    def __init__(self, embedding_api: EmbeddingAPI) -> None:
        self.embedding_api = embedding_api
        raise NotImplementedError()
