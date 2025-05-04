from doc_recoverers.interfaces.doc_embedder import DocEmbedder
from rar_engine.interfaces.text_text_llm import TextTextLLMQuerier

class LLMApi(DocEmbedder, TextTextLLMQuerier):
    def __init__(self) -> None:
        raise NotImplementedError()
