from controller.interfaces.doc_ranker import DocRanker, DocRankerResponse
from controller.interfaces.docset_builder import DocsetBuilder

class Controller:
    def __init__(self, docset_builder: DocsetBuilder, ranker: DocRanker):
        self.docset_builder = docset_builder
        self.ranker = ranker
        raise NotImplementedError()
