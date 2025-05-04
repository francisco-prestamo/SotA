from repository_initializer.interfaces.doc_repository import DocRepository

class RepositoryInitializer:
    def __init__(self, doc_repository: DocRepository) -> None:
        self.doc_repository = doc_repository
        raise NotImplementedError()
