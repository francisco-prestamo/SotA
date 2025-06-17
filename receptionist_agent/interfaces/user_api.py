from abc import ABC, abstractmethod


class UserAPI(ABC):
    @abstractmethod
    def query_user(self, query: str) -> str:
        pass

    @abstractmethod
    def message_user(self, text: str) -> None:
        pass

