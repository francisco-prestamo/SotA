from abc import ABC, abstractmethod


class UserAPI(ABC):
    @abstractmethod
    def receptionist_query_user(self, query: str) -> str:
        pass

    @abstractmethod
    def receptionist_message_user(self, text: str) -> None:
        pass

