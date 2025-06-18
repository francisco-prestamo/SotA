from abc import ABC, abstractmethod


class UserQuerier(ABC):

    @abstractmethod
    def ask_user(self, query: str) -> str:
        pass
