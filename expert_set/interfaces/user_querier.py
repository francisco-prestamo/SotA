from abc import ABC, abstractmethod


class UserQuerier(ABC):

    @abstractmethod
    def expert_set_query_user(self, query: str) -> str:
        pass
