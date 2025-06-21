from receptionist_agent.interfaces import UserAPI
from expert_set.interfaces import UserQuerier


class ConsoleUserApi(UserAPI, UserQuerier):
    def __init__(self):
        super().__init__()

    def query_user(self, query: str) -> str:
        return input(query)

    def message_user(self, text: str) -> None:
        print(text)

    def receptionist_message_user(self, text: str) -> None:
        self.message_user(text)

    def receptionist_query_user(self, query: str) -> str:
        return self.query_user(query)

    def expert_set_query_user(self, query: str) -> str:
        return self.query_user(query)


