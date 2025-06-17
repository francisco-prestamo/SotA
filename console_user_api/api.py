from receptionist_agent.interfaces import UserAPI
from expert_set.interfaces import UserQuerier


class ConsoleUserApi(UserAPI, UserQuerier):
    def __init__(self):
        super().__init__()

    def query_user(self, query: str) -> str:
        return input(query)

    def ask_user(self, query: str) -> str:
        return self.query_user(query)

    def message_user(self, text: str) -> None:
        print(text)


