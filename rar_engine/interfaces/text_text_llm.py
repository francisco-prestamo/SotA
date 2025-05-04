from abc import ABC, abstractmethod

class TextTextLLMQuerier(ABC):

    @abstractmethod
    def generate_text(self, query: str) -> str:
        """
        Send a query to the LLM and get back a text response.
        :param query: The input prompt or question.
        :return: A textual response from the model.
        """
        pass
