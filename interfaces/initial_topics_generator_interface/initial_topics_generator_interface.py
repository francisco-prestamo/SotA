import abc
from typing import Dict, Set

class InitialTopicsGeneratorInterface(abc.ABC):
    @abc.abstractmethod
    def generate_initial_topics(self, query: str) -> Dict[str, Set[str]]:
        """
        Generate initial topics for a given query.

        Args:
            query: The input query string.

        Returns:
            A dict mapping each topic (str) to a set of representative documents (Set[str]).
        """
        ...
