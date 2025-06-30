from abc import ABC, abstractmethod
from typing import List
from . import TestCase

class TestCaseRepository(ABC):
    @abstractmethod
    def get_test_cases(self) -> List[TestCase]:
        pass

    @abstractmethod
    def store_test_case(self, tc: TestCase):
        pass

    @abstractmethod
    def test_case_exists(self, id: str) -> bool:
        pass
