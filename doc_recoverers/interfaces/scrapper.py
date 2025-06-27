from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Scraper(ABC):
    """
    Abstract base class for web scrapers that extract article data from a URL.
    """

    @abstractmethod
    def get_metadata(self, url: str) -> Dict[str, object]:
        """
        Given an article URL, extract and return its metadata.
        Returns a dict with keys:
          - 'title': str
          - 'abstract': str
          - 'authors': List[str]
        """
        pass

    @abstractmethod
    def get_full_text(self, url: str) -> Optional[str]:
        """
        Given an article URL, locate and download the full text (PDF or HTML)
        and return it as a string. Returns None if not available.
        """
        pass

    @abstractmethod
    def get_reference_urls(self, url: str) -> List[str]:
        """
        Given an article URL, return a list of related article URLs.
        This could be a union of similar-articles and cited-by lists,
        without duplicates.
        """
        pass

    @abstractmethod
    def get_similar_urls(self, url: str) -> List[str]:
        """
        Given an article URL, scrape and return the list of URLs
        found in the 'Similar articles' section.
        """
        pass

    @abstractmethod
    def get_cited_by_urls(self, url: str) -> List[str]:
        """
        Given an article URL, scrape and return the list of URLs
        found in the 'Cited by' section.
        """
        pass
