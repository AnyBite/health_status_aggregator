from abc import ABC, abstractmethod
from typing import List
from app.domain.entities import RAGContext


class IRAGService(ABC):
    """Interface for RAG (Retrieval-Augmented Generation) service."""

    @abstractmethod
    def retrieve_similar(self, text: str, top_k: int = 5) -> List[str]:
        """Retrieve similar text entries from the knowledge base."""
        pass

    @abstractmethod
    def retrieve_similar_with_context(self, text: str, top_k: int = 5) -> RAGContext:
        """Retrieve similar cases with full context and timing metrics."""
        pass

    @abstractmethod
    def add_to_index(self, texts: List[str]) -> None:
        """Add new texts to the index."""
        pass
