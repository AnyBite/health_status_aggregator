from abc import ABC, abstractmethod
from typing import List
from app.domain.entities import Record, AnalysisResult


class ILLMService(ABC):
    """Interface for LLM-based health analysis service."""

    @abstractmethod
    def analyze_record(self, record: Record) -> AnalysisResult:
        """Analyze a health record and return analysis result."""
        pass

    @abstractmethod
    def reanalyze_with_context(
        self,
        record: Record,
        previous_result: AnalysisResult,
        similar_cases: List[dict],
    ) -> AnalysisResult:
        """Reanalyze a record with additional RAG context."""
        pass
