from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.domain.entities import AnalysisResult, AggregatedMetrics


class IReportingService(ABC):
    """Interface for reporting and aggregation service."""

    @abstractmethod
    def aggregate_by_account(self, results: List[AnalysisResult]) -> Dict[str, Dict]:
        """Aggregate results by account."""
        pass

    @abstractmethod
    def aggregate_by_project(self, results: List[AnalysisResult]) -> Dict[str, Dict]:
        """Aggregate results by project."""
        pass

    @abstractmethod
    def calculate_metrics(self, results: List[AnalysisResult]) -> AggregatedMetrics:
        """Calculate overall metrics."""
        pass

    @abstractmethod
    def export_to_csv(self, results: List[AnalysisResult], file_path: str) -> str:
        """Export results to CSV file."""
        pass

    @abstractmethod
    def generate_charts(self, metrics: AggregatedMetrics) -> Dict[str, Any]:
        """Generate chart data for visualization."""
        pass
