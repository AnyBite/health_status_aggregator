import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from app.domain import (
    IReportingService,
    IDataIngestionService,
    AnalysisResult,
    AggregatedMetrics,
)


logger = logging.getLogger(__name__)


class AggregateHealthUseCase:
    """Use case for aggregating health statistics."""

    def __init__(
        self,
        reporting_service: Optional[IReportingService] = None,
    ):
        self._reporting_service = reporting_service

    def execute(
        self,
        results: List[AnalysisResult],
        records: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate health statistics from analysis results.
        
        Args:
            results: List of analysis results.
            records: Optional list of original records for account/project mapping.
            
        Returns:
            Dictionary with aggregated statistics.
        """
        if not results:
            return {
                "total_records": 0,
                "health_distribution": {},
                "summary": "No records to aggregate",
            }
        
        # Basic aggregation
        health_dist = {"Good": 0, "Warning": 0, "Critical": 0}
        mismatches = 0
        total_confidence = 0.0
        
        for result in results:
            health_dist[result.recommended_health] = health_dist.get(result.recommended_health, 0) + 1
            total_confidence += result.confidence
            if result.mismatched:
                mismatches += 1
        
        aggregation = {
            "total_records": len(results),
            "health_distribution": health_dist,
            "mismatch_count": mismatches,
            "mismatch_rate": round(mismatches / len(results) * 100, 2),
            "average_confidence": round(total_confidence / len(results), 4),
        }
        
        # Add account/project aggregation if reporting service available
        if self._reporting_service and records:
            aggregation["by_account"] = self._reporting_service.aggregate_by_account(results, records)
            aggregation["by_project"] = self._reporting_service.aggregate_by_project(results, records)
        
        return aggregation


class GenerateReportUseCase:
    """Use case for generating comprehensive reports."""

    def __init__(
        self,
        reporting_service: IReportingService,
        data_ingestion_service: Optional[IDataIngestionService] = None,
    ):
        self._reporting_service = reporting_service
        self._data_ingestion_service = data_ingestion_service

    def execute(
        self,
        results: List[AnalysisResult],
        records: Optional[List[dict]] = None,
        export_csv: bool = True,
        csv_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report with metrics and visualizations.
        
        Args:
            results: List of analysis results.
            records: Optional list of original records for ground truth comparison.
            export_csv: Whether to export results to CSV.
            csv_path: Optional custom path for CSV export.
            
        Returns:
            Dictionary with metrics, charts, and export info.
        """
        if not results:
            return {"error": "No results to report"}
        
        # Calculate metrics
        metrics = self._reporting_service.calculate_metrics(results, records)
        
        # Generate chart data
        charts = self._reporting_service.generate_charts(metrics)
        
        # Export to CSV if requested
        csv_file = None
        if export_csv:
            csv_file = self._reporting_service.export_to_csv(results, csv_path)
        
        # Account-level aggregation
        account_aggregation = {}
        project_aggregation = {}
        if records:
            account_aggregation = self._reporting_service.aggregate_by_account(results, records)
            project_aggregation = self._reporting_service.aggregate_by_project(results, records)
        
        return {
            "metrics": {
                "total_records": metrics.total_records,
                "health_distribution": metrics.health_distribution,
                "flagged_mismatches": metrics.flagged_mismatches,
                "rag_invocations": metrics.rag_invocations,
                "human_reviews_required": metrics.human_reviews_required,
                "average_confidence": round(metrics.average_confidence, 4),
                "processing_time_total_ms": round(metrics.processing_time_total_ms, 2),
                "accuracy": metrics.accuracy_metrics,
            },
            "charts": charts,
            "aggregations": {
                "by_account": account_aggregation,
                "by_project": project_aggregation,
            },
            "exports": {
                "csv_file": csv_file,
            },
        }


class LoadAndValidateDataUseCase:
    """Use case for loading and validating dataset."""

    def __init__(self, data_ingestion_service: IDataIngestionService):
        self._data_ingestion_service = data_ingestion_service

    def execute(self, file_path: str) -> Dict[str, Any]:
        """
        Load and validate a dataset file.
        
        Args:
            file_path: Path to JSON or CSV file.
            
        Returns:
            Dictionary with valid records, invalid records, and statistics.
        """
        # Determine file type and load
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            raw_records = self._data_ingestion_service.load_from_json(file_path)
        elif path.suffix.lower() == ".csv":
            raw_records = self._data_ingestion_service.load_from_csv(file_path)
        else:
            return {"error": f"Unsupported file format: {path.suffix}"}
        
        # Validate and convert
        valid_records, invalid_records = self._data_ingestion_service.validate_and_convert(raw_records)
        
        return {
            "total_loaded": len(raw_records),
            "valid_count": len(valid_records),
            "invalid_count": len(invalid_records),
            "valid_records": valid_records,
            "invalid_records": invalid_records,
            "validation_rate": round(len(valid_records) / len(raw_records) * 100, 2) if raw_records else 0,
        }
