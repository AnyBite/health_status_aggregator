import csv
import logging
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from app.domain import IReportingService, AnalysisResult, AggregatedMetrics


logger = logging.getLogger(__name__)


class ReportingService(IReportingService):
    """Implementation of reporting and aggregation service."""

    def __init__(self, output_dir: Path | None = None):
        self._output_dir = output_dir or Path("reports")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_by_account(
        self, 
        results: List[AnalysisResult],
        records: List[dict] | None = None
    ) -> Dict[str, Dict]:
        """Aggregate results by account."""
        # We need record data to get account_id mapping
        if not records:
            logger.warning("No records provided for account aggregation")
            return {}
        
        record_map = {r.get("record_id", i): r for i, r in enumerate(records)}
        
        account_stats = defaultdict(lambda: {
            "total": 0,
            "Good": 0,
            "Warning": 0,
            "Critical": 0,
            "mismatches": 0,
            "rag_invocations": 0,
            "human_reviews": 0,
            "avg_confidence": 0.0,
            "total_confidence": 0.0,
        })
        
        for result in results:
            record = record_map.get(result.record_id, {})
            account_id = record.get("account_id", "UNKNOWN")
            
            stats = account_stats[account_id]
            stats["total"] += 1
            stats[result.recommended_health] += 1
            stats["total_confidence"] += result.confidence
            
            if result.mismatched:
                stats["mismatches"] += 1
            if result.requires_rag:
                stats["rag_invocations"] += 1
            if result.requires_human_review:
                stats["human_reviews"] += 1
        
        # Calculate averages
        for account_id, stats in account_stats.items():
            if stats["total"] > 0:
                stats["avg_confidence"] = stats["total_confidence"] / stats["total"]
            del stats["total_confidence"]
        
        return dict(account_stats)

    def aggregate_by_project(
        self, 
        results: List[AnalysisResult],
        records: List[dict] | None = None
    ) -> Dict[str, Dict]:
        """Aggregate results by project."""
        if not records:
            logger.warning("No records provided for project aggregation")
            return {}
        
        record_map = {r.get("record_id", i): r for i, r in enumerate(records)}
        
        project_stats = defaultdict(lambda: {
            "total": 0,
            "Good": 0,
            "Warning": 0,
            "Critical": 0,
            "mismatches": 0,
            "avg_confidence": 0.0,
            "total_confidence": 0.0,
            "account_id": None,
        })
        
        for result in results:
            record = record_map.get(result.record_id, {})
            project_id = record.get("project_id", "UNKNOWN")
            
            stats = project_stats[project_id]
            stats["total"] += 1
            stats[result.recommended_health] += 1
            stats["total_confidence"] += result.confidence
            stats["account_id"] = record.get("account_id")
            
            if result.mismatched:
                stats["mismatches"] += 1
        
        # Calculate averages
        for project_id, stats in project_stats.items():
            if stats["total"] > 0:
                stats["avg_confidence"] = stats["total_confidence"] / stats["total"]
            del stats["total_confidence"]
        
        return dict(project_stats)

    def calculate_metrics(
        self, 
        results: List[AnalysisResult],
        ground_truth: List[dict] | None = None
    ) -> AggregatedMetrics:
        """Calculate overall metrics."""
        if not results:
            return AggregatedMetrics(
                total_records=0,
                health_distribution={},
                flagged_mismatches=0,
                rag_invocations=0,
                human_reviews_required=0,
                average_confidence=0.0,
                processing_time_total_ms=0.0,
            )
        
        health_dist = {"Good": 0, "Warning": 0, "Critical": 0}
        mismatches = 0
        rag_count = 0
        human_review_count = 0
        total_confidence = 0.0
        total_time = 0.0
        
        for result in results:
            health_dist[result.recommended_health] = health_dist.get(result.recommended_health, 0) + 1
            total_confidence += result.confidence
            total_time += result.processing_time_ms or 0
            
            if result.mismatched:
                mismatches += 1
            if result.requires_rag:
                rag_count += 1
            if result.requires_human_review:
                human_review_count += 1
        
        # Calculate accuracy metrics if ground truth is available
        accuracy_metrics = None
        if ground_truth:
            accuracy_metrics = self._calculate_accuracy(results, ground_truth)
        
        return AggregatedMetrics(
            total_records=len(results),
            health_distribution=health_dist,
            flagged_mismatches=mismatches,
            rag_invocations=rag_count,
            human_reviews_required=human_review_count,
            average_confidence=total_confidence / len(results),
            processing_time_total_ms=total_time,
            accuracy_metrics=accuracy_metrics,
        )

    def _calculate_accuracy(
        self, 
        results: List[AnalysisResult], 
        ground_truth: List[dict]
    ) -> dict:
        """Calculate accuracy, precision, recall, F1 for each class."""
        truth_map = {r.get("record_id", i): r.get("predefined_health") for i, r in enumerate(ground_truth)}
        
        # Initialize confusion matrix components for each class
        classes = ["Good", "Warning", "Critical"]
        metrics = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in classes}
        
        correct = 0
        total = 0
        
        for result in results:
            actual = truth_map.get(result.record_id)
            predicted = result.recommended_health
            
            if actual is None:
                continue
            
            total += 1
            if actual == predicted:
                correct += 1
            
            for c in classes:
                if actual == c and predicted == c:
                    metrics[c]["tp"] += 1
                elif actual != c and predicted == c:
                    metrics[c]["fp"] += 1
                elif actual == c and predicted != c:
                    metrics[c]["fn"] += 1
                else:
                    metrics[c]["tn"] += 1
        
        # Calculate per-class metrics
        class_metrics = {}
        for c in classes:
            m = metrics[c]
            precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
            recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[c] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        
        # Calculate macro averages
        macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(classes)
        macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(classes)
        macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(classes)
        
        return {
            "accuracy": round(correct / total, 4) if total > 0 else 0,
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": class_metrics,
            "total_evaluated": total,
        }

    def export_to_csv(self, results: List[AnalysisResult], file_path: str | None = None) -> str:
        """Export results to CSV file."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self._output_dir / f"analysis_results_{timestamp}.csv")
        
        fieldnames = [
            "record_id", "recommended_health", "confidence", "mismatched",
            "requires_rag", "requires_human_review", "indicators", "reasoning",
            "processing_time_ms"
        ]
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    "record_id": result.record_id,
                    "recommended_health": result.recommended_health,
                    "confidence": result.confidence,
                    "mismatched": result.mismatched,
                    "requires_rag": result.requires_rag,
                    "requires_human_review": result.requires_human_review,
                    "indicators": "; ".join(result.indicators),
                    "reasoning": result.reasoning,
                    "processing_time_ms": result.processing_time_ms,
                })
        
        logger.info(f"Exported {len(results)} results to {file_path}")
        return file_path

    def generate_charts(self, metrics: AggregatedMetrics) -> Dict[str, Any]:
        """Generate chart data for visualization."""
        # Health distribution pie chart data
        health_pie = {
            "type": "pie",
            "title": "Health Status Distribution",
            "labels": list(metrics.health_distribution.keys()),
            "values": list(metrics.health_distribution.values()),
            "colors": ["#28a745", "#ffc107", "#dc3545"],  # Green, Yellow, Red
        }
        
        # Metrics bar chart data
        metrics_bar = {
            "type": "bar",
            "title": "Analysis Metrics",
            "labels": ["Total Records", "Mismatches", "RAG Invocations", "Human Reviews"],
            "values": [
                metrics.total_records,
                metrics.flagged_mismatches,
                metrics.rag_invocations,
                metrics.human_reviews_required,
            ],
        }
        
        # Confidence gauge data
        confidence_gauge = {
            "type": "gauge",
            "title": "Average Confidence",
            "value": round(metrics.average_confidence * 100, 1),
            "min": 0,
            "max": 100,
            "thresholds": [60, 85, 100],
            "colors": ["#dc3545", "#ffc107", "#28a745"],
        }
        
        # Accuracy metrics if available
        accuracy_chart = None
        if metrics.accuracy_metrics:
            accuracy_chart = {
                "type": "bar",
                "title": "Classification Accuracy Metrics",
                "labels": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "values": [
                    metrics.accuracy_metrics["accuracy"] * 100,
                    metrics.accuracy_metrics["macro_precision"] * 100,
                    metrics.accuracy_metrics["macro_recall"] * 100,
                    metrics.accuracy_metrics["macro_f1"] * 100,
                ],
            }
        
        return {
            "health_distribution": health_pie,
            "metrics_overview": metrics_bar,
            "confidence": confidence_gauge,
            "accuracy": accuracy_chart,
        }
