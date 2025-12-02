from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class HealthStatus(str, Enum):
    """Health status enumeration."""
    GOOD = "Good"
    WARNING = "Warning"
    CRITICAL = "Critical"


@dataclass
class Record:
    """Domain entity representing a health record."""
    record_id: int
    account_id: str
    project_id: str
    predefined_health: str
    free_text_details: str
    timestamp: Optional[str] = None
    data_source: Optional[str] = None

    def is_valid_health_status(self) -> bool:
        """Check if predefined_health is a valid status."""
        return self.predefined_health in [s.value for s in HealthStatus]


@dataclass
class AnalysisResult:
    """Domain entity representing the result of health analysis."""
    record_id: int
    indicators: List[str]
    recommended_health: str
    confidence: float
    mismatched: bool
    reasoning: Optional[str] = None
    processing_time_ms: Optional[float] = None
    requires_rag: bool = False
    requires_human_review: bool = False


@dataclass
class RAGContext:
    """Context retrieved from RAG for reanalysis."""
    similar_cases: List[dict]
    similarity_scores: List[float]
    retrieval_time_ms: float


@dataclass
class EnhancedAnalysisResult(AnalysisResult):
    """Analysis result enhanced with RAG context."""
    rag_context: Optional[RAGContext] = None
    reanalysis_reasoning: Optional[str] = None
    final_recommendation: Optional[str] = None
    final_confidence: Optional[float] = None


@dataclass
class Feedback:
    """Domain entity representing human feedback."""
    record_id: int
    human_decision: str
    reviewer_notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReviewRequest:
    """Request for human review of ambiguous record."""
    record_id: int
    original_input: dict
    llm_analysis: AnalysisResult
    rag_context: Optional[RAGContext]
    confidence_score: float
    reason_for_review: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending, reviewed, resolved


@dataclass
class ValidationResult:
    """Result of record validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass 
class AggregatedMetrics:
    """Aggregated metrics for reporting."""
    total_records: int
    health_distribution: dict
    flagged_mismatches: int
    rag_invocations: int
    human_reviews_required: int
    average_confidence: float
    processing_time_total_ms: float
    accuracy_metrics: Optional[dict] = None

