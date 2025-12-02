from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class RecordDTO(BaseModel):
    """Data transfer object for health records."""
    record_id: Optional[int] = None
    account_id: str
    project_id: str
    predefined_health: str
    free_text_details: str
    timestamp: Optional[str] = None
    data_source: Optional[str] = None


class AnalysisResultDTO(BaseModel):
    """Data transfer object for analysis results."""
    record_id: int
    indicators: List[str]
    recommended_health: str
    confidence: float
    mismatched: bool
    reasoning: Optional[str] = None
    processing_time_ms: Optional[float] = None
    requires_rag: bool = False
    requires_human_review: bool = False


class RAGContextDTO(BaseModel):
    """Data transfer object for RAG context."""
    similar_cases: List[Dict[str, Any]]
    similarity_scores: List[float]
    retrieval_time_ms: float


class AnalyzeResponseDTO(BaseModel):
    """Response DTO for analyze endpoint."""
    result: AnalysisResultDTO
    similar_cases: Optional[List[str]] = None
    rag_context: Optional[RAGContextDTO] = None
    review_created: bool = False


class FeedbackRequestDTO(BaseModel):
    """Request DTO for feedback endpoint."""
    record_id: int
    human_decision: str
    reviewer_notes: Optional[str] = None


class FeedbackResponseDTO(BaseModel):
    """Response DTO for feedback endpoint."""
    status: str
    record_id: Optional[int] = None
    message: Optional[str] = None
    review_status: Optional[str] = None


class HealthStatusDTO(BaseModel):
    """DTO for health status message."""
    message: str


class BatchAnalyzeRequestDTO(BaseModel):
    """Request DTO for batch analysis."""
    file_path: Optional[str] = None
    records: Optional[List[RecordDTO]] = None


class BatchAnalyzeResponseDTO(BaseModel):
    """Response DTO for batch analysis."""
    processed: int
    failed: int
    rag_invocations: int
    human_reviews_created: int
    total_time_seconds: float
    average_time_per_record: float
    results: List[AnalysisResultDTO]
    errors: List[Dict[str, Any]]


class MetricsDTO(BaseModel):
    """DTO for aggregated metrics."""
    total_records: int
    health_distribution: Dict[str, int]
    flagged_mismatches: int
    rag_invocations: int
    human_reviews_required: int
    average_confidence: float
    processing_time_total_ms: float
    accuracy: Optional[Dict[str, Any]] = None


class ReportResponseDTO(BaseModel):
    """Response DTO for report generation."""
    metrics: MetricsDTO
    charts: Dict[str, Any]
    aggregations: Dict[str, Any]
    exports: Dict[str, Optional[str]]


class PendingReviewDTO(BaseModel):
    """DTO for pending review item."""
    record_id: int
    original_input: Dict[str, Any]
    confidence_score: float
    reason_for_review: str
    created_at: str
    similar_cases_count: int


class PendingReviewsResponseDTO(BaseModel):
    """Response DTO for pending reviews list."""
    count: int
    reviews: List[PendingReviewDTO]


class ReviewDetailDTO(BaseModel):
    """DTO for detailed review information."""
    status: str
    record_id: Optional[int] = None
    original_input: Optional[Dict[str, Any]] = None
    llm_analysis: Optional[Dict[str, Any]] = None
    rag_context: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    reason_for_review: Optional[str] = None
    created_at: Optional[str] = None
    message: Optional[str] = None


class ValidationResultDTO(BaseModel):
    """DTO for data validation results."""
    total_loaded: int
    valid_count: int
    invalid_count: int
    validation_rate: float
    invalid_records: List[Dict[str, Any]]
