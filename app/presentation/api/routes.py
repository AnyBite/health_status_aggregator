from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional

from app.presentation.schemas import (
    RecordDTO,
    AnalyzeResponseDTO,
    AnalysisResultDTO,
    RAGContextDTO,
    FeedbackRequestDTO,
    FeedbackResponseDTO,
    HealthStatusDTO,
    BatchAnalyzeRequestDTO,
    BatchAnalyzeResponseDTO,
    ReportResponseDTO,
    MetricsDTO,
    PendingReviewsResponseDTO,
    PendingReviewDTO,
    ReviewDetailDTO,
    ValidationResultDTO,
)
from app.application import (
    AnalyzeRecordUseCase,
    BatchAnalyzeUseCase,
    SubmitFeedbackUseCase,
    GetPendingReviewsUseCase,
    GetReviewDetailsUseCase,
    AggregateHealthUseCase,
    GenerateReportUseCase,
    LoadAndValidateDataUseCase,
)
from app.domain import Record
from app.presentation.dependencies import (
    get_analyze_use_case,
    get_batch_analyze_use_case,
    get_feedback_use_case,
    get_pending_reviews_use_case,
    get_review_details_use_case,
    get_aggregate_use_case,
    get_report_use_case,
    get_load_data_use_case,
)


router = APIRouter()


@router.get("/", response_model=HealthStatusDTO)
def root() -> HealthStatusDTO:
    """Health check endpoint."""
    return HealthStatusDTO(message="Health Status Aggregator Running")


@router.post("/analyze/", response_model=AnalyzeResponseDTO)
def analyze(
    rec: RecordDTO,
    use_case: AnalyzeRecordUseCase = Depends(get_analyze_use_case)
) -> AnalyzeResponseDTO:
    """Analyze a single health record and return recommendations."""
    domain_record = Record(
        record_id=rec.record_id or 0,
        account_id=rec.account_id,
        project_id=rec.project_id,
        predefined_health=rec.predefined_health,
        free_text_details=rec.free_text_details,
        timestamp=rec.timestamp,
        data_source=rec.data_source,
    )
    
    result = use_case.execute(domain_record)
    
    rag_context = None
    if result.rag_context:
        rag_context = RAGContextDTO(
            similar_cases=result.rag_context.similar_cases,
            similarity_scores=result.rag_context.similarity_scores,
            retrieval_time_ms=result.rag_context.retrieval_time_ms,
        )
    
    return AnalyzeResponseDTO(
        result=AnalysisResultDTO(
            record_id=result.result.record_id,
            indicators=result.result.indicators,
            recommended_health=result.result.recommended_health,
            confidence=result.result.confidence,
            mismatched=result.result.mismatched,
            reasoning=result.result.reasoning,
            processing_time_ms=result.result.processing_time_ms,
            requires_rag=result.result.requires_rag,
            requires_human_review=result.result.requires_human_review,
        ),
        similar_cases=result.similar_cases,
        rag_context=rag_context,
        review_created=result.review_created,
    )


@router.post("/analyze/batch/", response_model=BatchAnalyzeResponseDTO)
def analyze_batch(
    request: BatchAnalyzeRequestDTO,
    use_case: BatchAnalyzeUseCase = Depends(get_batch_analyze_use_case),
    load_use_case: LoadAndValidateDataUseCase = Depends(get_load_data_use_case),
) -> BatchAnalyzeResponseDTO:
    """Analyze multiple health records in batch."""
    records = []
    
    if request.file_path:
        # Load from file
        load_result = load_use_case.execute(request.file_path)
        if "error" in load_result:
            raise HTTPException(status_code=400, detail=load_result["error"])
        records = load_result["valid_records"]
    elif request.records:
        # Use provided records
        records = [
            Record(
                record_id=rec.record_id or idx,
                account_id=rec.account_id,
                project_id=rec.project_id,
                predefined_health=rec.predefined_health,
                free_text_details=rec.free_text_details,
                timestamp=rec.timestamp,
                data_source=rec.data_source,
            )
            for idx, rec in enumerate(request.records)
        ]
    else:
        raise HTTPException(status_code=400, detail="Provide either file_path or records")
    
    batch_result = use_case.execute(records)
    
    return BatchAnalyzeResponseDTO(
        processed=batch_result["metrics"]["processed"],
        failed=batch_result["metrics"]["failed"],
        rag_invocations=batch_result["metrics"]["rag_invocations"],
        human_reviews_created=batch_result["metrics"]["human_reviews_created"],
        total_time_seconds=batch_result["metrics"]["total_time_seconds"],
        average_time_per_record=batch_result["metrics"]["average_time_per_record"],
        results=[
            AnalysisResultDTO(
                record_id=r.result.record_id,
                indicators=r.result.indicators,
                recommended_health=r.result.recommended_health,
                confidence=r.result.confidence,
                mismatched=r.result.mismatched,
                reasoning=r.result.reasoning,
                processing_time_ms=r.result.processing_time_ms,
                requires_rag=r.result.requires_rag,
                requires_human_review=r.result.requires_human_review,
            )
            for r in batch_result["results"]
        ],
        errors=batch_result["errors"],
    )


@router.post("/feedback/", response_model=FeedbackResponseDTO)
def feedback(
    entry: FeedbackRequestDTO,
    use_case: SubmitFeedbackUseCase = Depends(get_feedback_use_case)
) -> FeedbackResponseDTO:
    """Submit human feedback for a record."""
    result = use_case.execute(
        entry.record_id,
        entry.human_decision,
        entry.reviewer_notes,
    )
    return FeedbackResponseDTO(
        status=result.get("status", "unknown"),
        record_id=result.get("record_id"),
        message=result.get("message"),
        review_status=result.get("review_status"),
    )


@router.get("/reviews/pending/", response_model=PendingReviewsResponseDTO)
def get_pending_reviews(
    use_case: GetPendingReviewsUseCase = Depends(get_pending_reviews_use_case)
) -> PendingReviewsResponseDTO:
    """Get all pending human review requests."""
    result = use_case.execute()
    return PendingReviewsResponseDTO(
        count=result["count"],
        reviews=[
            PendingReviewDTO(**review)
            for review in result["reviews"]
        ],
    )


@router.get("/reviews/{record_id}/", response_model=ReviewDetailDTO)
def get_review_details(
    record_id: int,
    use_case: GetReviewDetailsUseCase = Depends(get_review_details_use_case)
) -> ReviewDetailDTO:
    """Get detailed review information for a specific record."""
    result = use_case.execute(record_id)
    return ReviewDetailDTO(**result)


@router.post("/reports/generate/", response_model=ReportResponseDTO)
def generate_report(
    file_path: Optional[str] = None,
    export_csv: bool = True,
    use_case: GenerateReportUseCase = Depends(get_report_use_case),
    load_use_case: LoadAndValidateDataUseCase = Depends(get_load_data_use_case),
    analyze_use_case: BatchAnalyzeUseCase = Depends(get_batch_analyze_use_case),
) -> ReportResponseDTO:
    """Generate a comprehensive report with metrics and visualizations."""
    # Use default data path if not provided
    if not file_path:
        from app.infrastructure.config import get_settings
        settings = get_settings()
        file_path = str(settings.data_path)
    
    # Load and validate data
    load_result = load_use_case.execute(file_path)
    if "error" in load_result:
        raise HTTPException(status_code=400, detail=load_result["error"])
    
    records = load_result["valid_records"]
    raw_records = [
        {
            "record_id": r.record_id,
            "account_id": r.account_id,
            "project_id": r.project_id,
            "predefined_health": r.predefined_health,
            "free_text_details": r.free_text_details,
        }
        for r in records
    ]
    
    # Analyze all records
    batch_result = analyze_use_case.execute(records)
    results = [r.result for r in batch_result["results"]]
    
    # Generate report
    report = use_case.execute(results, raw_records, export_csv)
    
    return ReportResponseDTO(
        metrics=MetricsDTO(**report["metrics"]),
        charts=report["charts"],
        aggregations=report["aggregations"],
        exports=report["exports"],
    )


@router.post("/data/validate/", response_model=ValidationResultDTO)
def validate_data(
    file_path: str,
    use_case: LoadAndValidateDataUseCase = Depends(get_load_data_use_case)
) -> ValidationResultDTO:
    """Validate a dataset file and return validation results."""
    result = use_case.execute(file_path)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return ValidationResultDTO(
        total_loaded=result["total_loaded"],
        valid_count=result["valid_count"],
        invalid_count=result["invalid_count"],
        validation_rate=result["validation_rate"],
        invalid_records=result["invalid_records"],
    )


@router.get("/aggregations/by-account/")
def get_aggregations_by_account(
    file_path: Optional[str] = None,
    use_case: AggregateHealthUseCase = Depends(get_aggregate_use_case),
    load_use_case: LoadAndValidateDataUseCase = Depends(get_load_data_use_case),
    analyze_use_case: BatchAnalyzeUseCase = Depends(get_batch_analyze_use_case),
):
    """Get health statistics aggregated by account."""
    if not file_path:
        from app.infrastructure.config import get_settings
        settings = get_settings()
        file_path = str(settings.data_path)
    
    load_result = load_use_case.execute(file_path)
    if "error" in load_result:
        raise HTTPException(status_code=400, detail=load_result["error"])
    
    records = load_result["valid_records"]
    raw_records = [
        {
            "record_id": r.record_id,
            "account_id": r.account_id,
            "project_id": r.project_id,
            "predefined_health": r.predefined_health,
        }
        for r in records
    ]
    
    batch_result = analyze_use_case.execute(records)
    results = [r.result for r in batch_result["results"]]
    
    aggregation = use_case.execute(results, raw_records)
    return aggregation
