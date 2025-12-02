import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from app.domain import (
    ILLMService,
    IRAGService,
    IReviewRepository,
    Record,
    AnalysisResult,
    EnhancedAnalysisResult,
    RAGContext,
    ReviewRequest,
)


logger = logging.getLogger(__name__)


@dataclass
class AnalyzeRecordResult:
    """Result of the analyze record use case."""
    result: AnalysisResult
    similar_cases: Optional[List[str]] = None
    rag_context: Optional[RAGContext] = None
    review_created: bool = False


class AnalyzeRecordUseCase:
    """Use case for analyzing health records with full RAG and review workflow."""

    def __init__(
        self,
        llm_service: ILLMService,
        rag_service: IRAGService,
        review_repository: Optional[IReviewRepository] = None,
        confidence_threshold: float = 0.85,
    ):
        self._llm_service = llm_service
        self._rag_service = rag_service
        self._review_repository = review_repository
        self._confidence_threshold = confidence_threshold

    def execute(self, record: Record) -> AnalyzeRecordResult:
        """
        Analyze a health record with RAG augmentation when needed.
        
        Flow:
        1. Initial LLM analysis
        2. If confidence < 85% or ambiguous -> RAG retrieval
        3. Reanalysis with RAG context
        4. If still ambiguous -> create human review request
        """
        # Step 1: Initial LLM analysis
        initial_result = self._llm_service.analyze_record(record)
        
        logger.info(
            f"Record {record.record_id}: Initial analysis - "
            f"Recommended: {initial_result.recommended_health}, "
            f"Confidence: {initial_result.confidence:.2f}, "
            f"Mismatched: {initial_result.mismatched}"
        )
        
        # Step 2: Check if RAG is needed
        if not initial_result.requires_rag:
            return AnalyzeRecordResult(
                result=initial_result,
                similar_cases=None,
                rag_context=None,
                review_created=False,
            )
        
        # Step 3: RAG retrieval
        logger.info(f"Record {record.record_id}: Invoking RAG (confidence: {initial_result.confidence:.2f})")
        
        rag_context = self._rag_service.retrieve_similar_with_context(
            record.free_text_details,
            top_k=5
        )
        
        similar_texts = [case.get("free_text_details", "") for case in rag_context.similar_cases]
        
        # Step 4: Reanalysis with RAG context
        if rag_context.similar_cases:
            refined_result = self._llm_service.reanalyze_with_context(
                record,
                initial_result,
                rag_context.similar_cases
            )
            
            logger.info(
                f"Record {record.record_id}: RAG reanalysis - "
                f"Recommended: {refined_result.recommended_health}, "
                f"Confidence: {refined_result.confidence:.2f}"
            )
        else:
            refined_result = initial_result
            refined_result.requires_human_review = True
        
        # Step 5: Create human review request if still ambiguous
        review_created = False
        if refined_result.requires_human_review and self._review_repository:
            review_request = ReviewRequest(
                record_id=record.record_id,
                original_input={
                    "account_id": record.account_id,
                    "project_id": record.project_id,
                    "predefined_health": record.predefined_health,
                    "free_text_details": record.free_text_details,
                },
                llm_analysis=refined_result,
                rag_context=rag_context if rag_context.similar_cases else None,
                confidence_score=refined_result.confidence,
                reason_for_review=self._get_review_reason(refined_result, initial_result),
            )
            self._review_repository.save_review_request(review_request)
            review_created = True
            
            logger.info(f"Record {record.record_id}: Human review request created")
        
        return AnalyzeRecordResult(
            result=refined_result,
            similar_cases=similar_texts if similar_texts else None,
            rag_context=rag_context,
            review_created=review_created,
        )

    def _get_review_reason(self, final_result: AnalysisResult, initial_result: AnalysisResult) -> str:
        """Determine the reason for human review."""
        reasons = []
        
        if final_result.confidence < self._confidence_threshold:
            reasons.append(f"Low confidence ({final_result.confidence:.2f} < {self._confidence_threshold})")
        
        if final_result.mismatched:
            reasons.append("Recommended status differs from predefined status")
        
        if initial_result.requires_rag and final_result.requires_human_review:
            reasons.append("RAG reanalysis did not resolve ambiguity")
        
        return "; ".join(reasons) if reasons else "Manual review required"


class BatchAnalyzeUseCase:
    """Use case for batch analysis of multiple records."""

    def __init__(
        self,
        analyze_use_case: AnalyzeRecordUseCase,
        max_processing_time_seconds: int = 1800,  # 30 minutes
    ):
        self._analyze_use_case = analyze_use_case
        self._max_processing_time = max_processing_time_seconds

    def execute(self, records: List[Record]) -> dict:
        """
        Process multiple records with timing and metrics.
        
        Returns:
            Dictionary with results and processing metrics.
        """
        start_time = time.time()
        results = []
        errors = []
        rag_invocations = 0
        human_reviews = 0
        
        for i, record in enumerate(records):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > self._max_processing_time:
                logger.warning(
                    f"Batch processing stopped at record {i}/{len(records)} "
                    f"due to time limit ({elapsed:.2f}s > {self._max_processing_time}s)"
                )
                break
            
            try:
                result = self._analyze_use_case.execute(record)
                results.append(result)
                
                if result.rag_context:
                    rag_invocations += 1
                if result.review_created:
                    human_reviews += 1
                    
            except Exception as e:
                logger.error(f"Error processing record {record.record_id}: {e}")
                errors.append({
                    "record_id": record.record_id,
                    "error": str(e),
                })
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "errors": errors,
            "metrics": {
                "total_records": len(records),
                "processed": len(results),
                "failed": len(errors),
                "rag_invocations": rag_invocations,
                "human_reviews_created": human_reviews,
                "total_time_seconds": round(total_time, 2),
                "average_time_per_record": round(total_time / max(len(results), 1), 3),
            }
        }
