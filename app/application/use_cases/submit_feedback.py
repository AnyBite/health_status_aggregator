import logging
from typing import Optional

from app.domain import IFeedbackRepository, IReviewRepository, Feedback


logger = logging.getLogger(__name__)


class SubmitFeedbackUseCase:
    """Use case for submitting human feedback."""

    def __init__(
        self,
        feedback_repository: IFeedbackRepository,
        review_repository: Optional[IReviewRepository] = None,
    ):
        self._feedback_repository = feedback_repository
        self._review_repository = review_repository

    def execute(
        self,
        record_id: int,
        human_decision: str,
        reviewer_notes: Optional[str] = None,
    ) -> dict:
        """
        Submit human feedback for a record.
        
        Args:
            record_id: The ID of the record being reviewed.
            human_decision: The human's decision about the health status.
            reviewer_notes: Optional notes from the reviewer.
            
        Returns:
            Status dictionary with feedback confirmation.
        """
        # Validate human decision
        valid_decisions = ["Good", "Warning", "Critical"]
        if human_decision not in valid_decisions:
            return {
                "status": "error",
                "message": f"Invalid decision. Must be one of: {valid_decisions}",
            }
        
        # Create and save feedback
        feedback = Feedback(
            record_id=record_id,
            human_decision=human_decision,
            reviewer_notes=reviewer_notes,
        )
        result = self._feedback_repository.save(feedback)
        
        # Update review status if review repository is available
        if self._review_repository:
            review = self._review_repository.get_review_by_record_id(record_id)
            if review:
                self._review_repository.update_review_status(
                    record_id=record_id,
                    status="resolved",
                    feedback=feedback,
                )
                result["review_status"] = "resolved"
        
        logger.info(
            f"Feedback submitted for record {record_id}: {human_decision}"
        )
        
        return result


class GetPendingReviewsUseCase:
    """Use case for retrieving pending human reviews."""

    def __init__(self, review_repository: IReviewRepository):
        self._review_repository = review_repository

    def execute(self) -> dict:
        """
        Get all pending review requests.
        
        Returns:
            Dictionary with pending reviews and count.
        """
        pending = self._review_repository.get_pending_reviews()
        
        return {
            "count": len(pending),
            "reviews": [
                {
                    "record_id": r.record_id,
                    "original_input": r.original_input,
                    "confidence_score": r.confidence_score,
                    "reason_for_review": r.reason_for_review,
                    "created_at": r.created_at,
                    "similar_cases_count": len(r.rag_context.similar_cases) if r.rag_context else 0,
                }
                for r in pending
            ],
        }


class GetReviewDetailsUseCase:
    """Use case for getting detailed review information."""

    def __init__(self, review_repository: IReviewRepository):
        self._review_repository = review_repository

    def execute(self, record_id: int) -> dict:
        """
        Get detailed review information for a specific record.
        
        Returns:
            Dictionary with full review details or error.
        """
        review = self._review_repository.get_review_by_record_id(record_id)
        
        if not review:
            return {
                "status": "not_found",
                "message": f"No review found for record {record_id}",
            }
        
        return {
            "status": "found",
            "record_id": review.record_id,
            "original_input": review.original_input,
            "llm_analysis": {
                "recommended_health": review.llm_analysis.recommended_health,
                "confidence": review.llm_analysis.confidence,
                "indicators": review.llm_analysis.indicators,
                "reasoning": review.llm_analysis.reasoning,
            },
            "rag_context": {
                "similar_cases": review.rag_context.similar_cases if review.rag_context else [],
                "similarity_scores": review.rag_context.similarity_scores if review.rag_context else [],
            } if review.rag_context else None,
            "confidence_score": review.confidence_score,
            "reason_for_review": review.reason_for_review,
            "created_at": review.created_at,
            "status": review.status,
        }
