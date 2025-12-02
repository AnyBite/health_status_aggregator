from typing import List, Optional
import logging
from datetime import datetime

from app.domain import IFeedbackRepository, IReviewRepository, Feedback, ReviewRequest


logger = logging.getLogger(__name__)


class InMemoryFeedbackRepository(IFeedbackRepository):
    """In-memory implementation of feedback repository."""

    def __init__(self):
        self._storage: List[Feedback] = []

    def save(self, feedback: Feedback) -> dict:
        """Save feedback entry to in-memory storage."""
        self._storage.append(feedback)
        logger.info(f"Feedback saved for record {feedback.record_id}")
        return {"status": "feedback stored", "record_id": feedback.record_id}

    def get_by_record_id(self, record_id: int) -> Optional[Feedback]:
        """Get feedback by record ID."""
        for feedback in self._storage:
            if feedback.record_id == record_id:
                return feedback
        return None

    def get_all(self) -> List[Feedback]:
        """Get all feedback entries."""
        return self._storage.copy()


class InMemoryReviewRepository(IReviewRepository):
    """In-memory implementation of review request repository."""

    def __init__(self):
        self._storage: List[ReviewRequest] = []

    def save_review_request(self, request: ReviewRequest) -> dict:
        """Save a review request."""
        # Check if already exists
        existing = self.get_review_by_record_id(request.record_id)
        if existing:
            # Update existing
            idx = next(i for i, r in enumerate(self._storage) if r.record_id == request.record_id)
            self._storage[idx] = request
            logger.info(f"Updated review request for record {request.record_id}")
        else:
            self._storage.append(request)
            logger.info(f"Created review request for record {request.record_id}")
        
        return {
            "status": "review request saved",
            "record_id": request.record_id,
            "reason": request.reason_for_review,
        }

    def get_pending_reviews(self) -> List[ReviewRequest]:
        """Get all pending review requests."""
        return [r for r in self._storage if r.status == "pending"]

    def get_review_by_record_id(self, record_id: int) -> Optional[ReviewRequest]:
        """Get review request by record ID."""
        for request in self._storage:
            if request.record_id == record_id:
                return request
        return None

    def update_review_status(
        self, 
        record_id: int, 
        status: str, 
        feedback: Optional[Feedback] = None
    ) -> dict:
        """Update review status after human decision."""
        request = self.get_review_by_record_id(record_id)
        if not request:
            return {"status": "error", "message": f"Review request not found for record {record_id}"}
        
        request.status = status
        logger.info(f"Review status updated for record {record_id}: {status}")
        
        return {
            "status": "updated",
            "record_id": record_id,
            "new_status": status,
        }

    def get_all_reviews(self) -> List[ReviewRequest]:
        """Get all review requests."""
        return self._storage.copy()

    def get_reviews_by_status(self, status: str) -> List[ReviewRequest]:
        """Get reviews by status."""
        return [r for r in self._storage if r.status == status]
