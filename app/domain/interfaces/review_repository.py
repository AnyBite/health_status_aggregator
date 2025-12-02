from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain.entities import ReviewRequest, Feedback


class IReviewRepository(ABC):
    """Interface for human review request persistence."""

    @abstractmethod
    def save_review_request(self, request: ReviewRequest) -> dict:
        """Save a review request."""
        pass

    @abstractmethod
    def get_pending_reviews(self) -> List[ReviewRequest]:
        """Get all pending review requests."""
        pass

    @abstractmethod
    def get_review_by_record_id(self, record_id: int) -> Optional[ReviewRequest]:
        """Get review request by record ID."""
        pass

    @abstractmethod
    def update_review_status(self, record_id: int, status: str, feedback: Optional[Feedback] = None) -> dict:
        """Update review status after human decision."""
        pass
