from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain.entities import Feedback


class IFeedbackRepository(ABC):
    """Interface for feedback persistence."""

    @abstractmethod
    def save(self, feedback: Feedback) -> dict:
        """Save feedback entry."""
        pass

    @abstractmethod
    def get_by_record_id(self, record_id: int) -> Optional[Feedback]:
        """Get feedback by record ID."""
        pass

    @abstractmethod
    def get_all(self) -> List[Feedback]:
        """Get all feedback entries."""
        pass
