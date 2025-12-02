from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain.entities import Record, ValidationResult


class IDataIngestionService(ABC):
    """Interface for data ingestion and validation service."""

    @abstractmethod
    def load_from_json(self, file_path: str) -> List[dict]:
        """Load records from JSON file."""
        pass

    @abstractmethod
    def load_from_csv(self, file_path: str) -> List[dict]:
        """Load records from CSV file."""
        pass

    @abstractmethod
    def validate_record(self, record_data: dict) -> ValidationResult:
        """Validate a single record against the schema."""
        pass

    @abstractmethod
    def validate_and_convert(self, records_data: List[dict]) -> tuple[List[Record], List[dict]]:
        """Validate records and convert valid ones to domain entities.
        
        Returns:
            Tuple of (valid_records, invalid_records_with_errors)
        """
        pass
