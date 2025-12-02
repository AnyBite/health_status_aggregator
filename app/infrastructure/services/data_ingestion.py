import json
import csv
import logging
from typing import List, Optional
from pathlib import Path

from app.domain import IDataIngestionService, Record, ValidationResult, HealthStatus


logger = logging.getLogger(__name__)


class DataIngestionService(IDataIngestionService):
    """Implementation of data ingestion and validation service."""

    REQUIRED_FIELDS = ["account_id", "project_id", "predefined_health", "free_text_details"]
    VALID_HEALTH_STATUSES = [s.value for s in HealthStatus]

    def load_from_json(self, file_path: str) -> List[dict]:
        """Load records from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of records")
        
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data

    def load_from_csv(self, file_path: str) -> List[dict]:
        """Load records from CSV file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        records = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert record_id to int if present
                if "record_id" in row and row["record_id"]:
                    try:
                        row["record_id"] = int(row["record_id"])
                    except ValueError:
                        pass
                records.append(dict(row))
        
        logger.info(f"Loaded {len(records)} records from {file_path}")
        return records

    def validate_record(self, record_data: dict) -> ValidationResult:
        """Validate a single record against the schema."""
        errors = []
        warnings = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in record_data:
                errors.append(f"Missing required field: {field}")
            elif not record_data[field]:
                errors.append(f"Empty required field: {field}")

        # Validate account_id format
        if "account_id" in record_data:
            acc_id = record_data["account_id"]
            if not (isinstance(acc_id, str) and acc_id.startswith("ACC")):
                warnings.append(f"account_id '{acc_id}' does not follow expected pattern 'ACC###'")

        # Validate project_id format
        if "project_id" in record_data:
            prj_id = record_data["project_id"]
            if not (isinstance(prj_id, str) and prj_id.startswith("PRJ")):
                warnings.append(f"project_id '{prj_id}' does not follow expected pattern 'PRJ###'")

        # Validate predefined_health
        if "predefined_health" in record_data:
            health = record_data["predefined_health"]
            if health not in self.VALID_HEALTH_STATUSES:
                errors.append(
                    f"Invalid predefined_health '{health}'. Must be one of: {self.VALID_HEALTH_STATUSES}"
                )

        # Validate free_text_details
        if "free_text_details" in record_data:
            text = record_data["free_text_details"]
            if len(text) < 5:
                warnings.append("free_text_details is very short (< 5 chars)")
            if len(text) > 5000:
                warnings.append("free_text_details is very long (> 5000 chars)")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def validate_and_convert(self, records_data: List[dict]) -> tuple[List[Record], List[dict]]:
        """Validate records and convert valid ones to domain entities."""
        valid_records = []
        invalid_records = []

        for idx, record_data in enumerate(records_data):
            # Ensure record_id exists
            if "record_id" not in record_data:
                record_data["record_id"] = idx + 1

            validation_result = self.validate_record(record_data)

            if validation_result.is_valid:
                try:
                    record = Record(
                        record_id=record_data.get("record_id", idx + 1),
                        account_id=record_data["account_id"],
                        project_id=record_data["project_id"],
                        predefined_health=record_data["predefined_health"],
                        free_text_details=record_data["free_text_details"],
                        timestamp=record_data.get("timestamp"),
                        data_source=record_data.get("data_source"),
                    )
                    valid_records.append(record)
                    
                    if validation_result.warnings:
                        logger.warning(
                            f"Record {record.record_id} has warnings: {validation_result.warnings}"
                        )
                except Exception as e:
                    invalid_records.append({
                        "record_data": record_data,
                        "errors": [str(e)],
                    })
            else:
                invalid_records.append({
                    "record_data": record_data,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                })
                logger.error(
                    f"Invalid record at index {idx}: {validation_result.errors}"
                )

        logger.info(
            f"Validation complete: {len(valid_records)} valid, {len(invalid_records)} invalid"
        )
        return valid_records, invalid_records
