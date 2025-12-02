from typing import Generator

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
from app.container import Container


_container: Container | None = None


def set_container(container: Container) -> None:
    """Set the global container for dependency injection."""
    global _container
    _container = container


def get_container() -> Container:
    """Get the global container."""
    if _container is None:
        raise RuntimeError("Container not initialized. Call set_container() first.")
    return _container


def get_analyze_use_case() -> Generator[AnalyzeRecordUseCase, None, None]:
    """Dependency provider for AnalyzeRecordUseCase."""
    container = get_container()
    yield container.analyze_record_use_case


def get_batch_analyze_use_case() -> Generator[BatchAnalyzeUseCase, None, None]:
    """Dependency provider for BatchAnalyzeUseCase."""
    container = get_container()
    yield container.batch_analyze_use_case


def get_feedback_use_case() -> Generator[SubmitFeedbackUseCase, None, None]:
    """Dependency provider for SubmitFeedbackUseCase."""
    container = get_container()
    yield container.submit_feedback_use_case


def get_pending_reviews_use_case() -> Generator[GetPendingReviewsUseCase, None, None]:
    """Dependency provider for GetPendingReviewsUseCase."""
    container = get_container()
    yield container.get_pending_reviews_use_case


def get_review_details_use_case() -> Generator[GetReviewDetailsUseCase, None, None]:
    """Dependency provider for GetReviewDetailsUseCase."""
    container = get_container()
    yield container.get_review_details_use_case


def get_aggregate_use_case() -> Generator[AggregateHealthUseCase, None, None]:
    """Dependency provider for AggregateHealthUseCase."""
    container = get_container()
    yield container.aggregate_health_use_case


def get_report_use_case() -> Generator[GenerateReportUseCase, None, None]:
    """Dependency provider for GenerateReportUseCase."""
    container = get_container()
    yield container.generate_report_use_case


def get_load_data_use_case() -> Generator[LoadAndValidateDataUseCase, None, None]:
    """Dependency provider for LoadAndValidateDataUseCase."""
    container = get_container()
    yield container.load_data_use_case
