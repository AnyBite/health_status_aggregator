"""
Dependency Injection Container

This module provides a simple DI container that wires together all the
application dependencies following the Clean Architecture pattern.
"""

from pathlib import Path

from app.domain import ILLMService, IRAGService, IFeedbackRepository, IReviewRepository, IReportingService, IDataIngestionService
from app.infrastructure import (
    Settings,
    get_settings,
    OpenAILLMService,
    MockLLMService,
    FAISSRAGService,
    DataIngestionService,
    ReportingService,
    InMemoryFeedbackRepository,
    InMemoryReviewRepository,
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


class Container:
    """
    Dependency Injection Container.
    
    Manages the lifecycle and wiring of all application dependencies.
    Following the Composition Root pattern, all dependencies are
    created and wired here.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        
        # Infrastructure layer - services
        self._llm_service: ILLMService | None = None
        self._rag_service: IRAGService | None = None
        self._feedback_repository: IFeedbackRepository | None = None
        self._review_repository: IReviewRepository | None = None
        self._reporting_service: IReportingService | None = None
        self._data_ingestion_service: IDataIngestionService | None = None
        
        # Application layer - use cases
        self._analyze_record_use_case: AnalyzeRecordUseCase | None = None
        self._batch_analyze_use_case: BatchAnalyzeUseCase | None = None
        self._submit_feedback_use_case: SubmitFeedbackUseCase | None = None
        self._get_pending_reviews_use_case: GetPendingReviewsUseCase | None = None
        self._get_review_details_use_case: GetReviewDetailsUseCase | None = None
        self._aggregate_health_use_case: AggregateHealthUseCase | None = None
        self._generate_report_use_case: GenerateReportUseCase | None = None
        self._load_data_use_case: LoadAndValidateDataUseCase | None = None

    @property
    def settings(self) -> Settings:
        """Get application settings."""
        return self._settings

    @property
    def llm_service(self) -> ILLMService:
        """Get or create LLM service instance."""
        if self._llm_service is None:
            if self._settings.use_mock_llm:
                self._llm_service = MockLLMService(self._settings)
            else:
                self._llm_service = OpenAILLMService(self._settings)
        return self._llm_service

    @property
    def rag_service(self) -> IRAGService:
        """Get or create RAG service instance."""
        if self._rag_service is None:
            self._rag_service = FAISSRAGService(
                self._settings,
                data_path=self._settings.data_path if self._settings.data_path.exists() else None
            )
        return self._rag_service

    @property
    def feedback_repository(self) -> IFeedbackRepository:
        """Get or create feedback repository instance."""
        if self._feedback_repository is None:
            self._feedback_repository = InMemoryFeedbackRepository()
        return self._feedback_repository

    @property
    def review_repository(self) -> IReviewRepository:
        """Get or create review repository instance."""
        if self._review_repository is None:
            self._review_repository = InMemoryReviewRepository()
        return self._review_repository

    @property
    def reporting_service(self) -> IReportingService:
        """Get or create reporting service instance."""
        if self._reporting_service is None:
            output_dir = Path(self._settings.data_path).parent / "reports"
            self._reporting_service = ReportingService(output_dir)
        return self._reporting_service

    @property
    def data_ingestion_service(self) -> IDataIngestionService:
        """Get or create data ingestion service instance."""
        if self._data_ingestion_service is None:
            self._data_ingestion_service = DataIngestionService()
        return self._data_ingestion_service

    @property
    def analyze_record_use_case(self) -> AnalyzeRecordUseCase:
        """Get or create AnalyzeRecordUseCase instance."""
        if self._analyze_record_use_case is None:
            self._analyze_record_use_case = AnalyzeRecordUseCase(
                llm_service=self.llm_service,
                rag_service=self.rag_service,
                review_repository=self.review_repository,
                confidence_threshold=self._settings.confidence_threshold,
            )
        return self._analyze_record_use_case

    @property
    def batch_analyze_use_case(self) -> BatchAnalyzeUseCase:
        """Get or create BatchAnalyzeUseCase instance."""
        if self._batch_analyze_use_case is None:
            self._batch_analyze_use_case = BatchAnalyzeUseCase(
                analyze_use_case=self.analyze_record_use_case,
                max_processing_time_seconds=1800,  # 30 minutes
            )
        return self._batch_analyze_use_case

    @property
    def submit_feedback_use_case(self) -> SubmitFeedbackUseCase:
        """Get or create SubmitFeedbackUseCase instance."""
        if self._submit_feedback_use_case is None:
            self._submit_feedback_use_case = SubmitFeedbackUseCase(
                feedback_repository=self.feedback_repository,
                review_repository=self.review_repository,
            )
        return self._submit_feedback_use_case

    @property
    def get_pending_reviews_use_case(self) -> GetPendingReviewsUseCase:
        """Get or create GetPendingReviewsUseCase instance."""
        if self._get_pending_reviews_use_case is None:
            self._get_pending_reviews_use_case = GetPendingReviewsUseCase(
                review_repository=self.review_repository,
            )
        return self._get_pending_reviews_use_case

    @property
    def get_review_details_use_case(self) -> GetReviewDetailsUseCase:
        """Get or create GetReviewDetailsUseCase instance."""
        if self._get_review_details_use_case is None:
            self._get_review_details_use_case = GetReviewDetailsUseCase(
                review_repository=self.review_repository,
            )
        return self._get_review_details_use_case

    @property
    def aggregate_health_use_case(self) -> AggregateHealthUseCase:
        """Get or create AggregateHealthUseCase instance."""
        if self._aggregate_health_use_case is None:
            self._aggregate_health_use_case = AggregateHealthUseCase(
                reporting_service=self.reporting_service,
            )
        return self._aggregate_health_use_case

    @property
    def generate_report_use_case(self) -> GenerateReportUseCase:
        """Get or create GenerateReportUseCase instance."""
        if self._generate_report_use_case is None:
            self._generate_report_use_case = GenerateReportUseCase(
                reporting_service=self.reporting_service,
                data_ingestion_service=self.data_ingestion_service,
            )
        return self._generate_report_use_case

    @property
    def load_data_use_case(self) -> LoadAndValidateDataUseCase:
        """Get or create LoadAndValidateDataUseCase instance."""
        if self._load_data_use_case is None:
            self._load_data_use_case = LoadAndValidateDataUseCase(
                data_ingestion_service=self.data_ingestion_service,
            )
        return self._load_data_use_case


def create_container(settings: Settings | None = None) -> Container:
    """Factory function to create a new container instance."""
    return Container(settings)
