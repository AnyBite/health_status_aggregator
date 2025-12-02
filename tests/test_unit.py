"""
Unit tests for Health Status Aggregator components.

These tests verify BEHAVIOR, not just execution. Each test documents:
1. What scenario is being tested
2. What the expected behavior is
3. Why this behavior is correct

Run with: pytest tests/test_unit.py -v
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Import domain entities
from app.domain.entities import Record, AnalysisResult
from app.domain.interfaces import ILLMService, IRAGService

# Import services
from app.infrastructure.services.llm_service import MockLLMService
from app.infrastructure.services.rag_service import FAISSRAGService
from app.infrastructure.config import Settings

# Import use cases
from app.application.use_cases.analyze_record import AnalyzeRecordUseCase, AnalyzeRecordResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def settings():
    """Create settings with test defaults."""
    return Settings(
        openai_api_key="test-key",
        confidence_threshold=0.85,
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
    )


@pytest.fixture
def mock_llm(settings):
    """Create MockLLMService for testing."""
    return MockLLMService(settings)


@pytest.fixture
def rag_service(settings):
    """Create RAG service without loading data."""
    return FAISSRAGService(settings, data_path=None)


# =============================================================================
# MockLLMService UNIT TESTS
# =============================================================================

class TestMockLLMServiceClassification:
    """Tests for LLM classification logic."""

    def test_clear_good_indicators_returns_good(self, mock_llm):
        """
        SCENARIO: Text contains strong positive indicators
        EXPECTED: Classification should be 'Good' with high confidence
        REASONING: Clear positive signals like 'on schedule' and 'under budget'
                   are definitive Good indicators with weight 3.0 each
        """
        record = Record(
            record_id=1,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Good",
            free_text_details="Project on schedule and under budget. Team morale is high.",
        )
        
        result = mock_llm.analyze_record(record)
        
        assert result.recommended_health == "Good"
        assert result.confidence >= 0.85, "Clear Good indicators should have high confidence"
        assert not result.mismatched, "Should match predefined Good status"

    def test_clear_critical_indicators_returns_critical(self, mock_llm):
        """
        SCENARIO: Text contains strong negative indicators
        EXPECTED: Classification should be 'Critical' regardless of predefined status
        REASONING: Risk-averse classification - critical indicators take priority
                   'budget overrun' (weight 3.5) alone exceeds threshold 3.0
        """
        record = Record(
            record_id=2,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Good",  # Intentional mismatch
            free_text_details="Major budget overrun. Project at risk of cancellation.",
        )
        
        result = mock_llm.analyze_record(record)
        
        assert result.recommended_health == "Critical"
        assert result.mismatched, "Should detect mismatch with predefined Good"
        assert "[X]" in str(result.indicators), "Should have critical indicator markers"

    def test_warning_indicators_returns_warning(self, mock_llm):
        """
        SCENARIO: Text contains moderate warning indicators
        EXPECTED: Classification should be 'Warning'
        REASONING: 'slight delay' (2.5) + 'resource constraints' (2.0) = 4.5
                   Exceeds warning threshold of 2.0, no critical indicators
        """
        record = Record(
            record_id=3,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Warning",
            free_text_details="Experiencing slight delays due to resource constraints.",
        )
        
        result = mock_llm.analyze_record(record)
        
        assert result.recommended_health == "Warning"
        assert result.confidence >= 0.70

    def test_no_indicators_uses_predefined_with_low_confidence(self, mock_llm):
        """
        SCENARIO: Text contains no recognized indicators
        EXPECTED: Falls back to predefined status with low confidence
        REASONING: When no keywords match, we can't determine status from text
                   Low confidence (0.60) ensures this gets flagged for review
        """
        record = Record(
            record_id=4,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Good",
            free_text_details="Lorem ipsum dolor sit amet.",
        )
        
        result = mock_llm.analyze_record(record)
        
        assert result.recommended_health == "Good", "Should use predefined"
        assert result.confidence == 0.60, "Should have low confidence"
        assert "No clear indicators" in str(result.indicators)

    def test_ambiguous_signals_caps_confidence(self, mock_llm):
        """
        SCENARIO: Text contains conflicting good AND critical signals
        EXPECTED: Confidence should be capped at 0.75
        REASONING: Conflicting signals indicate unreliable classification
                   Example: "excellent progress" (good) + "budget overrun" (critical)
        """
        record = Record(
            record_id=5,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Warning",
            free_text_details="Excellent client satisfaction but major budget overrun.",
        )
        
        result = mock_llm.analyze_record(record)
        
        assert result.confidence <= 0.75, "Ambiguous signals should cap confidence"
        assert "Ambiguity" in str(result.indicators)

    def test_critical_takes_priority_over_good(self, mock_llm):
        """
        SCENARIO: Text has both strong good AND critical indicators
        EXPECTED: Critical should win (risk-averse)
        REASONING: Safety first - even with positive signals, critical issues
                   need attention. This is intentional risk-averse design.
        """
        record = Record(
            record_id=6,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Good",
            free_text_details="On schedule and on budget but at risk of cancellation.",
        )
        
        result = mock_llm.analyze_record(record)
        
        # Critical takes priority due to weight 4.0 for "at risk of cancellation"
        assert result.recommended_health == "Critical"


class TestMockLLMServiceEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text_uses_predefined(self, mock_llm):
        """Empty text should use predefined status with low confidence."""
        record = Record(
            record_id=7,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Warning",
            free_text_details="",
        )
        
        result = mock_llm.analyze_record(record)
        
        assert result.recommended_health == "Warning"
        assert result.confidence == 0.60

    def test_case_insensitive_matching(self, mock_llm):
        """Keywords should match regardless of case."""
        record = Record(
            record_id=8,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Good",
            free_text_details="PROJECT ON SCHEDULE AND UNDER BUDGET",
        )
        
        result = mock_llm.analyze_record(record)
        
        assert result.recommended_health == "Good"
        assert result.confidence >= 0.85

    def test_requires_rag_when_low_confidence(self, mock_llm):
        """Low confidence results should flag for RAG analysis."""
        record = Record(
            record_id=9,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Good",
            free_text_details="Some concerns noted.",  # Weak warning indicator
        )
        
        result = mock_llm.analyze_record(record)
        
        # With only "concern" (weight 1.5), confidence should be low
        assert result.requires_rag or result.confidence < 0.85

    def test_requires_rag_when_mismatched(self, mock_llm):
        """Mismatched results should flag for RAG analysis."""
        record = Record(
            record_id=10,
            account_id="ACC001",
            project_id="PRJ001",
            predefined_health="Good",  # Says Good
            free_text_details="Budget overrun is severe.",  # But text is Critical
        )
        
        result = mock_llm.analyze_record(record)
        
        assert result.mismatched, "Should detect mismatch"
        assert result.requires_rag, "Mismatched records should require RAG"


# =============================================================================
# RAG SERVICE UNIT TESTS
# =============================================================================

class TestRAGServiceSimilarityConversion:
    """Tests for the L2 distance to similarity conversion."""

    def test_zero_distance_gives_similarity_one(self, rag_service):
        """Identical vectors (distance 0) should have similarity 1.0."""
        similarity = rag_service._l2_distance_to_similarity(0.0)
        assert similarity == 1.0

    def test_small_distance_gives_high_similarity(self, rag_service):
        """Small distances should give high similarity scores."""
        # Squared L2 distance of 1.0 -> sqrt(1.0) = 1.0 -> 1/(1+1) = 0.5
        similarity = rag_service._l2_distance_to_similarity(1.0)
        assert similarity == 0.5

    def test_large_distance_gives_low_similarity(self, rag_service):
        """Large distances should give low similarity scores."""
        # Squared L2 distance of 100 -> sqrt(100) = 10 -> 1/(1+10) ≈ 0.09
        similarity = rag_service._l2_distance_to_similarity(100.0)
        assert 0.08 < similarity < 0.10

    def test_similarity_monotonically_decreases(self, rag_service):
        """Similarity should decrease as distance increases."""
        distances = [0.0, 1.0, 4.0, 16.0, 100.0]
        similarities = [rag_service._l2_distance_to_similarity(d) for d in distances]
        
        for i in range(len(similarities) - 1):
            assert similarities[i] > similarities[i + 1], \
                f"Similarity should decrease: {similarities[i]} > {similarities[i+1]}"

    def test_negative_distance_handled_gracefully(self, rag_service):
        """Negative distances (edge case) should be handled."""
        # Should not raise, should clamp to 0
        similarity = rag_service._l2_distance_to_similarity(-5.0)
        assert similarity == 1.0  # Clamped to 0, so 1/(1+0) = 1


class TestRAGServiceRetrieval:
    """Tests for RAG retrieval functionality."""

    def test_empty_index_returns_empty_results(self, rag_service):
        """Empty index should return empty results, not error."""
        result = rag_service.retrieve_similar_with_context("test query", top_k=5)
        
        assert result.similar_cases == []
        assert result.similarity_scores == []

    def test_add_to_index_increases_size(self, rag_service):
        """Adding texts should increase index size."""
        assert rag_service.get_index_size() == 0
        
        rag_service.add_to_index(["Test text one", "Test text two"])
        
        assert rag_service.get_index_size() == 2

    def test_clear_index_resets_state(self, rag_service):
        """Clearing index should reset to empty state."""
        rag_service.add_to_index(["Some text"])
        rag_service.clear_index()
        
        assert rag_service.get_index_size() == 0
        assert rag_service._texts == []
        assert rag_service._records == []


# =============================================================================
# USE CASE UNIT TESTS
# =============================================================================

class TestAnalyzeRecordUseCase:
    """Tests for the analyze record use case."""

    def test_high_confidence_skips_rag(self):
        """
        SCENARIO: LLM returns high confidence result
        EXPECTED: RAG should NOT be invoked
        REASONING: RAG adds latency; skip if not needed
        """
        # Create mocks
        mock_llm = Mock(spec=ILLMService)
        mock_rag = Mock(spec=IRAGService)
        
        # Configure LLM to return high confidence
        mock_llm.analyze_record.return_value = AnalysisResult(
            record_id=1,
            indicators=["On schedule"],
            recommended_health="Good",
            confidence=0.95,
            mismatched=False,
            reasoning="Clear good indicators",
            processing_time_ms=50.0,
            requires_rag=False,
            requires_human_review=False,
        )
        
        use_case = AnalyzeRecordUseCase(
            llm_service=mock_llm,
            rag_service=mock_rag,
            confidence_threshold=0.85,
        )
        
        record = Record(
            record_id=1, account_id="A1", project_id="P1",
            predefined_health="Good",
            free_text_details="On schedule",
        )
        
        result = use_case.execute(record)
        
        # Verify RAG was NOT called
        mock_rag.retrieve_similar_with_context.assert_not_called()
        assert result.rag_context is None

    def test_low_confidence_triggers_rag(self):
        """
        SCENARIO: LLM returns low confidence result
        EXPECTED: RAG should be invoked for additional context
        REASONING: Low confidence needs similar cases to improve decision
        """
        from app.domain.entities import RAGContext
        
        mock_llm = Mock(spec=ILLMService)
        mock_rag = Mock(spec=IRAGService)
        
        # Configure LLM to return low confidence
        mock_llm.analyze_record.return_value = AnalysisResult(
            record_id=1,
            indicators=["Unclear"],
            recommended_health="Warning",
            confidence=0.65,
            mismatched=True,
            reasoning="Uncertain",
            processing_time_ms=50.0,
            requires_rag=True,
            requires_human_review=False,
        )
        
        # Configure RAG to return similar cases
        mock_rag.retrieve_similar_with_context.return_value = RAGContext(
            similar_cases=[{"predefined_health": "Warning", "free_text_details": "Similar case"}],
            similarity_scores=[0.85],
            retrieval_time_ms=10.0,
        )
        
        # Configure reanalysis
        mock_llm.reanalyze_with_context.return_value = AnalysisResult(
            record_id=1,
            indicators=["With RAG context"],
            recommended_health="Warning",
            confidence=0.88,
            mismatched=False,
            reasoning="RAG helped",
            processing_time_ms=60.0,
            requires_rag=False,
            requires_human_review=False,
        )
        
        use_case = AnalyzeRecordUseCase(
            llm_service=mock_llm,
            rag_service=mock_rag,
            confidence_threshold=0.85,
        )
        
        record = Record(
            record_id=1, account_id="A1", project_id="P1",
            predefined_health="Good",
            free_text_details="Ambiguous text",
        )
        
        result = use_case.execute(record)
        
        # Verify RAG WAS called
        mock_rag.retrieve_similar_with_context.assert_called_once()
        assert result.rag_context is not None


# =============================================================================
# INTEGRATION-LIKE TESTS (real services, mocked data)
# =============================================================================

class TestEndToEndClassification:
    """End-to-end tests using real services."""

    def test_batch_classification_accuracy(self, mock_llm):
        """
        SCENARIO: Process multiple records with known expected outcomes
        EXPECTED: At least 80% accuracy on clear cases
        REASONING: Validates that keyword scoring works on realistic data
        """
        test_cases = [
            # (free_text, expected_status)
            ("Project on schedule and under budget.", "Good"),
            ("All milestones achieved ahead of schedule.", "Good"),
            ("Slight delays due to resource constraints.", "Warning"),
            ("Budget is close to the limit.", "Warning"),
            ("Major budget overrun, at risk of cancellation.", "Critical"),
            ("Failed acceptance testing, urgent intervention needed.", "Critical"),
        ]
        
        correct = 0
        for text, expected in test_cases:
            record = Record(
                record_id=0, account_id="A", project_id="P",
                predefined_health=expected,
                free_text_details=text,
            )
            result = mock_llm.analyze_record(record)
            if result.recommended_health == expected:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.80, f"Accuracy {accuracy:.0%} < 80% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
