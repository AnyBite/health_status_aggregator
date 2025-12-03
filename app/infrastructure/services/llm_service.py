import json
import time
import logging
from typing import Optional

from openai import OpenAI

from app.domain import ILLMService, Record, AnalysisResult
from app.infrastructure.config import Settings


logger = logging.getLogger(__name__)


ANALYSIS_PROMPT_TEMPLATE = """You are an expert health status analyst for project management. Analyze the following project health record and extract key indicators.

## Record Details:
- Account ID: {account_id}
- Project ID: {project_id}
- Current Predefined Health Status: {predefined_health}
- Free Text Details: "{free_text_details}"

## Your Task:
1. Extract key health indicators from the free text (timeline, budget, quality, stakeholder satisfaction, risks)
2. Determine the recommended health status based ONLY on the free text content
3. Provide a confidence score for your recommendation
4. Explain your reasoning step by step

## Chain of Thought Analysis:
Think through this step by step:
1. First, identify all positive indicators in the text
2. Then, identify all negative indicators or risks
3. Look for any contradictions or ambiguity
4. Consider the severity of any issues mentioned
5. Based on the balance of indicators, determine the health status

## Response Format (JSON):
{{
    "indicators": ["indicator1", "indicator2", ...],
    "positive_signals": ["signal1", "signal2", ...],
    "negative_signals": ["signal1", "signal2", ...],
    "recommended_health": "Good" | "Warning" | "Critical",
    "confidence": 0.0-1.0,
    "reasoning": "Step by step explanation of your analysis",
    "ambiguity_detected": true | false,
    "ambiguity_reason": "Explanation if ambiguous"
}}

Respond ONLY with valid JSON, no additional text."""


RAG_REANALYSIS_PROMPT_TEMPLATE = """You are an expert health status analyst. Your previous analysis was uncertain. Here is additional context from similar historical cases to help refine your recommendation.

## Original Record:
- Account ID: {account_id}
- Project ID: {project_id}
- Current Predefined Health Status: {predefined_health}
- Free Text Details: "{free_text_details}"

## Your Previous Analysis:
- Recommended Health: {previous_recommendation}
- Confidence: {previous_confidence}
- Reasoning: {previous_reasoning}

## Similar Historical Cases:
{similar_cases}

## Your Task:
Based on the patterns from similar historical cases, refine your analysis. Pay special attention to:
1. How similar descriptions were classified
2. Common patterns in the language used
3. Whether your initial assessment aligns with historical precedents

## Response Format (JSON):
{{
    "final_recommendation": "Good" | "Warning" | "Critical",
    "final_confidence": 0.0-1.0,
    "reasoning": "Explanation incorporating historical context",
    "historical_pattern_match": true | false,
    "requires_human_review": true | false,
    "review_reason": "Why human review is needed, if applicable"
}}

Respond ONLY with valid JSON, no additional text."""


class OpenAILLMService(ILLMService):
    """OpenAI-based implementation of the LLM service."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )

    def analyze_record(self, record: Record) -> AnalysisResult:
        """Analyze a health record using OpenAI with chain-of-thought prompting."""
        start_time = time.time()
        
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            account_id=record.account_id,
            project_id=record.project_id,
            predefined_health=record.predefined_health,
            free_text_details=record.free_text_details,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent analysis
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            
            processing_time = (time.time() - start_time) * 1000

            confidence = float(parsed.get("confidence", 0.5))
            recommended_health = parsed.get("recommended_health", "Warning")
            
            mismatched = (
                recommended_health != record.predefined_health 
                or confidence < self._settings.confidence_threshold
            )
            
            requires_rag = (
                confidence < self._settings.confidence_threshold
                or parsed.get("ambiguity_detected", False)
                or mismatched
            )

            return AnalysisResult(
                record_id=record.record_id,
                indicators=parsed.get("indicators", []),
                recommended_health=recommended_health,
                confidence=confidence,
                mismatched=mismatched,
                reasoning=parsed.get("reasoning", ""),
                processing_time_ms=processing_time,
                requires_rag=requires_rag,
                requires_human_review=False,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_analysis(record, start_time)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(record, start_time)

    def reanalyze_with_context(
        self, 
        record: Record, 
        previous_result: AnalysisResult,
        similar_cases: list[dict]
    ) -> AnalysisResult:
        """Reanalyze a record with RAG context."""
        start_time = time.time()
        
        # Format similar cases for the prompt
        cases_text = "\n".join([
            f"Case {i+1}:\n  Status: {case.get('predefined_health', 'Unknown')}\n  Description: {case.get('free_text_details', 'N/A')}"
            for i, case in enumerate(similar_cases)
        ])
        
        prompt = RAG_REANALYSIS_PROMPT_TEMPLATE.format(
            account_id=record.account_id,
            project_id=record.project_id,
            predefined_health=record.predefined_health,
            free_text_details=record.free_text_details,
            previous_recommendation=previous_result.recommended_health,
            previous_confidence=previous_result.confidence,
            previous_reasoning=previous_result.reasoning,
            similar_cases=cases_text,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            
            processing_time = (time.time() - start_time) * 1000

            final_confidence = float(parsed.get("final_confidence", previous_result.confidence))
            final_recommendation = parsed.get("final_recommendation", previous_result.recommended_health)
            requires_human_review = (
                parsed.get("requires_human_review", False)
                or final_confidence < self._settings.confidence_threshold
            )

            return AnalysisResult(
                record_id=record.record_id,
                indicators=previous_result.indicators,
                recommended_health=final_recommendation,
                confidence=final_confidence,
                mismatched=final_recommendation != record.predefined_health,
                reasoning=parsed.get("reasoning", previous_result.reasoning),
                processing_time_ms=previous_result.processing_time_ms + processing_time,
                requires_rag=False,
                requires_human_review=requires_human_review,
            )

        except Exception as e:
            logger.error(f"RAG reanalysis failed: {e}")
            previous_result.requires_human_review = True
            return previous_result

    def _fallback_analysis(self, record: Record, start_time: float) -> AnalysisResult:
        """Provide fallback analysis when LLM fails."""
        processing_time = (time.time() - start_time) * 1000
        return AnalysisResult(
            record_id=record.record_id,
            indicators=["Analysis failed - fallback mode"],
            recommended_health="Warning",
            confidence=0.0,
            mismatched=True,
            reasoning="LLM analysis failed, manual review required",
            processing_time_ms=processing_time,
            requires_rag=True,
            requires_human_review=True,
        )


class MockLLMService(ILLMService):
    """
    Mock LLM implementation using weighted keyword scoring for classification.
    
    DESIGN RATIONALE:
    -----------------
    This mock simulates LLM behavior for development/testing without API costs.
    It uses a weighted keyword scoring approach based on text analysis principles:
    
    1. KEYWORD WEIGHTS (1.5 - 4.0 scale):
       - 4.0: Definitive indicators (e.g., "at risk of cancellation")
       - 3.0-3.5: Strong indicators (e.g., "budget overrun", "on schedule")
       - 2.0-2.5: Moderate indicators (e.g., "slight delay", "on track")
       - 1.5: Weak indicators that need context (e.g., "concern", "healthy")
       
       Weights are calibrated against the synthetic data phrases in generate_data.py.
    
    2. CLASSIFICATION THRESHOLDS:
       - Critical: score >= 3.0 OR ratio > 40% (single strong indicator is enough)
       - Warning: score >= 2.0 OR ratio > 35% (needs moderate evidence)
       - Good: score >= 2.0 OR ratio > 50% (needs clear positive evidence)
       
       These thresholds prevent over-classification and require meaningful evidence.
    
    3. CONFIDENCE SCORING:
       - Base: 0.70-0.75 (starts conservative)
       - Boost: +0.20-0.25 based on ratio dominance
       - Cap: 0.95-0.98 (never 100% certain)
       - Ambiguity penalty: capped at 0.75 if mixed signals
       
       This ensures uncertain cases get flagged for RAG/human review.
    
    4. AMBIGUITY DETECTION:
       Records with strong signals in multiple categories (score >= 2.0 in
       conflicting categories) are flagged. This catches contradictory statements
       like "budget overrun but client satisfaction excellent."
    
    LIMITATIONS:
    - No semantic understanding (just substring matching)
    - No context awareness (word order doesn't matter)
    - Susceptible to negation issues ("not on schedule" matches "on schedule")
    - For production, use OpenAILLMService with actual GPT-4
    """

    # Threshold constants with documentation
    CRITICAL_SCORE_THRESHOLD = 3.0  # Single strong critical indicator triggers Critical
    CRITICAL_RATIO_THRESHOLD = 0.40  # 40% of evidence being critical triggers Critical
    WARNING_SCORE_THRESHOLD = 2.0   # Moderate warning evidence needed
    WARNING_RATIO_THRESHOLD = 0.35  # 35% warning evidence threshold
    GOOD_SCORE_THRESHOLD = 2.0     # Clear positive evidence needed
    GOOD_RATIO_THRESHOLD = 0.50    # Good needs to be dominant (50%+)
    AMBIGUITY_SCORE_THRESHOLD = 2.0  # Mixed signals threshold
    
    # Confidence bounds
    CONFIDENCE_BASE_LOW = 0.70
    CONFIDENCE_BASE_MED = 0.75
    CONFIDENCE_NO_INDICATORS = 0.60
    CONFIDENCE_AMBIGUOUS_CAP = 0.75
    CONFIDENCE_MAX = 0.98

    def __init__(self, settings: Settings):
        self._settings = settings
        self._init_keyword_dictionaries()
    
    def _init_keyword_dictionaries(self) -> None:
        """
        Initialize keyword dictionaries with weights.
        
        Keywords are sourced from generate_data.py phrases to ensure alignment.
        Weights reflect indicator strength based on domain knowledge:
        - Project management status phrases have known severity implications
        - Multi-word phrases weighted higher (more specific)
        - Single words weighted lower (need context)
        """
        # Good indicators - from GOOD_PHRASES in generate_data.py
        self._good_keywords = {
            # Strong indicators (3.0) - definitive positive status
            "on schedule": 3.0,
            "under budget": 3.0,
            "ahead of schedule": 3.0,
            "all milestones achieved": 3.0,
            "10/10": 3.0,
            # Moderate indicators (2.5) - clear positive signals
            "progressing well": 2.5,
            "excellent": 2.5,
            "on time": 2.5,
            "exceeding expectations": 2.5,
            "no blockers": 2.5,
            "smooth execution": 2.5,
            "client satisfaction": 2.5,
            "9/10": 2.5,
            # Weaker indicators (2.0) - positive but need context
            "on track": 2.0,
            "team morale is high": 2.0,
            "positive stakeholder": 2.0,
            "optimally allocated": 2.0,
            "efficiency improved": 2.0,
            "good progress": 2.0,
            # Context-dependent (1.5) - can be misleading alone
            "healthy": 1.5,
            "positive": 1.5,
        }
        
        # Warning indicators - from WARNING_PHRASES in generate_data.py
        self._warning_keywords = {
            # Moderate-high indicators (2.5) - clear warning signals
            "slight delay": 2.5,
            "slight delays": 2.5,
            "budget is close": 2.5,
            "close to the limit": 2.5,
            "minor scope creep": 2.5,
            "timeline at risk": 2.5,
            # Moderate indicators (2.0) - warning signs
            "resource constraints": 2.0,
            "small delays": 2.0,
            "delayed by": 2.0,
            "budget variance": 2.0,
            "corrective action": 2.0,
            "team capacity stretched": 2.0,
            "overtime required": 2.0,
            "slightly below target": 2.0,
            "communication gaps": 2.0,
            "risk identified": 2.0,
            "morale declining": 2.0,
            # Context-dependent (1.5)
            "monitoring closely": 1.5,
            "concern": 1.5,
            "skills gap": 1.5,
        }
        
        # Critical indicators - from CRITICAL_PHRASES in generate_data.py
        self._critical_keywords = {
            # Severe indicators (4.0) - definitive critical status
            "major budget overrun": 4.0,
            "at risk of cancellation": 4.0,
            "critical resource resignation": 4.0,
            "failed acceptance testing": 4.0,
            "legal compliance issues": 4.0,
            "security vulnerability": 4.0,
            "exceeded by 40%": 4.0,
            # Strong indicators (3.5) - serious issues
            "budget overrun": 3.5,
            "slipped significantly": 3.5,
            "cancellation": 3.5,
            "urgent intervention": 3.5,
            "client escalation": 3.5,
            "budget exceeded": 3.5,
            "stakeholder confidence severely": 3.5,
            "critical path impacted": 3.5,
            # Moderate-strong indicators (3.0)
            "overrun": 3.0,
            "no additional funding": 3.0,
            "failed": 3.0,
            "severely impacted": 3.0,
            "major issue": 3.0,
            # Context-dependent (2.5) - can be serious
            "resignation": 2.5,
            "escalation": 2.5,
        }

    def analyze_record(self, record: Record) -> AnalysisResult:
        """
        Analyze a health record using weighted keyword scoring.
        
        Algorithm:
        1. Scan text for keywords in each category, sum weights
        2. Apply decision rules based on scores and ratios
        3. Calculate confidence based on evidence strength
        4. Detect ambiguity from conflicting signals
        5. Flag for RAG if uncertain or mismatched
        
        Returns:
            AnalysisResult with classification, confidence, and metadata
        """
        import time
        start_time = time.time()
        
        text_lower = record.free_text_details.lower()
        
        # Step 1: Calculate weighted scores for each category
        good_score, good_indicators = self._score_category(text_lower, self._good_keywords, "[+]")
        warning_score, warning_indicators = self._score_category(text_lower, self._warning_keywords, "[!]")
        critical_score, critical_indicators = self._score_category(text_lower, self._critical_keywords, "[X]")
        
        indicators = good_indicators + warning_indicators + critical_indicators
        total_score = good_score + warning_score + critical_score
        
        # Step 2: Apply classification logic
        if total_score == 0:
            # No indicators found - fall back to predefined with low confidence
            detected_status = record.predefined_health
            confidence = self.CONFIDENCE_NO_INDICATORS
            indicators.append("No clear indicators found - using predefined status")
        else:
            detected_status, confidence = self._classify(
                good_score, warning_score, critical_score, total_score
            )
        
        # Step 3: Check for ambiguity (conflicting strong signals)
        is_ambiguous = self._detect_ambiguity(good_score, warning_score, critical_score)
        if is_ambiguous:
            confidence = min(confidence, self.CONFIDENCE_AMBIGUOUS_CAP)
            indicators.append("⚠ Ambiguity: Conflicting signals detected")
        
        # Step 4: Determine if further analysis needed
        mismatched = detected_status != record.predefined_health
        requires_rag = (
            confidence < self._settings.confidence_threshold or
            is_ambiguous or
            mismatched
        )
        
        processing_time = (time.time() - start_time) * 1000 + 50  # +50ms simulated LLM delay
        
        return AnalysisResult(
            record_id=record.record_id,
            indicators=indicators[:10],  # Limit to top 10 for readability
            recommended_health=detected_status,
            confidence=confidence,
            mismatched=mismatched,
            reasoning=f"Weighted scores: Good={good_score:.1f}, Warning={warning_score:.1f}, Critical={critical_score:.1f}",
            processing_time_ms=processing_time,
            requires_rag=requires_rag,
            requires_human_review=False,
        )
    
    def _score_category(
        self, text: str, keywords: dict[str, float], prefix: str
    ) -> tuple[float, list[str]]:
        """
        Calculate total score for a category by summing matched keyword weights.
        
        Args:
            text: Lowercase text to search
            keywords: Dict mapping keyword -> weight
            prefix: Indicator prefix for logging (e.g., "[+]" for good)
        
        Returns:
            Tuple of (total_score, list_of_indicator_strings)
        """
        score = 0.0
        indicators = []
        for keyword, weight in keywords.items():
            if keyword in text:
                score += weight
                indicators.append(f"{prefix} '{keyword}' (w={weight})")
        return score, indicators
    
    def _classify(
        self, good: float, warning: float, critical: float, total: float
    ) -> tuple[str, float]:
        """
        Determine health status and confidence from scores.
        
        Decision priority (based on risk-averse classification):
        1. Critical - if strong critical evidence (safety first)
        2. Warning - if moderate warning evidence
        3. Good - if dominant positive evidence
        4. Fallback - pick highest score with moderate confidence
        
        Returns:
            Tuple of (status_string, confidence_float)
        """
        # Calculate ratios for relative comparison
        good_ratio = good / total
        warning_ratio = warning / total
        critical_ratio = critical / total
        
        # Critical takes priority (risk-averse)
        if critical >= self.CRITICAL_SCORE_THRESHOLD or critical_ratio > self.CRITICAL_RATIO_THRESHOLD:
            confidence = min(
                self.CONFIDENCE_BASE_LOW + (critical_ratio * 0.25),
                self.CONFIDENCE_MAX
            )
            return "Critical", confidence
        
        # Warning next
        if warning >= self.WARNING_SCORE_THRESHOLD or warning_ratio > self.WARNING_RATIO_THRESHOLD:
            # Lower confidence if critical signals also present
            base = self.CONFIDENCE_BASE_LOW if critical > 0 else self.CONFIDENCE_BASE_MED
            confidence = min(base + (warning_ratio * 0.20), 0.92)
            return "Warning", confidence
        
        # Good if clearly dominant
        if good >= self.GOOD_SCORE_THRESHOLD or good_ratio > self.GOOD_RATIO_THRESHOLD:
            confidence = min(
                self.CONFIDENCE_BASE_MED + (good_ratio * 0.20),
                0.95
            )
            return "Good", confidence
        
        # Fallback: pick highest score with moderate confidence
        scores = {"Good": good, "Warning": warning, "Critical": critical}
        best = max(scores, key=scores.get)
        return best, 0.70
    
    def _detect_ambiguity(self, good: float, warning: float, critical: float) -> bool:
        """
        Detect if record has conflicting strong signals.
        
        Ambiguity exists when:
        - Good + Warning/Critical both have significant evidence
        - Warning + Critical both have significant evidence
        
        This catches contradictory statements like:
        "Client satisfaction excellent but project at risk of cancellation"
        """
        threshold = self.AMBIGUITY_SCORE_THRESHOLD
        
        good_strong = good >= threshold
        warning_strong = warning >= threshold
        critical_strong = critical >= threshold
        
        return (good_strong and (warning_strong or critical_strong)) or \
               (warning_strong and critical_strong)

    def reanalyze_with_context(
        self, 
        record: Record, 
        previous_result: AnalysisResult,
        similar_cases: list[dict]
    ) -> AnalysisResult:
        """Mock reanalysis with RAG context."""
        import time
        start_time = time.time()
        
        # Simulate learning from similar cases
        case_statuses = [c.get("predefined_health", "Warning") for c in similar_cases]
        if case_statuses:
            from collections import Counter
            most_common = Counter(case_statuses).most_common(1)[0][0]
            
            # Increase confidence if similar cases agree
            new_confidence = min(previous_result.confidence + 0.1, 0.95)
            
            processing_time = (time.time() - start_time) * 1000 + 30
            
            return AnalysisResult(
                record_id=record.record_id,
                indicators=previous_result.indicators + [f"RAG: {len(similar_cases)} similar cases analyzed"],
                recommended_health=most_common,
                confidence=new_confidence,
                mismatched=most_common != record.predefined_health,
                reasoning=f"Refined based on {len(similar_cases)} similar cases. Most common status: {most_common}",
                processing_time_ms=previous_result.processing_time_ms + processing_time,
                requires_rag=False,
                requires_human_review=new_confidence < self._settings.confidence_threshold,
            )
        
        previous_result.requires_human_review = True
        return previous_result
