# agents/context_analyzer/validators.py

"""
Context Analyzer Input/Output Validation - Production Ready

Production-ready validation utilities for the Context Analyzer Agent providing
comprehensive input validation, output validation, bias analysis result checking,
and scoring consistency validation with enhanced error reporting and session tracking.
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass, field

from .exceptions import (
    InputValidationError,
    BiasDetectionError,
    ManipulationDetectionError,
    ScoringConsistencyError,
    DataFormatError,
    raise_input_validation_error
)


@dataclass
class ValidationResult:
    """
    Enhanced validation result container with comprehensive feedback.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of validation error messages
        warnings: List of validation warnings
        suggestions: List of suggestions for fixing issues
        score: Validation quality score (0-100)
        details: Additional validation details and metrics
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: float = 100.0
    details: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str, suggestion: str = None):
        """Add an error with optional suggestion for fixing it."""
        self.errors.append(error)
        if suggestion:
            self.suggestions.append(suggestion)
        self.is_valid = False

    def add_warning(self, warning: str, suggestion: str = None):
        """Add a warning with optional suggestion for improvement."""
        self.warnings.append(warning)
        if suggestion:
            self.suggestions.append(suggestion)


class InputValidator:
    """
    Production-ready input validation for context analyzer processing.
    
    Validates article text, previous analysis data, and configuration
    parameters before processing with enhanced error reporting and
    performance tracking.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize input validator with production configuration.

        Args:
            config: Optional configuration for validation rules and thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.InputValidator")
        
        # Text validation thresholds with production defaults
        self.min_article_length = self.config.get('min_article_length', 50)
        self.max_article_length = self.config.get('max_article_length', 50000)
        self.recommended_min_length = self.config.get('recommended_min_length', 200)
        
        # Analysis validation thresholds
        self.min_analysis_confidence = self.config.get('min_analysis_confidence', 0.0)
        self.low_confidence_threshold = self.config.get('low_confidence_threshold', 0.5)
        
        # Performance metrics
        self.validation_count = 0
        self.total_processing_time = 0.0
        self.validation_failures = 0
        
        self.logger.info(f"InputValidator initialized with production settings")

    def validate_article_text(self, text: Any, session_id: str = None) -> ValidationResult:
        """
        Validate article text input with comprehensive feedback.

        Args:
            text: Article text to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and detailed feedback
        """
        result = ValidationResult()
        start_time = time.time()
        
        self.logger.debug(f"Validating article text", extra={'session_id': session_id})

        try:
            # Basic type validation
            if not isinstance(text, str):
                result.add_error(
                    f"Article text must be string, got {type(text).__name__}",
                    "Convert the input to a string before processing"
                )
                return result

            # Content existence validation
            if not text.strip():
                result.add_error(
                    "Article text cannot be empty or whitespace only",
                    f"Provide meaningful article content with at least {self.min_article_length} characters"
                )
                return result

            # Length validations with detailed feedback
            text_length = len(text.strip())
            result.details['text_length'] = text_length
            result.details['word_count'] = len(text.strip().split())
            
            if text_length < self.min_article_length:
                result.add_error(
                    f"Article text too short: {text_length} chars (minimum: {self.min_article_length})",
                    f"Article should be at least {self.min_article_length} characters for meaningful context analysis"
                )
                result.score -= 40

            if text_length > self.max_article_length:
                result.add_warning(
                    f"Article text very long: {text_length} chars (maximum recommended: {self.max_article_length})",
                    f"Consider summarizing to {self.max_article_length} characters for optimal processing"
                )
                result.score -= 10

            # Quality assessments
            if text_length < self.recommended_min_length:
                result.add_warning(
                    f"Article text may be too short for comprehensive context analysis: {text_length} chars",
                    f"For best analysis results, provide at least {self.recommended_min_length} characters"
                )
                result.score -= 15

            # Structural quality checks
            sentence_count = len(re.findall(r'[.!?]+', text.strip()))
            result.details['sentence_count'] = sentence_count
            
            if sentence_count < text_length / 200:
                result.add_warning(
                    "Article may lack proper sentence structure",
                    "Ensure the text contains complete sentences with proper punctuation"
                )
                result.score -= 10

            # Content quality indicators
            newline_ratio = text.count('\n') / text_length if text_length > 0 else 0
            if newline_ratio > 0.1:
                result.add_warning(
                    "Article contains excessive line breaks which may affect analysis quality",
                    "Consider cleaning up formatting before processing"
                )
                result.score -= 5

            # Check for suspicious patterns that might affect analysis
            if len(re.findall(r'[^\w\s]', text)) / text_length > 0.3:
                result.add_warning(
                    "Article contains high ratio of special characters",
                    "Verify text encoding and consider cleaning special characters"
                )

            # Final validation
            result.is_valid = len(result.errors) == 0
            result.score = max(0.0, min(100.0, result.score))

            processing_time = time.time() - start_time
            self.validation_count += 1
            self.total_processing_time += processing_time
            
            if not result.is_valid:
                self.validation_failures += 1

            self.logger.info(
                f"Article text validation completed",
                extra={
                    'session_id': session_id,
                    'text_length': text_length,
                    'is_valid': result.is_valid,
                    'score': result.score,
                    'processing_time': round(processing_time * 1000, 2)
                }
            )

        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            self.logger.error(f"Article text validation failed: {str(e)}", extra={'session_id': session_id})

        return result

    def validate_previous_analysis(self, analysis: Any, session_id: str = None) -> ValidationResult:
        """
        Validate previous analysis input with comprehensive feedback.

        Args:
            analysis: Previous analysis results to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and suggestions
        """
        result = ValidationResult()
        
        self.logger.debug(f"Validating previous analysis", extra={'session_id': session_id})

        try:
            # Type validation
            if not isinstance(analysis, dict):
                result.add_error(
                    f"Previous analysis must be a dictionary, got {type(analysis).__name__}",
                    "Ensure previous analysis is provided as a structured dictionary"
                )
                return result

            # Required fields validation with helpful messages
            required_fields = ['prediction', 'confidence']
            for field in required_fields:
                if field not in analysis:
                    result.add_error(
                        f"Missing required field in previous analysis: '{field}'",
                        f"Add required field: '{field}': 'appropriate_value'"
                    )

            # Confidence validation with detailed feedback
            if 'confidence' in analysis:
                confidence = analysis['confidence']
                result.details['confidence'] = confidence
                
                if not isinstance(confidence, (int, float)):
                    result.add_error(
                        f"Confidence must be numeric, got {type(confidence).__name__}",
                        "Use a decimal number between 0 and 1 for confidence"
                    )
                elif not (0 <= confidence <= 1):
                    result.add_error(
                        f"Invalid confidence value: {confidence} (must be between 0 and 1)",
                        f"Adjust confidence to valid range: {max(0, min(1, confidence))}"
                    )
                elif confidence < self.min_analysis_confidence:
                    result.add_warning(
                        f"Very low confidence score: {confidence:.2f}",
                        "Consider re-analyzing with higher confidence threshold"
                    )
                elif confidence < self.low_confidence_threshold:
                    result.add_warning(
                        f"Low confidence score: {confidence:.2f}",
                        "Additional context analysis may be beneficial"
                    )

            # Prediction validation
            if 'prediction' in analysis:
                prediction = analysis['prediction']
                if not isinstance(prediction, str) or not prediction.strip():
                    result.add_error(
                        "Prediction must be a non-empty string",
                        "Provide prediction as string: 'REAL', 'FAKE', etc."
                    )
                else:
                    result.details['prediction'] = prediction

            # Optional field validation with suggestions
            optional_fields = {
                'source': str,
                'topic_domain': str,
                'processing_time': (int, float)
            }

            for field_name, expected_type in optional_fields.items():
                if field_name in analysis:
                    value = analysis[field_name]
                    if not isinstance(value, expected_type):
                        result.add_warning(
                            f"'{field_name}' should be {expected_type.__name__}, got {type(value).__name__}",
                            f"Convert to expected type: '{field_name}': {expected_type.__name__.lower()}_value"
                        )

            # Final validation
            result.is_valid = len(result.errors) == 0
            result.score = max(0.0, 100.0 - (len(result.errors) * 25) - (len(result.warnings) * 5))

            self.logger.info(
                f"Previous analysis validation completed",
                extra={
                    'session_id': session_id,
                    'is_valid': result.is_valid,
                    'has_confidence': 'confidence' in analysis,
                    'has_prediction': 'prediction' in analysis
                }
            )

        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            self.logger.error(f"Previous analysis validation failed: {str(e)}", extra={'session_id': session_id})

        return result

    def validate_processing_input(self, input_data: Dict[str, Any], session_id: str = None) -> ValidationResult:
        """
        Validate complete input data for context analysis processing.

        Args:
            input_data: Complete input data dictionary
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with comprehensive validation results
        """
        all_errors = []
        all_warnings = []
        all_suggestions = []
        
        self.logger.info(f"Starting comprehensive input validation", extra={'session_id': session_id})

        try:
            # Basic structure validation
            if not isinstance(input_data, dict):
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Input must be a dictionary, got {type(input_data).__name__}"],
                    suggestions=["Provide input as: {'text': 'article text', 'previous_analysis': {...}}"]
                )

            if not input_data:
                return ValidationResult(
                    is_valid=False,
                    errors=["Input data cannot be empty"],
                    suggestions=["Provide at least 'text' field with article content"]
                )

            # Required field validation
            if 'text' not in input_data:
                all_errors.append("Missing required 'text' field")
                all_suggestions.append("Add article text: 'text': 'your article content here'")
            else:
                text_result = self.validate_article_text(input_data['text'], session_id)
                all_errors.extend([f"Text validation: {error}" for error in text_result.errors])
                all_warnings.extend([f"Text warning: {warning}" for warning in text_result.warnings])
                all_suggestions.extend(text_result.suggestions)

            # Optional fields validation
            if 'previous_analysis' in input_data:
                analysis_result = self.validate_previous_analysis(input_data['previous_analysis'], session_id)
                # Previous analysis errors are warnings for context analysis (not critical)
                all_warnings.extend([f"Previous analysis: {error}" for error in analysis_result.errors])
                all_warnings.extend([f"Previous analysis: {warning}" for warning in analysis_result.warnings])
                all_suggestions.extend(analysis_result.suggestions)
            else:
                all_warnings.append("No previous analysis provided - context analysis will be more generic")
                all_suggestions.append("Add previous analysis for better context: 'previous_analysis': {'prediction': 'REAL', 'confidence': 0.85}")

            # Additional field validation
            if 'include_detailed_analysis' in input_data:
                detailed = input_data['include_detailed_analysis']
                if not isinstance(detailed, bool):
                    all_warnings.append(f"'include_detailed_analysis' should be boolean, got {type(detailed).__name__}")
                    all_suggestions.append("Use boolean value: 'include_detailed_analysis': true or false")

            # Calculate overall score
            error_penalty = len(all_errors) * 25
            warning_penalty = len(all_warnings) * 5
            score = max(0.0, 100.0 - error_penalty - warning_penalty)

            result = ValidationResult(
                is_valid=len(all_errors) == 0,
                errors=all_errors,
                warnings=all_warnings,
                suggestions=all_suggestions,
                score=score,
                details={
                    'has_text': 'text' in input_data,
                    'has_previous_analysis': 'previous_analysis' in input_data,
                    'has_detailed_flag': 'include_detailed_analysis' in input_data,
                    'total_fields': len(input_data)
                }
            )

            self.logger.info(
                f"Comprehensive input validation completed",
                extra={
                    'session_id': session_id,
                    'is_valid': result.is_valid,
                    'error_count': len(all_errors),
                    'warning_count': len(all_warnings),
                    'score': score
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}", extra={'session_id': session_id})
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                suggestions=["Check input format and try again"]
            )


class BiasAnalysisValidator:
    """
    Production-ready validation for bias analysis results and scoring consistency.
    
    Ensures bias detection results are properly formatted and consistent
    with textual analysis while providing detailed feedback for improvements.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize bias analysis validator with production configuration.

        Args:
            config: Optional configuration for validation rules and thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.BiasAnalysisValidator")

    def validate_bias_scores(self, scores: Any, session_id: str = None) -> ValidationResult:
        """
        Validate bias scoring results with detailed feedback.

        Args:
            scores: Bias scores to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and improvement suggestions
        """
        result = ValidationResult()

        try:
            # Type validation
            if not isinstance(scores, dict):
                result.add_error(
                    f"Bias scores must be a dictionary, got {type(scores).__name__}",
                    "Provide scores as: {'bias': 75, 'manipulation': 60, 'credibility': 40, 'risk': 70}"
                )
                return result

            # Required score fields with validation
            required_scores = {
                'bias': 'Political and emotional bias intensity',
                'manipulation': 'Emotional manipulation and propaganda level',
                'credibility': 'Information credibility and reliability',
                'risk': 'Overall misinformation risk level'
            }

            scores_found = 0
            for score_field, description in required_scores.items():
                if score_field not in scores:
                    result.add_error(
                        f"Missing required score field: {score_field}",
                        f"Add {description.lower()}: '{score_field}': 0-100"
                    )
                else:
                    score = scores[score_field]
                    scores_found += 1
                    
                    if not isinstance(score, (int, float)):
                        result.add_error(
                            f"{score_field} score must be number between 0-100, got {type(score).__name__}",
                            f"Use integer: '{score_field}': 75"
                        )
                    elif not (0 <= score <= 100):
                        result.add_error(
                            f"{score_field} score must be between 0-100, got {score}",
                            f"Adjust to valid range: '{score_field}': {max(0, min(100, score))}"
                        )
                    else:
                        result.details[score_field] = score

            # Score consistency checks with helpful feedback
            if scores_found >= 3:
                bias_score = scores.get('bias', 50)
                manipulation_score = scores.get('manipulation', 50)
                credibility_score = scores.get('credibility', 50)

                # Logical consistency checks
                if bias_score > 75 and credibility_score > 75:
                    result.add_warning(
                        "High bias score with high credibility may be inconsistent",
                        "Consider: high bias typically correlates with lower credibility"
                    )

                if manipulation_score > 75 and bias_score < 25:
                    result.add_warning(
                        "High manipulation score with low bias may be inconsistent",
                        "High manipulation often indicates underlying bias"
                    )

                # Score range recommendations
                if all(25 <= scores.get(field, 50) <= 75 for field in required_scores):
                    result.add_warning(
                        "All scores in middle range - consider more decisive analysis",
                        "Look for stronger indicators to differentiate content quality"
                    )

            # Final validation
            result.is_valid = len(result.errors) == 0
            result.score = max(0.0, 100.0 - (len(result.errors) * 20) - (len(result.warnings) * 5))

            self.logger.info(
                f"Bias scores validation completed",
                extra={
                    'session_id': session_id,
                    'is_valid': result.is_valid,
                    'scores_found': scores_found,
                    'score_consistency': len(result.warnings) == 0
                }
            )

        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            self.logger.error(f"Bias scores validation failed: {str(e)}", extra={'session_id': session_id})

        return result

    def validate_political_bias_analysis(self, analysis: Any, session_id: str = None) -> ValidationResult:
        """
        Validate political bias analysis results with comprehensive feedback.

        Args:
            analysis: Political bias analysis to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and detailed suggestions
        """
        result = ValidationResult()

        try:
            if not isinstance(analysis, dict):
                result.add_error(
                    f"Political bias analysis must be dictionary, got {type(analysis).__name__}",
                    "Provide as: {'political_leaning': 'left/right/center', 'confidence': 0.85}"
                )
                return result

            # Check required fields with specific guidance
            required_fields = {
                'political_leaning': 'Political orientation classification',
                'confidence': 'Classification confidence level'
            }

            for field, description in required_fields.items():
                if field not in analysis:
                    result.add_error(
                        f"Missing required field: {field}",
                        f"Add {description.lower()}: '{field}': appropriate_value"
                    )

            # Validate political leaning with comprehensive options
            if 'political_leaning' in analysis:
                leaning = analysis['political_leaning']
                valid_leanings = ['left', 'right', 'center', 'mixed', 'neutral', 'unknown']
                if leaning not in valid_leanings:
                    result.add_error(
                        f"Invalid political leaning: '{leaning}'",
                        f"Use one of: {valid_leanings}"
                    )
                else:
                    result.details['political_leaning'] = leaning

            # Validate confidence with detailed feedback
            if 'confidence' in analysis:
                confidence = analysis['confidence']
                if not isinstance(confidence, (int, float)):
                    result.add_error(
                        f"Confidence must be numeric, got {type(confidence).__name__}",
                        "Use decimal between 0 and 1: 'confidence': 0.85"
                    )
                elif not (0 <= confidence <= 1):
                    result.add_error(
                        f"Confidence must be between 0-1, got {confidence}",
                        f"Adjust to valid range: 'confidence': {max(0, min(1, confidence))}"
                    )
                else:
                    result.details['confidence'] = confidence
                    
                    # Confidence quality feedback
                    if confidence < 0.3:
                        result.add_warning(
                            f"Very low confidence: {confidence:.2f}",
                            "Consider 'unknown' classification for very uncertain cases"
                        )
                    elif confidence < 0.5:
                        result.add_warning(
                            f"Low confidence: {confidence:.2f}",
                            "Additional analysis may be needed for reliable classification"
                        )

            # Optional fields validation
            optional_fields = {
                'intensity': (int, float),
                'evidence': list,
                'keywords_found': list
            }

            for field_name, expected_type in optional_fields.items():
                if field_name in analysis:
                    value = analysis[field_name]
                    if not isinstance(value, expected_type):
                        result.add_warning(
                            f"'{field_name}' should be {expected_type.__name__}, got {type(value).__name__}",
                            f"Use appropriate type for better analysis quality"
                        )

            # Final validation
            result.is_valid = len(result.errors) == 0
            result.score = max(0.0, 100.0 - (len(result.errors) * 25) - (len(result.warnings) * 10))

            self.logger.info(
                f"Political bias analysis validation completed",
                extra={
                    'session_id': session_id,
                    'is_valid': result.is_valid,
                    'has_leaning': 'political_leaning' in analysis,
                    'has_confidence': 'confidence' in analysis
                }
            )

        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            self.logger.error(f"Political bias analysis validation failed: {str(e)}", extra={'session_id': session_id})

        return result


class ManipulationAnalysisValidator:
    """
    Production-ready validation for manipulation analysis results and technique detection.
    
    Ensures manipulation detection results are properly formatted and
    technique classifications are accurate with detailed feedback.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize manipulation analysis validator."""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ManipulationAnalysisValidator")

    def validate_manipulation_report(self, report: Any, session_id: str = None) -> ValidationResult:
        """
        Validate manipulation detection report with comprehensive feedback.

        Args:
            report: Manipulation report to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and improvement suggestions
        """
        result = ValidationResult()

        try:
            if not isinstance(report, dict):
                result.add_error(
                    f"Manipulation report must be dictionary, got {type(report).__name__}",
                    "Provide as structured dictionary with required sections"
                )
                return result

            # Check required sections with descriptions
            required_sections = {
                'propaganda_techniques': 'Detected propaganda methods',
                'manipulation_patterns': 'Emotional manipulation patterns',
                'logical_fallacies': 'Logical reasoning errors'
            }

            for section, description in required_sections.items():
                if section not in report:
                    result.add_error(
                        f"Missing required section: {section}",
                        f"Add {description.lower()}: '{section}': analysis_results"
                    )

            # Validate overall manipulation score
            if 'overall_manipulation_score' in report:
                score = report['overall_manipulation_score']
                if not isinstance(score, (int, float)):
                    result.add_error(
                        f"Manipulation score must be numeric, got {type(score).__name__}",
                        "Use number between 0-10: 'overall_manipulation_score': 7.5"
                    )
                elif not (0 <= score <= 10):
                    result.add_error(
                        f"Manipulation score must be between 0-10, got {score}",
                        f"Adjust to valid range: 'overall_manipulation_score': {max(0, min(10, score))}"
                    )
                else:
                    result.details['manipulation_score'] = score
                    
                    # Score interpretation feedback
                    if score >= 8:
                        result.add_warning(
                            "Very high manipulation score detected",
                            "Verify analysis with multiple detection methods"
                        )
                    elif score <= 2:
                        result.add_warning(
                            "Very low manipulation score - may indicate minimal content",
                            "Ensure sufficient content for meaningful analysis"
                        )

            # Validate risk level
            if 'risk_level' in report:
                risk_level = report['risk_level']
                valid_levels = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                if risk_level not in valid_levels:
                    result.add_error(
                        f"Invalid risk level: '{risk_level}'",
                        f"Use one of: {valid_levels}"
                    )
                else:
                    result.details['risk_level'] = risk_level

            # Check for technique summaries
            if 'techniques_summary' in report:
                summary = report['techniques_summary']
                if isinstance(summary, dict):
                    expected_keys = ['total_techniques_detected', 'high_severity_count']
                    missing_keys = [key for key in expected_keys if key not in summary]
                    if missing_keys:
                        result.add_warning(
                            f"Technique summary missing keys: {missing_keys}",
                            "Include comprehensive technique statistics"
                        )

            # Final validation
            result.is_valid = len(result.errors) == 0
            result.score = max(0.0, 100.0 - (len(result.errors) * 20) - (len(result.warnings) * 8))

            self.logger.info(
                f"Manipulation report validation completed",
                extra={
                    'session_id': session_id,
                    'is_valid': result.is_valid,
                    'has_score': 'overall_manipulation_score' in report,
                    'has_risk_level': 'risk_level' in report
                }
            )

        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            self.logger.error(f"Manipulation report validation failed: {str(e)}", extra={'session_id': session_id})

        return result

    def validate_propaganda_techniques(self, techniques: Any, session_id: str = None) -> ValidationResult:
        """
        Validate detected propaganda techniques with detailed feedback.

        Args:
            techniques: Propaganda techniques detection results
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and suggestions
        """
        result = ValidationResult()

        try:
            if not isinstance(techniques, dict):
                result.add_error(
                    f"Propaganda techniques must be dictionary, got {type(techniques).__name__}",
                    "Provide as: {'detected': {technique_name: technique_data}}"
                )
                return result

            if 'detected' not in techniques:
                result.add_error(
                    "Missing 'detected' field in propaganda techniques",
                    "Add: 'detected': {technique_name: {confidence, severity, description}}"
                )
                return result

            detected = techniques['detected']
            if not isinstance(detected, dict):
                result.add_error(
                    "'detected' field must be dictionary",
                    "Use dictionary format for technique details"
                )
                return result

            # Validate each detected technique
            valid_techniques = 0
            for technique_name, technique_data in detected.items():
                if not isinstance(technique_data, dict):
                    result.add_error(
                        f"Technique '{technique_name}' data must be dictionary",
                        f"Provide structured data: '{technique_name}': {{confidence, severity, description}}"
                    )
                    continue

                # Check required fields for each technique
                required_fields = {
                    'confidence': 'Detection confidence level',
                    'severity': 'Technique severity assessment'
                }

                technique_valid = True
                for field, description in required_fields.items():
                    if field not in technique_data:
                        result.add_error(
                            f"Technique '{technique_name}' missing required field: {field}",
                            f"Add {description.lower()}: '{field}': appropriate_value"
                        )
                        technique_valid = False

                # Validate confidence
                if 'confidence' in technique_data:
                    confidence = technique_data['confidence']
                    if not isinstance(confidence, (int, float)):
                        result.add_error(
                            f"Invalid confidence type for '{technique_name}': {type(confidence).__name__}",
                            f"Use decimal: 'confidence': 0.85"
                        )
                        technique_valid = False
                    elif not (0 <= confidence <= 1):
                        result.add_error(
                            f"Invalid confidence range for '{technique_name}': {confidence}",
                            f"Use 0-1 range: 'confidence': {max(0, min(1, confidence))}"
                        )
                        technique_valid = False

                # Validate severity
                if 'severity' in technique_data:
                    severity = technique_data['severity']
                    valid_severities = ['low', 'medium', 'high', 'critical']
                    if severity not in valid_severities:
                        result.add_error(
                            f"Invalid severity for '{technique_name}': '{severity}'",
                            f"Use one of: {valid_severities}"
                        )
                        technique_valid = False

                if technique_valid:
                    valid_techniques += 1

            result.details['total_techniques'] = len(detected)
            result.details['valid_techniques'] = valid_techniques

            # Quality assessment
            if len(detected) == 0:
                result.add_warning(
                    "No propaganda techniques detected",
                    "Verify analysis sensitivity or content complexity"
                )
            elif valid_techniques == 0:
                result.add_error(
                    "No valid technique detections found",
                    "Review technique detection format and data structure"
                )

            # Final validation
            result.is_valid = len(result.errors) == 0 and valid_techniques > 0
            result.score = max(0.0, 100.0 - (len(result.errors) * 15) - (len(result.warnings) * 5))

            self.logger.info(
                f"Propaganda techniques validation completed",
                extra={
                    'session_id': session_id,
                    'is_valid': result.is_valid,
                    'total_techniques': len(detected),
                    'valid_techniques': valid_techniques
                }
            )

        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            self.logger.error(f"Propaganda techniques validation failed: {str(e)}", extra={'session_id': session_id})

        return result


class ScoringConsistencyValidator:
    """
    Production-ready validator for checking consistency between textual analysis and numerical scores.
    
    Addresses the main issue where LLM text analysis doesn't match the provided scores
    with enhanced pattern matching and detailed feedback.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize scoring consistency validator."""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ScoringConsistencyValidator")
        
        # Consistency thresholds
        self.major_inconsistency_threshold = self.config.get('major_inconsistency_threshold', 40)
        self.minor_inconsistency_threshold = self.config.get('minor_inconsistency_threshold', 20)

    def validate_score_text_consistency(self, analysis_text: str, scores: Dict[str, int], session_id: str = None) -> ValidationResult:
        """
        Validate consistency between textual analysis and numerical scores with enhanced detection.

        Args:
            analysis_text: Generated analysis text
            scores: Dictionary of numerical scores
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with consistency validation and specific suggestions
        """
        result = ValidationResult()

        try:
            if not isinstance(analysis_text, str) or not analysis_text.strip():
                result.add_error(
                    "Analysis text cannot be empty",
                    "Provide meaningful textual analysis for consistency checking"
                )
                return result

            if not isinstance(scores, dict):
                result.add_error(
                    f"Scores must be dictionary, got {type(scores).__name__}",
                    "Provide scores as: {'bias': 75, 'manipulation': 60, 'credibility': 40}"
                )
                return result

            text_lower = analysis_text.lower()
            inconsistencies = []
            
            # Enhanced consistency checking with specific patterns
            consistency_patterns = {
                'bias': {
                    'low_indicators': ['minimal bias', 'low bias', 'neutral', 'unbiased', 'balanced', 'fair', 'objective'],
                    'high_indicators': ['high bias', 'significant bias', 'extreme bias', 'heavily biased', 'partisan', 'slanted'],
                    'moderate_indicators': ['some bias', 'moderate bias', 'slight bias', 'somewhat biased']
                },
                'manipulation': {
                    'low_indicators': ['minimal manipulation', 'no manipulation', 'straightforward', 'honest', 'direct'],
                    'high_indicators': ['high manipulation', 'extreme manipulation', 'heavily manipulative', 'propaganda', 'misleading'],
                    'moderate_indicators': ['some manipulation', 'moderate manipulation', 'emotional appeals', 'persuasive']
                },
                'credibility': {
                    'high_indicators': ['highly credible', 'very reliable', 'trustworthy', 'authoritative', 'credible', 'reliable'],
                    'low_indicators': ['not credible', 'unreliable', 'questionable', 'dubious', 'suspect', 'untrustworthy'],
                    'moderate_indicators': ['somewhat credible', 'moderately reliable', 'fairly reliable']
                }
            }

            # Check each score type for consistency
            for score_type, patterns in consistency_patterns.items():
                if score_type not in scores:
                    continue
                    
                score = scores[score_type]
                inconsistency_found = False
                
                # Check for low score with high indicators
                if score <= 30:
                    high_patterns = patterns.get('high_indicators', [])
                    for pattern in high_patterns:
                        if pattern in text_lower:
                            inconsistencies.append(f"Low {score_type} score ({score}) conflicts with text: '{pattern}'")
                            inconsistency_found = True
                            break
                
                # Check for high score with low indicators (except credibility which is inverted)
                elif score >= 70:
                    if score_type == 'credibility':
                        # High credibility score conflicting with low credibility text
                        low_patterns = patterns.get('low_indicators', [])
                        for pattern in low_patterns:
                            if pattern in text_lower:
                                inconsistencies.append(f"High {score_type} score ({score}) conflicts with text: '{pattern}'")
                                inconsistency_found = True
                                break
                    else:
                        # High bias/manipulation score conflicting with low intensity text
                        low_patterns = patterns.get('low_indicators', [])
                        for pattern in low_patterns:
                            if pattern in text_lower:
                                inconsistencies.append(f"High {score_type} score ({score}) conflicts with text: '{pattern}'")
                                inconsistency_found = True
                                break
                
                # Additional pattern checks for edge cases
                if not inconsistency_found:
                    # Check for absolute statements that conflict with moderate scores
                    if 30 < score < 70:
                        extreme_patterns = patterns.get('high_indicators', []) + patterns.get('low_indicators', [])
                        for pattern in extreme_patterns:
                            if pattern in text_lower:
                                result.add_warning(
                                    f"Moderate {score_type} score ({score}) with absolute language: '{pattern}'",
                                    f"Consider adjusting score or using more nuanced language"
                                )

            # Report major inconsistencies
            for inconsistency in inconsistencies:
                result.add_error(inconsistency, "Adjust either the score or the textual analysis for consistency")

            # Additional semantic consistency checks
            semantic_issues = self._check_semantic_consistency(text_lower, scores)
            for issue in semantic_issues:
                result.add_warning(issue, "Review overall narrative consistency")

            # Final validation
            result.is_valid = len(result.errors) == 0
            result.details['inconsistencies_found'] = len(inconsistencies)
            result.details['semantic_issues'] = len(semantic_issues)
            
            if inconsistencies:
                result.score = max(0.0, 100.0 - (len(inconsistencies) * 25))
            else:
                result.score = max(80.0, 100.0 - (len(semantic_issues) * 5))

            self.logger.info(
                f"Score consistency validation completed",
                extra={
                    'session_id': session_id,
                    'is_consistent': result.is_valid,
                    'inconsistencies': len(inconsistencies),
                    'semantic_issues': len(semantic_issues)
                }
            )

        except Exception as e:
            result.add_error(f"Consistency validation error: {str(e)}")
            self.logger.error(f"Score consistency validation failed: {str(e)}", extra={'session_id': session_id})

        return result

    def _check_semantic_consistency(self, text_lower: str, scores: Dict[str, int]) -> List[str]:
        """Check for semantic consistency issues in the overall narrative."""
        issues = []
        
        try:
            # Check for contradictory statements
            contradiction_patterns = [
                (r'no\s+\w+.*but.*\w+', "Contradictory 'no X but Y' statements"),
                (r'minimal\s+\w+.*however.*significant', "Contradictory minimal/significant statements"),
                (r'low\s+\w+.*despite.*high', "Contradictory low/high statements")
            ]
            
            for pattern, description in contradiction_patterns:
                if re.search(pattern, text_lower):
                    issues.append(f"Potential contradiction detected: {description}")
            
            # Check score relationships
            bias_score = scores.get('bias', 50)
            manipulation_score = scores.get('manipulation', 50)
            credibility_score = scores.get('credibility', 50)
            
            # Logical relationship checks
            if bias_score > 80 and credibility_score > 80:
                issues.append("High bias and high credibility scores seem contradictory")
            
            if manipulation_score > 80 and credibility_score > 80:
                issues.append("High manipulation and high credibility scores seem contradictory")
            
            if bias_score < 20 and manipulation_score > 80:
                issues.append("Very low bias with very high manipulation seems unlikely")
                
        except Exception:
            # Don't fail consistency check on semantic analysis errors
            pass
            
        return issues

    def validate_risk_assessment_consistency(self, analysis_text: str, risk_level: str, risk_score: int, session_id: str = None) -> ValidationResult:
        """
        Validate consistency between risk assessment text and risk classification.

        Args:
            analysis_text: Risk assessment text
            risk_level: Assigned risk level (LOW, MEDIUM, HIGH, etc.)
            risk_score: Numerical risk score
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with consistency validation and suggestions
        """
        result = ValidationResult()

        try:
            text_lower = analysis_text.lower() if analysis_text else ""
            
            # Risk level to score mapping
            risk_mappings = {
                'MINIMAL': (0, 20),
                'LOW': (20, 40),
                'MEDIUM': (40, 60),
                'HIGH': (60, 80),
                'CRITICAL': (80, 100)
            }

            # Check if score matches level
            if risk_level in risk_mappings:
                min_score, max_score = risk_mappings[risk_level]
                if not (min_score <= risk_score <= max_score):
                    result.add_error(
                        f"Risk level '{risk_level}' inconsistent with score {risk_score}",
                        f"Adjust score to {min_score}-{max_score} range or change risk level"
                    )

            # Check text consistency with level
            risk_text_indicators = {
                'MINIMAL': ['no risk', 'minimal risk', 'very low risk', 'safe'],
                'LOW': ['low risk', 'minor concerns', 'slight risk', 'limited threat'],
                'MEDIUM': ['moderate risk', 'some concerns', 'medium risk', 'potential issues'],
                'HIGH': ['high risk', 'significant concerns', 'major risk', 'serious threat'],
                'CRITICAL': ['critical risk', 'severe risk', 'dangerous', 'urgent', 'immediate threat']
            }

            expected_indicators = risk_text_indicators.get(risk_level, [])
            if expected_indicators and not any(indicator in text_lower for indicator in expected_indicators):
                result.add_warning(
                    f"Text doesn't contain expected language for risk level '{risk_level}'",
                    f"Consider including terms like: {', '.join(expected_indicators[:3])}"
                )

            # Final validation
            result.is_valid = len(result.errors) == 0
            result.score = max(0.0, 100.0 - (len(result.errors) * 30) - (len(result.warnings) * 10))

            self.logger.info(
                f"Risk assessment consistency validation completed",
                extra={
                    'session_id': session_id,
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'is_consistent': result.is_valid
                }
            )

        except Exception as e:
            result.add_error(f"Risk consistency validation error: {str(e)}")
            self.logger.error(f"Risk assessment consistency validation failed: {str(e)}", extra={'session_id': session_id})

        return result


class OutputValidator:
    """
    Production-ready comprehensive output validation for context analyzer results.
    
    Validates all aspects of context analysis output including scores,
    analysis text, result consistency, and completeness with detailed feedback.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize output validator with production configuration.

        Args:
            config: Optional configuration for validation rules and thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.OutputValidator")
        
        self.bias_validator = BiasAnalysisValidator(config)
        self.manipulation_validator = ManipulationAnalysisValidator(config)
        self.consistency_validator = ScoringConsistencyValidator(config)

    def validate_context_analysis_output(self, output: Any, session_id: str = None) -> ValidationResult:
        """
        Validate complete context analysis output with comprehensive checks.

        Args:
            output: Context analysis output to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with comprehensive validation results and suggestions
        """
        result = ValidationResult()
        
        self.logger.info(f"Starting comprehensive output validation", extra={'session_id': session_id})

        try:
            # Basic structure validation
            if not isinstance(output, dict):
                result.add_error(
                    f"Context analysis output must be dictionary, got {type(output).__name__}",
                    "Ensure output is structured as dictionary with required sections"
                )
                return result

            # Required sections validation with descriptions
            required_sections = {
                'llm_analysis': 'Textual analysis from language model',
                'llm_scores': 'Numerical scores from analysis',
                'context_scores': 'Processed context evaluation scores'
            }

            missing_sections = []
            for section, description in required_sections.items():
                if section not in output:
                    missing_sections.append(section)
                    result.add_error(
                        f"Missing required section: '{section}'",
                        f"Add {description.lower()}: '{section}': analysis_content"
                    )

            if missing_sections:
                return result

            # Validate LLM scores with detailed feedback
            if 'llm_scores' in output:
                scores_result = self.bias_validator.validate_bias_scores(output['llm_scores'], session_id)
                result.errors.extend([f"LLM scores: {error}" for error in scores_result.errors])
                result.warnings.extend([f"LLM scores: {warning}" for warning in scores_result.warnings])
                result.suggestions.extend(scores_result.suggestions)

            # Validate manipulation report if present
            if 'manipulation_report' in output:
                manipulation_result = self.manipulation_validator.validate_manipulation_report(
                    output['manipulation_report'], session_id
                )
                result.errors.extend([f"Manipulation report: {error}" for error in manipulation_result.errors])
                result.warnings.extend([f"Manipulation report: {warning}" for warning in manipulation_result.warnings])
                result.suggestions.extend(manipulation_result.suggestions)

            # Validate scoring consistency between text and numbers
            if 'llm_analysis' in output and 'llm_scores' in output:
                consistency_result = self.consistency_validator.validate_score_text_consistency(
                    output['llm_analysis'], output['llm_scores'], session_id
                )
                result.errors.extend([f"Consistency check: {error}" for error in consistency_result.errors])
                result.warnings.extend([f"Consistency check: {warning}" for warning in consistency_result.warnings])
                result.suggestions.extend(consistency_result.suggestions)

            # Validate context scores structure
            if 'context_scores' in output:
                context_scores = output['context_scores']
                if isinstance(context_scores, dict):
                    expected_context_fields = ['bias_score', 'manipulation_score', 'credibility', 'risk_level']
                    for field in expected_context_fields:
                        if field not in context_scores:
                            result.add_warning(
                                f"Context scores missing field: '{field}'",
                                f"Add processed score: '{field}': calculated_value"
                            )
                else:
                    result.add_error(
                        f"Context scores must be dictionary, got {type(context_scores).__name__}",
                        "Provide context scores as: {'bias_score': 75, 'manipulation_score': 60, ...}"
                    )

            # Check for completeness indicators
            completeness_score = 100.0
            optional_sections = ['bias_analysis', 'framing_analysis', 'emotional_analysis', 'propaganda_analysis']
            present_optional = sum(1 for section in optional_sections if section in output)
            
            if present_optional == 0:
                result.add_warning(
                    "No detailed analysis sections present",
                    "Consider adding detailed bias or manipulation analysis for comprehensive results"
                )
                completeness_score -= 20
            elif present_optional < len(optional_sections) // 2:
                result.add_warning(
                    f"Limited detailed analysis: {present_optional}/{len(optional_sections)} sections present",
                    "Additional detailed analysis sections may improve result quality"
                )
                completeness_score -= 10

            # Validate metadata if present
            if 'metadata' in output:
                metadata = output['metadata']
                if isinstance(metadata, dict):
                    recommended_metadata = ['analysis_timestamp', 'model_used', 'processing_time', 'agent_version']
                    missing_metadata = [field for field in recommended_metadata if field not in metadata]
                    if missing_metadata:
                        result.add_warning(
                            f"Metadata missing recommended fields: {missing_metadata}",
                            "Include comprehensive metadata for better traceability"
                        )

            # Final validation scoring
            error_penalty = len(result.errors) * 20
            warning_penalty = len(result.warnings) * 5
            final_score = max(0.0, min(completeness_score - error_penalty - warning_penalty, 100.0))
            
            result.is_valid = len(result.errors) == 0
            result.score = final_score
            result.details = {
                'required_sections_present': len(required_sections) - len(missing_sections),
                'total_required_sections': len(required_sections),
                'optional_sections_present': present_optional,
                'consistency_checked': 'llm_analysis' in output and 'llm_scores' in output,
                'completeness_score': completeness_score
            }

            self.logger.info(
                f"Comprehensive output validation completed",
                extra={
                    'session_id': session_id,
                    'is_valid': result.is_valid,
                    'final_score': final_score,
                    'error_count': len(result.errors),
                    'warning_count': len(result.warnings)
                }
            )

        except Exception as e:
            result.add_error(f"Output validation error: {str(e)}")
            self.logger.error(f"Output validation failed: {str(e)}", extra={'session_id': session_id})

        return result

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics for monitoring."""
        try:
            return {
                'validators_active': {
                    'input_validator': bool(hasattr(self, 'input_validator')),
                    'bias_validator': bool(self.bias_validator),
                    'manipulation_validator': bool(self.manipulation_validator),
                    'consistency_validator': bool(self.consistency_validator)
                },
                'validation_capabilities': [
                    'Input structure validation',
                    'Article text quality assessment', 
                    'Previous analysis validation',
                    'Bias scoring validation',
                    'Manipulation report validation',
                    'Score-text consistency checking',
                    'Output completeness validation',
                    'Semantic consistency checking'
                ],
                'configuration': self.config,
                'validation_active': True
            }
        except Exception as e:
            self.logger.error(f"Failed to get validation statistics: {str(e)}")
            return {'error': str(e), 'validation_active': False}


# Convenience validation functions for easy access
def validate_context_input(input_data: Dict[str, Any], config: Dict[str, Any] = None, session_id: str = None) -> ValidationResult:
    """
    Validate complete context analyzer input with comprehensive feedback.

    Args:
        input_data: Input data to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult with validation status, errors, warnings, and suggestions
    """
    validator = InputValidator(config)
    return validator.validate_processing_input(input_data, session_id)


def validate_bias_analysis(analysis_text: str, scores: Dict[str, int], config: Dict[str, Any] = None, session_id: str = None) -> bool:
    """
    Quick validation for bias analysis consistency.

    Args:
        analysis_text: Analysis text to validate
        scores: Numerical scores to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        True if consistent, False if inconsistent
    """
    validator = ScoringConsistencyValidator(config)
    result = validator.validate_score_text_consistency(analysis_text, scores, session_id)
    return result.is_valid


def validate_context_output(output: Dict[str, Any], config: Dict[str, Any] = None, session_id: str = None) -> ValidationResult:
    """
    Validate context analysis output with comprehensive checks.

    Args:
        output: Context analysis output to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult with validation status, errors, warnings, and suggestions
    """
    validator = OutputValidator(config)
    return validator.validate_context_analysis_output(output, session_id)


def validate_manipulation_techniques(techniques: Dict[str, Any], config: Dict[str, Any] = None, session_id: str = None) -> ValidationResult:
    """
    Validate manipulation detection techniques with detailed feedback.

    Args:
        techniques: Manipulation techniques to validate
        config: Optional validation configuration  
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult with validation status and suggestions
    """
    validator = ManipulationAnalysisValidator(config)
    return validator.validate_propaganda_techniques(techniques, session_id)


# Testing functionality
if __name__ == "__main__":
    """Test context analyzer validation functionality with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=== CONTEXT ANALYZER VALIDATION TEST ===")
    
    try:
        # Test input validation
        test_input = {
            'text': 'This is a test article with sufficient length for context analysis purposes and proper structure with multiple sentences. It contains enough content for meaningful bias and manipulation detection.',
            'previous_analysis': {
                'prediction': 'REAL',
                'confidence': 0.85,
                'source': 'Test Source',
                'topic_domain': 'general'
            },
            'include_detailed_analysis': True
        }

        print("--- Input Validation Test ---")
        input_result = validate_context_input(test_input, session_id="validation_test_001")
        print(f"Input validation: {'✅ PASSED' if input_result.is_valid else '❌ FAILED'}")
        print(f"Score: {input_result.score:.1f}/100")
        if input_result.errors:
            print(f"Errors: {input_result.errors}")
        if input_result.warnings:
            print(f"Warnings: {input_result.warnings[:2]}")  # Show first 2 warnings

        # Test bias analysis consistency
        print("\n--- Bias Analysis Consistency Test ---")
        test_analysis = "This article shows minimal bias and straightforward presentation with high credibility and low manipulation."
        test_scores = {'bias': 15, 'manipulation': 10, 'credibility': 85, 'risk': 20}
        
        is_consistent = validate_bias_analysis(test_analysis, test_scores, session_id="validation_test_002")
        print(f"Consistent analysis: {'✅ PASSED' if is_consistent else '❌ FAILED'}")

        # Test inconsistent scoring (should fail)
        print("\n--- Inconsistent Analysis Detection Test ---")
        inconsistent_analysis = "This article shows extreme bias and heavy manipulation with questionable credibility."
        inconsistent_scores = {'bias': 10, 'manipulation': 15, 'credibility': 90, 'risk': 5}
        
        is_inconsistent = validate_bias_analysis(inconsistent_analysis, inconsistent_scores, session_id="validation_test_003")
        print(f"Inconsistency detected: {'✅ PASSED' if not is_inconsistent else '❌ FAILED'}")

        # Test output validation
        print("\n--- Output Validation Test ---")
        test_output = {
            'llm_analysis': test_analysis,
            'llm_scores': test_scores,
            'context_scores': {
                'bias_score': test_scores['bias'],
                'manipulation_score': test_scores['manipulation'],
                'credibility': test_scores['credibility'],
                'risk_level': 'LOW',
                'risk_score': test_scores['risk']
            },
            'metadata': {
                'analysis_timestamp': '2025-09-11T12:30:00Z',
                'model_used': 'gemini-1.5-pro',
                'agent_version': '3.1.0'
            }
        }

        output_result = validate_context_output(test_output, session_id="validation_test_004")
        print(f"Output validation: {'✅ PASSED' if output_result.is_valid else '❌ FAILED'}")
        print(f"Score: {output_result.score:.1f}/100")
        if output_result.errors:
            print(f"Errors: {output_result.errors}")
        if output_result.warnings:
            print(f"Warnings: {output_result.warnings[:2]}")

        # Test manipulation techniques validation
        print("\n--- Manipulation Techniques Validation Test ---")
        test_techniques = {
            'detected': {
                'fear_mongering': {
                    'confidence': 0.8,
                    'severity': 'high',
                    'description': 'Creates fear to motivate specific actions'
                },
                'false_dilemma': {
                    'confidence': 0.6,
                    'severity': 'medium',
                    'description': 'Presents only two options when more exist'
                }
            }
        }

        techniques_result = validate_manipulation_techniques(test_techniques, session_id="validation_test_005")
        print(f"Techniques validation: {'✅ PASSED' if techniques_result.is_valid else '❌ FAILED'}")
        print(f"Valid techniques: {techniques_result.details.get('valid_techniques', 0)}/{techniques_result.details.get('total_techniques', 0)}")

        # Test edge cases
        print("\n--- Edge Cases Test ---")
        
        # Empty input
        empty_result = validate_context_input({}, session_id="validation_test_006")
        print(f"Empty input rejected: {'✅ PASSED' if not empty_result.is_valid else '❌ FAILED'}")
        
        # Short text
        short_input = {'text': 'Short text'}
        short_result = validate_context_input(short_input, session_id="validation_test_007")
        print(f"Short text rejected: {'✅ PASSED' if not short_result.is_valid else '❌ FAILED'}")

        print("\n✅ Context analyzer validation tests completed successfully!")

    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
