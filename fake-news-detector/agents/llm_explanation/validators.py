# agents/llm_explanation/validators.py

"""
LLM Explanation Agent Validators - Production Ready

Comprehensive validation system for ensuring data quality and consistency
in explanation generation workflows. Features detailed error reporting,
security validation, quality assessment, and performance tracking for
reliable production use in fake news explanation systems.
"""

import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from .exceptions import (
    InputValidationError,
    DataFormatError,
    raise_input_validation_error
)


@dataclass
class ValidationResult:
    """
    Comprehensive validation result container with enhanced metadata and suggestions.
    
    Attributes:
        is_valid: Whether validation passed successfully
        errors: List of validation error messages
        warnings: List of validation warnings
        score: Validation quality score (0-100)
        details: Additional validation details and metrics
        suggestions: Actionable suggestions for fixing validation issues
        session_id: Optional session ID for tracking
        validation_time: Time taken for validation in seconds
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    score: float = 0.0
    details: Dict[str, Any] = None
    suggestions: List[str] = None
    session_id: str = None
    validation_time: float = 0.0

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.details is None:
            self.details = {}
        if self.suggestions is None:
            self.suggestions = []

    def add_error(self, error: str, suggestion: str = None) -> None:
        """Add validation error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False
        if suggestion:
            self.suggestions.append(suggestion)

    def add_warning(self, warning: str, suggestion: str = None) -> None:
        """Add validation warning."""
        self.warnings.append(warning)
        if suggestion:
            self.suggestions.append(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses and logging."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'score': self.score,
            'details': self.details,
            'suggestions': self.suggestions,
            'session_id': self.session_id,
            'validation_time_seconds': self.validation_time,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'suggestion_count': len(self.suggestions)
        }


class InputValidator:
    """
    Comprehensive input validator for explanation generation requests.
    
    Validates article text, predictions, confidence scores, metadata,
    and other input parameters with enhanced security checks, quality
    assessment, and detailed error reporting for production environments.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize input validator with production configuration.

        Args:
            config: Optional validation configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.InputValidator")
        
        # Text validation thresholds
        self.min_text_length = self.config.get('min_text_length', 100)
        self.max_text_length = self.config.get('max_text_length', 15000)
        self.min_word_count = self.config.get('min_word_count', 15)
        self.max_word_count = self.config.get('max_word_count', 2500)
        
        # Confidence validation thresholds
        self.min_confidence = self.config.get('min_confidence', 0.0)
        self.max_confidence = self.config.get('max_confidence', 1.0)
        
        # Quality assessment thresholds
        self.min_sentence_count = self.config.get('min_sentence_count', 3)
        self.max_repeated_chars = self.config.get('max_repeated_chars', 10)
        self.min_unique_words_ratio = self.config.get('min_unique_words_ratio', 0.25)
        
        # Security validation settings
        self.enable_security_checks = self.config.get('enable_security_checks', True)
        self.blocked_patterns = self.config.get('blocked_patterns', [])
        
        # Performance tracking
        self.validation_count = 0
        self.total_validation_time = 0.0
        
        self.logger.info(f"InputValidator initialized with security checks: {self.enable_security_checks}")

    def validate_explanation_input(self, input_data: Dict[str, Any], session_id: str = None) -> ValidationResult:
        """
        Validate complete explanation input data with comprehensive checks.

        Args:
            input_data: Input dictionary containing explanation parameters
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with detailed validation results and suggestions
        """
        start_time = time.time()
        self.validation_count += 1

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            session_id=session_id
        )

        try:
            # Basic structure validation
            if not isinstance(input_data, dict):
                result.add_error(
                    "Input data must be a dictionary",
                    "Provide input as a dictionary with required fields"
                )
                return result

            if not input_data:
                result.add_error(
                    "Input data cannot be empty",
                    "Provide input data with at least the 'text' field"
                )
                return result

            # Validate required fields
            required_fields = ['text', 'prediction', 'confidence']
            for field in required_fields:
                if field not in input_data:
                    result.add_error(
                        f"Missing required field: {field}",
                        f"Include '{field}' field in input data"
                    )

            # Validate article text
            if 'text' in input_data:
                text_validation = self.validate_article_text(input_data['text'], session_id)
                result.details['text_validation'] = text_validation.details
                
                if not text_validation.is_valid:
                    result.errors.extend([f"Text validation: {error}" for error in text_validation.errors])
                    result.suggestions.extend(text_validation.suggestions)
                    result.is_valid = False
                
                if text_validation.warnings:
                    result.warnings.extend([f"Text warning: {warning}" for warning in text_validation.warnings])

            # Validate prediction
            if 'prediction' in input_data:
                pred_validation = self.validate_prediction(input_data['prediction'], session_id)
                result.details['prediction_validation'] = pred_validation.details
                
                if not pred_validation.is_valid:
                    result.errors.extend([f"Prediction validation: {error}" for error in pred_validation.errors])
                    result.suggestions.extend(pred_validation.suggestions)
                    result.is_valid = False
                
                if pred_validation.warnings:
                    result.warnings.extend([f"Prediction warning: {warning}" for warning in pred_validation.warnings])

            # Validate confidence score
            if 'confidence' in input_data:
                conf_validation = self.validate_confidence(input_data['confidence'], session_id)
                result.details['confidence_validation'] = conf_validation.details
                
                if not conf_validation.is_valid:
                    result.errors.extend([f"Confidence validation: {error}" for error in conf_validation.errors])
                    result.suggestions.extend(conf_validation.suggestions)
                    result.is_valid = False
                
                if conf_validation.warnings:
                    result.warnings.extend([f"Confidence warning: {warning}" for warning in conf_validation.warnings])

            # Validate optional metadata
            if 'metadata' in input_data:
                meta_validation = self.validate_metadata(input_data['metadata'], session_id)
                result.details['metadata_validation'] = meta_validation.details
                
                if not meta_validation.is_valid:
                    result.warnings.extend([f"Metadata validation: {error}" for error in meta_validation.errors])
                
                if meta_validation.warnings:
                    result.warnings.extend([f"Metadata warning: {warning}" for warning in meta_validation.warnings])

            # Validate optional parameters
            optional_fields = {
                'require_detailed_analysis': bool,
                'session_id': str,
                'priority_level': (str, ['high', 'medium', 'low']),
                'domain': (str, ['health', 'politics', 'science', 'technology', 'economics', 'general'])
            }

            for field, validation_spec in optional_fields.items():
                if field in input_data:
                    field_validation = self._validate_optional_field(field, input_data[field], validation_spec)
                    if field_validation.warnings:
                        result.warnings.extend([f"{field.title()} warning: {warning}" for warning in field_validation.warnings])

            # Overall quality assessment
            quality_score = self._assess_input_quality(input_data)
            result.score = quality_score
            result.details['quality_score'] = quality_score
            
            if quality_score < 60:
                result.add_warning(
                    f"Input quality score is low: {quality_score:.1f}/100",
                    "Review input data completeness and quality"
                )

            # Processing time tracking
            validation_time = time.time() - start_time
            result.validation_time = validation_time
            self.total_validation_time += validation_time
            result.details['validation_time_ms'] = round(validation_time * 1000, 2)

            # Final status logging
            self.logger.info(
                f"Input validation completed: {'PASSED' if result.is_valid else 'FAILED'}",
                extra={
                    'session_id': session_id,
                    'validation_score': result.score,
                    'error_count': len(result.errors),
                    'warning_count': len(result.warnings),
                    'validation_time_ms': round(validation_time * 1000, 2)
                }
            )

            return result

        except Exception as e:
            validation_time = time.time() - start_time
            result.validation_time = validation_time
            self.logger.error(f"Input validation failed: {str(e)}", extra={'session_id': session_id})
            
            result.add_error(
                f"Validation error: {str(e)}",
                "Check input format and try again"
            )
            return result

    def validate_article_text(self, text: Any, session_id: str = None) -> ValidationResult:
        """
        Validate article text content with comprehensive quality checks.

        Args:
            text: Article text to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult for text validation with detailed feedback
        """
        result = ValidationResult(is_valid=True, session_id=session_id)
        
        try:
            # Basic type validation
            if not isinstance(text, str):
                result.add_error(
                    f"Article text must be string, got {type(text).__name__}",
                    "Provide text input as a string type"
                )
                return result

            if not text or not text.strip():
                result.add_error(
                    "Article text cannot be empty",
                    "Provide non-empty article text for analysis"
                )
                return result

            text_clean = text.strip()
            text_length = len(text_clean)
            result.details['original_length'] = len(text)
            result.details['clean_length'] = text_length

            # Length validation
            if text_length < self.min_text_length:
                result.add_error(
                    f"Article text too short: {text_length} characters (minimum: {self.min_text_length})",
                    f"Provide at least {self.min_text_length} characters of meaningful content"
                )

            if text_length > self.max_text_length:
                result.add_warning(
                    f"Article text very long: {text_length} characters (may be truncated)",
                    f"Consider summarizing content to under {self.max_text_length} characters"
                )

            # Word count validation
            words = text_clean.split()
            word_count = len(words)
            result.details['word_count'] = word_count

            if word_count < self.min_word_count:
                result.add_error(
                    f"Article has too few words: {word_count} (minimum: {self.min_word_count})",
                    f"Provide at least {self.min_word_count} words for meaningful analysis"
                )

            if word_count > self.max_word_count:
                result.add_warning(
                    f"Article is very long: {word_count} words",
                    "Consider splitting very long articles for better processing"
                )

            # Sentence structure validation
            sentences = re.split(r'[.!?]+', text_clean)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            sentence_count = len(valid_sentences)
            result.details['sentence_count'] = sentence_count

            if sentence_count < self.min_sentence_count:
                result.add_error(
                    f"Article has too few sentences: {sentence_count} (minimum: {self.min_sentence_count})",
                    f"Provide at least {self.min_sentence_count} complete sentences"
                )

            # Content quality checks
            if words and sentence_count > 0:
                # Vocabulary diversity
                unique_words = set(word.lower().strip('.,!?;:"()[]{}') for word in words if word.isalnum())
                unique_ratio = len(unique_words) / len(words)
                result.details['unique_words_ratio'] = round(unique_ratio, 3)

                if unique_ratio < self.min_unique_words_ratio:
                    result.add_warning(
                        f"Low vocabulary diversity: {unique_ratio:.2f}",
                        "Ensure article contains varied vocabulary for better analysis"
                    )

                # Average sentence length assessment
                avg_sentence_length = len(words) / sentence_count
                result.details['avg_sentence_length'] = round(avg_sentence_length, 1)

                if avg_sentence_length < 5:
                    result.add_warning(
                        "Very short average sentence length",
                        "Check for sentence fragments or formatting issues"
                    )
                elif avg_sentence_length > 50:
                    result.add_warning(
                        "Very long average sentence length",
                        "Consider breaking up complex sentences for clarity"
                    )

            # Content format checks
            format_issues = self._check_content_format(text_clean)
            if format_issues:
                result.warnings.extend(format_issues)

            # Security validation
            if self.enable_security_checks:
                security_issues = self._check_security_patterns(text_clean, session_id)
                if security_issues:
                    result.errors.extend(security_issues)
                    result.suggestions.append("Remove potentially malicious content patterns")

            return result

        except Exception as e:
            self.logger.error(f"Article text validation failed: {str(e)}", extra={'session_id': session_id})
            result.add_error(
                f"Text validation error: {str(e)}",
                "Check text format and encoding"
            )
            return result

    def validate_prediction(self, prediction: Any, session_id: str = None) -> ValidationResult:
        """
        Validate prediction value with comprehensive format checking.

        Args:
            prediction: Prediction value to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult for prediction validation
        """
        result = ValidationResult(is_valid=True, session_id=session_id)

        try:
            # Type validation
            if not isinstance(prediction, str):
                result.add_error(
                    f"Prediction must be string, got {type(prediction).__name__}",
                    "Provide prediction as a string value"
                )
                return result

            # Value validation
            valid_predictions = ['FAKE', 'REAL', 'UNKNOWN', 'UNCERTAIN']
            prediction_clean = prediction.strip().upper()
            
            if prediction_clean not in valid_predictions:
                result.add_error(
                    f"Invalid prediction: '{prediction}' (valid options: {', '.join(valid_predictions)})",
                    f"Use one of: {', '.join(valid_predictions)}"
                )
            else:
                result.details['prediction_normalized'] = prediction_clean
                
                # Quality warnings
                if prediction_clean in ['UNKNOWN', 'UNCERTAIN']:
                    result.add_warning(
                        f"Prediction is {prediction_clean} - explanation quality may be limited",
                        "Consider providing additional context for uncertain predictions"
                    )

            return result

        except Exception as e:
            self.logger.error(f"Prediction validation failed: {str(e)}", extra={'session_id': session_id})
            result.add_error(
                f"Prediction validation error: {str(e)}",
                "Check prediction format and value"
            )
            return result

    def validate_confidence(self, confidence: Any, session_id: str = None) -> ValidationResult:
        """
        Validate confidence score with comprehensive range checking.

        Args:
            confidence: Confidence score to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult for confidence validation
        """
        result = ValidationResult(is_valid=True, session_id=session_id)

        try:
            # Type validation
            if not isinstance(confidence, (int, float)):
                result.add_error(
                    f"Confidence must be numeric, got {type(confidence).__name__}",
                    "Provide confidence as a float between 0.0 and 1.0"
                )
                return result

            # Range validation
            if confidence < self.min_confidence:
                result.add_error(
                    f"Confidence {confidence} below minimum {self.min_confidence}",
                    f"Ensure confidence is at least {self.min_confidence}"
                )
            
            if confidence > self.max_confidence:
                result.add_error(
                    f"Confidence {confidence} above maximum {self.max_confidence}",
                    f"Ensure confidence is at most {self.max_confidence}"
                )

            # Quality assessment warnings
            if self.min_confidence <= confidence <= self.max_confidence:
                result.details['confidence_value'] = confidence
                result.details['confidence_category'] = self._categorize_confidence(confidence)
                
                if confidence < 0.3:
                    result.add_warning(
                        f"Very low confidence: {confidence:.2f}",
                        "Low confidence predictions may require additional verification"
                    )
                elif confidence < 0.5:
                    result.add_warning(
                        f"Low confidence: {confidence:.2f}",
                        "Consider reviewing prediction logic and evidence"
                    )
                elif confidence > 0.95:
                    result.add_warning(
                        f"Very high confidence: {confidence:.2f}",
                        "Ensure high confidence is justified by strong evidence"
                    )

            return result

        except Exception as e:
            self.logger.error(f"Confidence validation failed: {str(e)}", extra={'session_id': session_id})
            result.add_error(
                f"Confidence validation error: {str(e)}",
                "Check confidence format and range"
            )
            return result

    def validate_metadata(self, metadata: Any, session_id: str = None) -> ValidationResult:
        """
        Validate metadata dictionary with comprehensive field checking.

        Args:
            metadata: Metadata dictionary to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult for metadata validation
        """
        result = ValidationResult(is_valid=True, session_id=session_id)

        try:
            # Type validation
            if not isinstance(metadata, dict):
                result.add_error(
                    f"Metadata must be dictionary, got {type(metadata).__name__}",
                    "Provide metadata as a dictionary with relevant fields"
                )
                return result

            # Validate common metadata fields
            expected_fields = {
                'source': str,
                'date': str,
                'subject': str,
                'author': str,
                'domain': str,
                'publication_type': str
            }

            present_fields = 0
            for field, expected_type in expected_fields.items():
                if field in metadata:
                    present_fields += 1
                    value = metadata[field]
                    
                    if not isinstance(value, expected_type):
                        result.add_warning(
                            f"Metadata '{field}' should be {expected_type.__name__}, got {type(value).__name__}",
                            f"Convert '{field}' to appropriate type"
                        )
                    elif isinstance(value, str) and not value.strip():
                        result.add_warning(
                            f"Metadata '{field}' is empty",
                            f"Provide meaningful value for '{field}'"
                        )

            # Completeness assessment
            completeness_ratio = present_fields / len(expected_fields)
            result.details['completeness_ratio'] = round(completeness_ratio, 2)
            result.details['present_fields'] = present_fields
            result.details['total_expected'] = len(expected_fields)

            if completeness_ratio < 0.5:
                result.add_warning(
                    f"Metadata is incomplete: {present_fields}/{len(expected_fields)} fields present",
                    "Consider providing additional metadata fields for better analysis"
                )

            # Specific field validation
            if 'date' in metadata:
                date_validation = self._validate_date_format(metadata['date'])
                if not date_validation:
                    result.add_warning(
                        f"Date format '{metadata['date']}' may not be standard",
                        "Use standard date format (YYYY-MM-DD or similar)"
                    )

            if 'source' in metadata:
                source_validation = self._validate_source_format(metadata['source'])
                if not source_validation:
                    result.add_warning(
                        "Source format may need verification",
                        "Ensure source is a valid publication name or URL"
                    )

            return result

        except Exception as e:
            self.logger.error(f"Metadata validation failed: {str(e)}", extra={'session_id': session_id})
            result.add_error(
                f"Metadata validation error: {str(e)}",
                "Check metadata format and content"
            )
            return result

    def _validate_optional_field(self, field_name: str, field_value: Any, 
                                validation_spec: Union[type, Tuple]) -> ValidationResult:
        """Validate optional field with type and value checking."""
        result = ValidationResult(is_valid=True)
        
        if isinstance(validation_spec, tuple):
            expected_type, valid_values = validation_spec
            
            if not isinstance(field_value, expected_type):
                result.add_warning(
                    f"Field {field_name} should be {expected_type.__name__}",
                    f"Convert {field_name} to {expected_type.__name__} type"
                )
            elif valid_values and field_value not in valid_values:
                result.add_warning(
                    f"Field {field_name} has invalid value: {field_value}",
                    f"Use one of: {', '.join(map(str, valid_values))}"
                )
        else:
            expected_type = validation_spec
            if not isinstance(field_value, expected_type):
                result.add_warning(
                    f"Field {field_name} should be {expected_type.__name__}",
                    f"Convert {field_name} to {expected_type.__name__} type"
                )
        
        return result

    def _check_content_format(self, text: str) -> List[str]:
        """Check for content format issues and patterns."""
        format_issues = []

        # Check for excessive whitespace
        if re.search(r'\s{10,}', text):
            format_issues.append("Contains excessive whitespace")

        # Check for non-printable characters
        non_printable = re.findall(r'[^\x20-\x7E\n\r\t]', text)
        if len(non_printable) > 20:
            format_issues.append(f"Contains {len(non_printable)} non-printable characters")

        # Check for HTML tags
        html_tags = re.findall(r'<[^>]+>', text)
        if len(html_tags) > 5:
            format_issues.append("Contains HTML tags - may need preprocessing")

        # Check for URL density
        urls = re.findall(r'https?://[^\s]+', text)
        url_ratio = len(urls) / max(len(text.split()), 1)
        if url_ratio > 0.1:
            format_issues.append(f"High URL density: {len(urls)} URLs found")

        return format_issues

    def _check_security_patterns(self, text: str, session_id: str = None) -> List[str]:
        """Check for security-related patterns in text."""
        security_issues = []

        # Script injection patterns
        script_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*='
        ]

        for pattern in script_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                security_issues.append("Potentially malicious script pattern detected")
                self.logger.warning("Script pattern detected", extra={'session_id': session_id})
                break

        # SQL injection patterns
        sql_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'--\s*$',
            r'/\*.*\*/'
        ]

        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                security_issues.append("Potentially malicious SQL pattern detected")
                self.logger.warning("SQL pattern detected", extra={'session_id': session_id})
                break

        return security_issues

    def _assess_input_quality(self, input_data: Dict[str, Any]) -> float:
        """Assess overall input quality score."""
        score = 70  # Base score
        
        # Text quality contribution (30 points)
        if 'text' in input_data:
            text = input_data['text']
            if isinstance(text, str):
                text_length = len(text.strip())
                if text_length >= self.min_text_length:
                    score += 15
                if text_length <= self.max_text_length:
                    score += 10
                
                words = text.split()
                if len(words) >= self.min_word_count:
                    score += 5

        # Metadata contribution (20 points)
        metadata = input_data.get('metadata', {})
        if isinstance(metadata, dict):
            field_count = len([v for v in metadata.values() if v and str(v).strip()])
            metadata_score = min(20, field_count * 3)
            score += metadata_score

        # Prediction and confidence validity (10 points)
        prediction = input_data.get('prediction', '')
        if isinstance(prediction, str) and prediction.upper() in ['FAKE', 'REAL']:
            score += 5
        
        confidence = input_data.get('confidence', 0)
        if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
            score += 5

        return min(100.0, max(0.0, score))

    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level for analysis."""
        if confidence < 0.3:
            return "very_low"
        elif confidence < 0.5:
            return "low"
        elif confidence < 0.7:
            return "moderate"
        elif confidence < 0.9:
            return "high"
        else:
            return "very_high"

    def _validate_date_format(self, date_str: str) -> bool:
        """Validate date string format."""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
        ]
        
        return any(re.match(pattern, date_str) for pattern in date_patterns)

    def _validate_source_format(self, source_str: str) -> bool:
        """Validate source string format."""
        # Basic checks for reasonable source format
        if len(source_str.strip()) < 3:
            return False
        
        # Check for URL format
        if re.match(r'https?://', source_str):
            return True
        
        # Check for publication name format
        if re.match(r'^[A-Za-z0-9\s\.\-_]+$', source_str):
            return True
        
        return False

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive input validation statistics."""
        avg_time = (self.total_validation_time / self.validation_count) if self.validation_count > 0 else 0
        
        return {
            'validation_count': self.validation_count,
            'total_validation_time': self.total_validation_time,
            'average_validation_time': avg_time,
            'configuration': {
                'min_text_length': self.min_text_length,
                'max_text_length': self.max_text_length,
                'min_word_count': self.min_word_count,
                'max_word_count': self.max_word_count,
                'security_checks_enabled': self.enable_security_checks
            }
        }


class OutputValidator:
    """
    Comprehensive output validator for explanation generation results.
    
    Validates explanation content, analysis components, metadata, and
    quality metrics with detailed assessment and production monitoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize output validator with production configuration.

        Args:
            config: Optional validation configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.OutputValidator")
        
        # Content validation thresholds
        self.min_explanation_length = self.config.get('min_explanation_length', 150)
        self.max_explanation_length = self.config.get('max_explanation_length', 8000)
        self.min_word_count = self.config.get('min_explanation_words', 25)
        
        # Quality assessment thresholds
        self.min_readability_score = self.config.get('min_readability_score', 30)
        self.max_repetition_ratio = self.config.get('max_repetition_ratio', 0.3)
        
        # Performance tracking
        self.validation_count = 0
        self.total_validation_time = 0.0

    def validate_explanation_output(self, output_data: Dict[str, Any], session_id: str = None) -> ValidationResult:
        """
        Validate complete explanation output with comprehensive quality assessment.

        Args:
            output_data: Output dictionary containing explanation results
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with detailed validation results and quality metrics
        """
        start_time = time.time()
        self.validation_count += 1

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            session_id=session_id
        )

        try:
            # Basic structure validation
            if not isinstance(output_data, dict):
                result.add_error(
                    f"Output data must be dictionary, got {type(output_data).__name__}",
                    "Ensure output is properly formatted as a dictionary"
                )
                return result

            # Validate required fields
            required_fields = ['explanation', 'metadata']
            for field in required_fields:
                if field not in output_data:
                    result.add_error(
                        f"Missing required output field: {field}",
                        f"Include '{field}' field in output data"
                    )

            # Validate explanation content
            if 'explanation' in output_data:
                explanation_validation = self.validate_explanation_content(
                    output_data['explanation'], session_id
                )
                result.details['explanation_validation'] = explanation_validation.details
                
                if not explanation_validation.is_valid:
                    result.errors.extend([f"Explanation: {error}" for error in explanation_validation.errors])
                    result.suggestions.extend(explanation_validation.suggestions)
                    result.is_valid = False
                
                if explanation_validation.warnings:
                    result.warnings.extend([f"Explanation: {warning}" for warning in explanation_validation.warnings])

            # Validate optional analysis components
            analysis_components = ['detailed_analysis', 'confidence_analysis', 'source_assessment']
            for component in analysis_components:
                if component in output_data and output_data[component] is not None:
                    component_validation = self.validate_explanation_content(
                        output_data[component], session_id, component_name=component
                    )
                    
                    if not component_validation.is_valid:
                        result.warnings.extend([f"{component.title()}: {error}" for error in component_validation.errors])
                    
                    if component_validation.warnings:
                        result.warnings.extend([f"{component.title()}: {warning}" for warning in component_validation.warnings])

            # Validate metadata
            if 'metadata' in output_data:
                metadata_validation = self.validate_output_metadata(output_data['metadata'], session_id)
                result.details['metadata_validation'] = metadata_validation.details
                
                if not metadata_validation.is_valid:
                    result.warnings.extend([f"Metadata: {error}" for error in metadata_validation.errors])
                
                if metadata_validation.warnings:
                    result.warnings.extend([f"Metadata: {warning}" for warning in metadata_validation.warnings])

            # Quality indicators validation
            if 'quality_indicators' in output_data:
                quality_validation = self._validate_quality_indicators(output_data['quality_indicators'])
                if quality_validation.warnings:
                    result.warnings.extend([f"Quality: {warning}" for warning in quality_validation.warnings])

            # Overall quality assessment
            quality_score = self._assess_output_quality(output_data)
            result.score = quality_score
            result.details['quality_score'] = quality_score
            
            if quality_score < 70:
                result.add_warning(
                    f"Output quality score is moderate: {quality_score:.1f}/100",
                    "Review output completeness and content quality"
                )

            # Processing time tracking
            validation_time = time.time() - start_time
            result.validation_time = validation_time
            self.total_validation_time += validation_time
            result.details['validation_time_ms'] = round(validation_time * 1000, 2)

            self.logger.info(
                f"Output validation completed: {'PASSED' if result.is_valid else 'FAILED'}",
                extra={
                    'session_id': session_id,
                    'quality_score': result.score,
                    'error_count': len(result.errors),
                    'warning_count': len(result.warnings)
                }
            )

            return result

        except Exception as e:
            validation_time = time.time() - start_time
            result.validation_time = validation_time
            self.logger.error(f"Output validation failed: {str(e)}", extra={'session_id': session_id})
            
            result.add_error(
                f"Validation error: {str(e)}",
                "Check output format and structure"
            )
            return result

    def validate_explanation_content(self, content: Any, session_id: str = None, 
                                   component_name: str = "explanation") -> ValidationResult:
        """
        Validate explanation content quality with comprehensive checks.

        Args:
            content: Explanation content to validate
            session_id: Optional session ID for tracking
            component_name: Name of the component being validated

        Returns:
            ValidationResult for content validation with quality metrics
        """
        result = ValidationResult(is_valid=True, session_id=session_id)

        try:
            # Type validation
            if not isinstance(content, str):
                result.add_error(
                    f"{component_name.title()} content must be string, got {type(content).__name__}",
                    f"Provide {component_name} as text content"
                )
                return result

            if not content or not content.strip():
                result.add_error(
                    f"{component_name.title()} content cannot be empty",
                    f"Generate meaningful {component_name} content"
                )
                return result

            content_clean = content.strip()
            content_length = len(content_clean)
            result.details['content_length'] = content_length
            result.details['component_name'] = component_name

            # Length validation (more lenient for optional components)
            min_length = self.min_explanation_length if component_name == "explanation" else 50
            
            if content_length < min_length:
                if component_name == "explanation":
                    result.add_error(
                        f"{component_name.title()} too short: {content_length} characters (minimum: {min_length})",
                        f"Generate at least {min_length} characters of meaningful content"
                    )
                else:
                    result.add_warning(
                        f"{component_name.title()} is short: {content_length} characters",
                        f"Consider expanding {component_name} for better quality"
                    )

            if content_length > self.max_explanation_length:
                result.add_warning(
                    f"{component_name.title()} very long: {content_length} characters",
                    f"Consider condensing {component_name} for better readability"
                )

            # Word count validation
            words = content_clean.split()
            word_count = len(words)
            result.details['word_count'] = word_count

            if word_count < self.min_word_count and component_name == "explanation":
                result.add_error(
                    f"{component_name.title()} too few words: {word_count} (minimum: {self.min_word_count})",
                    f"Generate at least {self.min_word_count} words of analysis"
                )

            # Content quality checks
            if words and content_clean:
                # Check for error messages in content
                error_indicators = [
                    'error', 'failed', 'blocked', 'unavailable', 'not available',
                    'something went wrong', 'try again', 'could not generate'
                ]
                
                content_lower = content_clean.lower()
                error_found = any(indicator in content_lower for indicator in error_indicators)
                
                if error_found:
                    result.add_warning(
                        f"{component_name.title()} may contain error messages",
                        "Review content generation for proper completion"
                    )

                # Check for repetitive content
                sentences = re.split(r'[.!?]+', content_clean)
                if len(sentences) > 3:
                    sentence_starts = [s.strip()[:30] for s in sentences if len(s.strip()) > 10]
                    if len(sentence_starts) > 0:
                        unique_starts = set(sentence_starts)
                        repetition_ratio = 1 - (len(unique_starts) / len(sentence_starts))
                        result.details['repetition_ratio'] = round(repetition_ratio, 3)
                        
                        if repetition_ratio > self.max_repetition_ratio:
                            result.add_warning(
                                f"{component_name.title()} has repetitive content: {repetition_ratio:.1%}",
                                "Ensure content variety and avoid repetitive phrasing"
                            )

                # Readability assessment
                readability_score = self._assess_readability(content_clean, words, sentences)
                result.details['readability_score'] = readability_score
                
                if readability_score < self.min_readability_score:
                    result.add_warning(
                        f"{component_name.title()} may have readability issues: {readability_score}/100",
                        "Ensure content is clear and well-structured"
                    )

            return result

        except Exception as e:
            self.logger.error(f"Content validation failed: {str(e)}", extra={'session_id': session_id})
            result.add_error(
                f"Content validation error: {str(e)}",
                "Check content format and structure"
            )
            return result

    def validate_output_metadata(self, metadata: Any, session_id: str = None) -> ValidationResult:
        """
        Validate output metadata structure and completeness.

        Args:
            metadata: Metadata dictionary to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult for metadata validation
        """
        result = ValidationResult(is_valid=True, session_id=session_id)

        try:
            # Type validation
            if not isinstance(metadata, dict):
                result.add_error(
                    f"Output metadata must be dictionary, got {type(metadata).__name__}",
                    "Ensure metadata is properly formatted"
                )
                return result

            # Check expected metadata fields
            expected_fields = {
                'processing_details': dict,
                'analysis_components': dict,
                'system_info': dict,
                'input_parameters': dict
            }

            present_fields = 0
            for field, expected_type in expected_fields.items():
                if field in metadata:
                    present_fields += 1
                    if not isinstance(metadata[field], expected_type):
                        result.add_warning(
                            f"Metadata field '{field}' should be {expected_type.__name__}",
                            f"Ensure '{field}' is properly structured"
                        )

            # Completeness assessment
            completeness_ratio = present_fields / len(expected_fields)
            result.details['completeness_ratio'] = round(completeness_ratio, 2)
            
            if completeness_ratio < 0.5:
                result.add_warning(
                    f"Output metadata incomplete: {present_fields}/{len(expected_fields)} sections",
                    "Include comprehensive metadata for better tracking"
                )

            # Validate specific fields if present
            if 'processing_details' in metadata:
                processing_details = metadata['processing_details']
                if isinstance(processing_details, dict):
                    if 'response_time_seconds' in processing_details:
                        response_time = processing_details['response_time_seconds']
                        if not isinstance(response_time, (int, float)) or response_time < 0:
                            result.add_warning(
                                "Invalid response time in metadata",
                                "Ensure response time is a positive number"
                            )
                        elif response_time > 120:  # 2 minutes
                            result.add_warning(
                                f"Very long response time: {response_time}s",
                                "Consider optimizing processing time"
                            )

            return result

        except Exception as e:
            self.logger.error(f"Metadata validation failed: {str(e)}", extra={'session_id': session_id})
            result.add_error(
                f"Metadata validation error: {str(e)}",
                "Check metadata format and content"
            )
            return result

    def _validate_quality_indicators(self, quality_indicators: Dict[str, Any]) -> ValidationResult:
        """Validate quality indicators structure and values."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(quality_indicators, dict):
            result.add_warning(
                "Quality indicators should be a dictionary",
                "Structure quality indicators properly"
            )
            return result

        expected_indicators = {
            'explanation_length': (int, lambda x: x > 0),
            'explanation_word_count': (int, lambda x: x > 0),
            'overall_quality_score': ((int, float), lambda x: 0 <= x <= 100)
        }

        for indicator, (expected_type, validator) in expected_indicators.items():
            if indicator in quality_indicators:
                value = quality_indicators[indicator]
                if not isinstance(value, expected_type):
                    result.add_warning(
                        f"Quality indicator '{indicator}' should be {expected_type}",
                        f"Ensure '{indicator}' is properly typed"
                    )
                elif not validator(value):
                    result.add_warning(
                        f"Quality indicator '{indicator}' has invalid value: {value}",
                        f"Check '{indicator}' calculation and range"
                    )

        return result

    def _assess_readability(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Assess content readability using multiple metrics."""
        if not words or not sentences:
            return 0.0

        try:
            # Basic readability metrics
            avg_words_per_sentence = len(words) / max(len(sentences), 1)
            avg_chars_per_word = sum(len(word) for word in words) / len(words)

            # Simple readability scoring
            readability = 80.0  # Base score

            # Adjust for sentence complexity
            if avg_words_per_sentence > 30:
                readability -= 25  # Very complex
            elif avg_words_per_sentence > 20:
                readability -= 15  # Complex
            elif avg_words_per_sentence < 8:
                readability -= 10  # Too simple

            # Adjust for word complexity
            if avg_chars_per_word > 7:
                readability -= 15  # Complex words
            elif avg_chars_per_word < 4:
                readability -= 5   # Very simple words

            # Check for formatting issues
            if text.count('\n') > len(sentences) * 0.5:
                readability += 5  # Good paragraph structure

            return max(0.0, min(100.0, readability))

        except Exception:
            return 50.0  # Default middle score if assessment fails

    def _assess_output_quality(self, output_data: Dict[str, Any]) -> float:
        """Assess overall output quality score."""
        score = 60  # Base score
        
        # Main explanation quality (40 points)
        explanation = output_data.get('explanation', '')
        if isinstance(explanation, str) and explanation.strip():
            explanation_length = len(explanation.strip())
            word_count = len(explanation.split())
            
            if explanation_length >= self.min_explanation_length:
                score += 20
            if word_count >= self.min_word_count:
                score += 10
            if explanation_length <= self.max_explanation_length:
                score += 10

        # Analysis components (30 points)
        components = ['detailed_analysis', 'confidence_analysis', 'source_assessment']
        present_components = sum(1 for comp in components 
                               if comp in output_data and output_data[comp])
        component_score = (present_components / len(components)) * 30
        score += component_score

        # Metadata completeness (10 points)
        metadata = output_data.get('metadata', {})
        if isinstance(metadata, dict) and metadata:
            expected_sections = ['processing_details', 'analysis_components', 'system_info']
            present_sections = sum(1 for section in expected_sections if section in metadata)
            metadata_score = (present_sections / len(expected_sections)) * 10
            score += metadata_score

        return min(100.0, max(0.0, score))

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive output validation statistics."""
        avg_time = (self.total_validation_time / self.validation_count) if self.validation_count > 0 else 0
        
        return {
            'validation_count': self.validation_count,
            'total_validation_time': self.total_validation_time,
            'average_validation_time': avg_time,
            'configuration': {
                'min_explanation_length': self.min_explanation_length,
                'max_explanation_length': self.max_explanation_length,
                'min_word_count': self.min_word_count,
                'min_readability_score': self.min_readability_score
            }
        }


class BatchValidator:
    """
    Comprehensive batch validator for processing multiple explanation requests.
    
    Provides efficient validation of batch operations with detailed error
    reporting, quality assessment, and performance optimization for
    high-throughput production scenarios.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize batch validator with production configuration.

        Args:
            config: Optional validation configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.BatchValidator")
        
        # Batch processing limits
        self.max_batch_size = self.config.get('max_batch_size', 100)
        self.min_batch_size = self.config.get('min_batch_size', 1)
        
        # Initialize component validators
        self.input_validator = InputValidator(self.config)
        self.output_validator = OutputValidator(self.config)
        
        # Performance tracking
        self.batch_count = 0
        self.total_items_processed = 0
        self.total_processing_time = 0.0

    def validate_batch_input(self, batch_data: List[Dict[str, Any]], session_id: str = None) -> ValidationResult:
        """
        Validate batch input data with comprehensive error reporting.

        Args:
            batch_data: List of input dictionaries to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult for batch input validation with detailed statistics
        """
        start_time = time.time()
        self.batch_count += 1

        result = ValidationResult(is_valid=True, session_id=session_id)

        try:
            # Basic structure validation
            if not isinstance(batch_data, list):
                result.add_error(
                    f"Batch data must be list, got {type(batch_data).__name__}",
                    "Provide batch data as a list of input dictionaries"
                )
                return result

            batch_size = len(batch_data)
            result.details['batch_size'] = batch_size

            # Size validation
            if batch_size < self.min_batch_size:
                result.add_error(
                    f"Batch too small: {batch_size} items (minimum: {self.min_batch_size})",
                    f"Provide at least {self.min_batch_size} items in batch"
                )

            if batch_size > self.max_batch_size:
                result.add_error(
                    f"Batch too large: {batch_size} items (maximum: {self.max_batch_size})",
                    f"Limit batch to {self.max_batch_size} items or split into smaller batches"
                )

            # Validate individual items
            valid_items = 0
            error_items = 0
            warning_items = 0

            for i, item in enumerate(batch_data):
                item_validation = self.input_validator.validate_explanation_input(
                    item, f"{session_id}_item_{i}" if session_id else None
                )
                
                if item_validation.is_valid:
                    valid_items += 1
                else:
                    error_items += 1
                    for error in item_validation.errors[:2]:  # Limit to 2 errors per item
                        result.add_error(f"Item {i}: {error}")

                if item_validation.warnings:
                    warning_items += 1
                    for warning in item_validation.warnings[:1]:  # Limit to 1 warning per item
                        result.add_warning(f"Item {i}: {warning}")

            # Batch quality assessment
            if batch_size > 0:
                success_rate = valid_items / batch_size
                result.details['valid_items'] = valid_items
                result.details['error_items'] = error_items
                result.details['warning_items'] = warning_items
                result.details['success_rate'] = round(success_rate, 3)
                result.score = success_rate * 100

                if success_rate < 0.8:  # Less than 80% valid
                    result.add_warning(
                        f"Low batch success rate: {success_rate:.1%}",
                        "Review input data quality and fix common errors"
                    )

            # Processing time tracking
            validation_time = time.time() - start_time
            result.validation_time = validation_time
            self.total_processing_time += validation_time
            self.total_items_processed += batch_size
            
            self.logger.info(
                f"Batch input validation completed: {valid_items}/{batch_size} items valid",
                extra={'session_id': session_id, 'batch_size': batch_size}
            )

            return result

        except Exception as e:
            validation_time = time.time() - start_time
            result.validation_time = validation_time
            self.logger.error(f"Batch input validation failed: {str(e)}", extra={'session_id': session_id})
            
            result.add_error(
                f"Batch validation error: {str(e)}",
                "Check batch format and structure"
            )
            return result

    def validate_batch_output(self, batch_results: List[Dict[str, Any]], session_id: str = None) -> ValidationResult:
        """
        Validate batch output results with quality assessment.

        Args:
            batch_results: List of output dictionaries to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult for batch output validation with quality metrics
        """
        start_time = time.time()
        result = ValidationResult(is_valid=True, session_id=session_id)

        try:
            # Basic structure validation
            if not isinstance(batch_results, list):
                result.add_error(
                    f"Batch results must be list, got {type(batch_results).__name__}",
                    "Ensure batch results are properly formatted"
                )
                return result

            batch_size = len(batch_results)
            result.details['batch_size'] = batch_size

            # Validate individual results
            valid_results = 0
            error_results = 0
            successful_generations = 0

            for i, item in enumerate(batch_results):
                if not isinstance(item, dict):
                    result.add_error(f"Result {i} must be dictionary, got {type(item).__name__}")
                    error_results += 1
                    continue

                # Check for success indicators
                if item.get('success', True):
                    successful_generations += 1
                    
                    # Validate successful results
                    if 'result' in item:
                        item_validation = self.output_validator.validate_explanation_output(
                            item['result'], f"{session_id}_result_{i}" if session_id else None
                        )
                        
                        if item_validation.is_valid:
                            valid_results += 1
                        else:
                            error_results += 1
                            for error in item_validation.errors[:1]:  # Limit to 1 error per result
                                result.add_warning(f"Result {i}: {error}")
                else:
                    error_results += 1

            # Batch quality assessment
            if batch_size > 0:
                success_rate = successful_generations / batch_size
                validity_rate = valid_results / batch_size
                
                result.details['successful_generations'] = successful_generations
                result.details['valid_results'] = valid_results
                result.details['error_results'] = error_results
                result.details['success_rate'] = round(success_rate, 3)
                result.details['validity_rate'] = round(validity_rate, 3)
                result.score = (success_rate + validity_rate) / 2 * 100

                if success_rate < 0.8:
                    result.add_warning(
                        f"Low batch success rate: {success_rate:.1%}",
                        "Review explanation generation quality and error handling"
                    )

            # Processing time tracking
            validation_time = time.time() - start_time
            result.validation_time = validation_time
            self.total_processing_time += validation_time
            
            self.logger.info(
                f"Batch output validation completed: {valid_results}/{batch_size} results valid",
                extra={'session_id': session_id, 'batch_size': batch_size}
            )

            return result

        except Exception as e:
            validation_time = time.time() - start_time
            result.validation_time = validation_time
            self.logger.error(f"Batch output validation failed: {str(e)}", extra={'session_id': session_id})
            
            result.add_error(
                f"Batch validation error: {str(e)}",
                "Check batch results format and content"
            )
            return result

    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get comprehensive batch processing statistics."""
        avg_time_per_batch = (self.total_processing_time / self.batch_count) if self.batch_count > 0 else 0
        avg_time_per_item = (self.total_processing_time / self.total_items_processed) if self.total_items_processed > 0 else 0
        
        return {
            'batch_count': self.batch_count,
            'total_items_processed': self.total_items_processed,
            'total_processing_time': self.total_processing_time,
            'average_time_per_batch': avg_time_per_batch,
            'average_time_per_item': avg_time_per_item,
            'configuration': {
                'max_batch_size': self.max_batch_size,
                'min_batch_size': self.min_batch_size
            }
        }


# Utility validation functions
def validate_explanation_input(input_data: Dict[str, Any], config: Dict[str, Any] = None, 
                             session_id: str = None) -> ValidationResult:
    """
    Convenience function for quick input validation.

    Args:
        input_data: Input dictionary to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult for input validation
    """
    validator = InputValidator(config)
    return validator.validate_explanation_input(input_data, session_id)


def validate_explanation_output(output_data: Dict[str, Any], config: Dict[str, Any] = None, 
                              session_id: str = None) -> ValidationResult:
    """
    Convenience function for quick output validation.

    Args:
        output_data: Output dictionary to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult for output validation
    """
    validator = OutputValidator(config)
    return validator.validate_explanation_output(output_data, session_id)


# Testing functionality
if __name__ == "__main__":
    """Test LLM explanation validation functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== LLM EXPLANATION VALIDATORS TEST ===")
    
    # Initialize validators with test configuration
    test_config = {
        'min_text_length': 100,
        'max_text_length': 5000,
        'min_word_count': 20,
        'enable_security_checks': True
    }
    
    input_validator = InputValidator(test_config)
    output_validator = OutputValidator(test_config)
    batch_validator = BatchValidator(test_config)

    # Test input validation
    print("--- Input Validation Test ---")
    test_input = {
        'text': 'This is a comprehensive test article about health misinformation claims. '
                'It contains multiple sentences with various claims that need to be analyzed '
                'for credibility and accuracy using professional fact-checking standards.',
        'prediction': 'FAKE',
        'confidence': 0.85,
        'metadata': {
            'source': 'HealthBlog.net',
            'date': '2025-01-15',
            'subject': 'Health',
            'author': 'Test Author'
        },
        'require_detailed_analysis': True,
        'session_id': 'test_validation_001'
    }

    input_result = input_validator.validate_explanation_input(test_input, 'test_session_001')
    print(f"✅ Input validation: {'PASSED' if input_result.is_valid else 'FAILED'}")
    print(f"   Score: {input_result.score:.1f}/100")
    print(f"   Errors: {len(input_result.errors)}")
    print(f"   Warnings: {len(input_result.warnings)}")
    print(f"   Suggestions: {len(input_result.suggestions)}")

    if input_result.errors:
        print(f"   First error: {input_result.errors[0]}")
    if input_result.warnings:
        print(f"   First warning: {input_result.warnings[0]}")

    # Test output validation
    print("\n--- Output Validation Test ---")
    test_output = {
        'explanation': 'This article exhibits several characteristics of health misinformation. '
                      'The claims lack peer-reviewed sources and contradict established medical '
                      'research. The author credentials are questionable and the publication '
                      'source has a history of promoting unverified health claims.',
        'detailed_analysis': 'Forensic analysis reveals multiple red flags including lack of '
                           'scientific methodology, absence of expert consultation, and '
                           'sensationalized language patterns typical of health misinformation.',
        'confidence_analysis': 'The high confidence level is justified by multiple converging '
                             'indicators of misinformation including source analysis, content '
                             'evaluation, and pattern recognition.',
        'source_assessment': {
            'reliability_level': 'LOW',
            'bias_warning': 'Source exhibits strong commercial bias',
            'verification_recommendation': 'Seek verification from medical institutions'
        },
        'quality_indicators': {
            'explanation_length': 245,
            'explanation_word_count': 42,
            'overall_quality_score': 88
        },
        'metadata': {
            'processing_details': {
                'response_time_seconds': 3.2,
                'model_used': 'gemini-1.5-pro',
                'temperature_used': 0.3
            },
            'analysis_components': {
                'detailed_analysis_included': True,
                'confidence_analysis_included': True,
                'source_analysis_included': True
            },
            'system_info': {
                'analysis_timestamp': '2025-09-11T18:00:00',
                'agent_version': '4.0.0',
                'session_id': 'test_session_001'
            }
        }
    }

    output_result = output_validator.validate_explanation_output(test_output, 'test_session_001')
    print(f"✅ Output validation: {'PASSED' if output_result.is_valid else 'FAILED'}")
    print(f"   Score: {output_result.score:.1f}/100")
    print(f"   Errors: {len(output_result.errors)}")
    print(f"   Warnings: {len(output_result.warnings)}")

    if output_result.warnings:
        print(f"   First warning: {output_result.warnings[0]}")

    # Test batch validation
    print("\n--- Batch Validation Test ---")
    batch_input = [test_input.copy() for _ in range(3)]
    batch_input[1]['text'] = 'Short text'  # Intentionally invalid
    batch_input[2]['confidence'] = 1.5     # Intentionally invalid

    batch_input_result = batch_validator.validate_batch_input(batch_input, 'test_batch_001')
    print(f"✅ Batch input validation: {'PASSED' if batch_input_result.is_valid else 'FAILED'}")
    print(f"   Batch size: {batch_input_result.details.get('batch_size', 0)}")
    print(f"   Valid items: {batch_input_result.details.get('valid_items', 0)}")
    print(f"   Success rate: {batch_input_result.details.get('success_rate', 0):.1%}")

    # Test batch output validation
    batch_output = [
        {'success': True, 'result': test_output},
        {'success': True, 'result': test_output},
        {'success': False, 'error': {'type': 'TestError', 'message': 'Test error'}}
    ]

    batch_output_result = batch_validator.validate_batch_output(batch_output, 'test_batch_001')
    print(f"✅ Batch output validation: {'PASSED' if batch_output_result.is_valid else 'FAILED'}")
    print(f"   Success rate: {batch_output_result.details.get('success_rate', 0):.1%}")
    print(f"   Validity rate: {batch_output_result.details.get('validity_rate', 0):.1%}")

    # Test convenience functions
    print("\n--- Convenience Functions Test ---")
    quick_input_result = validate_explanation_input(test_input, test_config, 'test_conv_001')
    quick_output_result = validate_explanation_output(test_output, test_config, 'test_conv_002')

    print(f"✅ Quick input validation: {'PASSED' if quick_input_result.is_valid else 'FAILED'}")
    print(f"✅ Quick output validation: {'PASSED' if quick_output_result.is_valid else 'FAILED'}")

    # Show statistics
    print("\n--- Validation Statistics ---")
    input_stats = input_validator.get_validation_statistics()
    output_stats = output_validator.get_validation_statistics()
    batch_stats = batch_validator.get_batch_statistics()

    print(f"📊 Input validations: {input_stats['validation_count']}")
    print(f"📊 Output validations: {output_stats['validation_count']}")
    print(f"📊 Batch operations: {batch_stats['batch_count']}")
    print(f"📊 Total items processed: {batch_stats['total_items_processed']}")

    print("\n🎯 LLM explanation validators tests completed successfully!")
