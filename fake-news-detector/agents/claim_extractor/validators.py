# agents/claim_extractor/validators.py

"""
Claim Extractor Input and Output Validators - Production Ready

Production-ready validation system for claim extraction agent providing
comprehensive input validation, output verification, and data quality checks
with detailed error reporting, configurable validation rules, and enhanced
security measures for reliable production use.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
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
    Structured validation result container with enhanced metadata.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of validation error messages
        warnings: List of validation warnings
        score: Validation quality score (0-100)
        details: Additional validation details and metrics
        suggestions: Actionable suggestions for fixing issues
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    score: float = 0.0
    details: Dict[str, Any] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.details is None:
            self.details = {}
        if self.suggestions is None:
            self.suggestions = []


class InputValidator:
    """
    Comprehensive input validator for claim extraction requests.
    
    Validates article text, BERT results, configuration parameters,
    and other input data with configurable validation rules and
    enhanced security measures for production environments.
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
        self.min_text_length = self.config.get('min_text_length', 50)
        self.max_text_length = self.config.get('max_text_length', 100000)
        self.min_word_count = self.config.get('min_word_count', 10)
        self.max_word_count = self.config.get('max_word_count', 15000)
        
        # Content quality thresholds
        self.min_sentence_count = self.config.get('min_sentence_count', 3)
        self.max_repeated_chars = self.config.get('max_repeated_chars', 10)
        self.min_unique_words_ratio = self.config.get('min_unique_words_ratio', 0.3)
        self.max_uppercase_ratio = self.config.get('max_uppercase_ratio', 0.5)
        
        # Security validation
        self.enable_security_checks = self.config.get('enable_security_checks', True)
        self.blocked_patterns = self.config.get('blocked_patterns', [])
        
        # Performance tracking
        self.validation_count = 0
        self.total_validation_time = 0.0
        
        self.logger.info(f"InputValidator initialized with security checks: {self.enable_security_checks}")

    def validate_article_text(self, text: str, session_id: str = None) -> ValidationResult:
        """
        Validate article text content comprehensively with enhanced checks.

        Args:
            text: Article text to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with detailed validation results
        """
        import time
        start_time = time.time()
        
        errors = []
        warnings = []
        suggestions = []
        score = 100.0
        details = {}

        try:
            # Basic type and existence checks
            if not isinstance(text, str):
                errors.append("Article text must be a string")
                return ValidationResult(
                    False, errors, warnings, 0.0, details, 
                    ["Provide text input as a string type"]
                )

            if not text or not text.strip():
                errors.append("Article text cannot be empty")
                return ValidationResult(
                    False, errors, warnings, 0.0, details,
                    ["Provide non-empty article text for analysis"]
                )

            text_clean = text.strip()
            details['original_length'] = len(text)
            details['clean_length'] = len(text_clean)

            # Length validation with detailed feedback
            if len(text_clean) < self.min_text_length:
                errors.append(f"Article text too short: {len(text_clean)} < {self.min_text_length} characters")
                suggestions.append(f"Provide at least {self.min_text_length} characters of article content")
                score -= 30

            if len(text_clean) > self.max_text_length:
                errors.append(f"Article text too long: {len(text_clean)} > {self.max_text_length} characters")
                suggestions.append(f"Limit article text to {self.max_text_length} characters or less")
                score -= 20

            # Word count validation with enhanced analysis
            words = text_clean.split()
            word_count = len(words)
            details['word_count'] = word_count

            if word_count < self.min_word_count:
                errors.append(f"Article has too few words: {word_count} < {self.min_word_count}")
                suggestions.append(f"Provide at least {self.min_word_count} words for meaningful claim extraction")
                score -= 25

            if word_count > self.max_word_count:
                warnings.append(f"Article is very long: {word_count} words")
                suggestions.append("Consider splitting very long articles for better processing")
                score -= 5

            # Sentence structure validation with enhanced logic
            sentences = re.split(r'[.!?]+', text_clean)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            sentence_count = len(valid_sentences)
            details['sentence_count'] = sentence_count

            if sentence_count < self.min_sentence_count:
                errors.append(f"Article has too few sentences: {sentence_count} < {self.min_sentence_count}")
                suggestions.append(f"Provide at least {self.min_sentence_count} complete sentences")
                score -= 20

            # Enhanced content quality checks
            if words:
                # Vocabulary diversity check
                unique_words = set(word.lower().strip('.,!?;:"()[]{}') for word in words if word.isalnum())
                unique_ratio = len(unique_words) / len(words)
                details['unique_words_ratio'] = round(unique_ratio, 3)

                if unique_ratio < self.min_unique_words_ratio:
                    warnings.append(f"Low vocabulary diversity: {unique_ratio:.2f}")
                    suggestions.append("Ensure article contains varied vocabulary for better claim extraction")
                    score -= 10

                # Uppercase content check
                uppercase_chars = sum(1 for c in text_clean if c.isupper())
                uppercase_ratio = uppercase_chars / max(len(text_clean), 1)
                details['uppercase_ratio'] = round(uppercase_ratio, 3)

                if uppercase_ratio > self.max_uppercase_ratio:
                    warnings.append(f"Excessive uppercase content: {uppercase_ratio:.1%}")
                    suggestions.append("Reduce excessive uppercase text for better processing")
                    score -= 15

            # Repeated character check with enhanced detection
            repeated_char_pattern = r'(.)\1{' + str(self.max_repeated_chars) + ',}'
            repeated_matches = re.findall(repeated_char_pattern, text_clean)
            
            if repeated_matches:
                warnings.append(f"Contains {len(repeated_matches)} patterns of excessive repeated characters")
                suggestions.append("Remove excessive repeated characters (e.g., 'aaaaaaa', '!!!!!')")
                score -= 5

            # Content format validation
            format_issues = self._check_content_format(text_clean)
            if format_issues:
                warnings.extend(format_issues)
                score -= len(format_issues) * 3

            # Security validation with enhanced checks
            if self.enable_security_checks:
                security_issues = self._check_security_patterns(text_clean, session_id)
                if security_issues:
                    errors.extend(security_issues)
                    suggestions.append("Remove potentially malicious content patterns")
                    score -= 40

            # Language and readability detection
            readability_score = self._assess_readability(text_clean, words, valid_sentences)
            details['readability_score'] = readability_score
            
            if readability_score < 30:
                warnings.append("Content may have readability issues")
                suggestions.append("Ensure content is in a readable format with proper grammar")
                score -= 15

            # Update performance metrics
            processing_time = time.time() - start_time
            self.validation_count += 1
            self.total_validation_time += processing_time
            details['validation_time_ms'] = round(processing_time * 1000, 2)

            # Determine final validation status
            is_valid = len(errors) == 0 and score > 0
            final_score = max(0.0, min(100.0, score))

            self.logger.info(
                f"Article text validation completed: {'PASSED' if is_valid else 'FAILED'}",
                extra={
                    'session_id': session_id,
                    'text_length': len(text_clean),
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'score': final_score,
                    'validation_time': round(processing_time * 1000, 2)
                }
            )

            return ValidationResult(is_valid, errors, warnings, final_score, details, suggestions)

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Article validation failed: {str(e)}", extra={'session_id': session_id})
            return ValidationResult(
                False,
                [f"Validation error: {str(e)}"],
                [],
                0.0,
                {'validation_time_ms': round(processing_time * 1000, 2)},
                ["Check input format and try again"]
            )

    def validate_bert_results(self, bert_results: Dict[str, Any], session_id: str = None) -> ValidationResult:
        """
        Validate BERT classification results with enhanced checks.

        Args:
            bert_results: BERT results dictionary
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation results
        """
        errors = []
        warnings = []
        suggestions = []
        score = 100.0
        details = {}

        try:
            if not isinstance(bert_results, dict):
                errors.append("BERT results must be a dictionary")
                return ValidationResult(
                    False, errors, warnings, 0.0, details,
                    ["Provide BERT results as a dictionary with 'prediction' and 'confidence' fields"]
                )

            # Check required fields with enhanced validation
            required_fields = ['prediction', 'confidence']
            for field in required_fields:
                if field not in bert_results:
                    errors.append(f"Missing required BERT field: {field}")
                    suggestions.append(f"Include '{field}' field in BERT results")
                    score -= 30

            # Validate prediction with enhanced checks
            if 'prediction' in bert_results:
                prediction = bert_results['prediction']
                valid_predictions = ['REAL', 'FAKE', 'UNCERTAIN', 'UNKNOWN']
                
                if prediction not in valid_predictions:
                    errors.append(f"Invalid BERT prediction: {prediction}")
                    suggestions.append(f"Use one of: {', '.join(valid_predictions)}")
                    score -= 25
                else:
                    details['prediction'] = prediction
                    details['prediction_valid'] = True

            # Validate confidence with comprehensive checks
            if 'confidence' in bert_results:
                confidence = bert_results['confidence']
                
                if not isinstance(confidence, (int, float)):
                    errors.append("BERT confidence must be a number")
                    suggestions.append("Provide confidence as a float between 0.0 and 1.0")
                    score -= 20
                elif not 0 <= confidence <= 1:
                    errors.append(f"BERT confidence out of range: {confidence}")
                    suggestions.append("Ensure confidence is between 0.0 and 1.0")
                    score -= 15
                else:
                    details['confidence'] = confidence
                    details['confidence_valid'] = True
                    
                    # Confidence quality assessment
                    if confidence < 0.3:
                        warnings.append(f"Very low BERT confidence: {confidence:.2f}")
                        suggestions.append("Low confidence predictions may require additional verification")
                        score -= 10
                    elif confidence < 0.5:
                        warnings.append(f"Low BERT confidence: {confidence:.2f}")
                        score -= 5

            # Validate optional fields
            optional_fields = {
                'model_version': str,
                'processing_time': (int, float),
                'features_used': list
            }

            for field, expected_type in optional_fields.items():
                if field in bert_results:
                    if not isinstance(bert_results[field], expected_type):
                        warnings.append(f"Field {field} should be {expected_type.__name__}")
                        score -= 3
                    else:
                        details[f'{field}_valid'] = True

            # Check for additional metadata
            metadata_fields = ['timestamp', 'model_name', 'version']
            metadata_count = sum(1 for field in metadata_fields if field in bert_results)
            details['metadata_completeness'] = metadata_count / len(metadata_fields)

            if metadata_count == 0:
                warnings.append("No metadata fields found in BERT results")
                suggestions.append("Consider including model metadata for better traceability")

            is_valid = len(errors) == 0
            final_score = max(0.0, min(100.0, score))

            self.logger.info(
                f"BERT results validation: {'PASSED' if is_valid else 'FAILED'}",
                extra={'session_id': session_id, 'score': final_score}
            )

            return ValidationResult(is_valid, errors, warnings, final_score, details, suggestions)

        except Exception as e:
            self.logger.error(f"BERT validation failed: {str(e)}", extra={'session_id': session_id})
            return ValidationResult(
                False,
                [f"BERT validation error: {str(e)}"],
                [],
                0.0,
                {},
                ["Check BERT results format and try again"]
            )

    def validate_input_data(self, input_data: Dict[str, Any], session_id: str = None) -> ValidationResult:
        """
        Comprehensive input data validation for claim extraction with enhanced checks.

        Args:
            input_data: Complete input data dictionary
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with comprehensive validation results
        """
        errors = []
        warnings = []
        suggestions = []
        score = 100.0
        details = {}

        try:
            # Basic structure validation
            if not isinstance(input_data, dict):
                errors.append("Input data must be a dictionary")
                return ValidationResult(
                    False, errors, warnings, 0.0, details,
                    ["Provide input as a dictionary with required fields"]
                )

            if not input_data:
                errors.append("Input data cannot be empty")
                return ValidationResult(
                    False, errors, warnings, 0.0, details,
                    ["Provide input data with at least the 'text' field"]
                )

            # Validate required text field with sub-validation
            if 'text' not in input_data:
                errors.append("Missing required field: text")
                suggestions.append("Include 'text' field with article content")
                score -= 40
            else:
                text_validation = self.validate_article_text(input_data['text'], session_id)
                details['text_validation'] = text_validation.details
                
                if not text_validation.is_valid:
                    errors.extend([f"Text validation: {error}" for error in text_validation.errors])
                    suggestions.extend(text_validation.suggestions)
                    score -= 30
                
                if text_validation.warnings:
                    warnings.extend([f"Text warning: {warning}" for warning in text_validation.warnings])
                    score -= 5

            # Validate optional BERT results
            if 'bert_results' in input_data:
                bert_validation = self.validate_bert_results(input_data['bert_results'], session_id)
                details['bert_validation'] = bert_validation.details
                
                if not bert_validation.is_valid:
                    warnings.extend([f"BERT validation: {error}" for error in bert_validation.errors])
                    suggestions.extend(bert_validation.suggestions)
                    score -= 10  # Non-critical for claim extraction
                
                if bert_validation.warnings:
                    warnings.extend([f"BERT warning: {warning}" for warning in bert_validation.warnings])

            # Validate optional parameters with enhanced checks
            optional_fields = {
                'topic_domain': (str, ['health', 'politics', 'science', 'technology', 'business', 'general']),
                'include_verification_analysis': (bool, None),
                'max_claims': (int, range(1, 21)),
                'session_id': (str, None),
                'priority_mode': (str, ['speed', 'quality', 'comprehensive'])
            }

            for field, (expected_type, valid_values) in optional_fields.items():
                if field in input_data:
                    value = input_data[field]
                    
                    if not isinstance(value, expected_type):
                        warnings.append(f"Field {field} should be {expected_type.__name__}")
                        suggestions.append(f"Convert {field} to {expected_type.__name__} type")
                        score -= 3
                    elif valid_values is not None:
                        if isinstance(valid_values, (list, tuple)) and value not in valid_values:
                            warnings.append(f"Field {field} has invalid value: {value}")
                            suggestions.append(f"Use one of: {', '.join(map(str, valid_values))}")
                            score -= 5
                        elif hasattr(valid_values, '__contains__') and value not in valid_values:
                            warnings.append(f"Field {field} value out of range: {value}")
                            suggestions.append(f"Provide {field} within valid range")
                            score -= 5

            # Check for suspicious or malformed data
            suspicious_patterns = self._check_suspicious_input_patterns(input_data)
            if suspicious_patterns:
                warnings.extend(suspicious_patterns)
                suggestions.append("Review input data for potential issues")
                score -= len(suspicious_patterns) * 5

            # Data completeness assessment
            completeness_score = self._assess_input_completeness(input_data)
            details['completeness_score'] = completeness_score
            
            if completeness_score < 0.6:
                warnings.append("Input data appears incomplete")
                suggestions.append("Consider providing additional optional fields for better results")

            # Final validation assessment
            is_valid = len(errors) == 0
            final_score = max(0.0, min(100.0, score))

            self.logger.info(
                f"Input data validation completed: {'PASSED' if is_valid else 'FAILED'}",
                extra={
                    'session_id': session_id,
                    'score': final_score,
                    'errors': len(errors),
                    'warnings': len(warnings)
                }
            )

            return ValidationResult(is_valid, errors, warnings, final_score, details, suggestions)

        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}", extra={'session_id': session_id})
            return ValidationResult(
                False,
                [f"Input validation error: {str(e)}"],
                [],
                0.0,
                {},
                ["Check input data format and try again"]
            )

    def _check_security_patterns(self, text: str, session_id: str = None) -> List[str]:
        """Check for security-related patterns in text with enhanced detection."""
        security_issues = []

        # Script injection patterns
        script_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*='
        ]

        for pattern in script_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                security_issues.append(f"Potentially malicious script pattern detected")
                self.logger.warning(f"Script injection pattern found", extra={'session_id': session_id})
                break  # Don't report multiple script issues

        # SQL injection patterns
        sql_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.+set',
            r'--\s*$',
            r'/\*.*\*/'
        ]

        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                security_issues.append("Potentially malicious SQL pattern detected")
                self.logger.warning(f"SQL injection pattern found", extra={'session_id': session_id})
                break

        # Command injection patterns
        command_patterns = [
            r';\s*(rm|del|format|shutdown)',
            r'\|\s*(curl|wget|nc|netcat)',
            r'&&\s*(cat|ls|dir|type)',
            r'`[^`]*`',
            r'\$\([^)]*\)'
        ]

        for pattern in command_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                security_issues.append("Potentially malicious command pattern detected")
                self.logger.warning(f"Command injection pattern found", extra={'session_id': session_id})
                break

        # Custom blocked patterns from configuration
        for pattern in self.blocked_patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    security_issues.append(f"Blocked content pattern detected")
                    self.logger.warning(f"Custom blocked pattern found", extra={'session_id': session_id})
            except re.error:
                self.logger.warning(f"Invalid blocked pattern: {pattern}", extra={'session_id': session_id})

        return security_issues

    def _check_content_format(self, text: str) -> List[str]:
        """Check content format and structure issues."""
        format_issues = []

        # Check for excessive whitespace
        if re.search(r'\s{10,}', text):
            format_issues.append("Contains excessive whitespace")

        # Check for non-printable characters
        non_printable = re.findall(r'[^\x20-\x7E\n\r\t]', text)
        if len(non_printable) > 10:
            format_issues.append(f"Contains {len(non_printable)} non-printable characters")

        # Check for HTML tags (might indicate web content)
        html_tags = re.findall(r'<[^>]+>', text)
        if len(html_tags) > 5:
            format_issues.append("Contains HTML tags - consider preprocessing")

        # Check for URL patterns
        urls = re.findall(r'https?://[^\s]+', text)
        if len(urls) > 10:
            format_issues.append(f"Contains {len(urls)} URLs - may affect claim extraction")

        # Check for excessive punctuation
        punct_chars = sum(1 for c in text if c in '!@#$%^&*()_+-=[]{}|;:,.<>?')
        punct_ratio = punct_chars / max(len(text), 1)
        if punct_ratio > 0.15:
            format_issues.append(f"High punctuation ratio: {punct_ratio:.1%}")

        return format_issues

    def _assess_readability(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Assess content readability using multiple metrics."""
        if not words or not sentences:
            return 0.0

        try:
            # Simple readability metrics
            avg_words_per_sentence = len(words) / max(len(sentences), 1)
            avg_chars_per_word = sum(len(word) for word in words) / len(words)

            # Readability score based on complexity
            if avg_words_per_sentence > 25 or avg_chars_per_word > 8:
                readability = 30.0  # Complex text
            elif avg_words_per_sentence > 15 or avg_chars_per_word > 6:
                readability = 60.0  # Moderate complexity
            else:
                readability = 90.0  # Simple text

            # Adjust for content patterns
            if len([w for w in words if w.isupper()]) > len(words) * 0.1:
                readability -= 20  # Too much uppercase

            # Check for reasonable sentence structure
            very_short_sentences = sum(1 for s in sentences if len(s.split()) < 3)
            if very_short_sentences > len(sentences) * 0.3:
                readability -= 15  # Too many very short sentences

            return max(0.0, min(100.0, readability))

        except Exception:
            return 50.0  # Default middle score if assessment fails

    def _check_suspicious_input_patterns(self, input_data: Dict[str, Any]) -> List[str]:
        """Check for suspicious patterns in input data structure."""
        suspicious_patterns = []

        # Check for unusual field names
        suspicious_keys = ['__', 'eval', 'exec', 'system', 'shell', 'cmd']
        for key in input_data.keys():
            if any(sus in str(key).lower() for sus in suspicious_keys):
                suspicious_patterns.append(f"Suspicious field name: {key}")

        # Check for excessively nested data
        def check_depth(obj, current_depth=0):
            if current_depth > 10:
                return True
            if isinstance(obj, dict):
                return any(check_depth(v, current_depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                return any(check_depth(item, current_depth + 1) for item in obj)
            return False

        if check_depth(input_data):
            suspicious_patterns.append("Excessively nested data structure")

        # Check for unusually large string values
        for key, value in input_data.items():
            if isinstance(value, str) and len(value) > 1000000:  # 1MB
                suspicious_patterns.append(f"Unusually large string in field: {key}")

        return suspicious_patterns

    def _assess_input_completeness(self, input_data: Dict[str, Any]) -> float:
        """Assess completeness of input data."""
        required_fields = ['text']
        recommended_fields = ['bert_results', 'topic_domain', 'session_id']
        optional_fields = ['include_verification_analysis', 'max_claims', 'priority_mode']

        required_score = sum(1 for field in required_fields if field in input_data) / len(required_fields)
        recommended_score = sum(1 for field in recommended_fields if field in input_data) / len(recommended_fields)
        optional_score = sum(1 for field in optional_fields if field in input_data) / len(optional_fields)

        # Weighted completeness score
        completeness = (required_score * 0.6) + (recommended_score * 0.3) + (optional_score * 0.1)
        return round(completeness, 2)

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics for monitoring."""
        avg_validation_time = (
            self.total_validation_time / self.validation_count
            if self.validation_count > 0 else 0
        )

        return {
            'total_validations': self.validation_count,
            'average_validation_time_ms': round(avg_validation_time * 1000, 2),
            'total_validation_time_seconds': round(self.total_validation_time, 2),
            'security_checks_enabled': self.enable_security_checks,
            'validation_thresholds': {
                'min_text_length': self.min_text_length,
                'max_text_length': self.max_text_length,
                'min_word_count': self.min_word_count,
                'max_word_count': self.max_word_count,
                'min_sentence_count': self.min_sentence_count,
                'min_unique_words_ratio': self.min_unique_words_ratio
            },
            'blocked_patterns_count': len(self.blocked_patterns),
            'configuration_applied': bool(self.config)
        }


class OutputValidator:
    """
    Comprehensive output validator for claim extraction results.
    
    Validates extracted claims, verification analysis, and other outputs
    with quality assessment and consistency checking for production use.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize output validator with production configuration.

        Args:
            config: Optional validation configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.OutputValidator")
        
        # Claim validation thresholds
        self.min_claim_length = self.config.get('min_claim_length', 5)
        self.max_claim_length = self.config.get('max_claim_length', 500)
        self.min_verifiability_score = self.config.get('min_verifiability_score', 1)
        self.max_verifiability_score = self.config.get('max_verifiability_score', 10)
        
        # Quality thresholds
        self.min_quality_score = self.config.get('min_quality_score', 60.0)
        
        self.logger.info("OutputValidator initialized")

    def validate_extracted_claims(self, claims: List[Dict[str, Any]], session_id: str = None) -> ValidationResult:
        """
        Validate extracted claims with comprehensive quality checks.

        Args:
            claims: List of extracted claim dictionaries
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with detailed claim validation results
        """
        errors = []
        warnings = []
        suggestions = []
        score = 100.0
        details = {}

        try:
            if not isinstance(claims, list):
                errors.append("Claims must be provided as a list")
                return ValidationResult(
                    False, errors, warnings, 0.0, details,
                    ["Provide claims as a list of dictionaries"]
                )

            if not claims:
                warnings.append("No claims extracted")
                return ValidationResult(
                    True, errors, warnings, 80.0, 
                    {'claim_count': 0},
                    ["Input may not contain extractable claims"]
                )

            details['claim_count'] = len(claims)
            claim_issues = 0
            valid_claims = 0
            
            required_fields = ['text', 'claim_type', 'priority', 'verifiability_score']
            field_presence = {field: 0 for field in required_fields}

            for i, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    errors.append(f"Claim {i+1} is not a dictionary")
                    claim_issues += 1
                    continue

                claim_valid = True

                # Check required fields
                for field in required_fields:
                    if field in claim and claim[field] is not None:
                        field_presence[field] += 1
                    else:
                        warnings.append(f"Claim {i+1} missing field: {field}")
                        claim_valid = False

                # Validate claim text
                if 'text' in claim:
                    text = claim['text']
                    if not isinstance(text, str):
                        errors.append(f"Claim {i+1} text must be a string")
                        claim_valid = False
                    elif len(text.strip()) < self.min_claim_length:
                        warnings.append(f"Claim {i+1} text too short: {len(text.strip())} characters")
                        claim_valid = False
                    elif len(text) > self.max_claim_length:
                        warnings.append(f"Claim {i+1} text too long: {len(text)} characters")

                # Validate claim type
                if 'claim_type' in claim:
                    valid_types = ['Statistical', 'Attribution', 'Event', 'Research', 'Policy', 'Causal', 'Other']
                    if claim['claim_type'] not in valid_types:
                        warnings.append(f"Claim {i+1} has invalid type: {claim['claim_type']}")

                # Validate priority
                if 'priority' in claim:
                    priority = claim['priority']
                    if not isinstance(priority, int) or not 1 <= priority <= 3:
                        warnings.append(f"Claim {i+1} has invalid priority: {priority}")

                # Validate verifiability score
                if 'verifiability_score' in claim:
                    score_val = claim['verifiability_score']
                    if not isinstance(score_val, (int, float)) or not 1 <= score_val <= 10:
                        warnings.append(f"Claim {i+1} has invalid verifiability score: {score_val}")

                if claim_valid:
                    valid_claims += 1
                else:
                    claim_issues += 1

            # Calculate field coverage
            field_coverage = {field: (count / len(claims)) for field, count in field_presence.items()}
            details['field_coverage'] = field_coverage

            # Assess overall claim quality
            quality_score = (valid_claims / len(claims)) * 100 if claims else 0
            details['quality_score'] = round(quality_score, 1)
            details['valid_claims'] = valid_claims
            details['claim_issues'] = claim_issues

            # Adjust score based on quality
            if quality_score < 50:
                errors.append(f"Low claim quality: {quality_score:.1f}% valid claims")
                score -= 40
            elif quality_score < 80:
                warnings.append(f"Moderate claim quality: {quality_score:.1f}% valid claims")
                score -= 20

            # Check for field coverage issues
            for field, coverage in field_coverage.items():
                if coverage < 0.8:
                    warnings.append(f"Low coverage for field '{field}': {coverage:.1%}")
                    suggestions.append(f"Ensure all claims include '{field}' field")
                    score -= 10

            # Diversity assessment
            claim_types = [claim.get('claim_type', 'Unknown') for claim in claims]
            unique_types = len(set(claim_types))
            details['claim_type_diversity'] = unique_types
            
            if unique_types == 1 and len(claims) > 3:
                warnings.append("Low claim type diversity")
                suggestions.append("Consider extracting different types of claims")

            is_valid = len(errors) == 0
            final_score = max(0.0, min(100.0, score))

            self.logger.info(
                f"Claims validation: {'PASSED' if is_valid else 'FAILED'}",
                extra={
                    'session_id': session_id,
                    'claim_count': len(claims),
                    'valid_claims': valid_claims,
                    'quality_score': quality_score
                }
            )

            return ValidationResult(is_valid, errors, warnings, final_score, details, suggestions)

        except Exception as e:
            self.logger.error(f"Claims validation failed: {str(e)}", extra={'session_id': session_id})
            return ValidationResult(
                False,
                [f"Claims validation error: {str(e)}"],
                [],
                0.0,
                {},
                ["Check claim format and try again"]
            )

    def validate_extraction_output(self, output: Dict[str, Any], session_id: str = None) -> ValidationResult:
        """
        Validate complete extraction output with comprehensive checks.

        Args:
            output: Complete extraction output dictionary
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with comprehensive output validation
        """
        errors = []
        warnings = []
        suggestions = []
        score = 100.0
        details = {}

        try:
            if not isinstance(output, dict):
                errors.append("Output must be a dictionary")
                return ValidationResult(False, errors, warnings, 0.0, details)

            # Check required output fields
            required_fields = ['extracted_claims', 'metadata']
            for field in required_fields:
                if field not in output:
                    errors.append(f"Missing required output field: {field}")
                    score -= 25

            # Validate extracted claims
            if 'extracted_claims' in output:
                claims_validation = self.validate_extracted_claims(
                    output['extracted_claims'], session_id
                )
                details['claims_validation'] = claims_validation.details
                
                if not claims_validation.is_valid:
                    errors.extend(claims_validation.errors)
                    suggestions.extend(claims_validation.suggestions)
                    score -= 30
                
                if claims_validation.warnings:
                    warnings.extend(claims_validation.warnings)
                    score -= 10

            # Validate metadata
            if 'metadata' in output:
                metadata_validation = self._validate_metadata(output['metadata'])
                details['metadata_validation'] = metadata_validation
                
                if not metadata_validation['valid']:
                    warnings.extend(metadata_validation['issues'])
                    score -= 10

            # Validate optional fields
            optional_fields = ['verification_analysis', 'prioritization_analysis', 'pattern_analysis']
            for field in optional_fields:
                if field in output:
                    if output[field] is not None:
                        details[f'{field}_present'] = True
                    else:
                        warnings.append(f"Field {field} is null")

            # Check output consistency
            consistency_issues = self._check_output_consistency(output)
            if consistency_issues:
                warnings.extend(consistency_issues)
                suggestions.append("Review output consistency")
                score -= len(consistency_issues) * 5

            is_valid = len(errors) == 0
            final_score = max(0.0, min(100.0, score))

            return ValidationResult(is_valid, errors, warnings, final_score, details, suggestions)

        except Exception as e:
            self.logger.error(f"Output validation failed: {str(e)}", extra={'session_id': session_id})
            return ValidationResult(False, [f"Output validation error: {str(e)}"], [], 0.0, {})

    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extraction metadata."""
        validation_result = {'valid': True, 'issues': []}

        required_metadata = ['total_claims_found', 'processing_time_seconds']
        for field in required_metadata:
            if field not in metadata:
                validation_result['issues'].append(f"Missing metadata field: {field}")
                validation_result['valid'] = False

        # Validate specific metadata fields
        if 'total_claims_found' in metadata:
            if not isinstance(metadata['total_claims_found'], int) or metadata['total_claims_found'] < 0:
                validation_result['issues'].append("Invalid total_claims_found")

        if 'processing_time_seconds' in metadata:
            if not isinstance(metadata['processing_time_seconds'], (int, float)) or metadata['processing_time_seconds'] < 0:
                validation_result['issues'].append("Invalid processing_time_seconds")

        return validation_result

    def _check_output_consistency(self, output: Dict[str, Any]) -> List[str]:
        """Check consistency between different output components."""
        consistency_issues = []

        # Check claims count consistency
        if 'extracted_claims' in output and 'metadata' in output:
            claims = output['extracted_claims']
            metadata = output['metadata']
            
            if 'total_claims_found' in metadata:
                claimed_count = metadata['total_claims_found']
                actual_count = len(claims) if isinstance(claims, list) else 0
                
                if claimed_count != actual_count:
                    consistency_issues.append(
                        f"Claims count mismatch: metadata says {claimed_count}, actual {actual_count}"
                    )

        return consistency_issues


# Convenience validation functions
def validate_input(input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None, session_id: str = None) -> ValidationResult:
    """Convenience function for input validation."""
    validator = InputValidator(config)
    return validator.validate_input_data(input_data, session_id)


def validate_output(output_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None, session_id: str = None) -> ValidationResult:
    """Convenience function for output validation."""
    validator = OutputValidator(config)
    return validator.validate_extraction_output(output_data, session_id)


def validate_claims(claims: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None, session_id: str = None) -> ValidationResult:
    """Convenience function for claims validation."""
    validator = OutputValidator(config)
    return validator.validate_extracted_claims(claims, session_id)


# Testing functionality
if __name__ == "__main__":
    """Test validator functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== CLAIM EXTRACTOR VALIDATORS TEST ===")
    
    # Test input validation
    print("--- Input Validation Test ---")
    input_validator = InputValidator()
    
    test_input = {
        "text": "This is a comprehensive test article for claim extraction. It contains multiple sentences with factual statements. According to recent studies, this approach works well.",
        "bert_results": {
            "prediction": "REAL",
            "confidence": 0.85
        },
        "topic_domain": "general",
        "max_claims": 5
    }
    
    input_result = input_validator.validate_input_data(test_input, session_id="test_validator_001")
    print(f"✅ Input validation: {'PASSED' if input_result.is_valid else 'FAILED'}")
    print(f"✅ Score: {input_result.score:.1f}/100")
    print(f"✅ Errors: {len(input_result.errors)}")
    print(f"✅ Warnings: {len(input_result.warnings)}")
    
    if input_result.errors:
        print(f"   Errors: {input_result.errors}")
    if input_result.warnings:
        print(f"   Warnings: {input_result.warnings[:2]}")
    
    # Test output validation
    print("\n--- Output Validation Test ---")
    output_validator = OutputValidator()
    
    test_output = {
        "extracted_claims": [
            {
                "claim_id": 1,
                "text": "Recent studies show this approach works well",
                "claim_type": "Research",
                "priority": 1,
                "verifiability_score": 8,
                "source": "Studies mentioned",
                "verification_strategy": "Check recent research publications"
            },
            {
                "claim_id": 2,
                "text": "This contains multiple sentences",
                "claim_type": "Statistical",
                "priority": 3,
                "verifiability_score": 5,
                "source": "Article content",
                "verification_strategy": "Count sentences in article"
            }
        ],
        "metadata": {
            "total_claims_found": 2,
            "processing_time_seconds": 1.5,
            "model_used": "gemini-1.5-pro",
            "agent_version": "3.1.0"
        },
        "verification_analysis": "Sample verification analysis",
        "pattern_analysis": {"total_indicators": 5}
    }
    
    output_result = output_validator.validate_extraction_output(test_output, session_id="test_validator_002")
    print(f"✅ Output validation: {'PASSED' if output_result.is_valid else 'FAILED'}")
    print(f"✅ Score: {output_result.score:.1f}/100")
    print(f"✅ Claims quality: {output_result.details.get('claims_validation', {}).get('quality_score', 0):.1f}%")
    
    # Test individual components
    print("\n--- Individual Component Tests ---")
    
    # Test article text validation
    article_result = input_validator.validate_article_text(test_input["text"], session_id="test_validator_003")
    print(f"✅ Article text validation: {'PASSED' if article_result.is_valid else 'FAILED'}")
    print(f"   Text length: {article_result.details['clean_length']} characters")
    print(f"   Word count: {article_result.details['word_count']} words")
    print(f"   Sentence count: {article_result.details['sentence_count']} sentences")
    
    # Test BERT results validation
    bert_result = input_validator.validate_bert_results(test_input["bert_results"], session_id="test_validator_004")
    print(f"✅ BERT results validation: {'PASSED' if bert_result.is_valid else 'FAILED'}")
    print(f"   Prediction: {bert_result.details.get('prediction', 'N/A')}")
    print(f"   Confidence: {bert_result.details.get('confidence', 'N/A')}")
    
    # Test claims validation
    claims_result = output_validator.validate_extracted_claims(test_output["extracted_claims"], session_id="test_validator_005")
    print(f"✅ Claims validation: {'PASSED' if claims_result.is_valid else 'FAILED'}")
    print(f"   Valid claims: {claims_result.details['valid_claims']}/{claims_result.details['claim_count']}")
    print(f"   Field coverage: {len([c for c in claims_result.details['field_coverage'].values() if c >= 0.8])}/4 fields")
    
    # Test performance statistics
    print("\n--- Performance Statistics ---")
    input_stats = input_validator.get_validation_statistics()
    print(f"✅ Total validations: {input_stats['total_validations']}")
    print(f"✅ Average time: {input_stats['average_validation_time_ms']:.1f}ms")
    print(f"✅ Security checks: {'Enabled' if input_stats['security_checks_enabled'] else 'Disabled'}")
    
    print("\n🎯 Validator tests completed successfully!")
