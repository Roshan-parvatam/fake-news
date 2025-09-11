# agents/credible_source/validators.py

"""
Validators for Credible Source Agent - Production Ready

Comprehensive input and output validation with detailed error reporting,
configurable validation rules, enhanced error handling, and structured feedback
for production reliability.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .exceptions import (
    InputValidationError,
    DataFormatError,
    raise_input_validation_error
)


@dataclass
class ValidationResult:
    """
    Structured validation result container with enhanced feedback.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of validation error messages
        warnings: List of validation warnings
        suggestions: List of suggestions for fixing issues
        score: Validation quality score (0-100)
        details: Additional validation details
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
    Production-ready input validator for credible source agent.
    
    Validates article text, extracted claims, evidence evaluation,
    and other input data with configurable validation rules and
    comprehensive error reporting.
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
        self.max_text_length = self.config.get('max_text_length', 50000)
        self.recommended_text_length = self.config.get('recommended_text_length', 200)
        
        self.min_word_count = self.config.get('min_word_count', 10)
        self.min_sentence_count = self.config.get('min_sentence_count', 3)
        
        # Claims validation thresholds
        self.min_claims_count = self.config.get('min_claims_count', 0)  # Allow empty claims
        self.max_claims_count = self.config.get('max_claims_count', 15)
        self.min_claim_length = self.config.get('min_claim_length', 10)
        
        # Security validation
        self.enable_security_checks = self.config.get('enable_security_checks', True)
        
        self.logger.info(f"InputValidator initialized with production settings")

    def validate_article_text(self, text: Any, session_id: str = None) -> ValidationResult:
        """
        Validate article text content with comprehensive feedback.

        Args:
            text: Article text to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with detailed validation results and suggestions
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        self.logger.debug(f"Validating article text", extra={'session_id': session_id})

        # Basic type and existence checks
        if not isinstance(text, str):
            result.add_error(
                f"Article text must be a string, received {type(text).__name__}",
                "Convert the input to a string before processing"
            )
            self.logger.warning(f"Invalid article text type: {type(text).__name__}", 
                              extra={'session_id': session_id})
            return result

        if not text or not text.strip():
            result.add_error(
                "Article text cannot be empty or contain only whitespace",
                f"Provide meaningful article content with at least {self.min_text_length} characters"
            )
            return result

        # Content analysis
        text_clean = text.strip()
        text_length = len(text_clean)
        
        result.details['original_length'] = len(text)
        result.details['clean_length'] = text_length
        
        # Length validation with detailed feedback
        if text_length < self.min_text_length:
            result.add_error(
                f"Article text too short: {text_length} characters (minimum: {self.min_text_length})",
                f"Article should be at least {self.min_text_length} characters for meaningful source analysis"
            )
            result.score -= 40
        
        if text_length > self.max_text_length:
            result.add_warning(
                f"Article text very long: {text_length} characters (maximum recommended: {self.max_text_length})",
                f"Consider summarizing to {self.max_text_length} characters for optimal processing"
            )
            result.score -= 10
        
        if text_length < self.recommended_text_length:
            result.add_warning(
                f"Article text may be too short for comprehensive source analysis: {text_length} characters",
                f"For best source recommendations, provide at least {self.recommended_text_length} characters"
            )
            result.score -= 15

        # Word count validation
        words = text_clean.split()
        word_count = len(words)
        result.details['word_count'] = word_count

        if word_count < self.min_word_count:
            result.add_error(
                f"Article has too few words: {word_count} (minimum: {self.min_word_count})",
                f"Provide more detailed content with at least {self.min_word_count} words"
            )
            result.score -= 30

        # Sentence structure validation
        sentences = re.split(r'[.!?]+', text_clean)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentence_count = len(valid_sentences)
        result.details['sentence_count'] = sentence_count

        if sentence_count < self.min_sentence_count:
            result.add_warning(
                f"Article has few sentences: {sentence_count} (recommended: {self.min_sentence_count}+)",
                "Ensure the text contains complete sentences for better source matching"
            )
            result.score -= 15

        # Content quality checks
        if word_count > 0:
            avg_word_length = sum(len(word) for word in words) / word_count
            result.details['average_word_length'] = round(avg_word_length, 2)
            
            if avg_word_length < 3:
                result.add_warning(
                    f"Average word length seems short: {avg_word_length:.1f} characters",
                    "Consider using more descriptive words for better source analysis"
                )

        # Check for excessive formatting issues
        newline_ratio = text.count('\n') / text_length if text_length > 0 else 0
        if newline_ratio > 0.1:
            result.add_warning(
                "Article contains excessive line breaks which may affect analysis quality",
                "Consider cleaning up formatting before processing"
            )

        # Security validation if enabled
        if self.enable_security_checks:
            security_issues = self._check_security_patterns(text_clean, session_id)
            if security_issues:
                result.errors.extend(security_issues)
                result.score -= 40

        # Final validation
        result.is_valid = len(result.errors) == 0 and result.score > 0
        result.score = max(0.0, min(100.0, result.score))

        self.logger.info(f"Article text validation completed", 
                        extra={
                            'session_id': session_id,
                            'text_length': text_length,
                            'is_valid': result.is_valid,
                            'score': result.score,
                            'error_count': len(result.errors)
                        })

        return result

    def validate_extracted_claims(self, claims: Any, session_id: str = None) -> ValidationResult:
        """
        Validate extracted claims structure and content with detailed feedback.

        Args:
            claims: Claims list to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation results and improvement suggestions
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        self.logger.debug(f"Validating extracted claims", extra={'session_id': session_id})

        # Type validation
        if claims is None:
            result.add_warning(
                "No claims provided for validation",
                "Add claims for better source recommendations: [{'text': 'claim text', 'claim_type': 'Research'}]"
            )
            result.score = 80.0
            return result

        if not isinstance(claims, list):
            result.add_error(
                f"Claims must be a list, received {type(claims).__name__}",
                "Ensure claims are provided as a list of dictionaries"
            )
            self.logger.warning(f"Invalid claims type: {type(claims).__name__}", 
                              extra={'session_id': session_id})
            return result

        claims_count = len(claims)
        result.details['claims_count'] = claims_count

        # Count validation
        if claims_count < self.min_claims_count:
            result.add_error(
                f"Too few claims provided: {claims_count} (minimum: {self.min_claims_count})",
                f"Provide at least {self.min_claims_count} claim for source analysis"
            )
            result.score -= 30

        if claims_count > self.max_claims_count:
            result.add_warning(
                f"Many claims provided: {claims_count} (maximum recommended: {self.max_claims_count})",
                f"Consider focusing on the top {self.max_claims_count} most important claims"
            )
            result.score -= 10

        if claims_count == 0:
            result.add_warning(
                "Claims list is empty",
                "Add claims for targeted source recommendations"
            )
            result.score -= 20
            return result

        # Individual claim validation
        valid_claims = 0
        claim_types_found = set()
        
        for i, claim in enumerate(claims):
            claim_errors = self._validate_single_claim(claim, i, session_id)
            result.errors.extend(claim_errors)
            
            if not claim_errors:
                valid_claims += 1
                if isinstance(claim, dict) and 'claim_type' in claim:
                    claim_types_found.add(claim['claim_type'])

        result.details['valid_claims'] = valid_claims
        result.details['claim_types_found'] = list(claim_types_found)

        # Claims quality assessment
        if claims_count > 0:
            valid_ratio = valid_claims / claims_count
            if valid_ratio < 0.5:
                result.add_error(
                    f"More than half of the claims are invalid ({valid_claims}/{claims_count})",
                    "Fix claim formatting issues before processing"
                )
                result.score -= 30
            elif valid_ratio < 0.8:
                result.add_warning(
                    f"Some claims have formatting issues ({valid_claims}/{claims_count} valid)",
                    "Review and fix claim formatting for better source recommendations"
                )
                result.score -= 15

        # Diversity check
        if len(claim_types_found) == 1 and valid_claims > 1:
            result.add_warning(
                f"All claims are of the same type: {list(claim_types_found)[0]}",
                "Consider including different claim types for diverse source recommendations"
            )

        # Final validation
        result.is_valid = len(result.errors) == 0 and valid_claims > 0
        result.score = max(0.0, min(100.0, result.score))

        self.logger.info(f"Claims validation completed", 
                        extra={
                            'session_id': session_id,
                            'claims_count': claims_count,
                            'valid_claims': valid_claims,
                            'is_valid': result.is_valid,
                            'claim_types': len(claim_types_found)
                        })

        return result

    def _validate_single_claim(self, claim: Any, index: int, session_id: str = None) -> List[str]:
        """Validate a single claim object with detailed error messages."""
        errors = []

        # Type validation
        if not isinstance(claim, dict):
            errors.append(
                f"Claim {index + 1} must be a dictionary, received {type(claim).__name__}. "
                f"Expected format: {{'text': 'claim text', 'claim_type': 'Research', 'priority': 1}}"
            )
            return errors

        # Required fields validation
        if 'text' not in claim:
            errors.append(
                f"Claim {index + 1} missing required 'text' field. "
                f"Add: 'text': 'your claim content here'"
            )
        else:
            claim_text = claim['text']
            if not isinstance(claim_text, str):
                errors.append(
                    f"Claim {index + 1} 'text' must be a string, received {type(claim_text).__name__}. "
                    f"Convert to string: 'text': '{str(claim_text)}'"
                )
            elif not claim_text.strip():
                errors.append(
                    f"Claim {index + 1} 'text' cannot be empty. "
                    f"Provide meaningful claim text"
                )
            elif len(claim_text.strip()) < self.min_claim_length:
                errors.append(
                    f"Claim {index + 1} 'text' too short: '{claim_text.strip()}' ({len(claim_text.strip())} chars). "
                    f"Provide more descriptive claim text (at least {self.min_claim_length} characters)"
                )

        # Optional field validation with helpful messages
        if 'claim_type' not in claim:
            errors.append(
                f"Claim {index + 1} missing 'claim_type' field. "
                f"Add claim type: 'claim_type': 'Research' or 'Statistical' or 'Factual'"
            )
        else:
            claim_type = claim['claim_type']
            valid_types = ['Research', 'Statistical', 'Factual', 'Attribution', 'Policy', 'Medical', 'Technical']
            if claim_type not in valid_types:
                errors.append(
                    f"Claim {index + 1} has unknown claim_type: '{claim_type}'. "
                    f"Use one of: {valid_types}"
                )

        # Priority validation
        if 'priority' in claim:
            priority = claim['priority']
            if not isinstance(priority, int):
                errors.append(
                    f"Claim {index + 1} 'priority' must be an integer, received {type(priority).__name__}. "
                    f"Use integer 1-5, e.g., 'priority': 1"
                )
            elif not (1 <= priority <= 5):
                errors.append(
                    f"Claim {index + 1} 'priority' must be between 1-5, received {priority}. "
                    f"Adjust to valid range: 'priority': {max(1, min(5, priority))}"
                )

        # Verifiability score validation
        if 'verifiability_score' in claim:
            score = claim['verifiability_score']
            if not isinstance(score, (int, float)):
                errors.append(
                    f"Claim {index + 1} 'verifiability_score' must be a number, received {type(score).__name__}. "
                    f"Use a number between 0-10, e.g., 'verifiability_score': 7"
                )
            elif not (0 <= score <= 10):
                errors.append(
                    f"Claim {index + 1} 'verifiability_score' must be between 0-10, received {score}. "
                    f"Adjust to valid range: 'verifiability_score': {max(0, min(10, score))}"
                )

        return errors

    def validate_evidence_evaluation(self, evidence: Any, session_id: str = None) -> ValidationResult:
        """
        Validate evidence evaluation structure with helpful feedback.

        Args:
            evidence: Evidence evaluation dictionary to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and suggestions
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        self.logger.debug(f"Validating evidence evaluation", extra={'session_id': session_id})

        if evidence is None:
            result.add_warning(
                "No evidence evaluation provided",
                "Add evidence evaluation for better context: {'overall_evidence_score': 7.5, 'source_quality': 8.0}"
            )
            result.score = 80.0
            return result

        # Type validation
        if not isinstance(evidence, dict):
            result.add_error(
                f"Evidence evaluation must be a dictionary, received {type(evidence).__name__}",
                "Provide evidence as: {'overall_evidence_score': 7.5, 'source_quality': 8.0}"
            )
            return result

        # Score validations with detailed feedback
        score_fields = {
            'overall_evidence_score': 'Overall evidence quality score',
            'source_quality': 'Source quality assessment score',
            'factual_accuracy': 'Factual accuracy score',
            'supporting_evidence': 'Supporting evidence strength score'
        }

        scores_found = 0
        for field, description in score_fields.items():
            if field in evidence:
                score = evidence[field]
                scores_found += 1
                
                if not isinstance(score, (int, float)):
                    result.add_error(
                        f"{description} must be numeric, received {type(score).__name__}",
                        f"Use a decimal number: '{field}': 7.5"
                    )
                elif not (0 <= score <= 10):
                    result.add_error(
                        f"{description} must be between 0-10, received {score}",
                        f"Adjust to valid range: '{field}': {max(0, min(10, score))}"
                    )
                else:
                    result.details[field] = score

        # Provide helpful suggestions for missing fields
        if 'overall_evidence_score' not in evidence:
            result.add_warning(
                "Evidence evaluation missing 'overall_evidence_score'",
                "Add overall score for better source targeting: 'overall_evidence_score': 7.5"
            )
            result.score -= 15

        if scores_found == 0:
            result.add_warning(
                "No evidence scores found in evaluation",
                "Add at least 'overall_evidence_score' for meaningful assessment"
            )
            result.score -= 20

        # Final validation
        result.is_valid = len(result.errors) == 0
        result.score = max(0.0, min(100.0, result.score))

        self.logger.info(f"Evidence evaluation validation completed", 
                        extra={
                            'session_id': session_id,
                            'is_valid': result.is_valid,
                            'scores_found': scores_found
                        })

        return result

    def validate_input_data(self, input_data: Any, session_id: str = None) -> ValidationResult:
        """
        Comprehensive input data validation for credible source recommendations.

        Args:
            input_data: Complete input data dictionary
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with comprehensive validation results
        """
        all_errors = []
        all_warnings = []
        all_suggestions = []

        self.logger.info(f"Starting comprehensive input validation", 
                        extra={
                            'session_id': session_id,
                            'input_keys': list(input_data.keys()) if isinstance(input_data, dict) else None
                        })

        # Basic structure validation
        if not isinstance(input_data, dict):
            return ValidationResult(
                is_valid=False,
                errors=[f"Input must be a dictionary, received {type(input_data).__name__}"],
                suggestions=["Provide input as: {'text': 'article text', 'extracted_claims': [...], 'evidence_evaluation': {...}}"]
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
        if 'extracted_claims' in input_data:
            claims_result = self.validate_extracted_claims(input_data['extracted_claims'], session_id)
            # Claims errors are warnings for source recommendation (not critical)
            all_warnings.extend([f"Claims: {error}" for error in claims_result.errors])
            all_warnings.extend([f"Claims: {warning}" for warning in claims_result.warnings])
            all_suggestions.extend(claims_result.suggestions)
        else:
            all_warnings.append("No extracted claims provided - source analysis will be more generic")
            all_suggestions.append("Add claims for targeted recommendations: 'extracted_claims': [{'text': 'claim', 'claim_type': 'Research'}]")

        if 'evidence_evaluation' in input_data:
            evidence_result = self.validate_evidence_evaluation(input_data['evidence_evaluation'], session_id)
            # Evidence errors are warnings for source recommendation (not critical)
            all_warnings.extend([f"Evidence: {error}" for error in evidence_result.errors])
            all_warnings.extend([f"Evidence: {warning}" for warning in evidence_result.warnings])
            all_suggestions.extend(evidence_result.suggestions)

        # Additional field validation
        optional_fields = {
            'include_detailed_analysis': bool,
            'max_sources': int,
            'domain_hint': str
        }

        for field_name, expected_type in optional_fields.items():
            if field_name in input_data:
                value = input_data[field_name]
                if not isinstance(value, expected_type):
                    all_warnings.append(f"'{field_name}' should be {expected_type.__name__}, received {type(value).__name__}")
                    all_suggestions.append(f"Use correct type: '{field_name}': {expected_type.__name__.lower()} value")

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
                'has_claims': 'extracted_claims' in input_data,
                'has_evidence': 'evidence_evaluation' in input_data,
                'total_fields': len(input_data) if isinstance(input_data, dict) else 0
            }
        )

        self.logger.info(f"Comprehensive input validation completed", 
                        extra={
                            'session_id': session_id,
                            'is_valid': result.is_valid,
                            'error_count': len(all_errors),
                            'warning_count': len(all_warnings),
                            'score': score
                        })

        return result

    def _check_security_patterns(self, text: str, session_id: str = None) -> List[str]:
        """Check for security-related patterns in text."""
        security_issues = []
        
        try:
            # Check for script injection patterns
            script_patterns = [
                r'<script[^>]*>',
                r'javascript:',
                r'on\w+\s*=',
                r'eval\s*\(',
                r'exec\s*\('
            ]
            
            for pattern in script_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    security_issues.append(f"Potentially unsafe content detected: script-like patterns")
                    self.logger.warning(f"Security pattern detected: {pattern}", 
                                      extra={'session_id': session_id})
                    break
            
            # Check for excessive special characters (possible encoding attack)
            special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
            if special_char_ratio > 0.3:
                security_issues.append(f"Excessive special characters detected ({special_char_ratio:.1%})")
                
            return security_issues
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {str(e)}", extra={'session_id': session_id})
            return []


class URLValidator:
    """
    Enhanced URL validator for source recommendations with credibility assessment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize URL validator with production settings."""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.URLValidator")
        
        self.trusted_domains = self.config.get('trusted_domains', [
            'gov', 'edu', 'who.int', 'cdc.gov', 'nih.gov', 'pubmed.ncbi.nlm.nih.gov',
            'nature.com', 'science.org', 'reuters.com', 'apnews.com', 'bbc.com',
            'arxiv.org', 'jstor.org', 'springer.com', 'wiley.com'
        ])

    def validate_url_format(self, url: str, session_id: str = None) -> ValidationResult:
        """
        Validate URL format and structure with detailed feedback.

        Args:
            url: URL string to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and improvement suggestions
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Basic format validation
        if not isinstance(url, str):
            result.add_error(
                f"URL must be a string, received {type(url).__name__}",
                "Convert URL to string format"
            )
            return result

        if not url.strip():
            result.add_error(
                "URL cannot be empty",
                "Provide a valid URL starting with http:// or https://"
            )
            return result

        # URL parsing validation
        try:
            parsed = urlparse(url.strip())
        except Exception as e:
            result.add_error(
                f"Invalid URL format: {str(e)}",
                "Ensure URL follows format: https://domain.com/specific-page"
            )
            return result

        # Scheme validation
        if not parsed.scheme:
            result.add_error(
                "URL missing protocol (http/https)",
                f"Add protocol: https://{url.strip()}"
            )
        elif parsed.scheme not in ['http', 'https']:
            result.add_error(
                f"Invalid URL protocol: {parsed.scheme} (must be http or https)",
                f"Change to: https://{parsed.netloc}{parsed.path}"
            )

        # Domain validation
        if not parsed.netloc:
            result.add_error(
                "URL missing domain",
                "Provide complete URL: https://domain.com/page"
            )
        elif len(parsed.netloc) < 4:
            result.add_error(
                f"Invalid domain: {parsed.netloc}",
                "Ensure domain is complete: https://example.com"
            )

        # HTTPS preference
        if parsed.scheme == 'http':
            result.add_warning(
                "HTTP URL detected - HTTPS preferred for security",
                f"Use HTTPS version if available: https://{parsed.netloc}{parsed.path}"
            )

        # Check domain credibility
        domain_credibility = self._assess_domain_credibility(parsed.netloc)
        result.details['domain_credibility'] = domain_credibility

        self.logger.debug(f"URL format validation completed", 
                         extra={
                             'session_id': session_id,
                             'url': url[:100],  # Truncate for logging
                             'is_valid': result.is_valid,
                             'domain_credibility': domain_credibility
                         })

        return result

    def _assess_domain_credibility(self, domain: str) -> Dict[str, Any]:
        """Assess domain credibility for source recommendations."""
        credibility = {
            'is_trusted': False,
            'credibility_score': 5.0,
            'domain_type': 'unknown'
        }

        if not domain:
            return credibility

        domain_lower = domain.lower()

        # Check against trusted domains
        for trusted in self.trusted_domains:
            if trusted in domain_lower:
                credibility['is_trusted'] = True
                credibility['credibility_score'] = 9.0
                break

        # Domain type assessment
        if '.gov' in domain_lower:
            credibility['domain_type'] = 'government'
            credibility['credibility_score'] = max(credibility['credibility_score'], 9.0)
        elif '.edu' in domain_lower:
            credibility['domain_type'] = 'academic'
            credibility['credibility_score'] = max(credibility['credibility_score'], 8.5)
        elif '.org' in domain_lower:
            credibility['domain_type'] = 'organization'
            credibility['credibility_score'] = max(credibility['credibility_score'], 7.0)
        elif any(news in domain_lower for news in ['reuters', 'ap', 'bbc', 'npr']):
            credibility['domain_type'] = 'established_news'
            credibility['credibility_score'] = max(credibility['credibility_score'], 8.0)

        return credibility


class OutputValidator:
    """
    Production-ready output validator for source recommendation results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize output validator with production settings."""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.OutputValidator")
        self.url_validator = URLValidator(config)

    def validate_source_recommendations(self, recommendations: Any, session_id: str = None) -> ValidationResult:
        """
        Validate source recommendations output with quality assessment.

        Args:
            recommendations: Source recommendations to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and quality feedback
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Type validation
        if not isinstance(recommendations, dict):
            result.add_error(
                f"Recommendations must be a dictionary, received {type(recommendations).__name__}",
                "Ensure recommendations are returned as a structured dictionary"
            )
            return result

        # Check for required fields
        required_fields = ['contextual_sources', 'database_recommendations', 'recommendation_scores']
        for field in required_fields:
            if field not in recommendations:
                result.add_error(
                    f"Missing required field: '{field}'",
                    f"Include {field} in recommendation output"
                )

        # Validate contextual sources
        if 'contextual_sources' in recommendations:
            contextual_sources = recommendations['contextual_sources']
            if isinstance(contextual_sources, list):
                source_quality = self._assess_source_list_quality(contextual_sources, session_id)
                result.details['contextual_source_quality'] = source_quality
                
                if source_quality['high_quality_count'] == 0 and len(contextual_sources) > 0:
                    result.add_warning(
                        "No high-quality contextual sources found",
                        "Ensure sources include credible institutions and specific URLs"
                    )

        # Validate recommendation scores
        if 'recommendation_scores' in recommendations:
            scores = recommendations['recommendation_scores']
            if isinstance(scores, dict):
                score_validation = self._validate_recommendation_scores(scores)
                result.errors.extend(score_validation['errors'])
                result.warnings.extend(score_validation['warnings'])

        result.is_valid = len(result.errors) == 0

        self.logger.info(f"Source recommendations validation completed", 
                        extra={
                            'session_id': session_id,
                            'is_valid': result.is_valid,
                            'error_count': len(result.errors)
                        })

        return result

    def _assess_source_list_quality(self, sources: List[Any], session_id: str = None) -> Dict[str, Any]:
        """Assess the quality of a source list."""
        quality_metrics = {
            'total_sources': len(sources),
            'high_quality_count': 0,
            'has_urls': 0,
            'institutional_sources': 0
        }

        for source in sources:
            if isinstance(source, dict):
                # Check for URL presence
                if 'url' in source and source['url']:
                    quality_metrics['has_urls'] += 1

                # Check for institutional indicators
                name = source.get('name', '').lower()
                if any(inst in name for inst in ['university', 'institute', 'agency', 'department']):
                    quality_metrics['institutional_sources'] += 1

                # Check reliability score
                reliability = source.get('reliability_score', 0)
                if reliability >= 8:
                    quality_metrics['high_quality_count'] += 1

        return quality_metrics

    def _validate_recommendation_scores(self, scores: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate recommendation scores structure."""
        validation = {'errors': [], 'warnings': []}

        required_scores = ['overall_recommendation_score', 'source_quality_score', 'source_relevance_score']
        for score_field in required_scores:
            if score_field not in scores:
                validation['errors'].append(f"Missing required score: {score_field}")
            else:
                score = scores[score_field]
                if not isinstance(score, (int, float)) or not (0 <= score <= 10):
                    validation['errors'].append(f"Invalid {score_field}: must be 0-10, got {score}")

        return validation


# Convenience validation functions
def validate_credible_source_input(input_data: Dict[str, Any], 
                                 config: Dict[str, Any] = None,
                                 session_id: str = None) -> ValidationResult:
    """
    Validate complete credible source input with comprehensive feedback.

    Args:
        input_data: Input data to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult with validation status, errors, warnings, and suggestions
    """
    validator = InputValidator(config)
    return validator.validate_input_data(input_data, session_id)


def validate_source_url(url: str, 
                       config: Dict[str, Any] = None,
                       session_id: str = None) -> bool:
    """
    Quick URL validation for source recommendations.

    Args:
        url: URL to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        True if URL is valid, False otherwise
    """
    validator = URLValidator(config)
    result = validator.validate_url_format(url, session_id)
    return result.is_valid


# Testing functionality
if __name__ == "__main__":
    """Test validation functionality with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    test_session_id = "validation_test_789"
    
    # Test input validation
    test_input = {
        'text': 'This is a test article with sufficient length for comprehensive source analysis. It contains multiple sentences and should pass validation with good scores.',
        'extracted_claims': [
            {'text': 'This is a test claim with good length', 'claim_type': 'Research', 'priority': 1},
            {'text': 'Another test claim for verification', 'claim_type': 'Statistical', 'priority': 2}
        ],
        'evidence_evaluation': {
            'overall_evidence_score': 7.5,
            'source_quality': 8.0
        }
    }
    
    print("=== INPUT VALIDATION TEST ===")
    input_result = validate_credible_source_input(test_input, session_id=test_session_id)
    print(f"Input validation: {'✅ Valid' if input_result.is_valid else '❌ Invalid'}")
    print(f"Score: {input_result.score:.1f}/100")
    if input_result.errors:
        print(f"Errors: {input_result.errors[:2]}")  # Show first 2 errors
    if input_result.warnings:
        print(f"Warnings: {input_result.warnings[:2]}")  # Show first 2 warnings
    if input_result.suggestions:
        print(f"Suggestions: {input_result.suggestions[:2]}")  # Show first 2 suggestions
    
    # Test URL validation
    test_urls = [
        'https://www.cdc.gov/vaccines/covid-19/clinical-considerations/managing-anaphylaxis.html',  # Valid specific
        'https://pubmed.ncbi.nlm.nih.gov/34289274/',  # Valid specific
        'invalid-url',  # Invalid format
        'https://www.cdc.gov/'  # Valid but generic
    ]
    
    print("\n=== URL VALIDATION TEST ===")
    url_validator = URLValidator()
    for url in test_urls:
        result = url_validator.validate_url_format(url, test_session_id)
        print(f"\nURL: {url}")
        print(f"  Valid: {'✅' if result.is_valid else '❌'}")
        if result.details.get('domain_credibility'):
            cred = result.details['domain_credibility']
            print(f"  Credibility: {cred['credibility_score']:.1f}/10 ({cred['domain_type']})")
        if result.errors:
            print(f"  Issues: {result.errors[0]}")
    
    print("\n✅ Validation tests completed")
