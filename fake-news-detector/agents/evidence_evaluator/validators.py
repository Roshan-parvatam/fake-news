# agents/evidence_evaluator/validators.py

"""
Evidence Evaluator Input/Output Validation - Production Ready

Enhanced validation utilities with clear error messages, better debugging support,
and production-level validation feedback for reliable operation.
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from dataclasses import dataclass

from .exceptions import (
    InputValidationError,
    VerificationSourceError,
    DataFormatError,
    raise_input_validation_error,
    raise_verification_source_error
)


@dataclass
class ValidationResult:
    """Container for validation results with enhanced feedback."""
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    suggestions: List[str] = None  # Added suggestions for fixing issues

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []

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
    Production-ready input validation for evidence evaluator processing.
    
    Provides comprehensive validation with clear error messages, suggestions
    for fixing issues, and detailed feedback for debugging.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize input validator with production configuration.

        Args:
            config: Optional configuration for validation rules
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.InputValidator")
        
        # Configurable validation thresholds
        self.min_article_length = self.config.get('min_article_length', 50)
        self.max_article_length = self.config.get('max_article_length', 50000)
        self.recommended_article_length = self.config.get('recommended_article_length', 200)
        
        self.min_claims_count = self.config.get('min_claims_count', 1)
        self.max_claims_count = self.config.get('max_claims_count', 20)
        self.recommended_claims_count = self.config.get('recommended_claims_count', 3)

    def validate_article_text(self, text: Any, session_id: str = None) -> ValidationResult:
        """
        Validate article text input with comprehensive feedback.

        Args:
            text: Article text to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status, errors, warnings, and suggestions
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Type validation
        if not isinstance(text, str):
            result.add_error(
                f"Article text must be a string, received {type(text).__name__}",
                "Convert the input to a string before processing"
            )
            self.logger.warning(f"Invalid article text type: {type(text).__name__}", 
                              extra={'session_id': session_id})
            return result

        # Content validation
        if not text.strip():
            result.add_error(
                "Article text cannot be empty or contain only whitespace",
                "Provide meaningful article content with at least 50 characters"
            )
            return result

        # Length validation
        text_length = len(text.strip())
        
        if text_length < self.min_article_length:
            result.add_error(
                f"Article text too short: {text_length} characters (minimum: {self.min_article_length})",
                f"Article should be at least {self.min_article_length} characters for meaningful analysis"
            )
        
        if text_length > self.max_article_length:
            result.add_warning(
                f"Article text very long: {text_length} characters (maximum recommended: {self.max_article_length})",
                f"Consider truncating to {self.max_article_length} characters for optimal processing"
            )
        
        if text_length < self.recommended_article_length:
            result.add_warning(
                f"Article text may be too short for comprehensive analysis: {text_length} characters",
                f"For best results, provide at least {self.recommended_article_length} characters"
            )

        # Content quality checks
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count < 2:
            result.add_warning(
                "Article appears to contain very few sentences",
                "Ensure the text contains complete sentences for better analysis"
            )

        # Check for excessive formatting issues
        newline_ratio = text.count('\n') / text_length if text_length > 0 else 0
        if newline_ratio > 0.1:
            result.add_warning(
                "Article contains excessive line breaks which may affect analysis quality",
                "Consider cleaning up formatting before processing"
            )

        # Check for potential encoding issues
        if '\x00' in text or any(ord(char) > 127 and ord(char) < 160 for char in text[:100]):
            result.add_warning(
                "Article may contain encoding issues or special characters",
                "Ensure text is properly encoded in UTF-8"
            )

        self.logger.info(f"Article text validation completed", 
                        extra={
                            'session_id': session_id,
                            'text_length': text_length,
                            'is_valid': result.is_valid,
                            'error_count': len(result.errors),
                            'warning_count': len(result.warnings)
                        })

        return result

    def validate_extracted_claims(self, claims: Any, session_id: str = None) -> ValidationResult:
        """
        Validate extracted claims input with detailed feedback.

        Args:
            claims: Claims list to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and detailed feedback
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Type validation
        if not isinstance(claims, list):
            result.add_error(
                f"Claims must be a list, received {type(claims).__name__}",
                "Ensure claims are provided as a list of dictionaries"
            )
            self.logger.warning(f"Invalid claims type: {type(claims).__name__}", 
                              extra={'session_id': session_id})
            return result

        claims_count = len(claims)

        # Count validation
        if claims_count < self.min_claims_count:
            result.add_error(
                f"Too few claims provided: {claims_count} (minimum: {self.min_claims_count})",
                f"Provide at least {self.min_claims_count} claim for analysis"
            )

        if claims_count > self.max_claims_count:
            result.add_warning(
                f"Many claims provided: {claims_count} (maximum recommended: {self.max_claims_count})",
                f"Consider focusing on the top {self.max_claims_count} most important claims"
            )

        if claims_count < self.recommended_claims_count and claims_count >= self.min_claims_count:
            result.add_warning(
                f"Few claims provided: {claims_count} (recommended: {self.recommended_claims_count}+)",
                f"For comprehensive analysis, provide at least {self.recommended_claims_count} claims"
            )

        # Individual claim validation
        valid_claims = 0
        for i, claim in enumerate(claims):
            claim_errors = self._validate_single_claim(claim, i, session_id)
            result.errors.extend(claim_errors)
            if not claim_errors:
                valid_claims += 1

        # Claims quality assessment
        if claims_count > 0:
            valid_ratio = valid_claims / claims_count
            if valid_ratio < 0.5:
                result.add_error(
                    f"More than half of the claims are invalid ({valid_claims}/{claims_count})",
                    "Fix claim formatting issues before processing"
                )
            elif valid_ratio < 0.8:
                result.add_warning(
                    f"Some claims have formatting issues ({valid_claims}/{claims_count} valid)",
                    "Review and fix claim formatting for better results"
                )

        self.logger.info(f"Claims validation completed", 
                        extra={
                            'session_id': session_id,
                            'claims_count': claims_count,
                            'valid_claims': valid_claims,
                            'is_valid': result.is_valid,
                            'error_count': len(result.errors)
                        })

        return result

    def _validate_single_claim(self, claim: Any, index: int, session_id: str = None) -> List[str]:
        """Validate a single claim object with detailed error messages."""
        errors = []

        # Type validation
        if not isinstance(claim, dict):
            errors.append(
                f"Claim {index + 1} must be a dictionary, received {type(claim).__name__}. "
                f"Expected format: {{'text': 'claim text', 'verifiability_score': 5}}"
            )
            return errors

        # Required fields validation
        if 'text' not in claim:
            errors.append(
                f"Claim {index + 1} missing required 'text' field. "
                f"Add: 'text': 'your claim text'"
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
            elif len(claim_text.strip()) < 10:
                errors.append(
                    f"Claim {index + 1} 'text' too short: '{claim_text.strip()}'. "
                    f"Provide more descriptive claim text (at least 10 characters)"
                )

        # Optional field validation with helpful messages
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

        return errors

    def validate_context_analysis(self, context: Any, session_id: str = None) -> ValidationResult:
        """
        Validate context analysis input with helpful feedback.

        Args:
            context: Context analysis to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and suggestions
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Type validation
        if not isinstance(context, dict):
            result.add_error(
                f"Context analysis must be a dictionary, received {type(context).__name__}",
                "Provide context as: {'overall_context_score': 7.5, 'risk_level': 'MEDIUM'}"
            )
            return result

        # Score validation
        if 'overall_context_score' in context:
            score = context['overall_context_score']
            if not isinstance(score, (int, float)):
                result.add_error(
                    f"Context score must be a number, received {type(score).__name__}",
                    f"Use a decimal number: 'overall_context_score': 7.5"
                )
            elif not (0 <= score <= 10):
                result.add_error(
                    f"Context score must be between 0-10, received {score}",
                    f"Adjust to valid range: 'overall_context_score': {max(0, min(10, score))}"
                )

        # Risk level validation
        if 'risk_level' in context:
            risk_level = context['risk_level']
            valid_levels = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
            if risk_level not in valid_levels:
                result.add_error(
                    f"Risk level must be one of {valid_levels}, received '{risk_level}'",
                    f"Use valid risk level: 'risk_level': 'MEDIUM'"
                )

        # Helpful suggestions
        if 'overall_context_score' not in context:
            result.add_warning(
                "Context analysis missing 'overall_context_score'",
                "Add context score for better analysis depth determination"
            )

        self.logger.info(f"Context analysis validation completed", 
                        extra={
                            'session_id': session_id,
                            'is_valid': result.is_valid,
                            'has_score': 'overall_context_score' in context,
                            'has_risk_level': 'risk_level' in context
                        })

        return result

    def validate_processing_input(self, input_data: Dict[str, Any], session_id: str = None) -> ValidationResult:
        """
        Validate complete input data for processing with comprehensive feedback.

        Args:
            input_data: Complete input data dictionary
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and actionable feedback
        """
        all_errors = []
        all_warnings = []
        all_suggestions = []

        self.logger.info(f"Starting input validation", 
                        extra={
                            'session_id': session_id,
                            'input_keys': list(input_data.keys()) if isinstance(input_data, dict) else None
                        })

        # Basic structure validation
        if not isinstance(input_data, dict):
            return ValidationResult(
                is_valid=False,
                errors=[f"Input must be a dictionary, received {type(input_data).__name__}"],
                suggestions=["Provide input as: {'text': 'article text', 'extracted_claims': [...]}"]
            )

        # Required fields validation
        if 'text' not in input_data:
            all_errors.append("Missing required 'text' field")
            all_suggestions.append("Add article text: 'text': 'your article content here'")
        else:
            text_result = self.validate_article_text(input_data['text'], session_id)
            all_errors.extend(text_result.errors)
            all_warnings.extend(text_result.warnings)
            all_suggestions.extend(text_result.suggestions)

        # Optional fields validation
        if 'extracted_claims' in input_data:
            claims_result = self.validate_extracted_claims(input_data['extracted_claims'], session_id)
            all_errors.extend(claims_result.errors)
            all_warnings.extend(claims_result.warnings)
            all_suggestions.extend(claims_result.suggestions)
        else:
            all_warnings.append("No extracted claims provided - analysis will be limited")
            all_suggestions.append("Add claims for better analysis: 'extracted_claims': [{'text': 'claim', 'verifiability_score': 7}]")

        if 'context_analysis' in input_data:
            context_result = self.validate_context_analysis(input_data['context_analysis'], session_id)
            all_errors.extend(context_result.errors)
            all_warnings.extend(context_result.warnings)
            all_suggestions.extend(context_result.suggestions)

        # Additional validation
        if 'include_detailed_analysis' in input_data:
            detailed = input_data['include_detailed_analysis']
            if not isinstance(detailed, bool):
                all_errors.append(f"'include_detailed_analysis' must be boolean, received {type(detailed).__name__}")
                all_suggestions.append("Use: 'include_detailed_analysis': true or false")

        result = ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            suggestions=all_suggestions
        )

        self.logger.info(f"Input validation completed", 
                        extra={
                            'session_id': session_id,
                            'is_valid': result.is_valid,
                            'error_count': len(all_errors),
                            'warning_count': len(all_warnings),
                            'suggestion_count': len(all_suggestions)
                        })

        return result


class URLValidator:
    """
    Production-ready URL validation for verification sources.
    
    Ensures URLs are specific, accessible, and point to credible sources
    with clear feedback for debugging and improvement.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize URL validator with production settings.

        Args:
            config: Optional configuration for URL validation rules
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.URLValidator")
        
        self.trusted_domains = self.config.get('trusted_domains', [
            'gov', 'edu', 'who.int', 'cdc.gov', 'nih.gov', 'pubmed.ncbi.nlm.nih.gov',
            'nature.com', 'science.org', 'reuters.com', 'apnews.com', 'bbc.com',
            'arxiv.org', 'jstor.org', 'springer.com', 'wiley.com'
        ])
        self.min_path_segments = self.config.get('min_path_segments', 2)

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
                f"Use HTTPS version: https://{parsed.netloc}{parsed.path}"
            )

        self.logger.debug(f"URL format validation completed", 
                         extra={
                             'session_id': session_id,
                             'url': url[:100],  # Truncate for logging
                             'is_valid': result.is_valid,
                             'scheme': parsed.scheme if 'parsed' in locals() else None
                         })

        return result

    def validate_url_specificity(self, url: str, session_id: str = None) -> ValidationResult:
        """
        Validate that URL is specific and not a generic homepage.

        Args:
            url: URL to validate for specificity
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with specificity assessment and suggestions
        """
        # First validate format
        format_result = self.validate_url_format(url, session_id)
        if not format_result.is_valid:
            return format_result

        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        try:
            parsed = urlparse(url.strip())
        except:
            result.add_error(
                "Unable to parse URL for specificity check",
                "Ensure URL is properly formatted"
            )
            return result

        # Generic patterns to avoid
        generic_patterns = [
            (r'^/$', "Root path only"),
            (r'^/index\.html?$', "Index page"),
            (r'^/home/?$', "Home page"),
            (r'^/main/?$', "Main page"),
            (r'^/default\.html?$', "Default page"),
            (r'^/about/?$', "About page"),
            (r'^/news/?$', "General news section")
        ]

        path = parsed.path.lower()
        for pattern, description in generic_patterns:
            if re.match(pattern, path):
                result.add_error(
                    f"URL appears to be generic homepage: {description}",
                    f"Provide specific page URL, e.g., {url}/specific-article-title"
                )
                break

        # Path specificity checks
        if not path or path == '/':
            result.add_error(
                "URL lacks specific path - appears to be homepage",
                f"Add specific page path: {url}/article-title or {url}/research/study-name"
            )

        path_segments = [seg for seg in path.split('/') if seg]
        if len(path_segments) < self.min_path_segments:
            result.add_warning(
                f"URL path may not be specific enough: {len(path_segments)} segments",
                f"Consider more specific URL with at least {self.min_path_segments} path segments"
            )

        # Positive indicators of specificity
        if parsed.query:
            result.add_warning(
                "URL contains query parameters - good specificity indicator",
                None  # This is positive feedback
            )

        if parsed.fragment:
            result.add_warning(
                "URL contains fragment identifier - indicates specific section",
                None  # This is positive feedback
            )

        # Check for specific content indicators
        specific_indicators = ['article', 'study', 'research', 'report', 'publication', 'paper']
        if any(indicator in path.lower() for indicator in specific_indicators):
            result.add_warning(
                "URL appears to point to specific content - good",
                None  # This is positive feedback
            )

        self.logger.debug(f"URL specificity validation completed", 
                         extra={
                             'session_id': session_id,
                             'url': url[:100],
                             'path_segments': len(path_segments),
                             'is_specific': result.is_valid
                         })

        return result

    def validate_domain_credibility(self, url: str, session_id: str = None) -> ValidationResult:
        """
        Validate domain credibility for verification sources.

        Args:
            url: URL to validate for domain credibility
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with credibility assessment and suggestions
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        try:
            parsed = urlparse(url.strip())
            domain = parsed.netloc.lower()
        except:
            result.add_error(
                "Unable to parse URL for domain validation",
                "Ensure URL is properly formatted"
            )
            return result

        # Check against trusted domains
        is_trusted = any(trusted in domain for trusted in self.trusted_domains)
        if is_trusted:
            result.add_warning(
                f"Domain recognized as trusted source: {domain}",
                None  # This is positive feedback
            )

        # Domain quality indicators
        high_quality_indicators = ['.gov', '.edu', '.org']
        has_quality_indicator = any(indicator in domain for indicator in high_quality_indicators)
        
        if has_quality_indicator:
            result.add_warning(
                f"Domain has quality indicator: {domain}",
                None  # This is positive feedback
            )

        # Suspicious domain patterns
        suspicious_patterns = [
            (r'\.tk$', "Free .tk domain"),
            (r'\.ml$', "Free .ml domain"),
            (r'\.ga$', "Free .ga domain"),
            (r'\.cf$', "Free .cf domain"),
            (r'\d+\.\d+\.\d+\.\d+', "IP address instead of domain"),
            (r'[^a-zA-Z0-9\-\.]', "Special characters in domain")
        ]

        for pattern, description in suspicious_patterns:
            if re.search(pattern, domain):
                result.add_error(
                    f"Suspicious domain detected: {description}",
                    f"Use established domain instead of {domain}"
                )
                break

        # Domain structure validation
        if domain.count('.') < 1:
            result.add_error(
                f"Invalid domain structure: {domain}",
                "Ensure domain has proper format: subdomain.domain.com"
            )

        # Credibility suggestions
        if not is_trusted and not has_quality_indicator:
            result.add_warning(
                f"Domain credibility uncertain: {domain}",
                "Consider using established sources like .gov, .edu, or recognized institutions"
            )

        self.logger.debug(f"Domain credibility validation completed", 
                         extra={
                             'session_id': session_id,
                             'domain': domain,
                             'is_trusted': is_trusted,
                             'has_quality_indicator': has_quality_indicator
                         })

        return result


class OutputValidator:
    """
    Production-ready output validation for evidence evaluator results.
    
    Validates verification sources, evidence scores, and analysis outputs
    with clear feedback for quality assurance and debugging.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize output validator with production settings.

        Args:
            config: Optional configuration for output validation rules
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.OutputValidator")
        self.url_validator = URLValidator(config)

    def validate_verification_sources(self, sources: Any, session_id: str = None) -> ValidationResult:
        """
        Validate verification sources output with detailed feedback.

        Args:
            sources: Verification sources to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and quality feedback
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Type validation
        if not isinstance(sources, list):
            result.add_error(
                f"Verification sources must be a list, received {type(sources).__name__}",
                "Ensure sources are returned as a list of dictionaries"
            )
            return result

        if len(sources) == 0:
            result.add_warning(
                "No verification sources provided",
                "Generate at least 1-3 verification sources for better credibility"
            )
            return result

        # Validate each source
        high_quality_count = 0
        for i, source in enumerate(sources):
            source_errors, quality_score = self._validate_single_source(source, i, session_id)
            result.errors.extend(source_errors)
            
            if quality_score >= 0.8:
                high_quality_count += 1

        # Overall quality assessment
        if high_quality_count == 0:
            result.add_warning(
                "No high-quality verification sources found",
                "Include sources from trusted domains (.gov, .edu, established institutions)"
            )
        elif high_quality_count >= len(sources) * 0.6:
            result.add_warning(
                f"Good quality sources: {high_quality_count}/{len(sources)}",
                None  # This is positive feedback
            )

        # Diversity check
        domains = []
        for source in sources:
            if isinstance(source, dict) and 'url' in source:
                try:
                    domain = urlparse(source['url']).netloc
                    domains.append(domain)
                except:
                    continue
        
        unique_domains = len(set(domains))
        if unique_domains < len(sources) * 0.7:
            result.add_warning(
                f"Limited source diversity: {unique_domains} unique domains for {len(sources)} sources",
                "Include sources from different institutions for better verification"
            )

        self.logger.info(f"Verification sources validation completed", 
                        extra={
                            'session_id': session_id,
                            'sources_count': len(sources),
                            'high_quality_count': high_quality_count,
                            'unique_domains': unique_domains,
                            'is_valid': result.is_valid
                        })

        return result

    def _validate_single_source(self, source: Any, index: int, session_id: str = None) -> Tuple[List[str], float]:
        """Validate a single verification source with detailed feedback."""
        errors = []
        quality_score = 0.0

        # Type validation
        if not isinstance(source, dict):
            errors.append(
                f"Verification source {index + 1} must be a dictionary, received {type(source).__name__}. "
                f"Expected: {{'claim': '...', 'url': '...', 'institution': '...'}}"
            )
            return errors, quality_score

        # Required fields validation
        required_fields = ['claim', 'url', 'institution']
        for field in required_fields:
            if field not in source:
                errors.append(
                    f"Verification source {index + 1} missing required field: '{field}'. "
                    f"Add: '{field}': 'appropriate value'"
                )
            elif not isinstance(source[field], str) or not source[field].strip():
                errors.append(
                    f"Verification source {index + 1} field '{field}' must be a non-empty string. "
                    f"Current value: {source.get(field, 'None')}"
                )

        # URL validation if present
        if 'url' in source and source['url']:
            url_result = self.url_validator.validate_url_specificity(source['url'], session_id)
            if not url_result.is_valid:
                for error in url_result.errors:
                    errors.append(f"Source {index + 1} URL issue: {error}")
            
            # Domain credibility check
            domain_result = self.url_validator.validate_domain_credibility(source['url'], session_id)
            if not domain_result.is_valid:
                for error in domain_result.errors:
                    errors.append(f"Source {index + 1} domain issue: {error}")
            
            # Calculate quality score based on validation results
            if url_result.is_valid and domain_result.is_valid:
                quality_score = 0.8
                # Bonus for trusted domains
                if any("trusted" in warning.lower() for warning in domain_result.warnings):
                    quality_score = 0.9

        # Optional field validation
        if 'quality_score' in source:
            score = source['quality_score']
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                errors.append(
                    f"Source {index + 1} quality_score must be a number between 0-1, received {score}. "
                    f"Adjust to: 'quality_score': {max(0, min(1, float(score) if isinstance(score, (int, float)) else 0.5))}"
                )
            else:
                quality_score = max(quality_score, score)

        if 'confidence' in source:
            confidence = source['confidence']
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                errors.append(
                    f"Source {index + 1} confidence must be a number between 0-1, received {confidence}. "
                    f"Adjust to: 'confidence': {max(0, min(1, float(confidence) if isinstance(confidence, (int, float)) else 0.5))}"
                )

        return errors, quality_score

    def validate_evidence_scores(self, scores: Any, session_id: str = None) -> ValidationResult:
        """
        Validate evidence scores output with range and consistency checks.

        Args:
            scores: Evidence scores to validate
            session_id: Optional session ID for tracking

        Returns:
            ValidationResult with validation status and suggestions
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Type validation
        if not isinstance(scores, dict):
            result.add_error(
                f"Evidence scores must be a dictionary, received {type(scores).__name__}",
                "Ensure scores are returned as: {'overall_evidence_score': 7.5, ...}"
            )
            return result

        # Required score fields validation
        required_scores = [
            'overall_evidence_score',
            'source_quality_score', 
            'logical_consistency_score'
        ]

        for score_field in required_scores:
            if score_field not in scores:
                result.add_error(
                    f"Missing required score field: '{score_field}'",
                    f"Add: '{score_field}': 7.5"
                )
            else:
                score = scores[score_field]
                if not isinstance(score, (int, float)):
                    result.add_error(
                        f"'{score_field}' must be a number, received {type(score).__name__}",
                        f"Use number: '{score_field}': 7.5"
                    )
                elif not (0 <= score <= 10):
                    result.add_error(
                        f"'{score_field}' must be between 0-10, received {score}",
                        f"Adjust to valid range: '{score_field}': {max(0, min(10, score))}"
                    )

        # Quality level validation
        if 'quality_level' in scores:
            quality_level = scores['quality_level']
            valid_levels = ['EXCELLENT', 'HIGH QUALITY', 'MODERATE QUALITY', 'LOW QUALITY', 'POOR QUALITY', 'VERY POOR QUALITY']
            if quality_level not in valid_levels:
                result.add_error(
                    f"Invalid quality level: '{quality_level}'",
                    f"Use one of: {valid_levels}"
                )

        # Consistency checks
        overall_score = scores.get('overall_evidence_score', 0)
        quality_level = scores.get('quality_level', '')
        
        if overall_score >= 8 and quality_level not in ['EXCELLENT', 'HIGH QUALITY']:
            result.add_warning(
                f"High overall score ({overall_score}) but quality level is '{quality_level}'",
                "Ensure score and quality level are consistent"
            )
        elif overall_score <= 3 and quality_level not in ['POOR QUALITY', 'VERY POOR QUALITY']:
            result.add_warning(
                f"Low overall score ({overall_score}) but quality level is '{quality_level}'",
                "Ensure score and quality level are consistent"
            )

        self.logger.info(f"Evidence scores validation completed", 
                        extra={
                            'session_id': session_id,
                            'overall_score': overall_score,
                            'quality_level': quality_level,
                            'is_valid': result.is_valid
                        })

        return result


# Enhanced convenience validation functions

def validate_evidence_input(input_data: Dict[str, Any], 
                          config: Dict[str, Any] = None,
                          session_id: str = None) -> ValidationResult:
    """
    Validate complete evidence evaluator input with comprehensive feedback.

    Args:
        input_data: Input data to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult with validation status, errors, warnings, and suggestions
    """
    validator = InputValidator(config)
    return validator.validate_processing_input(input_data, session_id)


def validate_url_specificity(url: str, 
                           config: Dict[str, Any] = None,
                           session_id: str = None) -> bool:
    """
    Quick validation for URL specificity with logging.

    Args:
        url: URL to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        True if URL is specific, False if generic
    """
    validator = URLValidator(config)
    result = validator.validate_url_specificity(url, session_id)
    return result.is_valid


def validate_verification_output(sources: List[Dict[str, Any]], 
                               config: Dict[str, Any] = None,
                               session_id: str = None) -> ValidationResult:
    """
    Validate verification sources output with quality assessment.

    Args:
        sources: Verification sources to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult with validation status and quality feedback
    """
    validator = OutputValidator(config)
    return validator.validate_verification_sources(sources, session_id)


# Testing functionality
if __name__ == "__main__":
    """Test validation functionality with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    test_session_id = "validation_test_123"
    
    # Test input validation
    test_input = {
        'text': 'This is a test article with sufficient length for analysis purposes. It contains multiple sentences and should pass validation.',
        'extracted_claims': [
            {'text': 'This is a test claim with good length', 'verifiability_score': 8, 'priority': 1},
            {'text': 'Another test claim', 'verifiability_score': 6, 'priority': 2}
        ],
        'context_analysis': {
            'overall_context_score': 7.5,
            'risk_level': 'MEDIUM'
        }
    }
    
    print("=== INPUT VALIDATION TEST ===")
    input_result = validate_evidence_input(test_input, session_id=test_session_id)
    print(f"Input validation: {'✅ Valid' if input_result.is_valid else '❌ Invalid'}")
    if input_result.errors:
        print(f"Errors: {input_result.errors}")
    if input_result.warnings:
        print(f"Warnings: {input_result.warnings}")
    if input_result.suggestions:
        print(f"Suggestions: {input_result.suggestions}")
    
    # Test URL validation
    test_urls = [
        'https://www.cdc.gov/',  # Generic (should fail specificity)
        'https://www.cdc.gov/vaccines/covid-19/clinical-considerations/managing-anaphylaxis.html',  # Specific (should pass)
        'https://pubmed.ncbi.nlm.nih.gov/34289274/',  # Specific (should pass)
        'invalid-url',  # Invalid format
        'https://suspicious-domain.tk/article'  # Suspicious domain
    ]
    
    print("\n=== URL VALIDATION TEST ===")
    url_validator = URLValidator()
    for url in test_urls:
        specificity_result = url_validator.validate_url_specificity(url, test_session_id)
        credibility_result = url_validator.validate_domain_credibility(url, test_session_id)
        
        print(f"\nURL: {url}")
        print(f"  Specificity: {'✅ Specific' if specificity_result.is_valid else '❌ Generic'}")
        print(f"  Credibility: {'✅ Credible' if credibility_result.is_valid else '⚠️ Questionable'}")
        
        if specificity_result.errors:
            print(f"  Specificity Issues: {specificity_result.errors[0]}")
        if credibility_result.errors:
            print(f"  Credibility Issues: {credibility_result.errors[0]}")
    
    # Test verification sources validation
    test_sources = [
        {
            'claim': 'COVID-19 vaccine effectiveness data',
            'url': 'https://www.cdc.gov/vaccines/covid-19/effectiveness/',
            'institution': 'CDC',
            'quality_score': 0.9,
            'confidence': 0.85
        },
        {
            'claim': 'Research study on treatment outcomes',
            'url': 'https://suspicious-domain.tk/study',
            'institution': 'Unknown Institute',
            'quality_score': 0.3,
            'confidence': 0.4
        }
    ]
    
    print("\n=== VERIFICATION SOURCES VALIDATION TEST ===")
    sources_result = validate_verification_output(test_sources, session_id=test_session_id)
    print(f"Sources validation: {'✅ Valid' if sources_result.is_valid else '❌ Issues Found'}")
    if sources_result.errors:
        print(f"Errors: {sources_result.errors}")
    if sources_result.warnings:
        print(f"Warnings: {sources_result.warnings}")
    if sources_result.suggestions:
        print(f"Suggestions: {sources_result.suggestions}")
    
    print("\n✅ Validation tests completed")
