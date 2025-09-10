# agents/evidence_evaluator/validators.py

"""
Evidence Evaluator Input/Output Validation

Validation utilities for the Evidence Evaluator Agent providing comprehensive
input validation, output validation, and URL specificity checking.
"""

import re
import json
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
    """Container for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class InputValidator:
    """
    Input validation for evidence evaluator processing.
    
    Validates article text, extracted claims, context analysis,
    and other input parameters before processing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize input validator.
        
        Args:
            config: Optional configuration for validation rules
        """
        self.config = config or {}
        self.min_article_length = self.config.get('min_article_length', 50)
        self.max_article_length = self.config.get('max_article_length', 50000)
        self.min_claims_count = self.config.get('min_claims_count', 0)
        self.max_claims_count = self.config.get('max_claims_count', 20)

    def validate_article_text(self, text: Any) -> ValidationResult:
        """
        Validate article text input.
        
        Args:
            text: Article text to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(text, str):
            errors.append(f"Article text must be string, got {type(text).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Content validation
        if not text.strip():
            errors.append("Article text cannot be empty or whitespace only")
        
        # Length validation
        text_length = len(text.strip())
        if text_length < self.min_article_length:
            errors.append(f"Article text too short: {text_length} chars (minimum: {self.min_article_length})")
        
        if text_length > self.max_article_length:
            warnings.append(f"Article text very long: {text_length} chars (maximum recommended: {self.max_article_length})")
        
        # Content quality checks
        if text_length < 200:
            warnings.append("Article text may be too short for meaningful evidence analysis")
        
        # Check for suspicious patterns
        if text.count('\n') > text_length * 0.1:
            warnings.append("Article contains excessive line breaks - may affect analysis quality")
        
        if len(re.findall(r'[.!?]', text)) < text_length / 200:
            warnings.append("Article may lack proper sentence structure")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_extracted_claims(self, claims: Any) -> ValidationResult:
        """
        Validate extracted claims input.
        
        Args:
            claims: Claims list to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(claims, list):
            errors.append(f"Claims must be a list, got {type(claims).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Count validation
        claims_count = len(claims)
        if claims_count < self.min_claims_count:
            warnings.append(f"Very few claims provided: {claims_count} (minimum recommended: {self.min_claims_count})")
        
        if claims_count > self.max_claims_count:
            warnings.append(f"Many claims provided: {claims_count} (maximum recommended: {self.max_claims_count})")
        
        # Individual claim validation
        for i, claim in enumerate(claims):
            claim_errors = self._validate_single_claim(claim, i)
            errors.extend(claim_errors)
        
        # Claims quality checks
        if claims_count == 0:
            warnings.append("No claims provided - evidence evaluation will be limited")
        
        text_claims = [c for c in claims if isinstance(c, dict) and c.get('text', '').strip()]
        if len(text_claims) < claims_count * 0.8:
            warnings.append("Many claims lack text content - may reduce analysis quality")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def _validate_single_claim(self, claim: Any, index: int) -> List[str]:
        """Validate a single claim object."""
        errors = []
        
        # Type validation
        if not isinstance(claim, dict):
            errors.append(f"Claim {index} must be a dictionary, got {type(claim).__name__}")
            return errors
        
        # Required fields
        if 'text' not in claim:
            errors.append(f"Claim {index} missing required 'text' field")
        else:
            claim_text = claim['text']
            if not isinstance(claim_text, str):
                errors.append(f"Claim {index} 'text' must be string, got {type(claim_text).__name__}")
            elif not claim_text.strip():
                errors.append(f"Claim {index} 'text' cannot be empty")
            elif len(claim_text.strip()) < 10:
                errors.append(f"Claim {index} 'text' too short: '{claim_text.strip()}'")
        
        # Optional field validation
        if 'verifiability_score' in claim:
            score = claim['verifiability_score']
            if not isinstance(score, (int, float)) or score < 0 or score > 10:
                errors.append(f"Claim {index} 'verifiability_score' must be number between 0-10, got {score}")
        
        if 'priority' in claim:
            priority = claim['priority']
            if not isinstance(priority, int) or priority < 1 or priority > 5:
                errors.append(f"Claim {index} 'priority' must be integer between 1-5, got {priority}")
        
        return errors

    def validate_context_analysis(self, context: Any) -> ValidationResult:
        """
        Validate context analysis input.
        
        Args:
            context: Context analysis to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(context, dict):
            errors.append(f"Context analysis must be a dictionary, got {type(context).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Score validation
        if 'overall_context_score' in context:
            score = context['overall_context_score']
            if not isinstance(score, (int, float)) or score < 0 or score > 10:
                errors.append(f"Context score must be number between 0-10, got {score}")
        
        # Risk level validation
        if 'risk_level' in context:
            risk_level = context['risk_level']
            valid_levels = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
            if risk_level not in valid_levels:
                errors.append(f"Risk level must be one of {valid_levels}, got '{risk_level}'")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_processing_input(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete input data for processing.
        
        Args:
            input_data: Complete input data dictionary
            
        Returns:
            ValidationResult with validation status and any errors
        """
        all_errors = []
        all_warnings = []
        
        # Required fields
        if 'text' not in input_data:
            all_errors.append("Missing required 'text' field")
        else:
            text_result = self.validate_article_text(input_data['text'])
            all_errors.extend(text_result.errors)
            all_warnings.extend(text_result.warnings)
        
        # Optional fields
        if 'extracted_claims' in input_data:
            claims_result = self.validate_extracted_claims(input_data['extracted_claims'])
            all_errors.extend(claims_result.errors)
            all_warnings.extend(claims_result.warnings)
        
        if 'context_analysis' in input_data:
            context_result = self.validate_context_analysis(input_data['context_analysis'])
            all_errors.extend(context_result.errors)
            all_warnings.extend(context_result.warnings)
        
        # Additional validation
        if 'include_detailed_analysis' in input_data:
            detailed = input_data['include_detailed_analysis']
            if not isinstance(detailed, bool):
                all_errors.append(f"'include_detailed_analysis' must be boolean, got {type(detailed).__name__}")
        
        return ValidationResult(len(all_errors) == 0, all_errors, all_warnings)


class URLValidator:
    """
    URL validation for verification sources.
    
    Ensures URLs are specific, accessible, and point to credible sources
    rather than generic homepages.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize URL validator.
        
        Args:
            config: Optional configuration for URL validation rules
        """
        self.config = config or {}
        self.trusted_domains = self.config.get('trusted_domains', [
            'gov', 'edu', 'who.int', 'cdc.gov', 'nih.gov', 'pubmed.ncbi.nlm.nih.gov',
            'nature.com', 'science.org', 'reuters.com', 'apnews.com', 'bbc.com'
        ])
        self.min_path_segments = self.config.get('min_path_segments', 2)

    def validate_url_format(self, url: str) -> ValidationResult:
        """
        Validate URL format and structure.
        
        Args:
            url: URL string to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Basic format validation
        if not isinstance(url, str):
            errors.append(f"URL must be string, got {type(url).__name__}")
            return ValidationResult(False, errors, warnings)
        
        if not url.strip():
            errors.append("URL cannot be empty")
            return ValidationResult(False, errors, warnings)
        
        # URL parsing validation
        try:
            parsed = urlparse(url.strip())
        except Exception as e:
            errors.append(f"Invalid URL format: {str(e)}")
            return ValidationResult(False, errors, warnings)
        
        # Scheme validation
        if not parsed.scheme:
            errors.append("URL missing protocol (http/https)")
        elif parsed.scheme not in ['http', 'https']:
            errors.append(f"Invalid URL protocol: {parsed.scheme} (must be http or https)")
        
        # Domain validation
        if not parsed.netloc:
            errors.append("URL missing domain")
        elif len(parsed.netloc) < 4:
            errors.append(f"Invalid domain: {parsed.netloc}")
        
        # HTTPS preference warning
        if parsed.scheme == 'http':
            warnings.append("HTTP URL detected - HTTPS preferred for security")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_url_specificity(self, url: str) -> ValidationResult:
        """
        Validate that URL is specific and not a generic homepage.
        
        Args:
            url: URL to validate for specificity
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # First validate format
        format_result = self.validate_url_format(url)
        if not format_result.is_valid:
            return format_result
        
        try:
            parsed = urlparse(url.strip())
        except:
            errors.append("Unable to parse URL for specificity check")
            return ValidationResult(False, errors, warnings)
        
        # Generic patterns to avoid
        generic_patterns = [
            r'^/$',  # Just root path
            r'^/index\.html?$',  # Index pages
            r'^/home/?$',  # Home pages
            r'^/main/?$',  # Main pages
            r'^/default\.html?$',  # Default pages
        ]
        
        path = parsed.path.lower()
        for pattern in generic_patterns:
            if re.match(pattern, path):
                errors.append(f"URL appears to be generic homepage: {url}")
                break
        
        # Path specificity checks
        if not path or path == '/':
            errors.append("URL lacks specific path - appears to be homepage")
        
        path_segments = [seg for seg in path.split('/') if seg]
        if len(path_segments) < self.min_path_segments:
            warnings.append(f"URL path may not be specific enough: {len(path_segments)} segments")
        
        # Query parameters can indicate specificity
        if parsed.query:
            warnings.clear()  # Query params suggest specific content
        
        # Fragment identifiers can indicate specificity
        if parsed.fragment:
            warnings.clear()  # Fragment suggests specific section
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_domain_credibility(self, url: str) -> ValidationResult:
        """
        Validate domain credibility for verification sources.
        
        Args:
            url: URL to validate for domain credibility
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        try:
            parsed = urlparse(url.strip())
            domain = parsed.netloc.lower()
        except:
            errors.append("Unable to parse URL for domain validation")
            return ValidationResult(False, errors, warnings)
        
        # Check against trusted domains
        is_trusted = any(trusted in domain for trusted in self.trusted_domains)
        
        # Domain quality indicators
        high_quality_indicators = ['.gov', '.edu', '.org']
        has_quality_indicator = any(indicator in domain for indicator in high_quality_indicators)
        
        # Suspicious domain patterns
        suspicious_patterns = [
            r'\.tk$', r'\.ml$', r'\.ga$', r'\.cf$',  # Free domains
            r'\d+\.\d+\.\d+\.\d+',  # IP addresses
            r'[^a-zA-Z0-9\-\.]',  # Special characters in domain
        ]
        
        is_suspicious = any(re.search(pattern, domain) for pattern in suspicious_patterns)
        
        # Validation logic
        if is_suspicious:
            errors.append(f"Suspicious domain detected: {domain}")
        
        if not is_trusted and not has_quality_indicator:
            warnings.append(f"Domain credibility uncertain: {domain}")
        
        if domain.count('.') < 1:
            errors.append(f"Invalid domain structure: {domain}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class OutputValidator:
    """
    Output validation for evidence evaluator results.
    
    Validates verification sources, evidence scores, and analysis outputs
    to ensure they meet quality and format requirements.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize output validator.
        
        Args:
            config: Optional configuration for output validation rules
        """
        self.config = config or {}
        self.url_validator = URLValidator(config)

    def validate_verification_sources(self, sources: Any) -> ValidationResult:
        """
        Validate verification sources output.
        
        Args:
            sources: Verification sources to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(sources, list):
            errors.append(f"Verification sources must be list, got {type(sources).__name__}")
            return ValidationResult(False, errors, warnings)
        
        if len(sources) == 0:
            warnings.append("No verification sources provided")
            return ValidationResult(True, errors, warnings)
        
        # Validate each source
        for i, source in enumerate(sources):
            source_errors, source_warnings = self._validate_single_source(source, i)
            errors.extend(source_errors)
            warnings.extend(source_warnings)
        
        # Overall quality checks
        high_quality_count = sum(1 for s in sources if isinstance(s, dict) and s.get('quality_score', 0) >= 0.8)
        if high_quality_count == 0:
            warnings.append("No high-quality verification sources found")
        
        return ValidationResult(len(errors) == 0, errors, warnings)

    def _validate_single_source(self, source: Any, index: int) -> Tuple[List[str], List[str]]:
        """Validate a single verification source."""
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(source, dict):
            errors.append(f"Verification source {index} must be dictionary, got {type(source).__name__}")
            return errors, warnings
        
        # Required fields
        required_fields = ['claim', 'url', 'institution']
        for field in required_fields:
            if field not in source:
                errors.append(f"Verification source {index} missing required field: {field}")
            elif not isinstance(source[field], str) or not source[field].strip():
                errors.append(f"Verification source {index} field '{field}' must be non-empty string")
        
        # URL validation
        if 'url' in source:
            url_result = self.url_validator.validate_url_specificity(source['url'])
            if not url_result.is_valid:
                errors.extend([f"Source {index} URL: {error}" for error in url_result.errors])
            warnings.extend([f"Source {index} URL: {warning}" for warning in url_result.warnings])
            
            # Domain credibility check
            domain_result = self.url_validator.validate_domain_credibility(source['url'])
            if not domain_result.is_valid:
                errors.extend([f"Source {index} domain: {error}" for error in domain_result.errors])
            warnings.extend([f"Source {index} domain: {warning}" for warning in domain_result.warnings])
        
        # Optional field validation
        if 'quality_score' in source:
            score = source['quality_score']
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                errors.append(f"Source {index} quality_score must be number between 0-1, got {score}")
        
        if 'confidence' in source:
            confidence = source['confidence']
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                errors.append(f"Source {index} confidence must be number between 0-1, got {confidence}")
        
        return errors, warnings

    def validate_evidence_scores(self, scores: Any) -> ValidationResult:
        """
        Validate evidence scores output.
        
        Args:
            scores: Evidence scores to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(scores, dict):
            errors.append(f"Evidence scores must be dictionary, got {type(scores).__name__}")
            return ValidationResult(False, errors, warnings)
        
        # Required score fields
        required_scores = [
            'overall_evidence_score',
            'source_quality_score',
            'logical_consistency_score'
        ]
        
        for score_field in required_scores:
            if score_field not in scores:
                errors.append(f"Missing required score field: {score_field}")
            else:
                score = scores[score_field]
                if not isinstance(score, (int, float)) or score < 0 or score > 10:
                    errors.append(f"{score_field} must be number between 0-10, got {score}")
        
        # Quality level validation
        if 'quality_level' in scores:
            quality_level = scores['quality_level']
            valid_levels = ['EXCELLENT', 'HIGH QUALITY', 'MODERATE QUALITY', 'LOW QUALITY', 'POOR QUALITY', 'VERY POOR QUALITY']
            if quality_level not in valid_levels:
                errors.append(f"Invalid quality level: {quality_level}")
        
        # Consistency checks
        overall_score = scores.get('overall_evidence_score', 0)
        if overall_score >= 8 and scores.get('quality_level') not in ['EXCELLENT', 'HIGH QUALITY']:
            warnings.append("High overall score but quality level doesn't match")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class ConfigValidator:
    """
    Configuration validation for evidence evaluator settings.
    
    Validates agent configuration, API keys, and processing parameters.
    """
    
    @staticmethod
    def validate_agent_config(config: Dict[str, Any]) -> ValidationResult:
        """
        Validate evidence evaluator agent configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []
        
        # Model configuration
        if 'model_name' in config:
            model_name = config['model_name']
            if not isinstance(model_name, str) or not model_name.strip():
                errors.append("model_name must be non-empty string")
        
        if 'temperature' in config:
            temp = config['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append("temperature must be number between 0-2")
        
        if 'max_tokens' in config:
            tokens = config['max_tokens']
            if not isinstance(tokens, int) or tokens < 100 or tokens > 10000:
                errors.append("max_tokens must be integer between 100-10000")
        
        # Scoring weights validation
        if 'scoring_weights' in config:
            weights = config['scoring_weights']
            if isinstance(weights, dict):
                total_weight = sum(weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    warnings.append(f"Scoring weights sum to {total_weight:.3f}, should sum to 1.0")
            else:
                errors.append("scoring_weights must be dictionary")
        
        # Threshold validation
        threshold_fields = ['evidence_threshold', 'high_quality_threshold', 'medium_quality_threshold']
        for field in threshold_fields:
            if field in config:
                threshold = config[field]
                if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 10:
                    errors.append(f"{field} must be number between 0-10")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


# Convenience validation functions
def validate_evidence_input(input_data: Dict[str, Any], config: Dict[str, Any] = None) -> ValidationResult:
    """
    Validate complete evidence evaluator input.
    
    Args:
        input_data: Input data to validate
        config: Optional validation configuration
        
    Returns:
        ValidationResult with validation status and any errors
    """
    validator = InputValidator(config)
    return validator.validate_processing_input(input_data)


def validate_url_specificity(url: str, config: Dict[str, Any] = None) -> bool:
    """
    Quick validation for URL specificity.
    
    Args:
        url: URL to validate
        config: Optional validation configuration
        
    Returns:
        True if URL is specific, False if generic
    """
    validator = URLValidator(config)
    result = validator.validate_url_specificity(url)
    return result.is_valid


def validate_verification_output(sources: List[Dict[str, Any]], config: Dict[str, Any] = None) -> ValidationResult:
    """
    Validate verification sources output.
    
    Args:
        sources: Verification sources to validate
        config: Optional validation configuration
        
    Returns:
        ValidationResult with validation status and any errors
    """
    validator = OutputValidator(config)
    return validator.validate_verification_sources(sources)


# Testing functionality
if __name__ == "__main__":
    """Test validation functionality."""
    
    # Test input validation
    test_input = {
        'text': 'This is a test article with sufficient length for analysis purposes.',
        'extracted_claims': [
            {'text': 'This is a test claim', 'verifiability_score': 8, 'priority': 1},
            {'text': 'Another test claim', 'verifiability_score': 6, 'priority': 2}
        ],
        'context_analysis': {
            'overall_context_score': 7.5,
            'risk_level': 'MEDIUM'
        }
    }
    
    input_result = validate_evidence_input(test_input)
    print(f"Input validation: {'✓' if input_result.is_valid else '✗'}")
    if input_result.errors:
        print(f"Errors: {input_result.errors}")
    if input_result.warnings:
        print(f"Warnings: {input_result.warnings}")
    
    # Test URL validation
    test_urls = [
        'https://www.cdc.gov/',  # Generic (should fail)
        'https://www.cdc.gov/vaccines/covid-19/clinical-considerations/managing-anaphylaxis.html',  # Specific (should pass)
        'https://pubmed.ncbi.nlm.nih.gov/34289274/',  # Specific (should pass)
    ]
    
    print("\nURL Specificity Tests:")
    for url in test_urls:
        is_specific = validate_url_specificity(url)
        print(f"  {url}: {'✓ Specific' if is_specific else '✗ Generic'}")
    
    # Test verification sources validation
    test_sources = [
        {
            'claim': 'COVID-19 vaccine effectiveness',
            'url': 'https://www.cdc.gov/vaccines/covid-19/effectiveness/',
            'institution': 'CDC',
            'quality_score': 0.9,
            'confidence': 0.85
        }
    ]
    
    sources_result = validate_verification_output(test_sources)
    print(f"\nVerification sources validation: {'✓' if sources_result.is_valid else '✗'}")
    if sources_result.errors:
        print(f"Errors: {sources_result.errors}")
    if sources_result.warnings:
        print(f"Warnings: {sources_result.warnings}")
