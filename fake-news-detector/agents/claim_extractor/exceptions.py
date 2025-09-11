# agents/claim_extractor/exceptions.py

"""
Claim Extractor Agent Custom Exceptions - Production Ready

Custom exception classes for the claim extractor agent providing
specific error handling for claim extraction, parsing, verification,
and prioritization workflows with detailed error context, recovery
strategies, and comprehensive production monitoring capabilities.
"""

import time
import logging
import traceback
from typing import Any, Dict, List, Optional, Union


class ClaimExtractorError(Exception):
    """
    Base exception class for all Claim Extractor Agent errors.
    
    All custom exceptions in the claim extractor module should inherit
    from this base class to provide consistent error handling with
    enhanced metadata, session tracking, and recovery recommendations.
    """

    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None, 
                 session_id: str = None, recoverable: bool = None):
        """
        Initialize base claim extractor exception with enhanced context.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
            session_id: Optional session ID for tracking and debugging
            recoverable: Whether this error can be recovered from
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CLAIM_EXTRACTOR_ERROR"
        self.details = details or {}
        self.session_id = session_id
        self.timestamp = time.time()
        self.recoverable = recoverable if recoverable is not None else self._determine_recoverability()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'recoverable': self.recoverable,
            'traceback': traceback.format_exc() if hasattr(self, '__traceback__') else None
        }

    def _determine_recoverability(self) -> bool:
        """Determine if this error type is generally recoverable."""
        recoverable_codes = [
            'LLM_RESPONSE_ERROR', 'CLAIM_PARSING_ERROR', 'PATTERN_ANALYSIS_ERROR',
            'RATE_LIMIT_ERROR', 'PROCESSING_TIMEOUT_ERROR'
        ]
        return self.error_code in recoverable_codes

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendation for this error type."""
        return None  # Override in subclasses


class InputValidationError(ClaimExtractorError):
    """
    Exception raised when input data validation fails.
    
    Used for invalid article text, malformed BERT results, missing required fields,
    or security validation failures with detailed field-level error reporting.
    """

    def __init__(self, message: str, field_name: str = None, field_value: Any = None, 
                 validation_rules: List[str] = None, session_id: str = None):
        """
        Initialize input validation error with field-specific context.

        Args:
            message: Error description
            field_name: Name of the invalid field
            field_value: Value that caused the validation error
            validation_rules: List of validation rules that failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            # Truncate long values and sanitize sensitive data
            value_str = str(field_value)
            if 'password' in field_name.lower() if field_name else False or 'key' in field_name.lower() if field_name else False:
                details['field_value'] = '[REDACTED]'
            else:
                details['field_value'] = value_str[:200] + '...' if len(value_str) > 200 else value_str
        if validation_rules:
            details['failed_validation_rules'] = validation_rules

        super().__init__(message, "INPUT_VALIDATION_ERROR", details, session_id, False)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rules = validation_rules or []

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get specific recovery recommendations for input validation errors."""
        recommendations = []
        
        if self.field_name:
            recommendations.append(f"Check the '{self.field_name}' field format and value")
        
        for rule in self.validation_rules:
            if 'length' in rule.lower():
                recommendations.append("Adjust input length to meet requirements")
            elif 'format' in rule.lower():
                recommendations.append("Ensure input follows the expected format")
            elif 'type' in rule.lower():
                recommendations.append("Check data type compatibility")

        return {
            'action_type': 'input_correction',
            'recommendations': recommendations,
            'retry_suggested': False
        }


class LLMResponseError(ClaimExtractorError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles empty responses, API failures, content generation issues,
    safety filter blocks, and rate limiting with detailed response analysis.
    """

    def __init__(self, message: str, response_type: str = None, model_name: str = None, 
                 status_code: int = None, api_error: str = None, session_id: str = None):
        """
        Initialize LLM response error with API-specific context.

        Args:
            message: Error description
            response_type: Type of response that failed
            model_name: Name of the LLM model that generated the error
            status_code: HTTP status code if applicable
            api_error: Specific API error message
            session_id: Optional session ID for tracking
        """
        details = {}
        if response_type:
            details['response_type'] = response_type
        if model_name:
            details['model_name'] = model_name
        if status_code:
            details['status_code'] = status_code
        if api_error:
            details['api_error'] = api_error

        super().__init__(message, "LLM_RESPONSE_ERROR", details, session_id, True)
        self.response_type = response_type
        self.model_name = model_name
        self.status_code = status_code
        self.api_error = api_error

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for LLM response errors."""
        recommendations = []
        
        if self.status_code == 429:  # Rate limit
            recommendations.append("Wait before retrying with exponential backoff")
        elif self.status_code in [500, 502, 503, 504]:  # Server errors
            recommendations.append("Retry with shorter delay")
        elif 'safety' in self.api_error.lower() if self.api_error else False:
            recommendations.append("Try alternative prompt with institutional language")
        elif 'timeout' in self.api_error.lower() if self.api_error else False:
            recommendations.append("Reduce input size and retry")

        return {
            'action_type': 'api_retry',
            'recommendations': recommendations,
            'retry_suggested': True,
            'retry_delay': self._get_retry_delay(),
            'max_retries': 3
        }

    def _get_retry_delay(self) -> float:
        """Get recommended retry delay based on error type."""
        if self.status_code == 429:
            return 60.0  # Rate limit - longer wait
        elif self.status_code in [500, 502, 503, 504]:
            return 5.0   # Server error - moderate wait
        else:
            return 2.0   # General error - short wait


class ClaimParsingError(ClaimExtractorError):
    """
    Exception raised when claim parsing fails.
    
    Used for malformed LLM output, parsing failures, claim structure validation issues,
    and format conversion problems with multiple fallback strategy recommendations.
    """

    def __init__(self, message: str, parsing_stage: str = None, raw_output: str = None, 
                 expected_format: str = None, parsing_method: str = None, session_id: str = None):
        """
        Initialize claim parsing error with parsing context.

        Args:
            message: Error description
            parsing_stage: Stage of parsing that failed
            raw_output: Raw LLM output that failed to parse
            expected_format: Expected output format
            parsing_method: Parsing method that was attempted
            session_id: Optional session ID for tracking
        """
        details = {}
        if parsing_stage:
            details['parsing_stage'] = parsing_stage
        if raw_output:
            details['raw_output_sample'] = raw_output[:300] + '...' if len(raw_output) > 300 else raw_output
        if expected_format:
            details['expected_format'] = expected_format
        if parsing_method:
            details['parsing_method'] = parsing_method

        super().__init__(message, "CLAIM_PARSING_ERROR", details, session_id, True)
        self.parsing_stage = parsing_stage
        self.raw_output = raw_output
        self.expected_format = expected_format
        self.parsing_method = parsing_method

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for parsing errors."""
        recommendations = []
        
        if self.parsing_method == 'structured':
            recommendations.append("Try alternative parsing method")
            recommendations.append("Use regex-based extraction fallback")
        elif self.parsing_method == 'json':
            recommendations.append("Try structured text parsing instead")
        elif self.parsing_method == 'regex':
            recommendations.append("Use basic sentence extraction")

        return {
            'action_type': 'parsing_fallback',
            'recommendations': recommendations,
            'retry_suggested': True,
            'fallback_methods': ['alternative_parsing', 'regex_extraction', 'basic_extraction']
        }


class ClaimExtractionError(ClaimExtractorError):
    """
    Exception raised during the claim extraction process.
    
    Used for extraction algorithm failures, pattern analysis errors, processing issues,
    and claim validation failures with comprehensive diagnostic information.
    """

    def __init__(self, message: str, extraction_stage: str = None, claims_processed: int = None,
                 processing_time: float = None, article_length: int = None, session_id: str = None):
        """
        Initialize claim extraction error with processing context.

        Args:
            message: Error description
            extraction_stage: Stage of extraction that failed
            claims_processed: Number of claims processed when error occurred
            processing_time: Time spent processing before error
            article_length: Length of article being processed
            session_id: Optional session ID for tracking
        """
        details = {}
        if extraction_stage:
            details['extraction_stage'] = extraction_stage
        if claims_processed is not None:
            details['claims_processed'] = claims_processed
        if processing_time is not None:
            details['processing_time_seconds'] = round(processing_time, 2)
        if article_length is not None:
            details['article_length'] = article_length

        super().__init__(message, "CLAIM_EXTRACTION_ERROR", details, session_id, True)
        self.extraction_stage = extraction_stage
        self.claims_processed = claims_processed

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for extraction errors."""
        recommendations = []
        
        if self.extraction_stage == 'pattern_analysis':
            recommendations.append("Skip pattern preprocessing and use direct LLM extraction")
        elif self.extraction_stage == 'llm_generation':
            recommendations.append("Use pattern-based fallback extraction")
        elif self.extraction_stage == 'post_processing':
            recommendations.append("Return partial results with reduced validation")

        return {
            'action_type': 'extraction_fallback',
            'recommendations': recommendations,
            'retry_suggested': True,
            'fallback_strategies': ['skip_patterns', 'reduce_validation', 'partial_results']
        }


class VerificationAnalysisError(ClaimExtractorError):
    """
    Exception raised when verification analysis generation fails.
    
    Used for verification prompt failures, analysis generation errors, strategy creation issues,
    and verification resource access problems with detailed analysis context.
    """

    def __init__(self, message: str, analysis_type: str = None, claims_count: int = None,
                 verification_stage: str = None, session_id: str = None):
        """
        Initialize verification analysis error with analysis context.

        Args:
            message: Error description
            analysis_type: Type of verification analysis that failed
            claims_count: Number of claims being analyzed when error occurred
            verification_stage: Specific stage of verification that failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if analysis_type:
            details['analysis_type'] = analysis_type
        if claims_count is not None:
            details['claims_count'] = claims_count
        if verification_stage:
            details['verification_stage'] = verification_stage

        super().__init__(message, "VERIFICATION_ANALYSIS_ERROR", details, session_id, True)
        self.analysis_type = analysis_type
        self.claims_count = claims_count

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for verification analysis errors."""
        return {
            'action_type': 'verification_fallback',
            'recommendations': [
                "Use simplified verification analysis",
                "Skip detailed verification and provide basic recommendations",
                "Return standard verification templates"
            ],
            'retry_suggested': True,
            'fallback_analysis_types': ['basic', 'template', 'minimal']
        }


class PrioritizationError(ClaimExtractorError):
    """
    Exception raised when claim prioritization fails.
    
    Used for prioritization algorithm failures, ranking errors, priority assignment issues,
    and prioritization logic errors with comprehensive prioritization context.
    """

    def __init__(self, message: str, prioritization_method: str = None, claims_analyzed: int = None,
                 priority_criteria: List[str] = None, session_id: str = None):
        """
        Initialize prioritization error with prioritization context.

        Args:
            message: Error description
            prioritization_method: Method used for prioritization that failed
            claims_analyzed: Number of claims being prioritized when error occurred
            priority_criteria: Criteria used for prioritization
            session_id: Optional session ID for tracking
        """
        details = {}
        if prioritization_method:
            details['prioritization_method'] = prioritization_method
        if claims_analyzed is not None:
            details['claims_analyzed'] = claims_analyzed
        if priority_criteria:
            details['priority_criteria'] = priority_criteria

        super().__init__(message, "PRIORITIZATION_ERROR", details, session_id, True)
        self.prioritization_method = prioritization_method
        self.claims_analyzed = claims_analyzed

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for prioritization errors."""
        return {
            'action_type': 'prioritization_fallback',
            'recommendations': [
                "Use default priority assignment (all claims priority 2)",
                "Apply simple rule-based prioritization",
                "Skip prioritization analysis and return original order"
            ],
            'retry_suggested': False,
            'fallback_methods': ['default_priority', 'simple_rules', 'no_prioritization']
        }


class PatternAnalysisError(ClaimExtractorError):
    """
    Exception raised when pattern analysis fails.
    
    Used for pattern database errors, pattern matching failures, claim richness calculation issues,
    and pattern validation problems with detailed pattern analysis context.
    """

    def __init__(self, message: str, pattern_type: str = None, analysis_stage: str = None,
                 patterns_tested: int = None, session_id: str = None):
        """
        Initialize pattern analysis error with pattern context.

        Args:
            message: Error description
            pattern_type: Type of pattern analysis that failed
            analysis_stage: Stage of pattern analysis that failed
            patterns_tested: Number of patterns tested before failure
            session_id: Optional session ID for tracking
        """
        details = {}
        if pattern_type:
            details['pattern_type'] = pattern_type
        if analysis_stage:
            details['analysis_stage'] = analysis_stage
        if patterns_tested is not None:
            details['patterns_tested'] = patterns_tested

        super().__init__(message, "PATTERN_ANALYSIS_ERROR", details, session_id, True)
        self.pattern_type = pattern_type
        self.analysis_stage = analysis_stage

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for pattern analysis errors."""
        return {
            'action_type': 'pattern_fallback',
            'recommendations': [
                "Skip pattern preprocessing",
                "Use simplified pattern matching",
                "Proceed with direct LLM extraction only"
            ],
            'retry_suggested': False,
            'fallback_strategies': ['skip_patterns', 'simplified_matching', 'direct_extraction']
        }


class ConfigurationError(ClaimExtractorError):
    """
    Exception raised when agent configuration is invalid or missing.
    
    Used for missing API keys, invalid config values, setup failures,
    and configuration validation problems with secure error reporting.
    """

    def __init__(self, message: str, config_key: str = None, config_value: Any = None,
                 config_section: str = None, session_id: str = None):
        """
        Initialize configuration error with config context.

        Args:
            message: Error description
            config_key: Configuration key that is invalid or missing
            config_value: Invalid configuration value (sanitized for logging)
            config_section: Configuration section with the issue
            session_id: Optional session ID for tracking
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_section:
            details['config_section'] = config_section
        if config_value is not None:
            # Sanitize sensitive config values
            key_lower = (config_key or '').lower()
            if any(sensitive in key_lower for sensitive in ['key', 'token', 'password', 'secret']):
                details['config_value'] = '[REDACTED]'
            else:
                details['config_value'] = str(config_value)[:100]

        super().__init__(message, "CONFIGURATION_ERROR", details, session_id, False)
        self.config_key = config_key
        self.config_value = config_value

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for configuration errors."""
        recommendations = []
        
        if self.config_key:
            if 'api_key' in self.config_key.lower():
                recommendations.append("Check environment variables for API key")
                recommendations.append("Verify API key format and validity")
            elif 'timeout' in self.config_key.lower():
                recommendations.append("Adjust timeout configuration to reasonable values")
            elif 'model' in self.config_key.lower():
                recommendations.append("Verify model name and availability")

        return {
            'action_type': 'configuration_fix',
            'recommendations': recommendations,
            'retry_suggested': False
        }


class RateLimitError(ClaimExtractorError):
    """
    Exception raised when API rate limits are exceeded.
    
    Used for API throttling, quota exhaustion, rate limit violations,
    and service availability issues with retry timing recommendations.
    """

    def __init__(self, message: str, retry_after: int = None, service: str = None,
                 rate_limit_type: str = None, quota_remaining: int = None, session_id: str = None):
        """
        Initialize rate limit error with rate limiting context.

        Args:
            message: Error description
            retry_after: Seconds to wait before retrying (if known)
            service: Name of the service that rate limited the request
            rate_limit_type: Type of rate limit (requests, tokens, etc.)
            quota_remaining: Remaining quota if known
            session_id: Optional session ID for tracking
        """
        details = {}
        if retry_after:
            details['retry_after'] = retry_after
        if service:
            details['service'] = service
        if rate_limit_type:
            details['rate_limit_type'] = rate_limit_type
        if quota_remaining is not None:
            details['quota_remaining'] = quota_remaining

        super().__init__(message, "RATE_LIMIT_ERROR", details, session_id, True)
        self.retry_after = retry_after
        self.service = service

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for rate limit errors."""
        retry_delay = self.retry_after if self.retry_after else 60  # Default 1 minute
        
        return {
            'action_type': 'rate_limit_backoff',
            'recommendations': [
                f"Wait {retry_delay} seconds before retrying",
                "Implement exponential backoff for subsequent requests",
                "Consider reducing request frequency"
            ],
            'retry_suggested': True,
            'retry_delay': retry_delay,
            'exponential_backoff': True
        }


class ProcessingTimeoutError(ClaimExtractorError):
    """
    Exception raised when claim extraction exceeds time limits.
    
    Used for long-running extractions, API timeouts, processing deadlines,
    and performance threshold violations with processing optimization recommendations.
    """

    def __init__(self, message: str, timeout_seconds: float = None, operation: str = None,
                 actual_duration: float = None, session_id: str = None):
        """
        Initialize processing timeout error with timing context.

        Args:
            message: Error description
            timeout_seconds: Timeout limit that was exceeded
            operation: Operation that timed out
            actual_duration: Actual time spent before timeout
            session_id: Optional session ID for tracking
        """
        details = {}
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        if actual_duration:
            details['actual_duration'] = round(actual_duration, 2)

        super().__init__(message, "PROCESSING_TIMEOUT_ERROR", details, session_id, True)
        self.timeout_seconds = timeout_seconds
        self.operation = operation

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for timeout errors."""
        return {
            'action_type': 'timeout_optimization',
            'recommendations': [
                "Reduce input size and retry",
                "Use faster processing mode",
                "Skip optional analysis steps",
                "Increase timeout limit if appropriate"
            ],
            'retry_suggested': True,
            'optimization_strategies': ['reduce_input', 'fast_mode', 'skip_optional', 'increase_timeout']
        }


class DataFormatError(ClaimExtractorError):
    """
    Exception raised when data parsing or formatting fails.
    
    Used for JSON parsing errors, claim format issues, data structure problems,
    and format conversion failures with format-specific recovery strategies.
    """

    def __init__(self, message: str, data_type: str = None, expected_format: str = None,
                 actual_format: str = None, session_id: str = None):
        """
        Initialize data format error with format context.

        Args:
            message: Error description
            data_type: Type of data that failed to parse
            expected_format: Expected format description
            actual_format: Actual format encountered
            session_id: Optional session ID for tracking
        """
        details = {}
        if data_type:
            details['data_type'] = data_type
        if expected_format:
            details['expected_format'] = expected_format
        if actual_format:
            details['actual_format'] = actual_format

        super().__init__(message, "DATA_FORMAT_ERROR", details, session_id, True)
        self.data_type = data_type
        self.expected_format = expected_format

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for data format errors."""
        recommendations = []
        
        if self.data_type == 'json':
            recommendations.append("Try alternative JSON parsing with error recovery")
            recommendations.append("Fall back to structured text parsing")
        elif self.data_type == 'structured_text':
            recommendations.append("Use regex-based extraction")
        
        return {
            'action_type': 'format_conversion',
            'recommendations': recommendations,
            'retry_suggested': True,
            'alternative_formats': ['json', 'structured_text', 'regex_extraction']
        }


class PromptGenerationError(ClaimExtractorError):
    """
    Exception raised when prompt generation fails.
    
    Used for prompt template errors, content generation failures, and formatting issues.
    These errors are typically retryable as they may be due to transient issues.
    """

    def __init__(self, 
                 message: str, 
                 prompt_type: str = None, 
                 generation_stage: str = None,
                 template_name: str = None,
                 session_id: str = None):
        """
        Initialize prompt generation error.

        Args:
            message: Error description
            prompt_type: Type of prompt that failed to generate
            generation_stage: Stage where generation failed
            template_name: Name of the template that failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if prompt_type:
            details['prompt_type'] = prompt_type
        if generation_stage:
            details['generation_stage'] = generation_stage
        if template_name:
            details['template_name'] = template_name

        super().__init__(
            message, 
            "PROMPT_GENERATION_ERROR", 
            details,
            session_id,
            True  # Prompt generation errors are typically retryable
        )
        self.prompt_type = prompt_type
        self.generation_stage = generation_stage
        self.template_name = template_name

    def get_recovery_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get recovery recommendations for prompt generation errors."""
        recommendations = []
        
        if self.generation_stage == 'template_loading':
            recommendations.append("Verify template file exists and is accessible")
            recommendations.append("Check template syntax and formatting")
        elif self.generation_stage == 'parameter_substitution':
            recommendations.append("Validate all required parameters are provided")
            recommendations.append("Check parameter types and formats")
        elif self.generation_stage == 'content_generation':
            recommendations.append("Try alternative prompt templates")
            recommendations.append("Reduce prompt complexity or length")
        
        return {
            'action_type': 'prompt_regeneration',
            'recommendations': recommendations,
            'fallback_strategies': [
                'alternative_template',
                'simplified_prompt',
                'manual_prompt_generation'
            ],
            'retry_suggested': True,
            'optimization_strategies': ['template_validation', 'parameter_check', 'fallback_templates']
        }


# Enhanced convenience functions for raising exceptions

def raise_input_validation_error(field_name: str, message: str, field_value: Any = None, 
                                validation_rules: List[str] = None, session_id: str = None) -> None:
    """Raise a standardized input validation error with enhanced context."""
    raise InputValidationError(
        f"Invalid {field_name}: {message}", 
        field_name, field_value, validation_rules, session_id
    )


def raise_llm_response_error(response_type: str, message: str, model_name: str = None,
                           status_code: int = None, session_id: str = None) -> None:
    """Raise a standardized LLM response error with API context."""
    raise LLMResponseError(
        f"LLM {response_type} failed: {message}", 
        response_type, model_name, status_code, None, session_id
    )


def raise_claim_parsing_error(parsing_stage: str, message: str, raw_output: str = None,
                            expected_format: str = None, session_id: str = None) -> None:
    """Raise a standardized claim parsing error with parsing context."""
    raise ClaimParsingError(
        f"Claim parsing error ({parsing_stage}): {message}", 
        parsing_stage, raw_output, expected_format, None, session_id
    )


def raise_claim_extraction_error(extraction_stage: str, message: str, claims_processed: int = None,
                               session_id: str = None) -> None:
    """Raise a standardized claim extraction error with processing context."""
    raise ClaimExtractionError(
        f"Claim extraction error ({extraction_stage}): {message}",
        extraction_stage, claims_processed, None, None, session_id
    )


def raise_configuration_error(config_key: str, message: str, config_value: Any = None,
                            session_id: str = None) -> None:
    """Raise a standardized configuration error with config context."""
    raise ConfigurationError(
        f"Configuration error for {config_key}: {message}", 
        config_key, config_value, None, session_id
    )


def raise_prompt_generation_error(prompt_type: str, message: str, generation_stage: str = None,
                                template_name: str = None, session_id: str = None) -> None:
    """Raise a standardized prompt generation error with context."""
    raise PromptGenerationError(
        f"Prompt generation failed for {prompt_type}: {message}", 
        prompt_type, generation_stage, template_name, session_id
    )


# Enhanced exception handling utilities

def handle_claim_extractor_exception(exception: Exception, session_id: str = None) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format with recovery recommendations.

    Args:
        exception: Exception to handle
        session_id: Optional session ID for tracking

    Returns:
        Dictionary with standardized error information and recovery recommendations
    """
    logger = logging.getLogger(f"{__name__}.handle_claim_extractor_exception")
    
    try:
        if isinstance(exception, ClaimExtractorError):
            error_dict = exception.to_dict()
            # Add recovery recommendation if available
            recovery = exception.get_recovery_recommendation()
            if recovery:
                error_dict['recovery_recommendation'] = recovery
            
            logger.error(f"Claim extractor error handled: {exception.error_code}", 
                        extra={'session_id': session_id, 'error_code': exception.error_code})
            
            return error_dict
        else:
            # Handle non-custom exceptions
            error_dict = {
                'error_type': exception.__class__.__name__,
                'error_code': 'UNEXPECTED_ERROR',
                'message': str(exception),
                'details': {},
                'session_id': session_id,
                'timestamp': time.time(),
                'recoverable': False,
                'traceback': traceback.format_exc()
            }
            
            logger.error(f"Unexpected error handled: {exception.__class__.__name__}", 
                        extra={'session_id': session_id})
            
            return error_dict

    except Exception as e:
        # Fallback error handling
        logger.error(f"Error in exception handling: {str(e)}", extra={'session_id': session_id})
        return {
            'error_type': 'ExceptionHandlingError',
            'error_code': 'EXCEPTION_HANDLER_ERROR',
            'message': f"Failed to handle exception: {str(e)}",
            'session_id': session_id,
            'recoverable': False
        }


def is_recoverable_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a recoverable error with enhanced logic.

    Args:
        exception: Exception to evaluate

    Returns:
        True if the error is recoverable (retry possible), False otherwise
    """
    if isinstance(exception, ClaimExtractorError):
        return exception.recoverable
    
    # Check for known recoverable exception types
    recoverable_types = (
        ConnectionError, TimeoutError, OSError
    )
    
    return isinstance(exception, recoverable_types)


def get_retry_delay(exception: Exception, attempt_number: int = 1) -> Optional[float]:
    """
    Get appropriate retry delay for recoverable errors with exponential backoff.

    Args:
        exception: Exception to analyze
        attempt_number: Current retry attempt number for exponential backoff

    Returns:
        Delay in seconds, or None if not retryable
    """
    if isinstance(exception, ClaimExtractorError):
        recovery = exception.get_recovery_recommendation()
        if recovery and recovery.get('retry_suggested'):
            base_delay = recovery.get('retry_delay', 1.0)
            if recovery.get('exponential_backoff'):
                return min(base_delay * (2 ** (attempt_number - 1)), 300)  # Cap at 5 minutes
            return base_delay
    
    # Default retry delays for common exceptions
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return min(2.0 * (2 ** (attempt_number - 1)), 60)  # Exponential backoff, cap at 1 minute
    
    return None


def get_fallback_recommendation(exception: Exception) -> Optional[Dict[str, Any]]:
    """
    Get fallback recommendation for specific error types with enhanced strategies.

    Args:
        exception: Exception that occurred

    Returns:
        Fallback recommendation dictionary or None
    """
    if isinstance(exception, ClaimExtractorError):
        return exception.get_recovery_recommendation()
    
    # Default fallback recommendations for common exceptions
    if isinstance(exception, ConnectionError):
        return {
            'action_type': 'connection_fallback',
            'recommendations': ['Check network connectivity', 'Try alternative endpoint'],
            'retry_suggested': True
        }
    
    return None


def log_exception_with_context(exception: Exception, session_id: str = None, 
                              additional_context: Dict[str, Any] = None) -> None:
    """
    Log exception with comprehensive context for production debugging.

    Args:
        exception: Exception to log
        session_id: Optional session ID for tracking
        additional_context: Additional context information
    """
    logger = logging.getLogger(f"{__name__}.log_exception_with_context")
    
    context = {
        'session_id': session_id,
        'exception_type': exception.__class__.__name__,
        'exception_message': str(exception),
        'recoverable': is_recoverable_error(exception),
        'timestamp': time.time()
    }
    
    if additional_context:
        context.update(additional_context)
    
    if isinstance(exception, ClaimExtractorError):
        context.update(exception.details)
    
    logger.error(f"Exception occurred: {str(exception)}", extra=context, exc_info=True)


# Testing functionality
if __name__ == "__main__":
    """Test claim extractor exception functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== CLAIM EXTRACTOR EXCEPTIONS TEST ===")
    
    # Test input validation error
    try:
        raise_input_validation_error(
            "article_text", 
            "Text is too short", 
            "", 
            ["min_length", "non_empty"], 
            "test_exc_001"
        )
    except InputValidationError as e:
        print("✅ Input Validation Error:", e.to_dict()['message'])
        print(f"   Recovery: {len(e.get_recovery_recommendation()['recommendations'])} recommendations")

    # Test LLM response error
    try:
        raise_llm_response_error(
            "claim_extraction", 
            "Empty response from API", 
            "gemini-1.5-pro", 
            500, 
            "test_exc_002"
        )
    except LLMResponseError as e:
        print("✅ LLM Response Error:", e.to_dict()['message'])
        print(f"   Recovery: retry delay {e.get_recovery_recommendation()['retry_delay']}s")

    # Test claim parsing error
    try:
        raise_claim_parsing_error(
            "structured_format", 
            "Failed to parse claim structure", 
            "**Claim 1**: Invalid...", 
            "structured", 
            "test_exc_003"
        )
    except ClaimParsingError as e:
        print("✅ Claim Parsing Error:", e.to_dict()['message'])
        fallback = e.get_recovery_recommendation()
        print(f"   Fallback methods: {fallback['fallback_methods']}")

    # Test configuration error
    try:
        raise_configuration_error(
            "gemini_api_key", 
            "API key not found in settings", 
            None, 
            "test_exc_004"
        )
    except ConfigurationError as e:
        print("✅ Configuration Error:", e.to_dict()['message'])
        print(f"   Recoverable: {e.recoverable}")

    # Test rate limit error with recovery
    rate_limit_error = RateLimitError(
        "Rate limit exceeded", 
        retry_after=60, 
        service="gemini", 
        session_id="test_exc_005"
    )
    print("✅ Rate Limit Error:", rate_limit_error.message)
    print(f"   Is recoverable: {is_recoverable_error(rate_limit_error)}")
    print(f"   Retry delay: {get_retry_delay(rate_limit_error)} seconds")

    # Test exception handling utility
    handled_error = handle_claim_extractor_exception(rate_limit_error, "test_exc_006")
    print("✅ Exception handling utility:")
    print(f"   Error type: {handled_error['error_type']}")
    print(f"   Has recovery recommendation: {'recovery_recommendation' in handled_error}")
    
    # Test logging with context
    log_exception_with_context(
        rate_limit_error, 
        "test_exc_007", 
        {"operation": "claim_extraction", "article_length": 1500}
    )
    print("✅ Exception logging with context completed")

    print("\n🎯 Claim extractor exceptions tests completed successfully!")
