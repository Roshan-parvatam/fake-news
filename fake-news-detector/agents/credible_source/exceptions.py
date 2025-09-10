# agents/credible_source/exceptions.py

"""
Credible Source Agent Custom Exceptions

Custom exception classes for the Credible Source Agent providing
specific error handling for source database operations, domain classification,
reliability assessments, and contextual source recommendation workflows.
"""

from typing import Any, Dict, List, Optional


class CredibleSourceError(Exception):
    """
    Base exception class for all Credible Source Agent errors.
    
    All custom exceptions in the credible source module should inherit
    from this base class to provide consistent error handling.
    """
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        Initialize base credible source exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CREDIBLE_SOURCE_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


class InputValidationError(CredibleSourceError):
    """
    Exception raised when input data validation fails.
    
    Used for invalid article text, malformed claims, or missing required fields.
    """
    
    def __init__(self, message: str, field_name: str = None, field_value: Any = None):
        """
        Initialize input validation error.
        
        Args:
            message: Error description
            field_name: Name of the invalid field
            field_value: Value that caused the validation error
        """
        details = {}
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = str(field_value)[:200]  # Truncate long values
        
        super().__init__(message, "INPUT_VALIDATION_ERROR", details)
        self.field_name = field_name
        self.field_value = field_value


class LLMResponseError(CredibleSourceError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles safety filter blocks (major issue in original code), invalid responses, 
    and API communication failures.
    """
    
    def __init__(self, message: str, response_type: str = None, model_name: str = None, 
                 safety_blocked: bool = False):
        """
        Initialize LLM response error.
        
        Args:
            message: Error description
            response_type: Type of response that failed (contextual_analysis, reliability_assessment, etc.)
            model_name: Name of the LLM model that generated the error
            safety_blocked: Whether this was caused by safety filters
        """
        details = {}
        if response_type:
            details['response_type'] = response_type
        if model_name:
            details['model_name'] = model_name
        if safety_blocked:
            details['safety_blocked'] = True
            details['fallback_recommended'] = True
        
        super().__init__(message, "LLM_RESPONSE_ERROR", details)
        self.response_type = response_type
        self.model_name = model_name
        self.safety_blocked = safety_blocked


class SourceDatabaseError(CredibleSourceError):
    """
    Exception raised when source database operations fail.
    
    Used for source lookup failures, database corruption, or missing source data.
    """
    
    def __init__(self, message: str, source_key: str = None, operation: str = None):
        """
        Initialize source database error.
        
        Args:
            message: Error description
            source_key: Key of the source that caused the error
            operation: Database operation that failed
        """
        details = {}
        if source_key:
            details['source_key'] = source_key
        if operation:
            details['operation'] = operation
        
        super().__init__(message, "SOURCE_DATABASE_ERROR", details)
        self.source_key = source_key
        self.operation = operation


class DomainClassificationError(CredibleSourceError):
    """
    Exception raised when domain classification fails.
    
    Used for classification algorithm failures, unknown domains, or insufficient data.
    """
    
    def __init__(self, message: str, domain: str = None, confidence: float = None):
        """
        Initialize domain classification error.
        
        Args:
            message: Error description
            domain: Domain that caused classification to fail
            confidence: Classification confidence score if available
        """
        details = {}
        if domain:
            details['domain'] = domain
        if confidence is not None:
            details['confidence'] = confidence
        
        super().__init__(message, "DOMAIN_CLASSIFICATION_ERROR", details)
        self.domain = domain
        self.confidence = confidence


class ReliabilityAssessmentError(CredibleSourceError):
    """
    Exception raised when reliability assessment fails.
    
    Used for source reliability scoring failures, assessment algorithm errors, and validation issues.
    """
    
    def __init__(self, message: str, assessment_step: str = None, source_count: int = None):
        """
        Initialize reliability assessment error.
        
        Args:
            message: Error description
            assessment_step: Stage of assessment that failed
            source_count: Number of sources being assessed when error occurred
        """
        details = {}
        if assessment_step:
            details['assessment_step'] = assessment_step
        if source_count is not None:
            details['source_count'] = source_count
        
        super().__init__(message, "RELIABILITY_ASSESSMENT_ERROR", details)
        self.assessment_step = assessment_step
        self.source_count = source_count


class ContextualRecommendationError(CredibleSourceError):
    """
    Exception raised when contextual source recommendation fails.
    
    Used for contextual analysis failures, source parsing errors, and recommendation generation issues.
    This addresses the key issue from the original code where contextual recommendations failed.
    """
    
    def __init__(self, message: str, context_step: str = None, claims_count: int = None, 
                 safety_fallback_available: bool = True):
        """
        Initialize contextual recommendation error.
        
        Args:
            message: Error description
            context_step: Stage of contextual analysis that failed
            claims_count: Number of claims being processed when error occurred
            safety_fallback_available: Whether safety fallback sources are available
        """
        details = {}
        if context_step:
            details['context_step'] = context_step
        if claims_count is not None:
            details['claims_count'] = claims_count
        details['safety_fallback_available'] = safety_fallback_available
        
        super().__init__(message, "CONTEXTUAL_RECOMMENDATION_ERROR", details)
        self.context_step = context_step
        self.claims_count = claims_count
        self.safety_fallback_available = safety_fallback_available


class VerificationStrategyError(CredibleSourceError):
    """
    Exception raised when verification strategy generation fails.
    
    Used for strategy generation errors, workflow planning failures, and guidance creation issues.
    """
    
    def __init__(self, message: str, strategy_type: str = None, claims_processed: int = None):
        """
        Initialize verification strategy error.
        
        Args:
            message: Error description
            strategy_type: Type of strategy that failed to generate
            claims_processed: Number of claims processed when error occurred
        """
        details = {}
        if strategy_type:
            details['strategy_type'] = strategy_type
        if claims_processed is not None:
            details['claims_processed'] = claims_processed
        
        super().__init__(message, "VERIFICATION_STRATEGY_ERROR", details)
        self.strategy_type = strategy_type
        self.claims_processed = claims_processed


class SafetyFilterError(CredibleSourceError):
    """
    Exception raised when content safety filters block analysis.
    
    This addresses the major issue in the original code where Gemini safety filters
    were blocking contextual source generation.
    """
    
    def __init__(self, message: str, content_type: str = None, fallback_used: bool = False):
        """
        Initialize safety filter error.
        
        Args:
            message: Error description
            content_type: Type of content that triggered safety filters
            fallback_used: Whether fallback sources were successfully generated
        """
        details = {}
        if content_type:
            details['content_type'] = content_type
        details['fallback_used'] = fallback_used
        details['mitigation'] = "Institutional fallback sources recommended"
        
        super().__init__(message, "SAFETY_FILTER_ERROR", details)
        self.content_type = content_type
        self.fallback_used = fallback_used


class ConfigurationError(CredibleSourceError):
    """
    Exception raised when agent configuration is invalid or missing.
    
    Used for missing API keys, invalid config values, and setup failures.
    """
    
    def __init__(self, message: str, config_key: str = None, config_value: Any = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error description
            config_key: Configuration key that is invalid or missing
            config_value: Invalid configuration value (sanitized for logging)
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            # Sanitize sensitive config values
            if 'key' in str(config_key).lower() or 'token' in str(config_key).lower():
                details['config_value'] = '[REDACTED]'
            else:
                details['config_value'] = str(config_value)[:100]
        
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key
        self.config_value = config_value


class RateLimitError(CredibleSourceError):
    """
    Exception raised when API rate limits are exceeded.
    
    Used for API throttling, quota exhaustion, and rate limit violations.
    """
    
    def __init__(self, message: str, retry_after: int = None, service: str = None):
        """
        Initialize rate limit error.
        
        Args:
            message: Error description
            retry_after: Seconds to wait before retrying (if known)
            service: Name of the service that rate limited the request
        """
        details = {}
        if retry_after:
            details['retry_after'] = retry_after
        if service:
            details['service'] = service
        
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.retry_after = retry_after
        self.service = service


class ProcessingTimeoutError(CredibleSourceError):
    """
    Exception raised when source recommendation exceeds time limits.
    
    Used for long-running analysis, API timeouts, and processing deadlines.
    """
    
    def __init__(self, message: str, timeout_seconds: float = None, operation: str = None):
        """
        Initialize processing timeout error.
        
        Args:
            message: Error description
            timeout_seconds: Timeout limit that was exceeded
            operation: Operation that timed out
        """
        details = {}
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        
        super().__init__(message, "PROCESSING_TIMEOUT_ERROR", details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class DataFormatError(CredibleSourceError):
    """
    Exception raised when data parsing or formatting fails.
    
    Used for JSON parsing errors, response format issues, and data structure problems.
    """
    
    def __init__(self, message: str, data_type: str = None, expected_format: str = None):
        """
        Initialize data format error.
        
        Args:
            message: Error description
            data_type: Type of data that failed to parse
            expected_format: Expected format description
        """
        details = {}
        if data_type:
            details['data_type'] = data_type
        if expected_format:
            details['expected_format'] = expected_format
        
        super().__init__(message, "DATA_FORMAT_ERROR", details)
        self.data_type = data_type
        self.expected_format = expected_format


# Convenience functions for common exception scenarios
def raise_input_validation_error(field_name: str, message: str, field_value: Any = None) -> None:
    """
    Raise a standardized input validation error.
    
    Args:
        field_name: Name of the invalid field
        message: Validation error description
        field_value: Optional invalid value
    """
    raise InputValidationError(f"Invalid {field_name}: {message}", field_name, field_value)


def raise_llm_response_error(response_type: str, message: str, model_name: str = None, 
                           safety_blocked: bool = False) -> None:
    """
    Raise a standardized LLM response error.
    
    Args:
        response_type: Type of LLM response that failed
        message: Error description
        model_name: Optional model name
        safety_blocked: Whether this was caused by safety filters
    """
    raise LLMResponseError(f"LLM {response_type} failed: {message}", response_type, model_name, safety_blocked)


def raise_source_database_error(source_key: str, message: str, operation: str = None) -> None:
    """
    Raise a standardized source database error.
    
    Args:
        source_key: Source that caused the error
        message: Error description
        operation: Optional database operation
    """
    raise SourceDatabaseError(f"Source database error for {source_key}: {message}", source_key, operation)


def raise_contextual_recommendation_error(context_step: str, message: str, 
                                        claims_count: int = None) -> None:
    """
    Raise a standardized contextual recommendation error.
    
    Args:
        context_step: Stage of contextual analysis that failed
        message: Error description
        claims_count: Optional number of claims being processed
    """
    raise ContextualRecommendationError(
        f"Contextual recommendation error ({context_step}): {message}", 
        context_step, claims_count
    )


def raise_safety_filter_error(content_type: str, message: str, fallback_used: bool = False) -> None:
    """
    Raise a standardized safety filter error.
    
    Args:
        content_type: Type of content that triggered safety filters
        message: Error description
        fallback_used: Whether fallback sources were generated
    """
    raise SafetyFilterError(f"Safety filter blocked {content_type}: {message}", content_type, fallback_used)


def raise_configuration_error(config_key: str, message: str, config_value: Any = None) -> None:
    """
    Raise a standardized configuration error.
    
    Args:
        config_key: Configuration key that is problematic
        message: Error description
        config_value: Optional configuration value
    """
    raise ConfigurationError(f"Configuration error for {config_key}: {message}", config_key, config_value)


# Exception handling utilities
def handle_credible_source_exception(exception: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format.
    
    Args:
        exception: Exception to handle
        
    Returns:
        Dictionary with standardized error information
    """
    if isinstance(exception, CredibleSourceError):
        return exception.to_dict()
    else:
        # Handle non-custom exceptions
        return {
            'error_type': exception.__class__.__name__,
            'error_code': 'UNEXPECTED_ERROR',
            'message': str(exception),
            'details': {}
        }


def is_recoverable_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a recoverable error.
    
    Args:
        exception: Exception to evaluate
        
    Returns:
        True if the error is recoverable (retry possible), False otherwise
    """
    recoverable_errors = (
        RateLimitError,
        ProcessingTimeoutError,
        LLMResponseError,
        ContextualRecommendationError,  # Can use fallback sources
        SafetyFilterError  # Can use institutional fallbacks
    )
    
    return isinstance(exception, recoverable_errors)


def get_retry_delay(exception: Exception) -> Optional[float]:
    """
    Get appropriate retry delay for recoverable errors.
    
    Args:
        exception: Exception to analyze
        
    Returns:
        Delay in seconds, or None if not retryable
    """
    if isinstance(exception, RateLimitError) and exception.retry_after:
        return float(exception.retry_after)
    elif isinstance(exception, ProcessingTimeoutError):
        return 5.0  # Standard retry delay for timeouts
    elif isinstance(exception, LLMResponseError):
        return 2.0  # Short delay for LLM retries
    elif isinstance(exception, SafetyFilterError):
        return 0.0  # Immediate fallback to institutional sources
    else:
        return None


def get_fallback_recommendation(exception: Exception) -> Optional[Dict[str, Any]]:
    """
    Get fallback recommendation for specific error types.
    
    Args:
        exception: Exception that occurred
        
    Returns:
        Fallback recommendation dictionary or None
    """
    if isinstance(exception, SafetyFilterError):
        return {
            'fallback_type': 'institutional_sources',
            'recommendation': 'Use institutional database sources',
            'action': 'generate_institutional_fallback'
        }
    elif isinstance(exception, ContextualRecommendationError) and exception.safety_fallback_available:
        return {
            'fallback_type': 'database_sources',
            'recommendation': 'Use source database recommendations',
            'action': 'use_database_recommendations'
        }
    elif isinstance(exception, LLMResponseError) and exception.safety_blocked:
        return {
            'fallback_type': 'safe_institutional',
            'recommendation': 'Generate safe institutional source list',
            'action': 'generate_safe_sources'
        }
    else:
        return None


# Testing functionality
if __name__ == "__main__":
    """Test credible source exception functionality."""
    
    # Test input validation error
    try:
        raise_input_validation_error("extracted_claims", "Claims list is empty", [])
    except InputValidationError as e:
        print("Input Validation Error:", e.to_dict())
    
    # Test LLM response error with safety blocking (major issue from original code)
    try:
        raise_llm_response_error("contextual_analysis", "Response blocked by safety filters", 
                               "gemini-1.5-pro", safety_blocked=True)
    except LLMResponseError as e:
        print("LLM Response Error (Safety Blocked):", e.to_dict())
    
    # Test source database error
    try:
        raise_source_database_error("cdc", "Source data not found", "lookup")
    except SourceDatabaseError as e:
        print("Source Database Error:", e.to_dict())
    
    # Test contextual recommendation error
    try:
        raise_contextual_recommendation_error("generate_contextual_sources", 
                                            "Failed to parse contextual sources", 3)
    except ContextualRecommendationError as e:
        print("Contextual Recommendation Error:", e.to_dict())
    
    # Test safety filter error (addresses main original issue)
    try:
        raise_safety_filter_error("vaccine_misinformation", 
                                 "Content flagged as potentially harmful", fallback_used=True)
    except SafetyFilterError as e:
        print("Safety Filter Error:", e.to_dict())
        
        # Test fallback recommendation
        fallback = get_fallback_recommendation(e)
        print("Fallback Recommendation:", fallback)
    
    # Test configuration error
    try:
        raise_configuration_error("gemini_api_key", "API key not found in settings")
    except ConfigurationError as e:
        print("Configuration Error:", e.to_dict())
    
    print("\n=== EXCEPTION TESTING COMPLETED ===")
    print("All exception types tested successfully!")
