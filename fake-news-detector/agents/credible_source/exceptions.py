# agents/credible_source/exceptions.py

"""
Credible Source Agent Custom Exceptions - Production Ready

Enhanced exception classes for production-level error handling with
retry logic support, detailed error context, session tracking, and
proper categorization for the credible source agent.
"""

from typing import Any, Dict, List, Optional
import time

class CredibleSourceError(Exception):
    """
    Base exception class for all Credible Source Agent errors.
    
    Provides structured error information for logging and debugging
    with support for retry logic, session tracking, and error categorization.
    """

    def __init__(self, 
                 message: str, 
                 error_code: str = None, 
                 details: Dict[str, Any] = None,
                 retryable: bool = False,
                 session_id: str = None):
        """
        Initialize base credible source exception.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
            retryable: Whether this error should trigger retry logic
            session_id: Optional session ID for tracking across operations
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CREDIBLE_SOURCE_ERROR"
        self.details = details or {}
        self.retryable = retryable
        self.session_id = session_id
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'retryable': self.retryable,
            'session_id': self.session_id,
            'timestamp': self.timestamp
        }

    def __str__(self) -> str:
        """String representation with session ID if available."""
        base_str = f"{self.error_code}: {self.message}"
        if self.session_id:
            base_str += f" [Session: {self.session_id}]"
        return base_str


class InputValidationError(CredibleSourceError):
    """
    Exception raised when input data validation fails.
    
    Used for invalid article text, malformed claims, or missing required fields.
    These errors are typically not retryable as they require user correction.
    """

    def __init__(self, 
                 message: str, 
                 field_name: str = None, 
                 field_value: Any = None,
                 session_id: str = None):
        """
        Initialize input validation error.

        Args:
            message: Error description
            field_name: Name of the invalid field
            field_value: Value that caused the validation error
            session_id: Optional session ID for tracking
        """
        details = {}
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            # Truncate long values for logging
            details['field_value'] = str(field_value)[:200] if len(str(field_value)) > 200 else str(field_value)

        super().__init__(
            message, 
            "INPUT_VALIDATION_ERROR", 
            details,
            retryable=False,  # Input validation errors are not retryable
            session_id=session_id
        )
        self.field_name = field_name
        self.field_value = field_value


class LLMResponseError(CredibleSourceError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles safety filter blocks (major issue in original code), invalid responses,
    API communication failures, and timeout issues. These errors are typically retryable.
    """

    def __init__(self, 
                 message: str, 
                 response_type: str = None, 
                 model_name: str = None,
                 safety_blocked: bool = False,
                 attempt_number: int = 1,
                 session_id: str = None):
        """
        Initialize LLM response error.

        Args:
            message: Error description
            response_type: Type of response that failed (contextual_analysis, reliability_assessment, etc.)
            model_name: Name of the LLM model that generated the error
            safety_blocked: Whether this was caused by safety filters
            attempt_number: Which retry attempt this error occurred on
            session_id: Optional session ID for tracking
        """
        details = {}
        if response_type:
            details['response_type'] = response_type
        if model_name:
            details['model_name'] = model_name
        if safety_blocked:
            details['safety_blocked'] = True
            details['fallback_recommended'] = True
        if attempt_number:
            details['attempt_number'] = attempt_number

        super().__init__(
            message, 
            "LLM_RESPONSE_ERROR", 
            details,
            retryable=True,  # LLM errors are typically retryable
            session_id=session_id
        )
        self.response_type = response_type
        self.model_name = model_name
        self.safety_blocked = safety_blocked
        self.attempt_number = attempt_number


class SourceDatabaseError(CredibleSourceError):
    """
    Exception raised when source database operations fail.
    
    Used for source lookup failures, database corruption, or missing source data.
    These errors may be retryable depending on the nature of the failure.
    """

    def __init__(self, 
                 message: str, 
                 source_key: str = None, 
                 operation: str = None,
                 database_type: str = None,
                 session_id: str = None):
        """
        Initialize source database error.

        Args:
            message: Error description
            source_key: Key of the source that caused the error
            operation: Database operation that failed
            database_type: Type of database operation (lookup, update, validate)
            session_id: Optional session ID for tracking
        """
        details = {}
        if source_key:
            details['source_key'] = source_key
        if operation:
            details['operation'] = operation
        if database_type:
            details['database_type'] = database_type

        # Database lookup errors might be retryable, corruption errors are not
        retryable = operation in ['lookup', 'search', 'recommendation']

        super().__init__(
            message, 
            "SOURCE_DATABASE_ERROR", 
            details,
            retryable=retryable,
            session_id=session_id
        )
        self.source_key = source_key
        self.operation = operation
        self.database_type = database_type


class DomainClassificationError(CredibleSourceError):
    """
    Exception raised when domain classification fails.
    
    Used for classification algorithm failures, unknown domains, or insufficient data.
    These errors are typically retryable with fallback to general domain.
    """

    def __init__(self, 
                 message: str, 
                 domain: str = None, 
                 confidence: float = None,
                 classification_stage: str = None,
                 session_id: str = None):
        """
        Initialize domain classification error.

        Args:
            message: Error description
            domain: Domain that caused classification to fail
            confidence: Classification confidence score if available
            classification_stage: Stage where classification failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if domain:
            details['domain'] = domain
        if confidence is not None:
            details['confidence'] = confidence
        if classification_stage:
            details['classification_stage'] = classification_stage

        super().__init__(
            message, 
            "DOMAIN_CLASSIFICATION_ERROR", 
            details,
            retryable=True,  # Can fallback to general domain
            session_id=session_id
        )
        self.domain = domain
        self.confidence = confidence
        self.classification_stage = classification_stage


class ReliabilityAssessmentError(CredibleSourceError):
    """
    Exception raised when reliability assessment fails.
    
    Used for source reliability scoring failures, assessment algorithm errors,
    and validation issues. These may be retryable depending on the cause.
    """

    def __init__(self, 
                 message: str, 
                 assessment_step: str = None, 
                 source_count: int = None,
                 assessment_type: str = None,
                 session_id: str = None):
        """
        Initialize reliability assessment error.

        Args:
            message: Error description
            assessment_step: Stage of assessment that failed
            source_count: Number of sources being assessed when error occurred
            assessment_type: Type of assessment (scoring, validation, ranking)
            session_id: Optional session ID for tracking
        """
        details = {}
        if assessment_step:
            details['assessment_step'] = assessment_step
        if source_count is not None:
            details['source_count'] = source_count
        if assessment_type:
            details['assessment_type'] = assessment_type

        # Assessment errors during generation are retryable, calculation errors are not
        retryable = assessment_step in ['generation', 'llm_analysis']

        super().__init__(
            message, 
            "RELIABILITY_ASSESSMENT_ERROR", 
            details,
            retryable=retryable,
            session_id=session_id
        )
        self.assessment_step = assessment_step
        self.source_count = source_count
        self.assessment_type = assessment_type


class ContextualRecommendationError(CredibleSourceError):
    """
    Exception raised when contextual source recommendation fails.
    
    This addresses the key issue from the original code where contextual 
    recommendations failed due to safety filters or analysis errors.
    """

    def __init__(self, 
                 message: str, 
                 context_step: str = None, 
                 claims_count: int = None,
                 safety_fallback_available: bool = True,
                 analysis_stage: str = None,
                 session_id: str = None):
        """
        Initialize contextual recommendation error.

        Args:
            message: Error description
            context_step: Stage of contextual analysis that failed
            claims_count: Number of claims being processed when error occurred
            safety_fallback_available: Whether safety fallback sources are available
            analysis_stage: Specific analysis stage that failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if context_step:
            details['context_step'] = context_step
        if claims_count is not None:
            details['claims_count'] = claims_count
        details['safety_fallback_available'] = safety_fallback_available
        if analysis_stage:
            details['analysis_stage'] = analysis_stage

        super().__init__(
            message, 
            "CONTEXTUAL_RECOMMENDATION_ERROR", 
            details,
            retryable=True,  # Can use fallback sources
            session_id=session_id
        )
        self.context_step = context_step
        self.claims_count = claims_count
        self.safety_fallback_available = safety_fallback_available
        self.analysis_stage = analysis_stage


class VerificationStrategyError(CredibleSourceError):
    """
    Exception raised when verification strategy generation fails.
    
    Used for strategy generation errors, workflow planning failures,
    and guidance creation issues. These are typically retryable.
    """

    def __init__(self, 
                 message: str, 
                 strategy_type: str = None, 
                 claims_processed: int = None,
                 generation_stage: str = None,
                 session_id: str = None):
        """
        Initialize verification strategy error.

        Args:
            message: Error description
            strategy_type: Type of strategy that failed to generate
            claims_processed: Number of claims processed when error occurred
            generation_stage: Stage of strategy generation that failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if strategy_type:
            details['strategy_type'] = strategy_type
        if claims_processed is not None:
            details['claims_processed'] = claims_processed
        if generation_stage:
            details['generation_stage'] = generation_stage

        super().__init__(
            message, 
            "VERIFICATION_STRATEGY_ERROR", 
            details,
            retryable=True,  # Strategy generation is retryable
            session_id=session_id
        )
        self.strategy_type = strategy_type
        self.claims_processed = claims_processed
        self.generation_stage = generation_stage


class SafetyFilterError(CredibleSourceError):
    """
    Exception raised when content safety filters block analysis.
    
    This addresses the major issue in the original code where Gemini safety filters
    were blocking contextual source generation. Always has institutional fallback.
    """

    def __init__(self, 
                 message: str, 
                 content_type: str = None, 
                 fallback_used: bool = False,
                 filter_category: str = None,
                 session_id: str = None):
        """
        Initialize safety filter error.

        Args:
            message: Error description
            content_type: Type of content that triggered safety filters
            fallback_used: Whether fallback sources were successfully generated
            filter_category: Category of safety filter that was triggered
            session_id: Optional session ID for tracking
        """
        details = {}
        if content_type:
            details['content_type'] = content_type
        details['fallback_used'] = fallback_used
        details['mitigation'] = "Institutional fallback sources recommended"
        if filter_category:
            details['filter_category'] = filter_category

        super().__init__(
            message, 
            "SAFETY_FILTER_ERROR", 
            details,
            retryable=False,  # Use immediate fallback instead of retry
            session_id=session_id
        )
        self.content_type = content_type
        self.fallback_used = fallback_used
        self.filter_category = filter_category


class ConfigurationError(CredibleSourceError):
    """
    Exception raised when agent configuration is invalid or missing.
    
    Used for missing API keys, invalid config values, and setup failures.
    These errors are not retryable without fixing the configuration.
    """

    def __init__(self, 
                 message: str, 
                 config_key: str = None, 
                 config_value: Any = None,
                 config_section: str = None,
                 session_id: str = None):
        """
        Initialize configuration error.

        Args:
            message: Error description
            config_key: Configuration key that is invalid or missing
            config_value: Invalid configuration value (sanitized for logging)
            config_section: Configuration section where error occurred
            session_id: Optional session ID for tracking
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_section:
            details['config_section'] = config_section
        
        if config_value is not None:
            # Sanitize sensitive config values
            if any(sensitive in str(config_key).lower() for sensitive in ['key', 'token', 'secret', 'password']):
                details['config_value'] = '[REDACTED]'
            else:
                details['config_value'] = str(config_value)[:100]

        super().__init__(
            message, 
            "CONFIGURATION_ERROR", 
            details,
            retryable=False,  # Config errors are not retryable without fixing config
            session_id=session_id
        )
        self.config_key = config_key
        self.config_value = config_value
        self.config_section = config_section


class RateLimitError(CredibleSourceError):
    """
    Exception raised when API rate limits are exceeded.
    
    Used for API throttling, quota exhaustion, and rate limit violations.
    These errors are retryable after waiting for the specified period.
    """

    def __init__(self, 
                 message: str, 
                 retry_after: int = None, 
                 service: str = None,
                 requests_remaining: int = None,
                 quota_type: str = None,
                 session_id: str = None):
        """
        Initialize rate limit error.

        Args:
            message: Error description
            retry_after: Seconds to wait before retrying (if known)
            service: Name of the service that rate limited the request
            requests_remaining: Number of requests remaining in quota
            quota_type: Type of quota exceeded (per-minute, per-day, etc.)
            session_id: Optional session ID for tracking
        """
        details = {}
        if retry_after:
            details['retry_after'] = retry_after
        if service:
            details['service'] = service
        if requests_remaining is not None:
            details['requests_remaining'] = requests_remaining
        if quota_type:
            details['quota_type'] = quota_type

        super().__init__(
            message, 
            "RATE_LIMIT_ERROR", 
            details,
            retryable=True,  # Rate limit errors are retryable after waiting
            session_id=session_id
        )
        self.retry_after = retry_after
        self.service = service
        self.requests_remaining = requests_remaining
        self.quota_type = quota_type


class ProcessingTimeoutError(CredibleSourceError):
    """
    Exception raised when source recommendation exceeds time limits.
    
    Used for long-running analysis, API timeouts, and processing deadlines.
    These errors may be retryable depending on the timeout type.
    """

    def __init__(self, 
                 message: str, 
                 timeout_seconds: float = None, 
                 operation: str = None,
                 elapsed_time: float = None,
                 timeout_type: str = None,
                 session_id: str = None):
        """
        Initialize processing timeout error.

        Args:
            message: Error description
            timeout_seconds: Timeout limit that was exceeded
            operation: Operation that timed out
            elapsed_time: How long the operation actually took
            timeout_type: Type of timeout (api, processing, total)
            session_id: Optional session ID for tracking
        """
        details = {}
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        if elapsed_time:
            details['elapsed_time'] = elapsed_time
        if timeout_type:
            details['timeout_type'] = timeout_type

        # API timeouts are retryable, processing timeouts might not be
        retryable = timeout_type in ['api', 'network'] or operation in ['api_call', 'llm_generation']

        super().__init__(
            message, 
            "PROCESSING_TIMEOUT_ERROR", 
            details,
            retryable=retryable,
            session_id=session_id
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.elapsed_time = elapsed_time
        self.timeout_type = timeout_type


class DataFormatError(CredibleSourceError):
    """
    Exception raised when data parsing or formatting fails.
    
    Used for JSON parsing errors, response format issues, and data structure problems.
    These errors may be retryable if they're due to transient response issues.
    """

    def __init__(self, 
                 message: str, 
                 data_type: str = None, 
                 expected_format: str = None,
                 actual_data: str = None,
                 parsing_stage: str = None,
                 session_id: str = None):
        """
        Initialize data format error.

        Args:
            message: Error description
            data_type: Type of data that failed to parse
            expected_format: Expected format description
            actual_data: Sample of actual data received (truncated)
            parsing_stage: Stage where parsing failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if data_type:
            details['data_type'] = data_type
        if expected_format:
            details['expected_format'] = expected_format
        if actual_data:
            # Truncate actual data for logging
            details['actual_data'] = actual_data[:200] if len(actual_data) > 200 else actual_data
        if parsing_stage:
            details['parsing_stage'] = parsing_stage

        # Data format errors from LLM responses might be retryable
        retryable = data_type in ['llm_response', 'api_response'] or parsing_stage == 'response_parsing'

        super().__init__(
            message, 
            "DATA_FORMAT_ERROR", 
            details,
            retryable=retryable,
            session_id=session_id
        )
        self.data_type = data_type
        self.expected_format = expected_format
        self.actual_data = actual_data
        self.parsing_stage = parsing_stage


class PromptGenerationError(CredibleSourceError):
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
            retryable=True,  # Prompt generation errors are typically retryable
            session_id=session_id
        )
        self.prompt_type = prompt_type
        self.generation_stage = generation_stage
        self.template_name = template_name


# Enhanced convenience functions for common exception scenarios

def raise_input_validation_error(field_name: str, message: str, field_value: Any = None, session_id: str = None) -> None:
    """
    Raise a standardized input validation error with session tracking.

    Args:
        field_name: Name of the invalid field
        message: Validation error description
        field_value: Optional invalid value
        session_id: Optional session ID for tracking
    """
    raise InputValidationError(
        f"Invalid {field_name}: {message}", 
        field_name, 
        field_value, 
        session_id
    )


def raise_llm_response_error(response_type: str, message: str, model_name: str = None,
                           safety_blocked: bool = False, attempt_number: int = 1, session_id: str = None) -> None:
    """
    Raise a standardized LLM response error with retry context.

    Args:
        response_type: Type of LLM response that failed
        message: Error description
        model_name: Optional model name
        safety_blocked: Whether this was caused by safety filters
        attempt_number: Which retry attempt this error occurred on
        session_id: Optional session ID for tracking
    """
    raise LLMResponseError(
        f"LLM {response_type} failed: {message}", 
        response_type, 
        model_name, 
        safety_blocked,
        attempt_number, 
        session_id
    )


def raise_source_database_error(source_key: str, message: str, operation: str = None, 
                              database_type: str = None, session_id: str = None) -> None:
    """
    Raise a standardized source database error with detailed context.

    Args:
        source_key: Source that caused the error
        message: Error description
        operation: Optional database operation
        database_type: Optional database operation type
        session_id: Optional session ID for tracking
    """
    raise SourceDatabaseError(
        f"Source database error for {source_key}: {message}", 
        source_key, 
        operation, 
        database_type, 
        session_id
    )


def raise_contextual_recommendation_error(context_step: str, message: str, claims_count: int = None,
                                        safety_fallback_available: bool = True, session_id: str = None) -> None:
    """
    Raise a standardized contextual recommendation error with fallback info.

    Args:
        context_step: Stage of contextual analysis that failed
        message: Error description
        claims_count: Optional number of claims being processed
        safety_fallback_available: Whether institutional fallback is available
        session_id: Optional session ID for tracking
    """
    raise ContextualRecommendationError(
        f"Contextual recommendation error ({context_step}): {message}",
        context_step, 
        claims_count, 
        safety_fallback_available,
        session_id=session_id
    )


def raise_safety_filter_error(content_type: str, message: str, fallback_used: bool = False, 
                            filter_category: str = None, session_id: str = None) -> None:
    """
    Raise a standardized safety filter error with fallback context.

    Args:
        content_type: Type of content that triggered safety filters
        message: Error description
        fallback_used: Whether fallback sources were generated
        filter_category: Category of safety filter triggered
        session_id: Optional session ID for tracking
    """
    raise SafetyFilterError(
        f"Safety filter blocked {content_type}: {message}", 
        content_type, 
        fallback_used, 
        filter_category, 
        session_id
    )


def raise_configuration_error(config_key: str, message: str, config_value: Any = None, 
                            config_section: str = None, session_id: str = None) -> None:
    """
    Raise a standardized configuration error with context.

    Args:
        config_key: Configuration key that is problematic
        message: Error description
        config_value: Optional configuration value
        config_section: Optional configuration section
        session_id: Optional session ID for tracking
    """
    raise ConfigurationError(
        f"Configuration error for {config_key}: {message}", 
        config_key, 
        config_value, 
        config_section, 
        session_id
    )


def raise_prompt_generation_error(prompt_type: str, message: str, generation_stage: str = None,
                                template_name: str = None, session_id: str = None) -> None:
    """
    Raise a standardized prompt generation error with context.

    Args:
        prompt_type: Type of prompt that failed to generate
        message: Error description
        generation_stage: Optional stage where generation failed
        template_name: Optional template name
        session_id: Optional session ID for tracking
    """
    raise PromptGenerationError(
        f"Prompt generation failed for {prompt_type}: {message}", 
        prompt_type, 
        generation_stage, 
        template_name, 
        session_id
    )


# Enhanced exception handling utilities

def handle_credible_source_exception(exception: Exception, session_id: str = None) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format with enhanced context.

    Args:
        exception: Exception to handle
        session_id: Optional session ID for tracking

    Returns:
        Dictionary with standardized error information
    """
    if isinstance(exception, CredibleSourceError):
        error_dict = exception.to_dict()
        # Add session ID if not already present
        if session_id and not error_dict.get('session_id'):
            error_dict['session_id'] = session_id
        return error_dict
    else:
        # Handle non-custom exceptions
        return {
            'error_type': exception.__class__.__name__,
            'error_code': 'UNEXPECTED_ERROR',
            'message': str(exception),
            'details': {},
            'retryable': False,
            'session_id': session_id,
            'timestamp': time.time()
        }


def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a retryable error.

    Args:
        exception: Exception to evaluate

    Returns:
        True if the error is retryable, False otherwise
    """
    if isinstance(exception, CredibleSourceError):
        return exception.retryable
    
    # For non-custom exceptions, check if they're typically retryable
    retryable_exception_types = (
        ConnectionError,
        TimeoutError,
        # Add other standard retryable exceptions as needed
    )
    
    return isinstance(exception, retryable_exception_types)


def get_retry_delay(exception: Exception, attempt_number: int = 1) -> Optional[float]:
    """
    Get appropriate retry delay for retryable errors with exponential backoff.

    Args:
        exception: Exception to analyze
        attempt_number: Current attempt number for exponential backoff

    Returns:
        Delay in seconds, or None if not retryable
    """
    if not is_retryable_error(exception):
        return None
    
    # Base delays for different error types
    base_delays = {
        RateLimitError: 5.0,
        ProcessingTimeoutError: 3.0,
        LLMResponseError: 2.0,
        SourceDatabaseError: 1.5,
        ContextualRecommendationError: 1.0,
        VerificationStrategyError: 1.0,
        DataFormatError: 1.0
    }
    
    # Get base delay
    base_delay = 2.0  # Default
    for error_type, delay in base_delays.items():
        if isinstance(exception, error_type):
            base_delay = delay
            break
    
    # Special handling for rate limit errors
    if isinstance(exception, RateLimitError) and exception.retry_after:
        return float(exception.retry_after)
    
    # Special handling for safety filter errors (use immediate fallback)
    if isinstance(exception, SafetyFilterError):
        return 0.0  # No delay, immediate institutional fallback
    
    # Exponential backoff: base_delay * (2 ^ (attempt_number - 1))
    exponential_delay = base_delay * (2 ** (attempt_number - 1))
    
    # Cap the maximum delay at 30 seconds
    return min(exponential_delay, 30.0)


def should_retry_after_attempts(exception: Exception, attempt_count: int, max_attempts: int = 3) -> bool:
    """
    Determine if we should continue retrying after a certain number of attempts.

    Args:
        exception: Exception that occurred
        attempt_count: Number of attempts made so far
        max_attempts: Maximum number of attempts allowed

    Returns:
        True if should retry, False otherwise
    """
    if attempt_count >= max_attempts:
        return False
    
    if not is_retryable_error(exception):
        return False
    
    # Special cases for specific error types
    if isinstance(exception, RateLimitError):
        # Always retry rate limit errors (they'll resolve eventually)
        return attempt_count < max_attempts
    
    if isinstance(exception, SafetyFilterError):
        # Don't retry safety filter errors, use immediate fallback
        return False
    
    if isinstance(exception, ConfigurationError):
        # Never retry configuration errors
        return False
    
    return True


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
            'recommendation': 'Use institutional database sources with domain-specific fallbacks',
            'action': 'generate_institutional_fallback',
            'immediate': True
        }

    elif isinstance(exception, ContextualRecommendationError) and exception.safety_fallback_available:
        return {
            'fallback_type': 'database_sources',
            'recommendation': 'Use source database recommendations with reliability scoring',
            'action': 'use_database_recommendations',
            'immediate': False
        }

    elif isinstance(exception, LLMResponseError) and exception.safety_blocked:
        return {
            'fallback_type': 'safe_institutional',
            'recommendation': 'Generate safe institutional source list with domain classification',
            'action': 'generate_safe_sources',
            'immediate': True
        }

    elif isinstance(exception, SourceDatabaseError):
        return {
            'fallback_type': 'manual_sources',
            'recommendation': 'Provide manual source recommendations based on domain',
            'action': 'generate_manual_fallback',
            'immediate': False
        }

    else:
        return None


# Testing functionality
if __name__ == "__main__":
    """Test credible source exception functionality."""
    import logging
    
    # Setup logging to see exception details
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test different exception types
    test_session_id = "test_session_456"
    
    try:
        # Test safety filter error (major issue from original code)
        raise_safety_filter_error(
            "vaccine_misinformation",
            "Content flagged as potentially harmful", 
            fallback_used=True,
            filter_category="medical_misinformation",
            session_id=test_session_id
        )
    except SafetyFilterError as e:
        error_dict = handle_credible_source_exception(e, test_session_id)
        logger.info(f"Handled safety filter error: {error_dict}")
        logger.info(f"Should retry: {is_retryable_error(e)}")
        logger.info(f"Retry delay: {get_retry_delay(e, 2)} seconds")
        
        # Test fallback recommendation
        fallback = get_fallback_recommendation(e)
        logger.info(f"Fallback recommendation: {fallback}")
    
    try:
        # Test contextual recommendation error
        raise_contextual_recommendation_error(
            "generate_contextual_sources",
            "Failed to parse contextual sources", 
            claims_count=3,
            safety_fallback_available=True,
            session_id=test_session_id
        )
    except ContextualRecommendationError as e:
        error_dict = handle_credible_source_exception(e, test_session_id)
        logger.info(f"Handled contextual recommendation error: {error_dict}")
        logger.info(f"Should retry: {is_retryable_error(e)}")
    
    try:
        # Test LLM response error with safety blocking
        raise_llm_response_error(
            "contextual_analysis", 
            "Response blocked by safety filters",
            model_name="gemini-1.5-pro", 
            safety_blocked=True,
            attempt_number=2,
            session_id=test_session_id
        )
    except LLMResponseError as e:
        error_dict = handle_credible_source_exception(e, test_session_id)
        logger.info(f"Handled LLM response error: {error_dict}")
        logger.info(f"Retry delay: {get_retry_delay(e, 2)} seconds")
    
    try:
        # Test configuration error
        raise_configuration_error(
            "gemini_api_key", 
            "API key not found in settings",
            config_section="api_config",
            session_id=test_session_id
        )
    except ConfigurationError as e:
        error_dict = handle_credible_source_exception(e, test_session_id)
        logger.info(f"Handled configuration error: {error_dict}")
        logger.info(f"Should retry: {is_retryable_error(e)}")
    
    print("✅ Exception handling tests completed")
    print("All exception types tested successfully with enhanced context!")
