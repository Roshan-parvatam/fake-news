# agents/evidence_evaluator/exceptions.py

"""
Evidence Evaluator Custom Exceptions - Production Ready

Enhanced exception classes for production-level error handling with
retry logic support, detailed error context, and proper categorization.
"""

from typing import Any, Dict, List, Optional
import time

class EvidenceEvaluatorError(Exception):
    """
    Base exception class for all Evidence Evaluator Agent errors.
    
    Provides structured error information for logging and debugging
    with support for retry logic and error categorization.
    """

    def __init__(self, 
                 message: str, 
                 error_code: str = None, 
                 details: Dict[str, Any] = None,
                 retryable: bool = False,
                 session_id: str = None):
        """
        Initialize base evidence evaluator exception.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
            retryable: Whether this error should trigger retry logic
            session_id: Optional session ID for tracking across operations
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "EVIDENCE_EVALUATOR_ERROR"
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


class InputValidationError(EvidenceEvaluatorError):
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


class LLMResponseError(EvidenceEvaluatorError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles safety filter blocks, invalid responses, API communication failures,
    and timeout issues. These errors are typically retryable.
    """

    def __init__(self, 
                 message: str, 
                 response_type: str = None, 
                 model_name: str = None,
                 attempt_number: int = 1,
                 session_id: str = None):
        """
        Initialize LLM response error.

        Args:
            message: Error description
            response_type: Type of response that failed (verification, analysis, etc.)
            model_name: Name of the LLM model that generated the error
            attempt_number: Which retry attempt this error occurred on
            session_id: Optional session ID for tracking
        """
        details = {}
        if response_type:
            details['response_type'] = response_type
        if model_name:
            details['model_name'] = model_name
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
        self.attempt_number = attempt_number


class APIError(EvidenceEvaluatorError):
    """
    Exception raised when API calls fail due to network or service issues.
    
    Used for connection timeouts, network errors, and service unavailability.
    These errors are retryable with exponential backoff.
    """

    def __init__(self, 
                 message: str, 
                 api_endpoint: str = None, 
                 status_code: int = None,
                 response_body: str = None,
                 session_id: str = None):
        """
        Initialize API error.

        Args:
            message: Error description
            api_endpoint: API endpoint that failed
            status_code: HTTP status code if available
            response_body: Response body for debugging (truncated)
            session_id: Optional session ID for tracking
        """
        details = {}
        if api_endpoint:
            details['api_endpoint'] = api_endpoint
        if status_code:
            details['status_code'] = status_code
        if response_body:
            # Truncate response body for logging
            details['response_body'] = response_body[:500] if len(response_body) > 500 else response_body

        super().__init__(
            message, 
            "API_ERROR", 
            details,
            retryable=True,  # API errors are retryable
            session_id=session_id
        )
        self.api_endpoint = api_endpoint
        self.status_code = status_code


class PromptGenerationError(EvidenceEvaluatorError):
    """
    Exception raised when prompt generation or formatting fails.
    
    Used when prompt templates are invalid or prompt parameter substitution fails.
    These errors are typically not retryable without fixing the template.
    """

    def __init__(self, 
                 message: str, 
                 prompt_type: str = None, 
                 parameters: Dict[str, Any] = None,
                 session_id: str = None):
        """
        Initialize prompt generation error.

        Args:
            message: Error description
            prompt_type: Type of prompt that failed to generate
            parameters: Parameters that caused the generation failure
            session_id: Optional session ID for tracking
        """
        details = {}
        if prompt_type:
            details['prompt_type'] = prompt_type
        if parameters:
            # Truncate parameter values for logging
            details['parameters'] = {
                k: str(v)[:100] if len(str(v)) > 100 else str(v) 
                for k, v in parameters.items()
            }

        super().__init__(
            message, 
            "PROMPT_GENERATION_ERROR", 
            details,
            retryable=False,  # Prompt errors are not retryable without fixing template
            session_id=session_id
        )
        self.prompt_type = prompt_type
        self.parameters = parameters


class VerificationSourceError(EvidenceEvaluatorError):
    """
    Exception raised when verification source generation or validation fails.
    
    Used for URL validation failures, source parsing errors, and quality assessment issues.
    Some of these errors may be retryable depending on the cause.
    """

    def __init__(self, 
                 message: str, 
                 source_url: str = None, 
                 source_type: str = None,
                 validation_stage: str = None,
                 session_id: str = None):
        """
        Initialize verification source error.

        Args:
            message: Error description
            source_url: URL that caused the error
            source_type: Type of verification source that failed
            validation_stage: Stage where validation failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if source_url:
            details['source_url'] = source_url
        if source_type:
            details['source_type'] = source_type
        if validation_stage:
            details['validation_stage'] = validation_stage

        # URL parsing errors are not retryable, but generation errors might be
        retryable = validation_stage in ['generation', 'api_parsing']

        super().__init__(
            message, 
            "VERIFICATION_SOURCE_ERROR", 
            details,
            retryable=retryable,
            session_id=session_id
        )
        self.source_url = source_url
        self.source_type = source_type
        self.validation_stage = validation_stage


class EvidenceAssessmentError(EvidenceEvaluatorError):
    """
    Exception raised when evidence quality assessment fails.
    
    Used for criteria evaluation failures, scoring calculation errors, and assessment logic issues.
    These are typically not retryable as they indicate logic or data issues.
    """

    def __init__(self, 
                 message: str, 
                 assessment_type: str = None, 
                 score_component: str = None,
                 session_id: str = None):
        """
        Initialize evidence assessment error.

        Args:
            message: Error description
            assessment_type: Type of assessment that failed (quality, logical, etc.)
            score_component: Specific scoring component that caused the error
            session_id: Optional session ID for tracking
        """
        details = {}
        if assessment_type:
            details['assessment_type'] = assessment_type
        if score_component:
            details['score_component'] = score_component

        super().__init__(
            message, 
            "EVIDENCE_ASSESSMENT_ERROR", 
            details,
            retryable=False,  # Assessment errors are typically logic issues
            session_id=session_id
        )
        self.assessment_type = assessment_type
        self.score_component = score_component


class FallacyDetectionError(EvidenceEvaluatorError):
    """
    Exception raised when logical fallacy detection fails.
    
    Used for pattern matching failures, scoring calculation errors, and fallacy analysis issues.
    These are typically not retryable as they indicate logic or data issues.
    """

    def __init__(self, 
                 message: str, 
                 fallacy_type: str = None, 
                 detection_stage: str = None,
                 session_id: str = None):
        """
        Initialize fallacy detection error.

        Args:
            message: Error description
            fallacy_type: Type of fallacy that caused detection to fail
            detection_stage: Stage of detection process that failed
            session_id: Optional session ID for tracking
        """
        details = {}
        if fallacy_type:
            details['fallacy_type'] = fallacy_type
        if detection_stage:
            details['detection_stage'] = detection_stage

        super().__init__(
            message, 
            "FALLACY_DETECTION_ERROR", 
            details,
            retryable=False,  # Fallacy detection errors are typically logic issues
            session_id=session_id
        )
        self.fallacy_type = fallacy_type
        self.detection_stage = detection_stage


class ConfigurationError(EvidenceEvaluatorError):
    """
    Exception raised when agent configuration is invalid or missing.
    
    Used for missing API keys, invalid config values, and setup failures.
    These errors are not retryable without fixing the configuration.
    """

    def __init__(self, 
                 message: str, 
                 config_key: str = None, 
                 config_value: Any = None,
                 session_id: str = None):
        """
        Initialize configuration error.

        Args:
            message: Error description
            config_key: Configuration key that is invalid or missing
            config_value: Invalid configuration value (sanitized for logging)
            session_id: Optional session ID for tracking
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
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


class RateLimitError(EvidenceEvaluatorError):
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
                 session_id: str = None):
        """
        Initialize rate limit error.

        Args:
            message: Error description
            retry_after: Seconds to wait before retrying (if known)
            service: Name of the service that rate limited the request
            requests_remaining: Number of requests remaining in quota
            session_id: Optional session ID for tracking
        """
        details = {}
        if retry_after:
            details['retry_after'] = retry_after
        if service:
            details['service'] = service
        if requests_remaining is not None:
            details['requests_remaining'] = requests_remaining

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


class ProcessingTimeoutError(EvidenceEvaluatorError):
    """
    Exception raised when evidence evaluation exceeds time limits.
    
    Used for long-running analysis, API timeouts, and processing deadlines.
    These errors may be retryable depending on the timeout type.
    """

    def __init__(self, 
                 message: str, 
                 timeout_seconds: float = None, 
                 operation: str = None,
                 elapsed_time: float = None,
                 session_id: str = None):
        """
        Initialize processing timeout error.

        Args:
            message: Error description
            timeout_seconds: Timeout limit that was exceeded
            operation: Operation that timed out
            elapsed_time: How long the operation actually took
            session_id: Optional session ID for tracking
        """
        details = {}
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        if elapsed_time:
            details['elapsed_time'] = elapsed_time

        # API timeouts are retryable, processing timeouts might not be
        retryable = operation in ['api_call', 'llm_generation']

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


class DataFormatError(EvidenceEvaluatorError):
    """
    Exception raised when data parsing or formatting fails.
    
    Used for JSON parsing errors, response format issues, and data structure problems.
    These errors may be retryable if they're due to transient API response issues.
    """

    def __init__(self, 
                 message: str, 
                 data_type: str = None, 
                 expected_format: str = None,
                 actual_data: str = None,
                 session_id: str = None):
        """
        Initialize data format error.

        Args:
            message: Error description
            data_type: Type of data that failed to parse
            expected_format: Expected format description
            actual_data: Sample of actual data received (truncated)
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

        # Data format errors from API responses might be retryable
        retryable = data_type in ['api_response', 'llm_output']

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
                           attempt_number: int = 1, session_id: str = None) -> None:
    """
    Raise a standardized LLM response error with retry context.

    Args:
        response_type: Type of LLM response that failed
        message: Error description
        model_name: Optional model name
        attempt_number: Which retry attempt this error occurred on
        session_id: Optional session ID for tracking
    """
    raise LLMResponseError(
        f"LLM {response_type} failed: {message}", 
        response_type, 
        model_name, 
        attempt_number, 
        session_id
    )


def raise_api_error(endpoint: str, message: str, status_code: int = None, 
                   response_body: str = None, session_id: str = None) -> None:
    """
    Raise a standardized API error with detailed context.

    Args:
        endpoint: API endpoint that failed
        message: Error description
        status_code: HTTP status code if available
        response_body: Response body for debugging
        session_id: Optional session ID for tracking
    """
    raise APIError(
        f"API call to {endpoint} failed: {message}", 
        endpoint, 
        status_code, 
        response_body, 
        session_id
    )


def raise_verification_source_error(source_url: str, message: str, source_type: str = None, 
                                  validation_stage: str = None, session_id: str = None) -> None:
    """
    Raise a standardized verification source error with context.

    Args:
        source_url: URL that caused the error
        message: Error description
        source_type: Optional source type
        validation_stage: Stage where validation failed
        session_id: Optional session ID for tracking
    """
    raise VerificationSourceError(
        f"Verification source error for {source_url}: {message}", 
        source_url, 
        source_type, 
        validation_stage, 
        session_id
    )


def raise_configuration_error(config_key: str, message: str, config_value: Any = None, 
                            session_id: str = None) -> None:
    """
    Raise a standardized configuration error with context.

    Args:
        config_key: Configuration key that is problematic
        message: Error description
        config_value: Optional configuration value
        session_id: Optional session ID for tracking
    """
    raise ConfigurationError(
        f"Configuration error for {config_key}: {message}", 
        config_key, 
        config_value, 
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
    parameters = {}
    if generation_stage:
        parameters['generation_stage'] = generation_stage
    if template_name:
        parameters['template_name'] = template_name
    
    raise PromptGenerationError(
        f"Prompt generation failed for {prompt_type}: {message}", 
        prompt_type, 
        parameters, 
        session_id
    )


# Enhanced exception handling utilities

def handle_evidence_evaluator_exception(exception: Exception, session_id: str = None) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format with enhanced context.

    Args:
        exception: Exception to handle
        session_id: Optional session ID for tracking

    Returns:
        Dictionary with standardized error information
    """
    if isinstance(exception, EvidenceEvaluatorError):
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
    if isinstance(exception, EvidenceEvaluatorError):
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
        APIError: 2.0,
        LLMResponseError: 1.5,
        ProcessingTimeoutError: 3.0,
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
    
    if isinstance(exception, ConfigurationError):
        # Never retry configuration errors
        return False
    
    return True


# Testing functionality
if __name__ == "__main__":
    """Test exception handling functionality."""
    import logging
    
    # Setup logging to see exception details
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test different exception types
    test_session_id = "test_session_123"
    
    try:
        # Test retryable exception
        raise LLMResponseError(
            "API response was blocked by safety filter",
            response_type="verification_sources",
            model_name="gemini-1.5-pro",
            attempt_number=2,
            session_id=test_session_id
        )
    except Exception as e:
        error_dict = handle_evidence_evaluator_exception(e, test_session_id)
        logger.info(f"Handled retryable error: {error_dict}")
        logger.info(f"Should retry: {is_retryable_error(e)}")
        logger.info(f"Retry delay: {get_retry_delay(e, 2)} seconds")
    
    try:
        # Test non-retryable exception
        raise InputValidationError(
            "Article text is too short",
            field_name="text",
            field_value="Short text",
            session_id=test_session_id
        )
    except Exception as e:
        error_dict = handle_evidence_evaluator_exception(e, test_session_id)
        logger.info(f"Handled non-retryable error: {error_dict}")
        logger.info(f"Should retry: {is_retryable_error(e)}")
    
    print("✅ Exception handling tests completed")
