# agents/evidence_evaluator/exceptions.py

"""
Evidence Evaluator Custom Exceptions

Custom exception classes for the Evidence Evaluator Agent providing
specific error handling for evidence evaluation workflows.
"""

from typing import Any, Dict, List, Optional


class EvidenceEvaluatorError(Exception):
    """
    Base exception class for all Evidence Evaluator Agent errors.
    
    All custom exceptions in the evidence evaluator module should inherit
    from this base class to provide consistent error handling.
    """
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        Initialize base evidence evaluator exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "EVIDENCE_EVALUATOR_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


class InputValidationError(EvidenceEvaluatorError):
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


class LLMResponseError(EvidenceEvaluatorError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles safety filter blocks, invalid responses, and API communication failures.
    """
    
    def __init__(self, message: str, response_type: str = None, model_name: str = None):
        """
        Initialize LLM response error.
        
        Args:
            message: Error description
            response_type: Type of response that failed (verification, analysis, etc.)
            model_name: Name of the LLM model that generated the error
        """
        details = {}
        if response_type:
            details['response_type'] = response_type
        if model_name:
            details['model_name'] = model_name
        
        super().__init__(message, "LLM_RESPONSE_ERROR", details)
        self.response_type = response_type
        self.model_name = model_name


class PromptGenerationError(EvidenceEvaluatorError):
    """
    Exception raised when prompt generation or formatting fails.
    
    Used when prompt templates are invalid or prompt parameter substitution fails.
    """
    
    def __init__(self, message: str, prompt_type: str = None, parameters: Dict[str, Any] = None):
        """
        Initialize prompt generation error.
        
        Args:
            message: Error description
            prompt_type: Type of prompt that failed to generate
            parameters: Parameters that caused the generation failure
        """
        details = {}
        if prompt_type:
            details['prompt_type'] = prompt_type
        if parameters:
            details['parameters'] = {k: str(v)[:100] for k, v in parameters.items()}
        
        super().__init__(message, "PROMPT_GENERATION_ERROR", details)
        self.prompt_type = prompt_type
        self.parameters = parameters


class VerificationSourceError(EvidenceEvaluatorError):
    """
    Exception raised when verification source generation or validation fails.
    
    Used for URL validation failures, source parsing errors, and quality assessment issues.
    """
    
    def __init__(self, message: str, source_url: str = None, source_type: str = None):
        """
        Initialize verification source error.
        
        Args:
            message: Error description
            source_url: URL that caused the error
            source_type: Type of verification source that failed
        """
        details = {}
        if source_url:
            details['source_url'] = source_url
        if source_type:
            details['source_type'] = source_type
        
        super().__init__(message, "VERIFICATION_SOURCE_ERROR", details)
        self.source_url = source_url
        self.source_type = source_type


class EvidenceAssessmentError(EvidenceEvaluatorError):
    """
    Exception raised when evidence quality assessment fails.
    
    Used for criteria evaluation failures, scoring calculation errors, and assessment logic issues.
    """
    
    def __init__(self, message: str, assessment_type: str = None, score_component: str = None):
        """
        Initialize evidence assessment error.
        
        Args:
            message: Error description
            assessment_type: Type of assessment that failed (quality, logical, etc.)
            score_component: Specific scoring component that caused the error
        """
        details = {}
        if assessment_type:
            details['assessment_type'] = assessment_type
        if score_component:
            details['score_component'] = score_component
        
        super().__init__(message, "EVIDENCE_ASSESSMENT_ERROR", details)
        self.assessment_type = assessment_type
        self.score_component = score_component


class FallacyDetectionError(EvidenceEvaluatorError):
    """
    Exception raised when logical fallacy detection fails.
    
    Used for pattern matching failures, scoring calculation errors, and fallacy analysis issues.
    """
    
    def __init__(self, message: str, fallacy_type: str = None, detection_stage: str = None):
        """
        Initialize fallacy detection error.
        
        Args:
            message: Error description
            fallacy_type: Type of fallacy that caused detection to fail
            detection_stage: Stage of detection process that failed
        """
        details = {}
        if fallacy_type:
            details['fallacy_type'] = fallacy_type
        if detection_stage:
            details['detection_stage'] = detection_stage
        
        super().__init__(message, "FALLACY_DETECTION_ERROR", details)
        self.fallacy_type = fallacy_type
        self.detection_stage = detection_stage


class ConfigurationError(EvidenceEvaluatorError):
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


class RateLimitError(EvidenceEvaluatorError):
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


class ProcessingTimeoutError(EvidenceEvaluatorError):
    """
    Exception raised when evidence evaluation exceeds time limits.
    
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


class DataFormatError(EvidenceEvaluatorError):
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


def raise_llm_response_error(response_type: str, message: str, model_name: str = None) -> None:
    """
    Raise a standardized LLM response error.
    
    Args:
        response_type: Type of LLM response that failed
        message: Error description
        model_name: Optional model name
    """
    raise LLMResponseError(f"LLM {response_type} failed: {message}", response_type, model_name)


def raise_verification_source_error(source_url: str, message: str, source_type: str = None) -> None:
    """
    Raise a standardized verification source error.
    
    Args:
        source_url: URL that caused the error
        message: Error description
        source_type: Optional source type
    """
    raise VerificationSourceError(f"Verification source error for {source_url}: {message}", source_url, source_type)


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
def handle_evidence_evaluator_exception(exception: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format.
    
    Args:
        exception: Exception to handle
        
    Returns:
        Dictionary with standardized error information
    """
    if isinstance(exception, EvidenceEvaluatorError):
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
        LLMResponseError
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
    else:
        return None
