# agents/llm_explanation/exceptions.py

"""
LLM Explanation Agent Custom Exceptions

Comprehensive exception handling system for the LLM explanation agent
with detailed error categorization, metadata, and recovery guidance.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


class LLMExplanationError(Exception):
    """
    Base exception class for all LLM Explanation Agent errors.
    
    Provides structured error information with error codes, metadata,
    and recovery suggestions for robust error handling.
    """
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        Initialize base LLM explanation exception.
        
        Args:
            message: Human-readable error message
            error_code: Categorized error code for programmatic handling
            details: Additional error context and metadata
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "LLM_EXPLANATION_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


class InputValidationError(LLMExplanationError):
    """
    Exception raised when input validation fails.
    
    Used for invalid article text, missing required fields, or malformed input data.
    """
    
    def __init__(self, message: str, field_name: str = None, field_value: Any = None):
        """
        Initialize input validation error.
        
        Args:
            message: Validation error description
            field_name: Name of the invalid field
            field_value: Value that failed validation
        """
        details = {}
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = str(field_value)[:200]  # Truncate long values
        
        super().__init__(message, "INPUT_VALIDATION_ERROR", details)
        self.field_name = field_name
        self.field_value = field_value


class APIConfigurationError(LLMExplanationError):
    """
    Exception raised for API configuration issues.
    
    Handles missing API keys, invalid model configurations, and setup failures.
    """
    
    def __init__(self, message: str, config_issue: str = None):
        """
        Initialize API configuration error.
        
        Args:
            message: Configuration error description
            config_issue: Specific configuration problem
        """
        details = {'config_issue': config_issue} if config_issue else {}
        super().__init__(message, "API_CONFIGURATION_ERROR", details)
        self.config_issue = config_issue


class LLMResponseError(LLMExplanationError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles safety filter blocks, empty responses, and API failures.
    """
    
    def __init__(self, message: str, response_type: str = None, 
                 model_name: str = None, safety_blocked: bool = False):
        """
        Initialize LLM response error.
        
        Args:
            message: Response error description
            response_type: Type of response that failed
            model_name: Name of the LLM model
            safety_blocked: Whether content was blocked by safety filters
        """
        details = {}
        if response_type:
            details['response_type'] = response_type
        if model_name:
            details['model_name'] = model_name
        if safety_blocked:
            details['safety_blocked'] = True
        
        super().__init__(message, "LLM_RESPONSE_ERROR", details)
        self.response_type = response_type
        self.model_name = model_name
        self.safety_blocked = safety_blocked


class ExplanationGenerationError(LLMExplanationError):
    """
    Exception raised when explanation generation fails.
    
    Used for prompt formatting errors, generation timeouts, and processing failures.
    """
    
    def __init__(self, message: str, generation_stage: str = None, 
                 explanation_type: str = None):
        """
        Initialize explanation generation error.
        
        Args:
            message: Generation error description
            generation_stage: Stage of generation that failed
            explanation_type: Type of explanation being generated
        """
        details = {}
        if generation_stage:
            details['generation_stage'] = generation_stage
        if explanation_type:
            details['explanation_type'] = explanation_type
        
        super().__init__(message, "EXPLANATION_GENERATION_ERROR", details)
        self.generation_stage = generation_stage
        self.explanation_type = explanation_type


class RateLimitError(LLMExplanationError):
    """
    Exception raised when API rate limits are exceeded.
    
    Handles rate limiting, quota exhaustion, and throttling scenarios.
    """
    
    def __init__(self, message: str, retry_after: int = None, 
                 service: str = None):
        """
        Initialize rate limit error.
        
        Args:
            message: Rate limit error description
            retry_after: Seconds to wait before retrying
            service: Name of the service that rate limited
        """
        details = {}
        if retry_after:
            details['retry_after'] = retry_after
        if service:
            details['service'] = service
        
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.retry_after = retry_after
        self.service = service


class SourceAssessmentError(LLMExplanationError):
    """
    Exception raised when source reliability assessment fails.
    
    Used for source database errors and reliability evaluation issues.
    """
    
    def __init__(self, message: str, source_name: str = None, 
                 assessment_stage: str = None):
        """
        Initialize source assessment error.
        
        Args:
            message: Assessment error description
            source_name: Name of source being assessed
            assessment_stage: Stage of assessment that failed
        """
        details = {}
        if source_name:
            details['source_name'] = source_name
        if assessment_stage:
            details['assessment_stage'] = assessment_stage
        
        super().__init__(message, "SOURCE_ASSESSMENT_ERROR", details)
        self.source_name = source_name
        self.assessment_stage = assessment_stage


class PromptFormattingError(LLMExplanationError):
    """
    Exception raised when prompt formatting fails.
    
    Handles template errors, parameter substitution failures, and prompt validation issues.
    """
    
    def __init__(self, message: str, prompt_type: str = None, 
                 missing_params: List[str] = None):
        """
        Initialize prompt formatting error.
        
        Args:
            message: Formatting error description
            prompt_type: Type of prompt that failed
            missing_params: List of missing parameters
        """
        details = {}
        if prompt_type:
            details['prompt_type'] = prompt_type
        if missing_params:
            details['missing_params'] = missing_params
        
        super().__init__(message, "PROMPT_FORMATTING_ERROR", details)
        self.prompt_type = prompt_type
        self.missing_params = missing_params or []


class ProcessingTimeoutError(LLMExplanationError):
    """
    Exception raised when explanation processing exceeds time limits.
    
    Used for long-running explanations, API timeouts, and processing deadlines.
    """
    
    def __init__(self, message: str, timeout_seconds: float = None, 
                 operation: str = None):
        """
        Initialize processing timeout error.
        
        Args:
            message: Timeout error description
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


class DataFormatError(LLMExplanationError):
    """
    Exception raised when data format validation fails.
    
    Used for metadata parsing errors, response format issues, and data structure problems.
    """
    
    def __init__(self, message: str, data_type: str = None, 
                 expected_format: str = None):
        """
        Initialize data format error.
        
        Args:
            message: Format error description
            data_type: Type of data that failed validation
            expected_format: Expected data format description
        """
        details = {}
        if data_type:
            details['data_type'] = data_type
        if expected_format:
            details['expected_format'] = expected_format
        
        super().__init__(message, "DATA_FORMAT_ERROR", details)
        self.data_type = data_type
        self.expected_format = expected_format


# Convenience functions for common error scenarios
def raise_input_validation_error(field_name: str, message: str, field_value: Any = None) -> None:
    """Raise standardized input validation error."""
    raise InputValidationError(f"Invalid {field_name}: {message}", field_name, field_value)


def raise_api_configuration_error(message: str, config_issue: str = None) -> None:
    """Raise standardized API configuration error."""
    raise APIConfigurationError(f"API configuration error: {message}", config_issue)


def raise_llm_response_error(response_type: str, message: str, 
                           model_name: str = None, safety_blocked: bool = False) -> None:
    """Raise standardized LLM response error."""
    raise LLMResponseError(f"LLM {response_type} failed: {message}", response_type, model_name, safety_blocked)


def raise_explanation_generation_error(generation_stage: str, message: str, 
                                     explanation_type: str = None) -> None:
    """Raise standardized explanation generation error."""
    raise ExplanationGenerationError(
        f"Explanation generation error ({generation_stage}): {message}", 
        generation_stage, explanation_type
    )


# Error handling utilities
def handle_llm_explanation_exception(exception: Exception) -> LLMExplanationError:
    """
    Convert any exception to a standardized LLM explanation error.
    
    Args:
        exception: Exception to convert
        
    Returns:
        Standardized LLMExplanationError instance
    """
    if isinstance(exception, LLMExplanationError):
        return exception
    
    # Map common exception types
    error_mappings = {
        KeyError: lambda e: InputValidationError(f"Missing required key: {str(e)}"),
        ValueError: lambda e: InputValidationError(f"Invalid value: {str(e)}"),
        TypeError: lambda e: DataFormatError(f"Type error: {str(e)}"),
        TimeoutError: lambda e: ProcessingTimeoutError(f"Operation timed out: {str(e)}"),
        ConnectionError: lambda e: APIConfigurationError(f"Connection failed: {str(e)}"),
    }
    
    exception_type = type(exception)
    if exception_type in error_mappings:
        return error_mappings[exception_type](exception)
    
    # Default fallback
    return LLMExplanationError(f"Unexpected error: {str(exception)}", "UNEXPECTED_ERROR")


def is_recoverable_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a recoverable error.
    
    Args:
        exception: Exception to evaluate
        
    Returns:
        True if error is recoverable (retry possible), False otherwise
    """
    recoverable_types = (
        RateLimitError,
        ProcessingTimeoutError,
        LLMResponseError,
        APIConfigurationError
    )
    
    if isinstance(exception, recoverable_types):
        # Additional checks for specific error types
        if isinstance(exception, LLMResponseError):
            # Safety blocked content may not be recoverable
            return not exception.safety_blocked
        return True
    
    return False


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
        return 5.0  # Standard retry delay
    elif isinstance(exception, LLMResponseError) and not exception.safety_blocked:
        return 2.0  # Quick retry for non-safety issues
    elif isinstance(exception, APIConfigurationError):
        return 10.0  # Longer delay for config issues
    
    return None


def get_error_recovery_suggestion(exception: Exception) -> Optional[str]:
    """
    Get recovery suggestion for specific error types.
    
    Args:
        exception: Exception that occurred
        
    Returns:
        Human-readable recovery suggestion
    """
    if isinstance(exception, InputValidationError):
        return "Validate input data and ensure all required fields are provided"
    elif isinstance(exception, APIConfigurationError):
        return "Check API key configuration and model settings"
    elif isinstance(exception, LLMResponseError):
        if exception.safety_blocked:
            return "Content was blocked by safety filters - try rephrasing or using different content"
        else:
            return "Try the request again or use alternative prompt formatting"
    elif isinstance(exception, RateLimitError):
        retry_time = exception.retry_after or 60
        return f"Rate limit exceeded - wait {retry_time} seconds before retrying"
    elif isinstance(exception, ProcessingTimeoutError):
        return "Request timed out - try with shorter content or increase timeout limits"
    elif isinstance(exception, SourceAssessmentError):
        return "Source assessment failed - verify source name and database availability"
    elif isinstance(exception, PromptFormattingError):
        missing = ", ".join(exception.missing_params) if exception.missing_params else "parameters"
        return f"Fix prompt formatting issues - check {missing}"
    else:
        return "Review error details and try again"


# Error categorization for monitoring and alerting
def categorize_error_severity(exception: Exception) -> str:
    """
    Categorize error severity for monitoring and alerting.
    
    Args:
        exception: Exception to categorize
        
    Returns:
        Severity level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
    """
    if isinstance(exception, (InputValidationError, PromptFormattingError, DataFormatError)):
        return 'LOW'  # User/input errors
    elif isinstance(exception, (RateLimitError, ProcessingTimeoutError)):
        return 'MEDIUM'  # Recoverable system issues
    elif isinstance(exception, (LLMResponseError, SourceAssessmentError)):
        return 'HIGH'  # Service functionality issues
    elif isinstance(exception, APIConfigurationError):
        return 'CRITICAL'  # System configuration problems
    else:
        return 'MEDIUM'  # Unknown issues default to medium


# Testing functionality
if __name__ == "__main__":
    """Test LLM explanation exception functionality."""
    
    # Test input validation error
    try:
        raise_input_validation_error("article_text", "Text is too short", "")
    except InputValidationError as e:
        print("Input Validation Error:", e.to_dict())
    
    # Test LLM response error
    try:
        raise_llm_response_error("explanation", "Content blocked by safety filters", 
                               "gemini-1.5-pro", safety_blocked=True)
    except LLMResponseError as e:
        print("LLM Response Error:", e.to_dict())
        print(f"Is recoverable: {is_recoverable_error(e)}")
        print(f"Recovery suggestion: {get_error_recovery_suggestion(e)}")
    
    # Test rate limit error
    rate_error = RateLimitError("API rate limit exceeded", retry_after=120, service="gemini")
    print(f"Rate Limit - Retry delay: {get_retry_delay(rate_error)} seconds")
    print(f"Severity: {categorize_error_severity(rate_error)}")
    
    # Test exception conversion
    try:
        raise KeyError("missing_field")
    except Exception as e:
        converted = handle_llm_explanation_exception(e)
        print("Converted Exception:", converted.to_dict())
    
    print("\n=== LLM EXPLANATION EXCEPTIONS TESTING COMPLETED ===")
