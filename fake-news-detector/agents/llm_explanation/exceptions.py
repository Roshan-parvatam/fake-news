# agents/llm_explanation/exceptions.py

"""
LLM Explanation Agent Custom Exceptions - Production Ready

Comprehensive exception handling system for the LLM explanation agent with
detailed error categorization, recovery guidance, monitoring integration,
and production-ready error management for reliable operation in professional
fact-checking environments.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for systematic classification."""
    INPUT_VALIDATION = "input_validation"
    API_CONFIGURATION = "api_configuration"
    LLM_RESPONSE = "llm_response"
    EXPLANATION_GENERATION = "explanation_generation"
    RATE_LIMITING = "rate_limiting"
    SOURCE_ASSESSMENT = "source_assessment"
    PROMPT_FORMATTING = "prompt_formatting"
    PROCESSING_TIMEOUT = "processing_timeout"
    DATA_FORMAT = "data_format"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorContext:
    """Enhanced error context with detailed metadata."""
    session_id: Optional[str] = None
    operation: Optional[str] = None
    model_used: Optional[str] = None
    input_size: Optional[int] = None
    processing_time: Optional[float] = None
    retry_count: Optional[int] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and monitoring."""
        return {
            'session_id': self.session_id,
            'operation': self.operation,
            'model_used': self.model_used,
            'input_size': self.input_size,
            'processing_time': self.processing_time,
            'retry_count': self.retry_count,
            'user_id': self.user_id
        }


class LLMExplanationError(Exception):
    """
    Base exception class for all LLM Explanation Agent errors.
    
    Provides structured error information with error codes, detailed metadata,
    recovery suggestions, and monitoring integration for comprehensive
    error handling in production environments.
    """

    def __init__(self, 
                 message: str, 
                 error_code: str = None, 
                 details: Dict[str, Any] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                 context: ErrorContext = None,
                 recoverable: bool = False,
                 retry_delay: Optional[float] = None,
                 suggestion: str = None):
        """
        Initialize enhanced LLM explanation exception.

        Args:
            message: Human-readable error message
            error_code: Categorized error code for programmatic handling
            details: Additional error context and metadata
            severity: Error severity level for monitoring
            category: Error category for systematic classification
            context: Enhanced error context with operational details
            recoverable: Whether error can be recovered through retry
            retry_delay: Suggested delay before retry (seconds)
            suggestion: Actionable recovery suggestion
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "LLM_EXPLANATION_ERROR"
        self.details = details or {}
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.retry_delay = retry_delay
        self.suggestion = suggestion
        self.timestamp = datetime.now().isoformat()
        
        # Logging integration
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._log_exception()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to comprehensive dictionary for API responses and logging."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context.to_dict() if self.context else {},
            'recoverable': self.recoverable,
            'retry_delay': self.retry_delay,
            'suggestion': self.suggestion,
            'timestamp': self.timestamp
        }

    def to_user_message(self) -> str:
        """Generate user-friendly error message."""
        if self.suggestion:
            return f"{self.message} Suggestion: {self.suggestion}"
        return self.message

    def _log_exception(self) -> None:
        """Log exception with appropriate level based on severity."""
        extra = {
            'error_code': self.error_code,
            'severity': self.severity.value,
            'category': self.category.value,
            'recoverable': self.recoverable
        }
        
        if self.context:
            extra.update(self.context.to_dict())

        if self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Exception: {self.message}", extra=extra)
        elif self.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Exception: {self.message}", extra=extra)
        else:
            self.logger.info(f"Exception: {self.message}", extra=extra)

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', code='{self.error_code}', severity='{self.severity.value}')"


class InputValidationError(LLMExplanationError):
    """
    Exception raised when input validation fails.
    
    Used for invalid article text, missing required fields, malformed input data,
    or security validation failures during input processing.
    """

    def __init__(self, 
                 message: str, 
                 field_name: str = None, 
                 field_value: Any = None,
                 validation_rules: List[str] = None,
                 context: ErrorContext = None):
        """
        Initialize input validation error.

        Args:
            message: Validation error description
            field_name: Name of the invalid field
            field_value: Value that failed validation (truncated for logging)
            validation_rules: List of validation rules that failed
            context: Enhanced error context
        """
        details = {
            'field_name': field_name,
            'field_value': str(field_value)[:200] if field_value is not None else None,
            'validation_rules': validation_rules or []
        }
        
        suggestion = f"Correct the '{field_name}' field and retry" if field_name else "Review and correct input data"
        
        super().__init__(
            message=message,
            error_code="INPUT_VALIDATION_ERROR",
            details=details,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.INPUT_VALIDATION,
            context=context,
            recoverable=True,
            suggestion=suggestion
        )
        
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rules = validation_rules or []


class APIConfigurationError(LLMExplanationError):
    """
    Exception raised for API configuration issues.
    
    Handles missing API keys, invalid model configurations, network connectivity
    issues, and service setup failures requiring system-level intervention.
    """

    def __init__(self, 
                 message: str, 
                 config_issue: str = None,
                 service_name: str = None,
                 context: ErrorContext = None):
        """
        Initialize API configuration error.

        Args:
            message: Configuration error description
            config_issue: Specific configuration problem identifier
            service_name: Name of the affected service
            context: Enhanced error context
        """
        details = {
            'config_issue': config_issue,
            'service_name': service_name
        }
        
        suggestion = "Check API configuration, keys, and network connectivity"
        
        super().__init__(
            message=message,
            error_code="API_CONFIGURATION_ERROR",
            details=details,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.API_CONFIGURATION,
            context=context,
            recoverable=True,
            retry_delay=30.0,
            suggestion=suggestion
        )
        
        self.config_issue = config_issue
        self.service_name = service_name


class LLMResponseError(LLMExplanationError):
    """
    Exception raised when LLM API responses are invalid, blocked, or failed.
    
    Handles safety filter blocks, empty responses, malformed API responses,
    and model-specific errors during content generation.
    """

    def __init__(self, 
                 message: str, 
                 response_type: str = None,
                 model_name: str = None, 
                 safety_blocked: bool = False,
                 finish_reason: str = None,
                 context: ErrorContext = None):
        """
        Initialize LLM response error.

        Args:
            message: Response error description
            response_type: Type of response that failed
            model_name: Name of the LLM model
            safety_blocked: Whether content was blocked by safety filters
            finish_reason: API finish reason code
            context: Enhanced error context
        """
        details = {
            'response_type': response_type,
            'model_name': model_name,
            'safety_blocked': safety_blocked,
            'finish_reason': finish_reason
        }
        
        if safety_blocked:
            suggestion = "Modify prompt to avoid restricted content patterns"
            recoverable = False
        else:
            suggestion = "Retry with adjusted parameters or different model"
            recoverable = True
        
        super().__init__(
            message=message,
            error_code="LLM_RESPONSE_ERROR",
            details=details,
            severity=ErrorSeverity.HIGH if safety_blocked else ErrorSeverity.MEDIUM,
            category=ErrorCategory.LLM_RESPONSE,
            context=context,
            recoverable=recoverable,
            retry_delay=2.0 if not safety_blocked else None,
            suggestion=suggestion
        )
        
        self.response_type = response_type
        self.model_name = model_name
        self.safety_blocked = safety_blocked
        self.finish_reason = finish_reason


class ExplanationGenerationError(LLMExplanationError):
    """
    Exception raised when explanation generation fails.
    
    Used for prompt formatting errors, generation timeouts, processing failures,
    and quality issues during explanation content creation.
    """

    def __init__(self, 
                 message: str, 
                 generation_stage: str = None,
                 explanation_type: str = None,
                 processing_time: float = None,
                 context: ErrorContext = None):
        """
        Initialize explanation generation error.

        Args:
            message: Generation error description
            generation_stage: Stage of generation that failed
            explanation_type: Type of explanation being generated
            processing_time: Time spent before failure
            context: Enhanced error context
        """
        details = {
            'generation_stage': generation_stage,
            'explanation_type': explanation_type,
            'processing_time': processing_time
        }
        
        suggestion = f"Review {explanation_type} generation parameters and retry" if explanation_type else "Check generation configuration and retry"
        
        super().__init__(
            message=message,
            error_code="EXPLANATION_GENERATION_ERROR",
            details=details,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.EXPLANATION_GENERATION,
            context=context,
            recoverable=True,
            retry_delay=3.0,
            suggestion=suggestion
        )
        
        self.generation_stage = generation_stage
        self.explanation_type = explanation_type
        self.processing_time = processing_time


class RateLimitError(LLMExplanationError):
    """
    Exception raised when API rate limits are exceeded.
    
    Handles rate limiting, quota exhaustion, throttling scenarios, and
    provides intelligent retry scheduling based on service limits.
    """

    def __init__(self, 
                 message: str, 
                 retry_after: int = None,
                 service: str = None,
                 rate_limit_type: str = None,
                 context: ErrorContext = None):
        """
        Initialize rate limit error.

        Args:
            message: Rate limit error description
            retry_after: Seconds to wait before retrying (from API headers)
            service: Name of the rate-limited service
            rate_limit_type: Type of rate limit (requests, tokens, etc.)
            context: Enhanced error context
        """
        details = {
            'retry_after': retry_after,
            'service': service,
            'rate_limit_type': rate_limit_type
        }
        
        suggestion = f"Wait {retry_after or 60} seconds before retrying" if retry_after else "Implement exponential backoff retry strategy"
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=details,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RATE_LIMITING,
            context=context,
            recoverable=True,
            retry_delay=float(retry_after) if retry_after else 60.0,
            suggestion=suggestion
        )
        
        self.retry_after = retry_after
        self.service = service
        self.rate_limit_type = rate_limit_type


class SourceAssessmentError(LLMExplanationError):
    """
    Exception raised when source reliability assessment fails.
    
    Used for source database errors, reliability evaluation issues, and
    problems during source credibility analysis processing.
    """

    def __init__(self, 
                 message: str, 
                 source_name: str = None,
                 assessment_stage: str = None,
                 database_version: str = None,
                 context: ErrorContext = None):
        """
        Initialize source assessment error.

        Args:
            message: Assessment error description
            source_name: Name of source being assessed
            assessment_stage: Stage of assessment that failed
            database_version: Version of source database used
            context: Enhanced error context
        """
        details = {
            'source_name': source_name,
            'assessment_stage': assessment_stage,
            'database_version': database_version
        }
        
        suggestion = "Verify source information and check database connectivity"
        
        super().__init__(
            message=message,
            error_code="SOURCE_ASSESSMENT_ERROR",
            details=details,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SOURCE_ASSESSMENT,
            context=context,
            recoverable=True,
            retry_delay=5.0,
            suggestion=suggestion
        )
        
        self.source_name = source_name
        self.assessment_stage = assessment_stage
        self.database_version = database_version


class PromptFormattingError(LLMExplanationError):
    """
    Exception raised when prompt formatting fails.
    
    Handles template errors, parameter substitution failures, prompt validation
    issues, and malformed prompt construction problems.
    """

    def __init__(self, 
                 message: str, 
                 prompt_type: str = None,
                 missing_params: List[str] = None,
                 template_version: str = None,
                 context: ErrorContext = None):
        """
        Initialize prompt formatting error.

        Args:
            message: Formatting error description
            prompt_type: Type of prompt that failed formatting
            missing_params: List of missing required parameters
            template_version: Version of prompt template used
            context: Enhanced error context
        """
        details = {
            'prompt_type': prompt_type,
            'missing_params': missing_params or [],
            'template_version': template_version
        }
        
        if missing_params:
            suggestion = f"Provide missing parameters: {', '.join(missing_params)}"
        else:
            suggestion = "Check prompt template syntax and parameter formatting"
        
        super().__init__(
            message=message,
            error_code="PROMPT_FORMATTING_ERROR",
            details=details,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.PROMPT_FORMATTING,
            context=context,
            recoverable=True,
            suggestion=suggestion
        )
        
        self.prompt_type = prompt_type
        self.missing_params = missing_params or []
        self.template_version = template_version


class ProcessingTimeoutError(LLMExplanationError):
    """
    Exception raised when explanation processing exceeds time limits.
    
    Used for long-running explanations, API timeouts, processing deadlines,
    and resource exhaustion during explanation generation.
    """

    def __init__(self, 
                 message: str, 
                 timeout_seconds: float = None,
                 operation: str = None,
                 elapsed_time: float = None,
                 context: ErrorContext = None):
        """
        Initialize processing timeout error.

        Args:
            message: Timeout error description
            timeout_seconds: Timeout limit that was exceeded
            operation: Operation that timed out
            elapsed_time: Actual time elapsed before timeout
            context: Enhanced error context
        """
        details = {
            'timeout_seconds': timeout_seconds,
            'operation': operation,
            'elapsed_time': elapsed_time
        }
        
        suggestion = "Try with shorter input, increase timeout, or split processing"
        
        super().__init__(
            message=message,
            error_code="PROCESSING_TIMEOUT_ERROR",
            details=details,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING_TIMEOUT,
            context=context,
            recoverable=True,
            retry_delay=5.0,
            suggestion=suggestion
        )
        
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.elapsed_time = elapsed_time


class DataFormatError(LLMExplanationError):
    """
    Exception raised when data format validation fails.
    
    Used for metadata parsing errors, response format issues, data structure
    problems, and serialization/deserialization failures.
    """

    def __init__(self, 
                 message: str, 
                 data_type: str = None,
                 expected_format: str = None,
                 actual_format: str = None,
                 context: ErrorContext = None):
        """
        Initialize data format error.

        Args:
            message: Format error description
            data_type: Type of data that failed validation
            expected_format: Expected data format description
            actual_format: Actual format encountered
            context: Enhanced error context
        """
        details = {
            'data_type': data_type,
            'expected_format': expected_format,
            'actual_format': actual_format
        }
        
        suggestion = f"Convert data to {expected_format} format" if expected_format else "Check data format and structure"
        
        super().__init__(
            message=message,
            error_code="DATA_FORMAT_ERROR",
            details=details,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.DATA_FORMAT,
            context=context,
            recoverable=True,
            suggestion=suggestion
        )
        
        self.data_type = data_type
        self.expected_format = expected_format
        self.actual_format = actual_format


# Convenience exception raising functions with enhanced parameter handling

def raise_input_validation_error(field_name: str, 
                                message: str, 
                                field_value: Any = None,
                                validation_rules: List[str] = None,
                                context: ErrorContext = None) -> None:
    """Raise standardized input validation error with enhanced context."""
    raise InputValidationError(
        f"Invalid {field_name}: {message}", 
        field_name=field_name, 
        field_value=field_value,
        validation_rules=validation_rules,
        context=context
    )


def raise_api_configuration_error(message: str, 
                                 config_issue: str = None,
                                 service_name: str = None,
                                 context: ErrorContext = None) -> None:
    """Raise standardized API configuration error."""
    raise APIConfigurationError(
        f"API configuration error: {message}", 
        config_issue=config_issue,
        service_name=service_name,
        context=context
    )


def raise_llm_response_error(response_type: str, 
                            message: str,
                            model_name: str = None, 
                            safety_blocked: bool = False,
                            finish_reason: str = None,
                            context: ErrorContext = None) -> None:
    """Raise standardized LLM response error."""
    raise LLMResponseError(
        f"LLM {response_type} failed: {message}", 
        response_type=response_type, 
        model_name=model_name, 
        safety_blocked=safety_blocked,
        finish_reason=finish_reason,
        context=context
    )


def raise_explanation_generation_error(generation_stage: str, 
                                     message: str,
                                     explanation_type: str = None,
                                     processing_time: float = None,
                                     context: ErrorContext = None) -> None:
    """Raise standardized explanation generation error."""
    raise ExplanationGenerationError(
        f"Explanation generation error ({generation_stage}): {message}",
        generation_stage=generation_stage, 
        explanation_type=explanation_type,
        processing_time=processing_time,
        context=context
    )


def raise_source_assessment_error(message: str,
                                 source_name: str = None,
                                 assessment_stage: str = None,
                                 context: ErrorContext = None) -> None:
    """Raise standardized source assessment error."""
    raise SourceAssessmentError(
        f"Source assessment failed: {message}",
        source_name=source_name,
        assessment_stage=assessment_stage,
        context=context
    )


def raise_processing_timeout_error(operation: str,
                                  timeout_seconds: float,
                                  elapsed_time: float = None,
                                  context: ErrorContext = None) -> None:
    """Raise standardized processing timeout error."""
    raise ProcessingTimeoutError(
        f"Operation '{operation}' timed out after {timeout_seconds}s",
        timeout_seconds=timeout_seconds,
        operation=operation,
        elapsed_time=elapsed_time,
        context=context
    )


# Advanced error handling utilities with production features

def handle_llm_explanation_exception(exception: Exception, 
                                   context: ErrorContext = None) -> LLMExplanationError:
    """
    Convert any exception to a standardized LLM explanation error with context preservation.

    Args:
        exception: Exception to convert
        context: Enhanced error context to preserve

    Returns:
        Standardized LLMExplanationError instance with full context
    """
    if isinstance(exception, LLMExplanationError):
        # Update context if provided
        if context:
            exception.context = context
        return exception

    # Enhanced exception type mapping with context preservation
    error_mappings = {
        KeyError: lambda e, ctx: InputValidationError(
            f"Missing required key: {str(e)}", 
            field_name=str(e).strip("'\""), 
            context=ctx
        ),
        ValueError: lambda e, ctx: InputValidationError(
            f"Invalid value: {str(e)}", 
            context=ctx
        ),
        TypeError: lambda e, ctx: DataFormatError(
            f"Type error: {str(e)}", 
            context=ctx
        ),
        TimeoutError: lambda e, ctx: ProcessingTimeoutError(
            f"Operation timed out: {str(e)}", 
            context=ctx
        ),
        ConnectionError: lambda e, ctx: APIConfigurationError(
            f"Connection failed: {str(e)}", 
            config_issue="connectivity",
            context=ctx
        ),
        FileNotFoundError: lambda e, ctx: DataFormatError(
            f"File not found: {str(e)}",
            data_type="file",
            context=ctx
        ),
        PermissionError: lambda e, ctx: APIConfigurationError(
            f"Permission denied: {str(e)}",
            config_issue="permissions",
            context=ctx
        )
    }

    exception_type = type(exception)
    if exception_type in error_mappings:
        return error_mappings[exception_type](exception, context)

    # Default fallback with enhanced context
    return LLMExplanationError(
        f"Unexpected error: {str(exception)}", 
        error_code="UNEXPECTED_ERROR",
        details={'original_exception': exception_type.__name__},
        severity=ErrorSeverity.HIGH,
        context=context,
        suggestion="Review error details and contact support if issue persists"
    )


def is_recoverable_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a recoverable error with enhanced logic.

    Args:
        exception: Exception to evaluate

    Returns:
        True if error is recoverable (retry possible), False otherwise
    """
    if isinstance(exception, LLMExplanationError):
        return exception.recoverable
    
    # Enhanced recoverability analysis
    recoverable_types = (
        RateLimitError,
        ProcessingTimeoutError,
        APIConfigurationError,
        ExplanationGenerationError,
        SourceAssessmentError,
        PromptFormattingError
    )
    
    if isinstance(exception, recoverable_types):
        # Additional specific checks
        if isinstance(exception, LLMResponseError):
            return not exception.safety_blocked
        return True
    
    # Check for common recoverable system errors
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return True
    
    return False


def get_retry_delay(exception: Exception) -> Optional[float]:
    """
    Get appropriate retry delay for recoverable errors with intelligent scheduling.

    Args:
        exception: Exception to analyze

    Returns:
        Delay in seconds, or None if not retryable
    """
    if isinstance(exception, LLMExplanationError) and exception.retry_delay is not None:
        return exception.retry_delay
    
    # Enhanced retry delay logic
    if isinstance(exception, RateLimitError) and exception.retry_after:
        return float(exception.retry_after)
    elif isinstance(exception, ProcessingTimeoutError):
        return 5.0  # Standard retry delay for timeouts
    elif isinstance(exception, LLMResponseError) and not exception.safety_blocked:
        return 2.0  # Quick retry for non-safety issues
    elif isinstance(exception, APIConfigurationError):
        return 30.0  # Longer delay for config issues
    elif isinstance(exception, ExplanationGenerationError):
        return 3.0  # Moderate delay for generation issues
    elif isinstance(exception, SourceAssessmentError):
        return 5.0  # Standard delay for source issues
    elif isinstance(exception, (ConnectionError, TimeoutError)):
        return 10.0  # Network-related delays
    
    return None


def get_error_recovery_suggestion(exception: Exception) -> Optional[str]:
    """
    Get detailed recovery suggestion for specific error types with actionable guidance.

    Args:
        exception: Exception that occurred

    Returns:
        Human-readable recovery suggestion with specific steps
    """
    if isinstance(exception, LLMExplanationError) and exception.suggestion:
        return exception.suggestion
    
    # Enhanced recovery suggestions
    if isinstance(exception, InputValidationError):
        return "Validate input data format and ensure all required fields are provided with correct types"
    elif isinstance(exception, APIConfigurationError):
        return "Check API key validity, network connectivity, and service endpoint configuration"
    elif isinstance(exception, LLMResponseError):
        if hasattr(exception, 'safety_blocked') and exception.safety_blocked:
            return "Content was blocked by safety filters - modify prompt language and avoid restricted topics"
        else:
            return "Try the request again with adjusted parameters or alternative prompt phrasing"
    elif isinstance(exception, RateLimitError):
        retry_time = get_retry_delay(exception) or 60
        return f"Rate limit exceeded - implement exponential backoff and wait {retry_time} seconds before retrying"
    elif isinstance(exception, ProcessingTimeoutError):
        return "Request timed out - try with shorter content, increase timeout limits, or break into smaller requests"
    elif isinstance(exception, SourceAssessmentError):
        return "Source assessment failed - verify source name format and check database connectivity"
    elif isinstance(exception, PromptFormattingError):
        return "Fix prompt formatting issues - check template syntax and provide all required parameters"
    elif isinstance(exception, ExplanationGenerationError):
        return "Explanation generation failed - review input quality and try with different parameters"
    elif isinstance(exception, DataFormatError):
        return "Data format error - ensure data structure matches expected format and check encoding"
    else:
        return "Review error details, check system logs, and retry with corrected parameters"


def categorize_error_severity(exception: Exception) -> ErrorSeverity:
    """
    Categorize error severity for monitoring, alerting, and escalation procedures.

    Args:
        exception: Exception to categorize

    Returns:
        Severity level for monitoring and alerting systems
    """
    if isinstance(exception, LLMExplanationError):
        return exception.severity
    
    # Enhanced severity categorization
    if isinstance(exception, (InputValidationError, PromptFormattingError, DataFormatError)):
        return ErrorSeverity.LOW  # User/input errors - low severity
    elif isinstance(exception, (RateLimitError, ProcessingTimeoutError, SourceAssessmentError)):
        return ErrorSeverity.MEDIUM  # Recoverable system issues - medium severity
    elif isinstance(exception, (LLMResponseError, ExplanationGenerationError)):
        return ErrorSeverity.HIGH  # Service functionality issues - high severity  
    elif isinstance(exception, APIConfigurationError):
        return ErrorSeverity.CRITICAL  # System configuration problems - critical severity
    else:
        return ErrorSeverity.MEDIUM  # Unknown issues default to medium


def log_exception_with_context(exception: Exception, 
                              session_id: str = None,
                              additional_context: Dict[str, Any] = None) -> None:
    """
    Log exception with comprehensive context for production debugging.

    Args:
        exception: Exception to log
        session_id: Session identifier for request tracking
        additional_context: Additional context information
    """
    logger = logging.getLogger(f"{__name__}.exception_logger")
    
    # Build comprehensive context
    context = {
        'exception_type': type(exception).__name__,
        'session_id': session_id,
        'recoverable': is_recoverable_error(exception),
        'severity': categorize_error_severity(exception).value,
        'retry_delay': get_retry_delay(exception),
        'suggestion': get_error_recovery_suggestion(exception)
    }
    
    if additional_context:
        context.update(additional_context)
    
    if isinstance(exception, LLMExplanationError):
        context.update({
            'error_code': exception.error_code,
            'category': exception.category.value,
            'details': exception.details
        })
    
    # Log with appropriate level based on severity
    severity = categorize_error_severity(exception)
    
    if severity == ErrorSeverity.CRITICAL:
        logger.critical(f"Critical exception: {str(exception)}", extra=context)
    elif severity == ErrorSeverity.HIGH:
        logger.error(f"High severity exception: {str(exception)}", extra=context)
    elif severity == ErrorSeverity.MEDIUM:
        logger.warning(f"Medium severity exception: {str(exception)}", extra=context)
    else:
        logger.info(f"Low severity exception: {str(exception)}", extra=context)


# Production monitoring and metrics integration

class ErrorMetrics:
    """Error metrics collector for production monitoring."""
    
    def __init__(self):
        self.error_counts = {}
        self.severity_counts = {severity.value: 0 for severity in ErrorSeverity}
        self.category_counts = {category.value: 0 for category in ErrorCategory}
        self.recovery_attempts = 0
        self.successful_recoveries = 0

    def record_error(self, exception: Exception) -> None:
        """Record error occurrence for metrics."""
        error_type = type(exception).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        if isinstance(exception, LLMExplanationError):
            self.severity_counts[exception.severity.value] += 1
            self.category_counts[exception.category.value] += 1

    def record_recovery_attempt(self, successful: bool = False) -> None:
        """Record error recovery attempt."""
        self.recovery_attempts += 1
        if successful:
            self.successful_recoveries += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive error metrics."""
        total_errors = sum(self.error_counts.values())
        recovery_rate = (self.successful_recoveries / max(self.recovery_attempts, 1)) * 100
        
        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts,
            'severity_distribution': self.severity_counts,
            'category_distribution': self.category_counts,
            'recovery_attempts': self.recovery_attempts,
            'successful_recoveries': self.successful_recoveries,
            'recovery_rate_percent': round(recovery_rate, 2),
            'most_common_errors': sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }


# Global error metrics instance
error_metrics = ErrorMetrics()


# Testing functionality
if __name__ == "__main__":
    """Test LLM explanation exception functionality with comprehensive examples."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== LLM EXPLANATION EXCEPTIONS TEST ===")
    
    # Test input validation error
    print("--- Input Validation Error Test ---")
    try:
        raise_input_validation_error(
            "article_text", 
            "Text is too short", 
            "", 
            validation_rules=["min_length_100", "non_empty"],
            context=ErrorContext(session_id="test_001", operation="input_validation")
        )
    except InputValidationError as e:
        print(f"✅ Input validation error: {e}")
        print(f"   Severity: {e.severity.value}")
        print(f"   Recoverable: {e.recoverable}")
        print(f"   Suggestion: {e.suggestion}")
        print(f"   Details: {e.details}")
        error_metrics.record_error(e)

    # Test API configuration error
    print("\n--- API Configuration Error Test ---")
    try:
        raise_api_configuration_error(
            "Missing API key", 
            config_issue="MISSING_API_KEY",
            service_name="gemini",
            context=ErrorContext(session_id="test_002", operation="api_init")
        )
    except APIConfigurationError as e:
        print(f"✅ API configuration error: {e}")
        print(f"   Severity: {e.severity.value}")
        print(f"   Retry delay: {e.retry_delay}s")
        print(f"   Context: {e.context.to_dict()}")
        error_metrics.record_error(e)

    # Test LLM response error with safety blocking
    print("\n--- LLM Response Error Test ---")
    try:
        raise_llm_response_error(
            "explanation", 
            "Content blocked by safety filters",
            model_name="gemini-1.5-pro", 
            safety_blocked=True,
            finish_reason="SAFETY",
            context=ErrorContext(session_id="test_003", operation="explanation_generation", model_used="gemini-1.5-pro")
        )
    except LLMResponseError as e:
        print(f"✅ LLM response error: {e}")
        print(f"   Safety blocked: {e.safety_blocked}")
        print(f"   Recoverable: {e.recoverable}")
        print(f"   User message: {e.to_user_message()}")
        error_metrics.record_error(e)

    # Test rate limit error
    print("\n--- Rate Limit Error Test ---")
    try:
        rate_error = RateLimitError(
            "API rate limit exceeded", 
            retry_after=120, 
            service="gemini",
            rate_limit_type="requests_per_minute",
            context=ErrorContext(session_id="test_004", operation="batch_processing")
        )
        raise rate_error
    except RateLimitError as e:
        print(f"✅ Rate limit error: {e}")
        print(f"   Retry after: {e.retry_after}s")
        print(f"   Recovery delay: {get_retry_delay(e)}s")
        print(f"   Is recoverable: {is_recoverable_error(e)}")
        error_metrics.record_error(e)

    # Test exception conversion
    print("\n--- Exception Conversion Test ---")
    try:
        raise KeyError("missing_field")
    except Exception as e:
        converted = handle_llm_explanation_exception(
            e, 
            context=ErrorContext(session_id="test_005", operation="data_processing")
        )
        print(f"✅ Converted exception: {converted}")
        print(f"   Original type: {type(e).__name__}")
        print(f"   Converted type: {type(converted).__name__}")
        print(f"   Severity: {converted.severity.value}")
        error_metrics.record_error(converted)

    # Test error recovery utilities
    print("\n--- Error Recovery Utilities Test ---")
    timeout_error = ProcessingTimeoutError(
        "Operation timed out",
        timeout_seconds=30.0,
        operation="detailed_analysis",
        elapsed_time=35.2,
        context=ErrorContext(session_id="test_006", operation="timeout_test")
    )
    
    print(f"✅ Recovery utilities:")
    print(f"   Is recoverable: {is_recoverable_error(timeout_error)}")
    print(f"   Retry delay: {get_retry_delay(timeout_error)}s")
    print(f"   Severity: {categorize_error_severity(timeout_error).value}")
    print(f"   Suggestion: {get_error_recovery_suggestion(timeout_error)}")
    error_metrics.record_error(timeout_error)

    # Test error metrics
    print("\n--- Error Metrics Test ---")
    error_metrics.record_recovery_attempt(successful=True)
    error_metrics.record_recovery_attempt(successful=False)
    error_metrics.record_recovery_attempt(successful=True)
    
    metrics = error_metrics.get_metrics()
    print(f"✅ Error metrics:")
    print(f"   Total errors: {metrics['total_errors']}")
    print(f"   Recovery rate: {metrics['recovery_rate_percent']:.1f}%")
    print(f"   Most common: {metrics['most_common_errors'][:3]}")
    print(f"   Severity distribution: {metrics['severity_distribution']}")

    # Test context logging
    print("\n--- Context Logging Test ---")
    log_exception_with_context(
        timeout_error,
        session_id="test_logging_001",
        additional_context={
            'user_id': 'user123',
            'request_size': 1500,
            'processing_stage': 'analysis'
        }
    )
    print("✅ Context logging completed")

    print("\n🎯 LLM explanation exceptions tests completed successfully!")
