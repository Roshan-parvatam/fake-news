# agents/context_analyzer/exceptions.py

"""
Context Analyzer Custom Exceptions

Custom exception classes for the Context Analyzer Agent providing
specific error handling for bias detection, manipulation analysis,
and context evaluation workflows.
"""

from typing import Any, Dict, List, Optional


class ContextAnalyzerError(Exception):
    """
    Base exception class for all Context Analyzer Agent errors.
    
    All custom exceptions in the context analyzer module should inherit
    from this base class to provide consistent error handling.
    """
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        Initialize base context analyzer exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CONTEXT_ANALYZER_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


class InputValidationError(ContextAnalyzerError):
    """
    Exception raised when input data validation fails.
    
    Used for invalid article text, malformed analysis data, or missing required fields.
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


class LLMResponseError(ContextAnalyzerError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles safety filter blocks, invalid responses, and API communication failures.
    """
    
    def __init__(self, message: str, response_type: str = None, model_name: str = None):
        """
        Initialize LLM response error.
        
        Args:
            message: Error description
            response_type: Type of response that failed (bias_analysis, manipulation_detection, etc.)
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


class PromptGenerationError(ContextAnalyzerError):
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


class BiasDetectionError(ContextAnalyzerError):
    """
    Exception raised when bias detection analysis fails.
    
    Used for bias pattern matching failures, scoring calculation errors, and bias analysis issues.
    """
    
    def __init__(self, message: str, bias_type: str = None, detection_stage: str = None):
        """
        Initialize bias detection error.
        
        Args:
            message: Error description
            bias_type: Type of bias that caused detection to fail (political, emotional, etc.)
            detection_stage: Stage of detection process that failed
        """
        details = {}
        if bias_type:
            details['bias_type'] = bias_type
        if detection_stage:
            details['detection_stage'] = detection_stage
        
        super().__init__(message, "BIAS_DETECTION_ERROR", details)
        self.bias_type = bias_type
        self.detection_stage = detection_stage


class ManipulationDetectionError(ContextAnalyzerError):
    """
    Exception raised when manipulation detection fails.
    
    Used for propaganda analysis failures, emotional manipulation detection errors, and framing analysis issues.
    """
    
    def __init__(self, message: str, manipulation_type: str = None, detection_stage: str = None):
        """
        Initialize manipulation detection error.
        
        Args:
            message: Error description
            manipulation_type: Type of manipulation that caused detection to fail
            detection_stage: Stage of detection process that failed
        """
        details = {}
        if manipulation_type:
            details['manipulation_type'] = manipulation_type
        if detection_stage:
            details['detection_stage'] = detection_stage
        
        super().__init__(message, "MANIPULATION_DETECTION_ERROR", details)
        self.manipulation_type = manipulation_type
        self.detection_stage = detection_stage


class ScoringConsistencyError(ContextAnalyzerError):
    """
    Exception raised when LLM scoring is inconsistent with textual analysis.
    
    Used when numerical scores don't match the explanatory text analysis.
    """
    
    def __init__(self, message: str, score_type: str = None, text_analysis: str = None, numerical_score: int = None):
        """
        Initialize scoring consistency error.
        
        Args:
            message: Error description
            score_type: Type of score that's inconsistent (bias, manipulation, etc.)
            text_analysis: Text analysis excerpt showing inconsistency
            numerical_score: The inconsistent numerical score
        """
        details = {}
        if score_type:
            details['score_type'] = score_type
        if text_analysis:
            details['text_analysis'] = text_analysis[:200]  # Truncate for brevity
        if numerical_score is not None:
            details['numerical_score'] = numerical_score
        
        super().__init__(message, "SCORING_CONSISTENCY_ERROR", details)
        self.score_type = score_type
        self.text_analysis = text_analysis
        self.numerical_score = numerical_score


class FramingAnalysisError(ContextAnalyzerError):
    """
    Exception raised when framing analysis fails.
    
    Used for narrative structure analysis failures and framing technique detection errors.
    """
    
    def __init__(self, message: str, framing_type: str = None, analysis_stage: str = None):
        """
        Initialize framing analysis error.
        
        Args:
            message: Error description
            framing_type: Type of framing analysis that failed
            analysis_stage: Stage of analysis that failed
        """
        details = {}
        if framing_type:
            details['framing_type'] = framing_type
        if analysis_stage:
            details['analysis_stage'] = analysis_stage
        
        super().__init__(message, "FRAMING_ANALYSIS_ERROR", details)
        self.framing_type = framing_type
        self.analysis_stage = analysis_stage


class ConfigurationError(ContextAnalyzerError):
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


class RateLimitError(ContextAnalyzerError):
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


class ProcessingTimeoutError(ContextAnalyzerError):
    """
    Exception raised when context analysis exceeds time limits.
    
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


class DataFormatError(ContextAnalyzerError):
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


def raise_bias_detection_error(bias_type: str, message: str, detection_stage: str = None) -> None:
    """
    Raise a standardized bias detection error.
    
    Args:
        bias_type: Type of bias detection that failed
        message: Error description
        detection_stage: Optional detection stage
    """
    raise BiasDetectionError(f"Bias detection error ({bias_type}): {message}", bias_type, detection_stage)


def raise_manipulation_detection_error(manipulation_type: str, message: str, detection_stage: str = None) -> None:
    """
    Raise a standardized manipulation detection error.
    
    Args:
        manipulation_type: Type of manipulation detection that failed
        message: Error description
        detection_stage: Optional detection stage
    """
    raise ManipulationDetectionError(f"Manipulation detection error ({manipulation_type}): {message}", manipulation_type, detection_stage)


def raise_scoring_consistency_error(score_type: str, message: str, text_analysis: str = None, numerical_score: int = None) -> None:
    """
    Raise a standardized scoring consistency error.
    
    Args:
        score_type: Type of score that's inconsistent
        message: Error description
        text_analysis: Optional text analysis excerpt
        numerical_score: Optional inconsistent numerical score
    """
    raise ScoringConsistencyError(f"Scoring consistency error ({score_type}): {message}", score_type, text_analysis, numerical_score)


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
def handle_context_analyzer_exception(exception: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format.
    
    Args:
        exception: Exception to handle
        
    Returns:
        Dictionary with standardized error information
    """
    if isinstance(exception, ContextAnalyzerError):
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


def validate_score_consistency(text_analysis: str, scores: Dict[str, int]) -> None:
    """
    Validate that numerical scores match textual analysis.
    
    Args:
        text_analysis: Generated analysis text
        scores: Dictionary of numerical scores
        
    Raises:
        ScoringConsistencyError: If scores don't match text analysis
    """
    text_lower = text_analysis.lower()
    
    for score_type, score_value in scores.items():
        # Check for obvious inconsistencies
        if score_value <= 25:  # Low scores
            if any(word in text_lower for word in ['high', 'significant', 'extreme', 'severe']):
                raise_scoring_consistency_error(
                    score_type, 
                    f"Text indicates high {score_type} but score is {score_value}",
                    text_analysis[:200],
                    score_value
                )
        elif score_value >= 75:  # High scores
            if any(word in text_lower for word in ['minimal', 'low', 'slight', 'neutral']):
                raise_scoring_consistency_error(
                    score_type,
                    f"Text indicates low {score_type} but score is {score_value}",
                    text_analysis[:200],
                    score_value
                )
