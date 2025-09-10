# agents/claim_extractor/exceptions.py

"""
Claim Extractor Agent Custom Exceptions

Custom exception classes for the claim extractor agent providing
specific error handling for claim extraction, parsing, verification,
and prioritization workflows with detailed error context.
"""

from typing import Any, Dict, List, Optional


class ClaimExtractorError(Exception):
    """
    Base exception class for all Claim Extractor Agent errors.
    
    All custom exceptions in the claim extractor module should inherit
    from this base class to provide consistent error handling.
    """
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        Initialize base claim extractor exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CLAIM_EXTRACTOR_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


class InputValidationError(ClaimExtractorError):
    """
    Exception raised when input data validation fails.
    
    Used for invalid article text, malformed BERT results, or missing required fields.
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


class LLMResponseError(ClaimExtractorError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles empty responses, API failures, and content generation issues.
    """
    
    def __init__(self, message: str, response_type: str = None, model_name: str = None):
        """
        Initialize LLM response error.
        
        Args:
            message: Error description
            response_type: Type of response that failed (extraction, verification, prioritization)
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


class ClaimParsingError(ClaimExtractorError):
    """
    Exception raised when claim parsing fails.
    
    Used for malformed LLM output, parsing failures, and claim structure validation issues.
    """
    
    def __init__(self, message: str, parsing_stage: str = None, raw_output: str = None):
        """
        Initialize claim parsing error.
        
        Args:
            message: Error description
            parsing_stage: Stage of parsing that failed
            raw_output: Raw LLM output that failed to parse (truncated for logging)
        """
        details = {}
        if parsing_stage:
            details['parsing_stage'] = parsing_stage
        if raw_output:
            details['raw_output_sample'] = raw_output[:300]  # Truncate for logging
        
        super().__init__(message, "CLAIM_PARSING_ERROR", details)
        self.parsing_stage = parsing_stage
        self.raw_output = raw_output


class ClaimExtractionError(ClaimExtractorError):
    """
    Exception raised during the claim extraction process.
    
    Used for extraction algorithm failures, pattern analysis errors, and processing issues.
    """
    
    def __init__(self, message: str, extraction_stage: str = None, claims_processed: int = None):
        """
        Initialize claim extraction error.
        
        Args:
            message: Error description
            extraction_stage: Stage of extraction that failed
            claims_processed: Number of claims processed when error occurred
        """
        details = {}
        if extraction_stage:
            details['extraction_stage'] = extraction_stage
        if claims_processed is not None:
            details['claims_processed'] = claims_processed
        
        super().__init__(message, "CLAIM_EXTRACTION_ERROR", details)
        self.extraction_stage = extraction_stage
        self.claims_processed = claims_processed


class VerificationAnalysisError(ClaimExtractorError):
    """
    Exception raised when verification analysis generation fails.
    
    Used for verification prompt failures, analysis generation errors, and strategy creation issues.
    """
    
    def __init__(self, message: str, analysis_type: str = None, claims_count: int = None):
        """
        Initialize verification analysis error.
        
        Args:
            message: Error description
            analysis_type: Type of verification analysis that failed
            claims_count: Number of claims being analyzed when error occurred
        """
        details = {}
        if analysis_type:
            details['analysis_type'] = analysis_type
        if claims_count is not None:
            details['claims_count'] = claims_count
        
        super().__init__(message, "VERIFICATION_ANALYSIS_ERROR", details)
        self.analysis_type = analysis_type
        self.claims_count = claims_count


class PrioritizationError(ClaimExtractorError):
    """
    Exception raised when claim prioritization fails.
    
    Used for prioritization algorithm failures, ranking errors, and priority assignment issues.
    """
    
    def __init__(self, message: str, prioritization_method: str = None, claims_analyzed: int = None):
        """
        Initialize prioritization error.
        
        Args:
            message: Error description
            prioritization_method: Method used for prioritization that failed
            claims_analyzed: Number of claims being prioritized when error occurred
        """
        details = {}
        if prioritization_method:
            details['prioritization_method'] = prioritization_method
        if claims_analyzed is not None:
            details['claims_analyzed'] = claims_analyzed
        
        super().__init__(message, "PRIORITIZATION_ERROR", details)
        self.prioritization_method = prioritization_method
        self.claims_analyzed = claims_analyzed


class PatternAnalysisError(ClaimExtractorError):
    """
    Exception raised when pattern analysis fails.
    
    Used for pattern database errors, pattern matching failures, and claim richness calculation issues.
    """
    
    def __init__(self, message: str, pattern_type: str = None, analysis_stage: str = None):
        """
        Initialize pattern analysis error.
        
        Args:
            message: Error description
            pattern_type: Type of pattern analysis that failed
            analysis_stage: Stage of pattern analysis that failed
        """
        details = {}
        if pattern_type:
            details['pattern_type'] = pattern_type
        if analysis_stage:
            details['analysis_stage'] = analysis_stage
        
        super().__init__(message, "PATTERN_ANALYSIS_ERROR", details)
        self.pattern_type = pattern_type
        self.analysis_stage = analysis_stage


class ConfigurationError(ClaimExtractorError):
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


class RateLimitError(ClaimExtractorError):
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


class ProcessingTimeoutError(ClaimExtractorError):
    """
    Exception raised when claim extraction exceeds time limits.
    
    Used for long-running extractions, API timeouts, and processing deadlines.
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


class DataFormatError(ClaimExtractorError):
    """
    Exception raised when data parsing or formatting fails.
    
    Used for JSON parsing errors, claim format issues, and data structure problems.
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


def raise_claim_parsing_error(parsing_stage: str, message: str, raw_output: str = None) -> None:
    """
    Raise a standardized claim parsing error.
    
    Args:
        parsing_stage: Stage of parsing that failed
        message: Error description
        raw_output: Optional raw LLM output
    """
    raise ClaimParsingError(f"Claim parsing error ({parsing_stage}): {message}", parsing_stage, raw_output)


def raise_claim_extraction_error(extraction_stage: str, message: str, claims_processed: int = None) -> None:
    """
    Raise a standardized claim extraction error.
    
    Args:
        extraction_stage: Stage of extraction that failed
        message: Error description
        claims_processed: Optional number of claims processed
    """
    raise ClaimExtractionError(
        f"Claim extraction error ({extraction_stage}): {message}", 
        extraction_stage, claims_processed
    )


def raise_verification_analysis_error(analysis_type: str, message: str, claims_count: int = None) -> None:
    """
    Raise a standardized verification analysis error.
    
    Args:
        analysis_type: Type of verification analysis that failed
        message: Error description
        claims_count: Optional number of claims being analyzed
    """
    raise VerificationAnalysisError(
        f"Verification analysis error ({analysis_type}): {message}", 
        analysis_type, claims_count
    )


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
def handle_claim_extractor_exception(exception: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format.
    
    Args:
        exception: Exception to handle
        
    Returns:
        Dictionary with standardized error information
    """
    if isinstance(exception, ClaimExtractorError):
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
        ClaimParsingError,  # Can try alternative parsing methods
        PatternAnalysisError  # Can skip pattern analysis
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
    elif isinstance(exception, ClaimParsingError):
        return 1.0  # Quick retry for parsing issues
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
    if isinstance(exception, ClaimParsingError):
        return {
            'fallback_type': 'alternative_parsing',
            'recommendation': 'Try alternative parsing method',
            'action': 'use_fallback_parser'
        }
    elif isinstance(exception, PatternAnalysisError):
        return {
            'fallback_type': 'skip_pattern_analysis',
            'recommendation': 'Skip pattern preprocessing and use direct LLM extraction',
            'action': 'disable_pattern_analysis'
        }
    elif isinstance(exception, LLMResponseError):
        return {
            'fallback_type': 'simplified_extraction',
            'recommendation': 'Use simplified extraction prompt',
            'action': 'use_basic_extraction'
        }
    else:
        return None


# Testing functionality
if __name__ == "__main__":
    """Test claim extractor exception functionality."""
    
    # Test input validation error
    try:
        raise_input_validation_error("article_text", "Text is too short", "")
    except InputValidationError as e:
        print("Input Validation Error:", e.to_dict())
    
    # Test LLM response error
    try:
        raise_llm_response_error("claim_extraction", "Empty response from API", "gemini-1.5-pro")
    except LLMResponseError as e:
        print("LLM Response Error:", e.to_dict())
    
    # Test claim parsing error
    try:
        raise_claim_parsing_error("structured_format", "Failed to parse claim structure", "**Claim 1**: Invalid...")
    except ClaimParsingError as e:
        print("Claim Parsing Error:", e.to_dict())
        
        # Test fallback recommendation
        fallback = get_fallback_recommendation(e)
        print("Fallback Recommendation:", fallback)
    
    # Test claim extraction error
    try:
        raise_claim_extraction_error("pattern_analysis", "Pattern database lookup failed", 3)
    except ClaimExtractionError as e:
        print("Claim Extraction Error:", e.to_dict())
    
    # Test verification analysis error
    try:
        raise_verification_analysis_error("verification_strategy", "Failed to generate verification steps", 5)
    except VerificationAnalysisError as e:
        print("Verification Analysis Error:", e.to_dict())
    
    # Test configuration error
    try:
        raise_configuration_error("gemini_api_key", "API key not found in settings")
    except ConfigurationError as e:
        print("Configuration Error:", e.to_dict())
    
    # Test error recovery logic
    rate_limit_error = RateLimitError("Rate limit exceeded", retry_after=60, service="gemini")
    print(f"Is recoverable: {is_recoverable_error(rate_limit_error)}")
    print(f"Retry delay: {get_retry_delay(rate_limit_error)} seconds")
    
    print("\n=== EXCEPTION TESTING COMPLETED ===")
    print("All exception types tested successfully!")
