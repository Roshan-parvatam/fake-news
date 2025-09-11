# agents/context_analyzer/exceptions.py

"""
Context Analyzer Custom Exceptions - Production Ready

Production-ready custom exception classes for the Context Analyzer Agent providing
specific error handling for bias detection, manipulation analysis, context evaluation
workflows, enhanced logging integration, session tracking, and recovery mechanisms.
"""

import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime


class ContextAnalyzerError(Exception):
    """
    Base exception class for all Context Analyzer Agent errors.
    
    All custom exceptions in the context analyzer module should inherit
    from this base class to provide consistent error handling, logging
    integration, and session tracking for production environments.
    """

    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None, session_id: str = None):
        """
        Initialize base context analyzer exception with enhanced context.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
            session_id: Optional session ID for request tracking
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CONTEXT_ANALYZER_ERROR"
        self.details = details or {}
        self.session_id = session_id
        self.timestamp = datetime.now().isoformat()
        
        # Add session tracking to details if provided
        if session_id:
            self.details['session_id'] = session_id
            self.details['timestamp'] = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'session_id': self.session_id,
            'timestamp': self.timestamp
        }

    def __str__(self) -> str:
        """Enhanced string representation with session context."""
        base_str = self.message
        if self.session_id:
            base_str += f" (Session: {self.session_id})"
        return base_str


class InputValidationError(ContextAnalyzerError):
    """
    Exception raised when input data validation fails.
    
    Used for invalid article text, malformed analysis data, missing required fields,
    or input format issues with detailed field-level error context.
    """

    def __init__(self, message: str, field_name: str = None, field_value: Any = None, 
                 validation_type: str = None, session_id: str = None):
        """
        Initialize input validation error with field-specific context.

        Args:
            message: Error description
            field_name: Name of the invalid field
            field_value: Value that caused the validation error
            validation_type: Type of validation that failed (format, range, required, etc.)
            session_id: Optional session ID for tracking
        """
        details = {'validation_category': 'input_validation'}
        
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            # Truncate long values and sanitize sensitive data
            if isinstance(field_value, str):
                details['field_value'] = field_value[:200] + ('...' if len(field_value) > 200 else '')
            else:
                details['field_value'] = str(field_value)[:200]
        if validation_type:
            details['validation_type'] = validation_type

        super().__init__(message, "INPUT_VALIDATION_ERROR", details, session_id)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_type = validation_type


class LLMResponseError(ContextAnalyzerError):
    """
    Exception raised when LLM API responses are invalid or blocked.
    
    Handles safety filter blocks, invalid responses, API communication failures,
    timeout issues, and content generation problems with model-specific context.
    """

    def __init__(self, message: str, response_type: str = None, model_name: str = None, 
                 safety_filtered: bool = False, retry_possible: bool = True, session_id: str = None):
        """
        Initialize LLM response error with model and response context.

        Args:
            message: Error description
            response_type: Type of response that failed (bias_analysis, manipulation_detection, etc.)
            model_name: Name of the LLM model that generated the error
            safety_filtered: Whether error was caused by safety filters
            retry_possible: Whether the operation can be retried
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'llm_response'}
        
        if response_type:
            details['response_type'] = response_type
        if model_name:
            details['model_name'] = model_name
        if safety_filtered:
            details['safety_filtered'] = True
        if not retry_possible:
            details['retry_possible'] = False

        super().__init__(message, "LLM_RESPONSE_ERROR", details, session_id)
        self.response_type = response_type
        self.model_name = model_name
        self.safety_filtered = safety_filtered
        self.retry_possible = retry_possible


class PromptGenerationError(ContextAnalyzerError):
    """
    Exception raised when prompt generation or formatting fails.
    
    Used when prompt templates are invalid, prompt parameter substitution fails,
    or prompt length exceeds model limits with template-specific context.
    """

    def __init__(self, message: str, prompt_type: str = None, parameters: Dict[str, Any] = None,
                 template_error: bool = False, session_id: str = None):
        """
        Initialize prompt generation error with template context.

        Args:
            message: Error description
            prompt_type: Type of prompt that failed to generate
            parameters: Parameters that caused the generation failure
            template_error: Whether error was in template itself vs parameters
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'prompt_generation'}
        
        if prompt_type:
            details['prompt_type'] = prompt_type
        if parameters:
            # Sanitize parameters for logging (limit length and remove sensitive data)
            sanitized_params = {}
            for k, v in parameters.items():
                if 'key' in k.lower() or 'token' in k.lower():
                    sanitized_params[k] = '[REDACTED]'
                else:
                    sanitized_params[k] = str(v)[:100] + ('...' if len(str(v)) > 100 else '')
            details['parameters'] = sanitized_params
        if template_error:
            details['template_error'] = True

        super().__init__(message, "PROMPT_GENERATION_ERROR", details, session_id)
        self.prompt_type = prompt_type
        self.parameters = parameters
        self.template_error = template_error


class BiasDetectionError(ContextAnalyzerError):
    """
    Exception raised when bias detection analysis fails.
    
    Used for bias pattern matching failures, scoring calculation errors, political
    bias classification issues, and bias analysis inconsistencies.
    """

    def __init__(self, message: str, bias_type: str = None, detection_stage: str = None, 
                 confidence_score: float = None, session_id: str = None):
        """
        Initialize bias detection error with detection context.

        Args:
            message: Error description
            bias_type: Type of bias that caused detection to fail (political, emotional, etc.)
            detection_stage: Stage of detection process that failed
            confidence_score: Confidence score when error occurred (if applicable)
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'bias_detection'}
        
        if bias_type:
            details['bias_type'] = bias_type
        if detection_stage:
            details['detection_stage'] = detection_stage
        if confidence_score is not None:
            details['confidence_score'] = confidence_score

        super().__init__(message, "BIAS_DETECTION_ERROR", details, session_id)
        self.bias_type = bias_type
        self.detection_stage = detection_stage
        self.confidence_score = confidence_score


class ManipulationDetectionError(ContextAnalyzerError):
    """
    Exception raised when manipulation detection fails.
    
    Used for propaganda analysis failures, emotional manipulation detection errors,
    framing analysis issues, and manipulation technique classification problems.
    """

    def __init__(self, message: str, manipulation_type: str = None, detection_stage: str = None,
                 technique_count: int = None, session_id: str = None):
        """
        Initialize manipulation detection error with detection context.

        Args:
            message: Error description
            manipulation_type: Type of manipulation that caused detection to fail
            detection_stage: Stage of detection process that failed
            technique_count: Number of techniques detected when error occurred
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'manipulation_detection'}
        
        if manipulation_type:
            details['manipulation_type'] = manipulation_type
        if detection_stage:
            details['detection_stage'] = detection_stage
        if technique_count is not None:
            details['technique_count'] = technique_count

        super().__init__(message, "MANIPULATION_DETECTION_ERROR", details, session_id)
        self.manipulation_type = manipulation_type
        self.detection_stage = detection_stage
        self.technique_count = technique_count


class ScoringConsistencyError(ContextAnalyzerError):
    """
    Exception raised when LLM scoring is inconsistent with textual analysis.
    
    Used when numerical scores don't match the explanatory text analysis,
    addressing the main consistency issue in the original implementation.
    """

    def __init__(self, message: str, score_type: str = None, text_analysis: str = None, 
                 numerical_score: int = None, expected_range: tuple = None, session_id: str = None):
        """
        Initialize scoring consistency error with detailed mismatch context.

        Args:
            message: Error description
            score_type: Type of score that's inconsistent (bias, manipulation, etc.)
            text_analysis: Text analysis excerpt showing inconsistency
            numerical_score: The inconsistent numerical score
            expected_range: Expected score range based on text analysis
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'scoring_consistency'}
        
        if score_type:
            details['score_type'] = score_type
        if text_analysis:
            details['text_analysis'] = text_analysis[:200] + ('...' if len(text_analysis) > 200 else '')
        if numerical_score is not None:
            details['numerical_score'] = numerical_score
        if expected_range:
            details['expected_range'] = expected_range

        super().__init__(message, "SCORING_CONSISTENCY_ERROR", details, session_id)
        self.score_type = score_type
        self.text_analysis = text_analysis
        self.numerical_score = numerical_score
        self.expected_range = expected_range


class ReliabilityAssessmentError(ContextAnalyzerError):
    """
    Exception raised when reliability assessment fails.
    
    Used for credibility scoring failures, source reliability evaluation errors,
    and trustworthiness classification issues.
    """

    def __init__(self, message: str, assessment_type: str = None, reliability_score: int = None,
                 source_info: Dict[str, Any] = None, session_id: str = None):
        """
        Initialize reliability assessment error.

        Args:
            message: Error description
            assessment_type: Type of reliability assessment that failed
            reliability_score: Score when error occurred
            source_info: Information about source being assessed
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'reliability_assessment'}
        
        if assessment_type:
            details['assessment_type'] = assessment_type
        if reliability_score is not None:
            details['reliability_score'] = reliability_score
        if source_info:
            # Include basic source info but not full content
            details['source_info'] = {
                'source_type': source_info.get('type', 'unknown'),
                'domain': source_info.get('domain', 'unknown')
            }

        super().__init__(message, "RELIABILITY_ASSESSMENT_ERROR", details, session_id)
        self.assessment_type = assessment_type
        self.reliability_score = reliability_score
        self.source_info = source_info


class ContextualRecommendationError(ContextAnalyzerError):
    """
    Exception raised when contextual recommendation generation fails.
    
    Used for recommendation algorithm failures, context matching errors,
    and recommendation scoring inconsistencies.
    """

    def __init__(self, message: str, recommendation_type: str = None, context_data: Dict[str, Any] = None,
                 recommendations_count: int = None, session_id: str = None):
        """
        Initialize contextual recommendation error.

        Args:
            message: Error description
            recommendation_type: Type of recommendation that failed
            context_data: Context data when error occurred
            recommendations_count: Number of recommendations processed
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'contextual_recommendation'}
        
        if recommendation_type:
            details['recommendation_type'] = recommendation_type
        if context_data:
            # Include summary of context data
            details['context_summary'] = {
                'has_previous_analysis': 'previous_analysis' in context_data,
                'has_claims': 'extracted_claims' in context_data,
                'content_length': len(context_data.get('text', '')) if context_data.get('text') else 0
            }
        if recommendations_count is not None:
            details['recommendations_count'] = recommendations_count

        super().__init__(message, "CONTEXTUAL_RECOMMENDATION_ERROR", details, session_id)
        self.recommendation_type = recommendation_type
        self.context_data = context_data
        self.recommendations_count = recommendations_count


class VerificationStrategyError(ContextAnalyzerError):
    """
    Exception raised when verification strategy generation fails.
    
    Used for strategy algorithm failures, verification approach errors,
    and fact-checking workflow issues.
    """

    def __init__(self, message: str, strategy_type: str = None, verification_stage: str = None,
                 claims_processed: int = None, session_id: str = None):
        """
        Initialize verification strategy error.

        Args:
            message: Error description
            strategy_type: Type of strategy that failed
            verification_stage: Stage of verification that failed
            claims_processed: Number of claims processed when error occurred
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'verification_strategy'}
        
        if strategy_type:
            details['strategy_type'] = strategy_type
        if verification_stage:
            details['verification_stage'] = verification_stage
        if claims_processed is not None:
            details['claims_processed'] = claims_processed

        super().__init__(message, "VERIFICATION_STRATEGY_ERROR", details, session_id)
        self.strategy_type = strategy_type
        self.verification_stage = verification_stage
        self.claims_processed = claims_processed


class FramingAnalysisError(ContextAnalyzerError):
    """
    Exception raised when framing analysis fails.
    
    Used for narrative structure analysis failures and framing technique
    detection errors with analysis-specific context.
    """

    def __init__(self, message: str, framing_type: str = None, analysis_stage: str = None,
                 narrative_elements: List[str] = None, session_id: str = None):
        """
        Initialize framing analysis error.

        Args:
            message: Error description
            framing_type: Type of framing analysis that failed
            analysis_stage: Stage of analysis that failed
            narrative_elements: Narrative elements being analyzed
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'framing_analysis'}
        
        if framing_type:
            details['framing_type'] = framing_type
        if analysis_stage:
            details['analysis_stage'] = analysis_stage
        if narrative_elements:
            details['narrative_elements_count'] = len(narrative_elements)
            details['narrative_elements'] = narrative_elements[:5]  # First 5 elements

        super().__init__(message, "FRAMING_ANALYSIS_ERROR", details, session_id)
        self.framing_type = framing_type
        self.analysis_stage = analysis_stage
        self.narrative_elements = narrative_elements


class SafetyFilterError(ContextAnalyzerError):
    """
    Exception raised when content safety filters block analysis.
    
    Used when content is blocked by safety filters, content moderation
    systems, or inappropriate content detection mechanisms.
    """

    def __init__(self, message: str, filter_type: str = None, content_category: str = None,
                 safety_score: float = None, fallback_available: bool = True, session_id: str = None):
        """
        Initialize safety filter error.

        Args:
            message: Error description
            filter_type: Type of safety filter that triggered
            content_category: Category of flagged content
            safety_score: Safety assessment score
            fallback_available: Whether fallback processing is available
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'safety_filter'}
        
        if filter_type:
            details['filter_type'] = filter_type
        if content_category:
            details['content_category'] = content_category
        if safety_score is not None:
            details['safety_score'] = safety_score
        details['fallback_available'] = fallback_available

        super().__init__(message, "SAFETY_FILTER_ERROR", details, session_id)
        self.filter_type = filter_type
        self.content_category = content_category
        self.safety_score = safety_score
        self.fallback_available = fallback_available


class ConfigurationError(ContextAnalyzerError):
    """
    Exception raised when agent configuration is invalid or missing.
    
    Used for missing API keys, invalid config values, setup failures,
    and environment configuration issues with enhanced security.
    """

    def __init__(self, message: str, config_key: str = None, config_value: Any = None,
                 config_section: str = None, session_id: str = None):
        """
        Initialize configuration error with secure value handling.

        Args:
            message: Error description
            config_key: Configuration key that is invalid or missing
            config_value: Invalid configuration value (sanitized for logging)
            config_section: Configuration section where error occurred
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'configuration'}
        
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            # Sanitize sensitive config values for logging
            if any(keyword in str(config_key).lower() for keyword in ['key', 'token', 'password', 'secret']):
                details['config_value'] = '[REDACTED]'
            else:
                details['config_value'] = str(config_value)[:100] + ('...' if len(str(config_value)) > 100 else '')
        if config_section:
            details['config_section'] = config_section

        super().__init__(message, "CONFIGURATION_ERROR", details, session_id)
        self.config_key = config_key
        self.config_value = config_value
        self.config_section = config_section


class RateLimitError(ContextAnalyzerError):
    """
    Exception raised when API rate limits are exceeded.
    
    Used for API throttling, quota exhaustion, rate limit violations,
    and service unavailability with retry timing information.
    """

    def __init__(self, message: str, retry_after: int = None, service: str = None,
                 requests_made: int = None, rate_limit: int = None, session_id: str = None):
        """
        Initialize rate limit error with retry context.

        Args:
            message: Error description
            retry_after: Seconds to wait before retrying (if known)
            service: Name of the service that rate limited the request
            requests_made: Number of requests made when limit hit
            rate_limit: Rate limit that was exceeded
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'rate_limit'}
        
        if retry_after:
            details['retry_after'] = retry_after
        if service:
            details['service'] = service
        if requests_made is not None:
            details['requests_made'] = requests_made
        if rate_limit is not None:
            details['rate_limit'] = rate_limit

        super().__init__(message, "RATE_LIMIT_ERROR", details, session_id)
        self.retry_after = retry_after
        self.service = service
        self.requests_made = requests_made
        self.rate_limit = rate_limit


class ProcessingTimeoutError(ContextAnalyzerError):
    """
    Exception raised when context analysis exceeds time limits.
    
    Used for long-running analysis, API timeouts, processing deadlines,
    and performance constraint violations with timing context.
    """

    def __init__(self, message: str, timeout_seconds: float = None, operation: str = None,
                 elapsed_time: float = None, partial_results: bool = False, session_id: str = None):
        """
        Initialize processing timeout error with timing context.

        Args:
            message: Error description
            timeout_seconds: Timeout limit that was exceeded
            operation: Operation that timed out
            elapsed_time: Actual time elapsed when timeout occurred
            partial_results: Whether partial results are available
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'processing_timeout'}
        
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        if elapsed_time is not None:
            details['elapsed_time'] = elapsed_time
        details['partial_results'] = partial_results

        super().__init__(message, "PROCESSING_TIMEOUT_ERROR", details, session_id)
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.elapsed_time = elapsed_time
        self.partial_results = partial_results


class DataFormatError(ContextAnalyzerError):
    """
    Exception raised when data parsing or formatting fails.
    
    Used for JSON parsing errors, response format issues, data structure
    problems, and serialization failures with format-specific context.
    """

    def __init__(self, message: str, data_type: str = None, expected_format: str = None,
                 parse_error: str = None, data_sample: str = None, session_id: str = None):
        """
        Initialize data format error with parsing context.

        Args:
            message: Error description
            data_type: Type of data that failed to parse
            expected_format: Expected format description
            parse_error: Specific parsing error message
            data_sample: Sample of problematic data (truncated)
            session_id: Optional session ID for tracking
        """
        details = {'error_category': 'data_format'}
        
        if data_type:
            details['data_type'] = data_type
        if expected_format:
            details['expected_format'] = expected_format
        if parse_error:
            details['parse_error'] = parse_error
        if data_sample:
            details['data_sample'] = data_sample[:200] + ('...' if len(data_sample) > 200 else '')

        super().__init__(message, "DATA_FORMAT_ERROR", details, session_id)
        self.data_type = data_type
        self.expected_format = expected_format
        self.parse_error = parse_error
        self.data_sample = data_sample


# Convenience functions for common exception scenarios with enhanced logging

def raise_input_validation_error(field_name: str, message: str, field_value: Any = None, 
                                validation_type: str = None, session_id: str = None) -> None:
    """
    Raise a standardized input validation error with logging.

    Args:
        field_name: Name of the invalid field
        message: Validation error description
        field_value: Optional invalid value
        validation_type: Type of validation that failed
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.error(f"Input validation failed for field '{field_name}': {message}", 
                extra={'session_id': session_id, 'field_name': field_name, 'validation_type': validation_type})
    
    raise InputValidationError(f"Invalid {field_name}: {message}", field_name, field_value, validation_type, session_id)


def raise_llm_response_error(response_type: str, message: str, model_name: str = None, 
                            safety_filtered: bool = False, session_id: str = None) -> None:
    """
    Raise a standardized LLM response error with logging.

    Args:
        response_type: Type of LLM response that failed
        message: Error description
        model_name: Optional model name
        safety_filtered: Whether error was caused by safety filters
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.error(f"LLM {response_type} failed: {message}", 
                extra={'session_id': session_id, 'model_name': model_name, 'safety_filtered': safety_filtered})
    
    raise LLMResponseError(f"LLM {response_type} failed: {message}", response_type, model_name, safety_filtered, True, session_id)


def raise_prompt_generation_error(prompt_type: str, message: str, parameters: Dict[str, Any] = None,
                                template_error: bool = False, session_id: str = None) -> None:
    """
    Raise a standardized prompt generation error with logging.

    Args:
        prompt_type: Type of prompt that failed to generate
        message: Error description
        parameters: Optional parameters that caused the error
        template_error: Whether this was a template-related error
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.error(f"Prompt generation failed for {prompt_type}: {message}", 
                extra={'session_id': session_id, 'prompt_type': prompt_type, 'template_error': template_error})
    
    raise PromptGenerationError(f"Prompt generation failed for {prompt_type}: {message}", 
                               prompt_type, parameters, template_error, session_id)


def raise_bias_detection_error(bias_type: str, message: str, detection_stage: str = None, 
                              confidence_score: float = None, session_id: str = None) -> None:
    """
    Raise a standardized bias detection error with logging.

    Args:
        bias_type: Type of bias detection that failed
        message: Error description
        detection_stage: Optional detection stage
        confidence_score: Optional confidence score when error occurred
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.error(f"Bias detection error ({bias_type}): {message}", 
                extra={'session_id': session_id, 'bias_type': bias_type, 'detection_stage': detection_stage})
    
    raise BiasDetectionError(f"Bias detection error ({bias_type}): {message}", bias_type, detection_stage, confidence_score, session_id)


def raise_manipulation_detection_error(manipulation_type: str, message: str, detection_stage: str = None, 
                                     technique_count: int = None, session_id: str = None) -> None:
    """
    Raise a standardized manipulation detection error with logging.

    Args:
        manipulation_type: Type of manipulation detection that failed
        message: Error description
        detection_stage: Optional detection stage
        technique_count: Number of techniques detected when error occurred
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.error(f"Manipulation detection error ({manipulation_type}): {message}", 
                extra={'session_id': session_id, 'manipulation_type': manipulation_type, 'detection_stage': detection_stage})
    
    raise ManipulationDetectionError(f"Manipulation detection error ({manipulation_type}): {message}", 
                                   manipulation_type, detection_stage, technique_count, session_id)


def raise_scoring_consistency_error(score_type: str, message: str, text_analysis: str = None, 
                                  numerical_score: int = None, expected_range: tuple = None, 
                                  session_id: str = None) -> None:
    """
    Raise a standardized scoring consistency error with logging.

    Args:
        score_type: Type of score that's inconsistent
        message: Error description
        text_analysis: Optional text analysis excerpt
        numerical_score: Optional inconsistent numerical score
        expected_range: Expected score range based on text
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.error(f"Scoring consistency error ({score_type}): {message}", 
                extra={'session_id': session_id, 'score_type': score_type, 'numerical_score': numerical_score})
    
    raise ScoringConsistencyError(f"Scoring consistency error ({score_type}): {message}", 
                                 score_type, text_analysis, numerical_score, expected_range, session_id)


def raise_contextual_recommendation_error(recommendation_type: str, message: str, context_data: Dict[str, Any] = None,
                                        recommendations_count: int = None, session_id: str = None) -> None:
    """
    Raise a standardized contextual recommendation error with logging.

    Args:
        recommendation_type: Type of recommendation that failed
        message: Error description
        context_data: Optional context data
        recommendations_count: Number of recommendations processed
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.error(f"Contextual recommendation error ({recommendation_type}): {message}", 
                extra={'session_id': session_id, 'recommendation_type': recommendation_type})
    
    raise ContextualRecommendationError(f"Contextual recommendation error ({recommendation_type}): {message}",
                                      recommendation_type, context_data, recommendations_count, session_id)


def raise_safety_filter_error(filter_type: str, message: str, content_category: str = None,
                             fallback_available: bool = True, session_id: str = None) -> None:
    """
    Raise a standardized safety filter error with logging.

    Args:
        filter_type: Type of safety filter that triggered
        message: Error description
        content_category: Category of flagged content
        fallback_available: Whether fallback processing is available
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.warning(f"Safety filter triggered ({filter_type}): {message}", 
                  extra={'session_id': session_id, 'filter_type': filter_type, 'fallback_available': fallback_available})
    
    raise SafetyFilterError(f"Safety filter error ({filter_type}): {message}",
                          filter_type, content_category, None, fallback_available, session_id)


def raise_configuration_error(config_key: str, message: str, config_value: Any = None, 
                             config_section: str = None, session_id: str = None) -> None:
    """
    Raise a standardized configuration error with logging.

    Args:
        config_key: Configuration key that is problematic
        message: Error description
        config_value: Optional configuration value
        config_section: Configuration section where error occurred
        session_id: Optional session ID for tracking
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    logger.error(f"Configuration error for {config_key}: {message}", 
                extra={'session_id': session_id, 'config_key': config_key, 'config_section': config_section})
    
    raise ConfigurationError(f"Configuration error for {config_key}: {message}", 
                           config_key, config_value, config_section, session_id)


# Exception handling utilities with enhanced functionality

def handle_context_analyzer_exception(exception: Exception, session_id: str = None) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format with logging.

    Args:
        exception: Exception to handle
        session_id: Optional session ID for tracking

    Returns:
        Dictionary with standardized error information
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    
    if isinstance(exception, ContextAnalyzerError):
        error_dict = exception.to_dict()
        logger.error(f"Context analyzer error: {exception.message}", 
                    extra={'session_id': session_id, 'error_type': exception.__class__.__name__})
        return error_dict
    else:
        # Handle non-custom exceptions
        error_dict = {
            'error_type': exception.__class__.__name__,
            'error_code': 'UNEXPECTED_ERROR',
            'message': str(exception),
            'details': {'unexpected_error': True},
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        logger.error(f"Unexpected error: {str(exception)}", 
                    extra={'session_id': session_id, 'error_type': exception.__class__.__name__})
        return error_dict


def is_recoverable_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a recoverable error with enhanced logic.

    Args:
        exception: Exception to evaluate

    Returns:
        True if the error is recoverable (retry possible), False otherwise
    """
    # Recoverable errors that can be retried
    recoverable_errors = (
        RateLimitError,
        ProcessingTimeoutError,
        LLMResponseError
    )
    
    if isinstance(exception, recoverable_errors):
        return True
    
    # Special cases for LLM errors
    if isinstance(exception, LLMResponseError):
        return exception.retry_possible
    
    # Safety filter errors may be recoverable with fallback
    if isinstance(exception, SafetyFilterError):
        return exception.fallback_available
    
    return False


def get_retry_delay(exception: Exception, attempt_number: int = 1) -> Optional[float]:
    """
    Get appropriate retry delay for recoverable errors with exponential backoff.

    Args:
        exception: Exception to analyze
        attempt_number: Current attempt number for exponential backoff

    Returns:
        Delay in seconds, or None if not retryable
    """
    base_delay = 1.0
    max_delay = 30.0
    
    if isinstance(exception, RateLimitError) and exception.retry_after:
        return float(exception.retry_after)
    elif isinstance(exception, ProcessingTimeoutError):
        return min(max_delay, base_delay * (2 ** attempt_number))  # Exponential backoff
    elif isinstance(exception, LLMResponseError):
        return min(max_delay, base_delay * attempt_number)  # Linear backoff
    elif isinstance(exception, SafetyFilterError):
        return 2.0  # Short delay for safety fallback
    else:
        return None


def should_retry_after_attempts(exception: Exception, attempts_made: int, max_attempts: int = 3) -> bool:
    """
    Determine if retry should continue after specified attempts.

    Args:
        exception: Exception that occurred
        attempts_made: Number of attempts already made
        max_attempts: Maximum attempts allowed

    Returns:
        True if retry should continue, False otherwise
    """
    if attempts_made >= max_attempts:
        return False
    
    if not is_recoverable_error(exception):
        return False
    
    # Special logic for specific error types
    if isinstance(exception, RateLimitError):
        return attempts_made < (max_attempts + 2)  # Allow extra attempts for rate limits
    
    if isinstance(exception, ProcessingTimeoutError):
        return attempts_made < (max_attempts - 1)  # Fewer attempts for timeouts
    
    return True


def get_fallback_recommendation(exception: Exception, original_operation: str, 
                               session_id: str = None) -> Dict[str, Any]:
    """
    Get fallback recommendations when operations fail.

    Args:
        exception: Exception that occurred
        original_operation: Operation that failed
        session_id: Optional session ID for tracking

    Returns:
        Dictionary with fallback recommendations
    """
    logger = logging.getLogger('context_analyzer.exceptions')
    
    fallback = {
        'fallback_available': False,
        'fallback_type': 'none',
        'recommendations': [],
        'reduced_functionality': True,
        'session_id': session_id
    }
    
    if isinstance(exception, SafetyFilterError):
        fallback.update({
            'fallback_available': True,
            'fallback_type': 'pattern_based_analysis',
            'recommendations': [
                'Use pattern-based bias detection as fallback',
                'Apply rule-based manipulation detection',
                'Generate conservative scoring estimates'
            ],
            'reduced_functionality': True
        })
    elif isinstance(exception, LLMResponseError):
        fallback.update({
            'fallback_available': True,
            'fallback_type': 'alternative_model_or_rules',
            'recommendations': [
                'Try alternative model if available',
                'Use rule-based analysis as backup',
                'Return partial results with confidence indicators'
            ],
            'reduced_functionality': True
        })
    elif isinstance(exception, ProcessingTimeoutError):
        fallback.update({
            'fallback_available': exception.partial_results,
            'fallback_type': 'partial_results',
            'recommendations': [
                'Return partial analysis results',
                'Suggest reduced scope for faster processing',
                'Offer asynchronous processing option'
            ],
            'reduced_functionality': True
        })
    
    logger.info(f"Fallback recommendation generated for {original_operation}", 
               extra={'session_id': session_id, 'fallback_type': fallback['fallback_type']})
    
    return fallback


def validate_score_consistency(text_analysis: str, scores: Dict[str, int], session_id: str = None) -> None:
    """
    Validate that numerical scores match textual analysis and raise appropriate errors.

    Args:
        text_analysis: Generated analysis text
        scores: Dictionary of numerical scores
        session_id: Optional session ID for tracking

    Raises:
        ScoringConsistencyError: If scores don't match text analysis
    """
    if not isinstance(text_analysis, str) or not text_analysis.strip():
        raise_scoring_consistency_error(
            'general', 'Empty or invalid analysis text provided',
            text_analysis, None, None, session_id
        )
    
    if not isinstance(scores, dict) or not scores:
        raise_scoring_consistency_error(
            'general', 'Empty or invalid scores dictionary provided',
            text_analysis, None, None, session_id
        )
    
    text_lower = text_analysis.lower()
    
    # Check each score type for consistency
    for score_type, score_value in scores.items():
        if not isinstance(score_value, (int, float)) or not (0 <= score_value <= 100):
            raise_scoring_consistency_error(
                score_type, f'Invalid score value: {score_value} (must be 0-100)',
                text_analysis, score_value, (0, 100), session_id
            )
        
        # Check for obvious inconsistencies
        if score_value <= 25:  # Low scores
            high_indicators = ['high', 'significant', 'extreme', 'severe', 'major']
            for indicator in high_indicators:
                if f'{indicator} {score_type}' in text_lower:
                    raise_scoring_consistency_error(
                        score_type, f'Text indicates high {score_type} but score is {score_value}',
                        text_analysis[:200], score_value, (70, 100), session_id
                    )
        elif score_value >= 75:  # High scores  
            low_indicators = ['minimal', 'low', 'slight', 'neutral', 'little']
            for indicator in low_indicators:
                if f'{indicator} {score_type}' in text_lower:
                    raise_scoring_consistency_error(
                        score_type, f'Text indicates low {score_type} but score is {score_value}',
                        text_analysis[:200], score_value, (0, 30), session_id
                    )


# Testing functionality
if __name__ == "__main__":
    """Test context analyzer exception functionality with comprehensive examples."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=== CONTEXT ANALYZER EXCEPTIONS TEST ===")
    
    try:
        # Test input validation error
        print("--- Input Validation Error Test ---")
        try:
            raise_input_validation_error(
                'article_text', 
                'Article text too short: 15 chars (minimum: 50)',
                'Short article',
                'length_validation',
                'test_session_001'
            )
        except InputValidationError as e:
            print(f"✅ Input validation error caught: {e}")
            print(f"   Error dict: {e.to_dict()}")

        # Test LLM response error
        print("\n--- LLM Response Error Test ---")
        try:
            raise_llm_response_error(
                'bias_analysis',
                'Response blocked by safety filters',
                'gemini-1.5-pro',
                True,
                'test_session_002'
            )
        except LLMResponseError as e:
            print(f"✅ LLM response error caught: {e}")
            print(f"   Safety filtered: {e.safety_filtered}")
            print(f"   Retry possible: {e.retry_possible}")

        # Test scoring consistency error
        print("\n--- Scoring Consistency Error Test ---")
        try:
            validate_score_consistency(
                "This article shows minimal bias and is very reliable",
                {'bias': 85, 'credibility': 20},  # Inconsistent scores
                'test_session_003'
            )
        except ScoringConsistencyError as e:
            print(f"✅ Scoring consistency error caught: {e}")
            print(f"   Score type: {e.score_type}")
            print(f"   Numerical score: {e.numerical_score}")

        # Test error recovery utilities
        print("\n--- Error Recovery Utilities Test ---")
        
        rate_limit_error = RateLimitError(
            "API rate limit exceeded",
            retry_after=30,
            service="gemini-api",
            session_id='test_session_004'
        )
        
        print(f"Rate limit error recoverable: {'✅' if is_recoverable_error(rate_limit_error) else '❌'}")
        print(f"Retry delay: {get_retry_delay(rate_limit_error)} seconds")
        print(f"Should retry after 2 attempts: {'✅' if should_retry_after_attempts(rate_limit_error, 2) else '❌'}")

        # Test fallback recommendations
        print("\n--- Fallback Recommendations Test ---")
        
        safety_error = SafetyFilterError(
            "Content blocked by harassment filter",
            filter_type="harassment",
            content_category="political",
            session_id='test_session_005'
        )
        
        fallback = get_fallback_recommendation(safety_error, 'bias_analysis', 'test_session_005')
        print(f"Fallback available: {'✅' if fallback['fallback_available'] else '❌'}")
        print(f"Fallback type: {fallback['fallback_type']}")
        print(f"Recommendations: {fallback['recommendations'][:2]}")

        # Test exception handling utility
        print("\n--- Exception Handling Utility Test ---")
        
        try:
            # Simulate an unexpected error
            raise ValueError("Unexpected processing error")
        except Exception as e:
            error_dict = handle_context_analyzer_exception(e, 'test_session_006')
            print(f"✅ Unexpected error handled: {error_dict['error_type']}")
            print(f"   Error code: {error_dict['error_code']}")

        print("\n✅ Context analyzer exception tests completed successfully!")

    except Exception as e:
        print(f"❌ Exception test failed: {str(e)}")
        import traceback
        traceback.print_exc()
