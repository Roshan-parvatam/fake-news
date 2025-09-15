# agents/evidence_evaluator/__init__.py

"""
Evidence Evaluator Agent Module - Production Ready

Enhanced evidence evaluation system for assessing news article credibility,
source quality, and logical consistency with production-level reliability.

Features:
- Comprehensive evidence quality assessment with configurable scoring
- Advanced logical fallacy detection using rule-based pattern matching
- LLM-powered verification source generation with URL specificity validation
- Structured logging and session tracking for production monitoring
- Robust error handling with retry logic and graceful degradation
- Enhanced validation with clear error messages and suggestions
- State-of-the-art prompt engineering with Chain-of-Thought reasoning
- Hybrid architecture combining LLM reasoning with traditional reliability

Production Improvements:
- Session-based tracking and structured logging
- Comprehensive error handling with fallback mechanisms  
- Configurable parameters and scoring weights
- Performance metrics and monitoring
- Input/output validation with helpful feedback
- Retry logic for API failures with exponential backoff
"""

import logging
from typing import Dict, Any, Optional

# Core agent import
from .evaluator_agent import EvidenceEvaluatorAgent

# Assessment components
from .criteria import EvidenceQualityCriteria
from .fallacy_detection import LogicalFallacyDetector

# Prompt system
from .prompts import (
    get_prompt_template, 
    PromptValidator,
    EvidenceVerificationPrompts,
    LogicalConsistencyPrompts,
    EvidenceGapPrompts,
    StructuredOutputPrompts,
    DomainSpecificPrompts
)

# Validation utilities
from .validators import (
    validate_evidence_input,
    validate_url_specificity,
    validate_verification_output,
    InputValidator,
    URLValidator,
    OutputValidator,
    ValidationResult
)

# Exception classes
from .exceptions import (
    # Base exception
    EvidenceEvaluatorError,
    
    # Specific exceptions
    InputValidationError,
    LLMResponseError,
    APIError,
    PromptGenerationError,
    VerificationSourceError,
    EvidenceAssessmentError,
    FallacyDetectionError,
    ConfigurationError,
    RateLimitError,
    ProcessingTimeoutError,
    DataFormatError,
    
    # Utility functions
    handle_evidence_evaluator_exception,
    is_retryable_error,
    get_retry_delay,
    should_retry_after_attempts,
    
    # Convenience functions
    raise_input_validation_error,
    raise_llm_response_error,
    raise_api_error,
    raise_verification_source_error,
    raise_configuration_error
)

# Module metadata
__version__ = "3.1.0"
__description__ = "Production-ready evidence evaluation with LLM-powered verification and rule-based assessment"
__author__ = "Evidence Evaluator Team"

# Main exports - organized by category
__all__ = [
    # Core agent
    'EvidenceEvaluatorAgent',
    
    # Assessment components  
    'EvidenceQualityCriteria',
    'LogicalFallacyDetector',
    
    # Prompt system
    'get_prompt_template',
    'PromptValidator', 
    'EvidenceVerificationPrompts',
    'LogicalConsistencyPrompts',
    'EvidenceGapPrompts',
    'StructuredOutputPrompts',
    'DomainSpecificPrompts',
    
    # Validation system
    'validate_evidence_input',
    'validate_url_specificity', 
    'validate_verification_output',
    'InputValidator',
    'URLValidator',
    'OutputValidator',
    'ValidationResult',
    
    # Exception hierarchy
    'EvidenceEvaluatorError',
    'InputValidationError',
    'LLMResponseError',
    'APIError', 
    'PromptGenerationError',
    'VerificationSourceError',
    'EvidenceAssessmentError',
    'FallacyDetectionError',
    'ConfigurationError',
    'RateLimitError',
    'ProcessingTimeoutError',
    'DataFormatError',
    
    # Exception utilities
    'handle_evidence_evaluator_exception',
    'is_retryable_error',
    'get_retry_delay',
    'should_retry_after_attempts',
    
    # Convenience functions
    'raise_input_validation_error',
    'raise_llm_response_error', 
    'raise_api_error',
    'raise_verification_source_error',
    'raise_configuration_error',
    'create_evidence_evaluator',
    'get_module_info',
    'get_default_config'
]

# Module logger
logger = logging.getLogger(__name__)


def create_evidence_evaluator(config: Optional[Dict[str, Any]] = None, 
                            session_id: Optional[str] = None) -> EvidenceEvaluatorAgent:
    """
    Create a new Evidence Evaluator Agent instance with production configuration.
    
    Args:
        config: Optional configuration dictionary for customizing agent behavior
        session_id: Optional session ID for tracking and logging
        
    Returns:
        EvidenceEvaluatorAgent: Configured agent instance ready for production use
        
    Example:
        >>> # Basic usage
        >>> agent = create_evidence_evaluator()
        >>> 
        >>> # With custom configuration
        >>> config = {
        >>>     'temperature': 0.2,
        >>>     'max_retries': 5,
        >>>     'enable_fallacy_detection': True,
        >>>     'scoring_weights': {
        >>>         'source_quality': 0.4,
        >>>         'logical_consistency': 0.3,
        >>>         'evidence_completeness': 0.2,
        >>>         'verification_quality': 0.1
        >>>     }
        >>> }
        >>> agent = create_evidence_evaluator(config, session_id="eval_001")
    """
    try:
        agent = EvidenceEvaluatorAgent(config)
        
        logger.info(f"Evidence evaluator created successfully", 
                   extra={
                       'session_id': session_id,
                       'version': __version__,
                       'config_provided': config is not None
                   })
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create evidence evaluator: {str(e)}", 
                    extra={'session_id': session_id})
        raise ConfigurationError(
            f"Evidence evaluator creation failed: {str(e)}",
            config_key="agent_initialization",
            session_id=session_id
        )


def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive module information and capabilities.
    
    Returns:
        dict: Module metadata, features, and component information
        
    Example:
        >>> info = get_module_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Features: {info['features']}")
    """
    return {
        'version': __version__,
        'description': __description__,
        'features': [
            'Evidence quality assessment with configurable scoring',
            'Logical fallacy detection using pattern matching',
            'LLM-powered verification source generation',
            'URL specificity validation and domain credibility checking',
            'Structured logging with session tracking',
            'Comprehensive error handling with retry logic',
            'Input/output validation with helpful feedback',
            'State-of-the-art prompt engineering',
            'Production-level monitoring and metrics'
        ],
        'components': {
            'main_agent': 'EvidenceEvaluatorAgent - Core evaluation orchestrator',
            'criteria_assessment': 'EvidenceQualityCriteria - Rule-based quality scoring', 
            'fallacy_detection': 'LogicalFallacyDetector - Pattern-based fallacy identification',
            'prompt_system': 'Enhanced prompts with Chain-of-Thought reasoning',
            'validation_system': 'Comprehensive input/output validation',
            'exception_system': 'Structured error handling with retry support'
        },
        'architecture': {
            'type': 'Hybrid (LLM + Rule-based)',
            'llm_components': ['Verification sources', 'Logical analysis', 'Evidence gaps'],
            'rule_based_components': ['Fallacy detection', 'Quality scoring', 'URL validation'],
            'benefits': ['Cost-effective', 'Reliable scoring', 'Fast processing', 'Deterministic results']
        },
        'production_features': {
            'error_handling': 'Comprehensive with fallback mechanisms',
            'logging': 'Structured with session tracking',
            'monitoring': 'Performance metrics and error rates',
            'validation': 'Input/output with helpful suggestions',
            'retry_logic': 'Exponential backoff for API failures',
            'configuration': 'Fully configurable scoring and parameters'
        }
    }


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for the Evidence Evaluator Agent.
    
    Returns:
        dict: Default configuration with recommended production settings
        
    Example:
        >>> config = get_default_config()
        >>> # Customize specific settings
        >>> config['temperature'] = 0.1  # More deterministic
        >>> config['max_retries'] = 5     # More resilient
        >>> agent = create_evidence_evaluator(config)
    """
    return {
        # Core LLM settings
        'model_name': 'gemini-1.5-pro',
        'temperature': 0.3,
        'max_tokens': 3072,
        'top_p': 0.9,
        'top_k': 40,
        
        # Processing settings
        'enable_detailed_analysis': True,
        'evidence_threshold': 6.0,
        'enable_fallacy_detection': True,
        'max_verification_sources': 50,  # Much higher limit to allow more RAG sources
        'max_article_length': 4000,
        'max_claims_for_processing': 20,
        
        # Scoring weights (must sum to 1.0)
        'scoring_weights': {
            'source_quality': 0.35,
            'logical_consistency': 0.3,
            'evidence_completeness': 0.25,
            'verification_quality': 0.1
        },
        
        # Retry and rate limiting
        'max_retries': 3,
        'retry_delay': 2.0,
        'rate_limit_seconds': 1.0,
        
        # Fallacy detection weights  
        'fallacy_weights': {
            'confidence_multiplier': 0.15,
            'fallacy_penalty_multiplier': 1.5
        },
        
        # Health thresholds
        'health_thresholds': {
            'excellent': 8.0,
            'good': 6.0,
            'fair': 4.0, 
            'poor': 2.0
        },
        
        # Evidence strength weights
        'strength_weights': {
            'strong_evidence': 3.0,
            'moderate_evidence': 1.5,
            'weak_evidence': -1.0,
            'quantitative_indicators': 2.0
        },
        
        # Overall quality weights
        'overall_weights': {
            'source_quality': 0.3,
            'evidence_strength': 0.25,
            'verification': 0.2,
            'transparency': 0.15,
            'claims_quality': 0.1
        },
        
        # Safety settings for Gemini
        'safety_settings': [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"}, 
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]
    }


def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate Evidence Evaluator configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        ValidationResult: Validation status with errors and suggestions
        
    Example:
        >>> config = get_default_config()
        >>> config['temperature'] = 5.0  # Invalid value
        >>> result = validate_config(config)
        >>> if not result.is_valid:
        >>>     print(f"Errors: {result.errors}")
        >>>     print(f"Suggestions: {result.suggestions}")
    """
    from .validators import ValidationResult
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
    
    # Validate scoring weights
    if 'scoring_weights' in config:
        weights = config['scoring_weights']
        if isinstance(weights, dict):
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                result.add_error(
                    f"Scoring weights sum to {total_weight:.3f}, must sum to 1.0",
                    "Adjust weights so they total exactly 1.0"
                )
        else:
            result.add_error(
                "Scoring weights must be a dictionary",
                "Use format: {'source_quality': 0.35, 'logical_consistency': 0.3, ...}"
            )
    
    # Validate temperature
    if 'temperature' in config:
        temp = config['temperature']
        if not isinstance(temp, (int, float)) or not (0 <= temp <= 2):
            result.add_error(
                f"Temperature must be between 0-2, got {temp}",
                "Use value between 0.0 (deterministic) and 2.0 (creative)"
            )
    
    # Validate max_retries
    if 'max_retries' in config:
        retries = config['max_retries']
        if not isinstance(retries, int) or retries < 0:
            result.add_error(
                f"Max retries must be non-negative integer, got {retries}",
                "Use integer value >= 0, recommended: 3"
            )
    
    # Validate thresholds
    threshold_fields = ['evidence_threshold']
    for field in threshold_fields:
        if field in config:
            threshold = config[field]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 10):
                result.add_error(
                    f"{field} must be between 0-10, got {threshold}",
                    f"Use value between 0.0 and 10.0 for {field}"
                )
    
    logger.debug(f"Configuration validation completed: {result.is_valid}")
    return result


# Module initialization with production logging
def _initialize_module():
    """Initialize module with production logging setup."""
    try:
        logger.info(f"Evidence Evaluator Module v{__version__} initializing")
        
        # Verify core dependencies
        required_components = [
            EvidenceEvaluatorAgent,
            EvidenceQualityCriteria, 
            LogicalFallacyDetector,
            get_prompt_template
        ]
        
        missing_components = []
        for component in required_components:
            if component is None:
                missing_components.append(component.__name__ if hasattr(component, '__name__') else str(component))
        
        if missing_components:
            logger.error(f"Missing required components: {missing_components}")
            raise ImportError(f"Failed to import required components: {missing_components}")
        
        logger.info(f"Evidence Evaluator Module v{__version__} initialized successfully")
        logger.debug(f"Available components: {len(__all__)} exports")
        
    except Exception as e:
        logger.error(f"Module initialization failed: {str(e)}")
        raise


# Initialize module on import
_initialize_module()


# Example usage for documentation
if __name__ == "__main__":
    """
    Example usage of the Evidence Evaluator Module.
    
    This demonstrates the main functionality and production features
    of the enhanced evidence evaluation system.
    """
    import json
    
    print("=" * 60)
    print("EVIDENCE EVALUATOR MODULE - PRODUCTION DEMO")  
    print("=" * 60)
    
    # Display module information
    print("\n📋 MODULE INFORMATION")
    info = get_module_info()
    print(f"Version: {info['version']}")
    print(f"Architecture: {info['architecture']['type']}")
    print("\nKey Features:")
    for feature in info['features'][:5]:  # Show first 5 features
        print(f"  • {feature}")
    
    # Show configuration example
    print("\n⚙️  DEFAULT CONFIGURATION")
    config = get_default_config()
    print("Key settings:")
    print(f"  • Model: {config['model_name']}")
    print(f"  • Temperature: {config['temperature']}")
    print(f"  • Max retries: {config['max_retries']}")
    print(f"  • Fallacy detection: {config['enable_fallacy_detection']}")
    
    # Demonstrate agent creation
    print("\n🚀 AGENT CREATION DEMO")
    try:
        # Create agent with default config
        agent = create_evidence_evaluator(session_id="demo_session")
        print("✅ Agent created successfully")
        
        # Show performance metrics
        metrics = agent.get_performance_metrics()
        print(f"✅ Agent ready - Model: {metrics.get('model_config', {}).get('model_name', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {str(e)}")
    
    # Demonstrate validation
    print("\n✅ VALIDATION DEMO") 
    test_config = get_default_config()
    test_config['temperature'] = 5.0  # Invalid value
    
    validation = validate_config(test_config)
    if not validation.is_valid:
        print(f"❌ Config validation failed:")
        for error in validation.errors[:2]:  # Show first 2 errors
            print(f"   • {error}")
    else:
        print("✅ Configuration valid")
    
    # Show available exceptions
    print("\n🔧 ERROR HANDLING")
    print("Available exception types:")
    exception_types = [
        'InputValidationError', 'LLMResponseError', 'ConfigurationError',
        'VerificationSourceError', 'RateLimitError'
    ]
    for exc_type in exception_types:
        print(f"  • {exc_type}")
    
    print(f"\n✅ Demo completed - Evidence Evaluator Module v{__version__} ready for production")
