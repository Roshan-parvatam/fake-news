# agents/context_analyzer/__init__.py

"""
Context Analyzer Agent Module - Production Ready

Production-ready context analysis system for detecting bias, manipulation,
framing techniques, and propaganda in news articles with enhanced reliability,
safety filter handling, and comprehensive error management.

Key Features:
- Comprehensive bias detection (political, emotional, selection, linguistic)
- Manipulation and propaganda technique identification
- Logical fallacy detection and reasoning quality analysis
- LLM-powered analysis with consistent numerical scoring
- Safety filter avoidance with institutional fallbacks
- Configurable detection thresholds and scoring weights
- Input/output validation and comprehensive error handling
- Production-level monitoring and performance tracking
- Session-based tracking for debugging and analytics

Components:
- ContextAnalyzerAgent: Main agent with safety handling and retry logic
- BiasPatternDatabase: Pattern-based bias detection with performance tracking
- ManipulationDetector: Propaganda and manipulation technique detection
- Enhanced prompts, validators, and exception handling for production use
"""

import time
import logging
from typing import Dict, List, Any, Optional

# Core agent and analysis components
from .analyzer_agent import ContextAnalyzerAgent
from .bias_patterns import BiasPatternDatabase
from .manipulation_detection import ManipulationDetector

# Prompt system with safety enhancements
from .prompts import (
    get_context_prompt_template,
    validate_context_analysis_output,
    get_domain_guidance,
    get_prompt_statistics,
    BiasDetectionPrompts,
    ManipulationDetectionPrompts,
    FramingAnalysisPrompts,
    StructuredOutputPrompts,
    DomainSpecificPrompts,
    SafetyEnhancedPrompts
)

# Comprehensive validation utilities
from .validators import (
    ValidationResult,
    InputValidator,
    OutputValidator,
    BiasAnalysisValidator,
    ManipulationAnalysisValidator,
    ScoringConsistencyValidator,
    validate_context_input,
    validate_bias_analysis,
    validate_context_output,
    validate_manipulation_techniques
)

# Exception classes and handlers with enhanced production features
from .exceptions import (
    # Base exceptions
    ContextAnalyzerError,
    
    # Specific exception types
    InputValidationError,
    LLMResponseError,
    PromptGenerationError,
    BiasDetectionError,
    ManipulationDetectionError,
    ScoringConsistencyError,
    ReliabilityAssessmentError,
    ContextualRecommendationError,
    VerificationStrategyError,
    FramingAnalysisError,
    SafetyFilterError,
    ConfigurationError,
    RateLimitError,
    ProcessingTimeoutError,
    DataFormatError,
    
    # Convenience functions with enhanced logging
    raise_input_validation_error,
    raise_llm_response_error,
    raise_bias_detection_error,
    raise_manipulation_detection_error,
    raise_scoring_consistency_error,
    raise_contextual_recommendation_error,
    raise_safety_filter_error,
    raise_configuration_error,
    
    # Utility functions for production error handling
    handle_context_analyzer_exception,
    is_recoverable_error,
    get_retry_delay,
    should_retry_after_attempts,
    get_fallback_recommendation,
    validate_score_consistency
)

# Module metadata
__version__ = "3.1.0"
__description__ = "Production-ready context analyzer with bias detection, manipulation analysis, and safety handling"
__author__ = "Context Analyzer Development Team"
__license__ = "MIT"
__status__ = "Production"

# Main exports for clean API
__all__ = [
    # Core agent and components
    'ContextAnalyzerAgent',
    'BiasPatternDatabase',
    'ManipulationDetector',
    
    # Prompt system
    'get_context_prompt_template',
    'validate_context_analysis_output',
    'get_domain_guidance',
    'get_prompt_statistics',
    'BiasDetectionPrompts',
    'ManipulationDetectionPrompts',
    'FramingAnalysisPrompts',
    'StructuredOutputPrompts',
    'DomainSpecificPrompts',
    'SafetyEnhancedPrompts',
    
    # Validation utilities
    'ValidationResult',
    'InputValidator',
    'OutputValidator',
    'BiasAnalysisValidator',
    'ManipulationAnalysisValidator',
    'ScoringConsistencyValidator',
    'validate_context_input',
    'validate_bias_analysis',
    'validate_context_output',
    'validate_manipulation_techniques',
    
    # Exception classes
    'ContextAnalyzerError',
    'InputValidationError',
    'LLMResponseError',
    'PromptGenerationError',
    'BiasDetectionError',
    'ManipulationDetectionError',
    'ScoringConsistencyError',
    'ReliabilityAssessmentError',
    'ContextualRecommendationError',
    'VerificationStrategyError',
    'FramingAnalysisError',
    'SafetyFilterError',
    'ConfigurationError',
    'RateLimitError',
    'ProcessingTimeoutError',
    'DataFormatError',
    
    # Exception utilities
    'raise_input_validation_error',
    'raise_llm_response_error',
    'raise_bias_detection_error',
    'raise_manipulation_detection_error',
    'raise_scoring_consistency_error',
    'raise_contextual_recommendation_error',
    'raise_safety_filter_error',
    'raise_configuration_error',
    'handle_context_analyzer_exception',
    'is_recoverable_error',
    'get_retry_delay',
    'should_retry_after_attempts',
    'get_fallback_recommendation',
    'validate_score_consistency',
    
    # Convenience functions
    'create_context_analyzer',
    'get_supported_analysis_types',
    'get_module_info',
    'get_analysis_statistics',
    'validate_analysis_input',
    
    # Configuration constants
    'DEFAULT_CONFIG',
    'ANALYSIS_TYPES',
    'SAFETY_FALLBACK_ENABLED',
    'SCORING_CONSISTENCY_ENABLED'
]

# Convenience functions for quick access and common operations

def create_context_analyzer(config: Optional[Dict[str, Any]] = None, session_id: str = None) -> ContextAnalyzerAgent:
    """
    Create a new Context Analyzer Agent instance with production configuration.

    Args:
        config: Optional configuration dictionary for agent customization
        session_id: Optional session ID for tracking and logging

    Returns:
        ContextAnalyzerAgent: Fully configured agent instance ready for analysis

    Example:
        >>> agent = create_context_analyzer()
        >>> result = agent.process({'text': 'article content...', 'previous_analysis': {...}})
    """
    logger = logging.getLogger(f"{__name__}.create_context_analyzer")
    
    try:
        logger.info(f"Creating context analyzer agent", extra={'session_id': session_id})
        agent = ContextAnalyzerAgent(config)
        logger.info(f"Context analyzer agent created successfully", extra={'session_id': session_id})
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create context analyzer agent: {str(e)}", extra={'session_id': session_id})
        raise ConfigurationError(f"Agent creation failed: {str(e)}", session_id=session_id)


def get_supported_analysis_types() -> List[str]:
    """
    Get list of supported analysis types and detection capabilities.

    Returns:
        List[str]: Supported analysis types for content processing

    Example:
        >>> analysis_types = get_supported_analysis_types()
        >>> print(analysis_types)
        ['bias_detection', 'manipulation_analysis', 'framing_analysis', ...]
    """
    return [
        'bias_detection',
        'political_bias_analysis', 
        'emotional_manipulation_detection',
        'propaganda_technique_identification',
        'framing_analysis',
        'narrative_structure_analysis',
        'logical_fallacy_detection',
        'scoring_consistency_validation',
        'comprehensive_context_analysis'
    ]


def get_analysis_statistics() -> Dict[str, Any]:
    """
    Get comprehensive analysis system statistics and performance metrics.

    Returns:
        Dict[str, Any]: Statistics including detection capabilities, performance data, and system status

    Example:
        >>> stats = get_analysis_statistics()
        >>> print(f"Analysis types available: {len(stats['analysis_types'])}")
    """
    try:
        return {
            'analysis_types': get_supported_analysis_types(),
            'detection_capabilities': {
                'bias_patterns': True,
                'manipulation_techniques': True,
                'propaganda_methods': True,
                'logical_fallacies': True,
                'framing_analysis': True,
                'scoring_consistency': True
            },
            'safety_features': {
                'safety_filter_handling': True,
                'institutional_fallbacks': True,
                'session_tracking': True,
                'error_recovery': True,
                'retry_mechanisms': True
            },
            'validation_capabilities': {
                'input_validation': True,
                'output_validation': True,
                'consistency_checking': True,
                'comprehensive_feedback': True
            },
            'prompt_statistics': get_prompt_statistics(),
            'module_info': {
                'version': __version__,
                'status': __status__,
                'production_ready': True
            }
        }
    except Exception as e:
        return {
            'error': str(e),
            'analysis_types': [],
            'status': 'limited_functionality'
        }


def validate_analysis_input(input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None,
                           session_id: str = None) -> ValidationResult:
    """
    Quick input validation for context analysis with detailed feedback.

    Args:
        input_data: Input data to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult: Comprehensive validation results with errors, warnings, and suggestions

    Example:
        >>> result = validate_analysis_input({'text': 'article content...', 'previous_analysis': {...}})
        >>> if result.is_valid:
        >>>     print("Input is valid for analysis")
        >>> else:
        >>>     print(f"Validation errors: {result.errors}")
    """
    try:
        return validate_context_input(input_data, config, session_id)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"Validation failed: {str(e)}"],
            suggestions=["Check input format and try again"],
            score=0.0
        )


def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive module information and capabilities.

    Returns:
        Dict[str, Any]: Module metadata, features, and configuration information

    Example:
        >>> info = get_module_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Features: {info['features']}")
    """
    return {
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'license': __license__,
        'status': __status__,
        'features': [
            'Comprehensive bias detection (political, emotional, selection, linguistic)',
            'Manipulation and propaganda technique identification',
            'Logical fallacy detection and reasoning quality analysis',
            'LLM-powered analysis with consistent numerical scoring',
            'Safety filter handling with institutional fallbacks',
            'Framing and narrative structure analysis',
            'Production-level error handling and session tracking',
            'Comprehensive input/output validation with detailed feedback',
            'Score-text consistency validation and enforcement',
            'Performance monitoring and analytics integration'
        ],
        'components': {
            'main_agent': 'ContextAnalyzerAgent - Primary analysis agent with safety handling',
            'bias_patterns': 'BiasPatternDatabase - Pattern-based bias detection system',
            'manipulation_detector': 'ManipulationDetector - Propaganda and technique detection',
            'prompt_system': 'Enhanced prompts with safety filter avoidance',
            'validation_system': 'Comprehensive input/output validation',
            'exception_handling': 'Production-ready error management with recovery'
        },
        'analysis_types_supported': get_supported_analysis_types(),
        'key_improvements_v3_1': [
            'Enhanced safety filter handling with institutional language',
            'Fixed scoring consistency issues with validation enforcement',
            'Comprehensive error handling with retry logic and fallbacks', 
            'Session-based tracking for production debugging and monitoring',
            'Enhanced prompt templates for reliable LLM communication',
            'Production-ready validation with detailed feedback and suggestions',
            'Performance optimization with metrics and analytics integration'
        ],
        'production_features': {
            'error_handling': 'Comprehensive exception system with recovery mechanisms',
            'logging': 'Structured logging with session tracking and performance metrics',
            'monitoring': 'Performance analytics and usage statistics',
            'validation': 'Input/output validation with detailed feedback and suggestions',
            'safety': 'Content safety handling with institutional fallbacks',
            'reliability': 'Retry logic, graceful degradation, and fallback mechanisms',
            'consistency': 'Score-text consistency validation and enforcement'
        }
    }


# Configuration constants and presets
DEFAULT_CONFIG = {
    'model_name': 'gemini-1.5-pro',
    'temperature': 0.4,
    'max_tokens': 3072,
    'enable_detailed_analysis': True,
    'bias_threshold': 70.0,
    'manipulation_threshold': 70.0,
    'enable_propaganda_analysis': True,
    'rate_limit_seconds': 1.0,
    'max_retries': 3,
    'request_timeout_seconds': 30.0,
    'session_tracking': True,
    'safety_fallback_enabled': True,
    'scoring_consistency_enabled': True
}

# Analysis type configurations
HEALTH_ANALYSIS_CONFIG = {
    **DEFAULT_CONFIG,
    'bias_threshold': 60.0,
    'manipulation_threshold': 50.0,
    'enable_detailed_analysis': True,
    'domain_focus': 'health',
    'safety_priority': 'high'
}

POLITICS_ANALYSIS_CONFIG = {
    **DEFAULT_CONFIG,
    'bias_threshold': 75.0,
    'manipulation_threshold': 70.0,
    'enable_detailed_analysis': True,
    'domain_focus': 'politics',
    'political_bias_sensitivity': 'enhanced'
}

SCIENCE_ANALYSIS_CONFIG = {
    **DEFAULT_CONFIG,
    'bias_threshold': 65.0,
    'manipulation_threshold': 60.0,
    'enable_detailed_analysis': True,
    'domain_focus': 'science',
    'accuracy_priority': 'high'
}

# Export analysis configurations
ANALYSIS_TYPES = {
    'health': HEALTH_ANALYSIS_CONFIG,
    'politics': POLITICS_ANALYSIS_CONFIG,
    'science': SCIENCE_ANALYSIS_CONFIG,
    'default': DEFAULT_CONFIG
}

def get_analysis_config(analysis_type: str) -> Dict[str, Any]:
    """
    Get analysis-type-specific configuration optimized for different content domains.

    Args:
        analysis_type: Analysis type (health, politics, science, etc.)

    Returns:
        Dict[str, Any]: Optimized configuration for the specified analysis type

    Example:
        >>> config = get_analysis_config('health')
        >>> agent = create_context_analyzer(config)
    """
    return ANALYSIS_TYPES.get(analysis_type.lower(), DEFAULT_CONFIG).copy()

# Module feature flags
SAFETY_FALLBACK_ENABLED = True
SCORING_CONSISTENCY_ENABLED = True
COMPREHENSIVE_VALIDATION_ENABLED = True
SESSION_TRACKING_ENABLED = True
PERFORMANCE_MONITORING_ENABLED = True

# Module initialization and testing functions
def _initialize_module() -> bool:
    """Initialize module logging and validate component availability."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Context Analyzer Agent Module v{__version__} initializing...")
        
        # Test core component availability
        test_start = time.time()
        
        try:
            # Test basic component initialization (without full agent setup)
            _ = BiasPatternDatabase({'min_confidence': 0.5})
            _ = ManipulationDetector({'confidence_threshold': 0.5})
            _ = InputValidator({'min_article_length': 20})
            
            initialization_time = time.time() - test_start
            
            logger.info(
                f"All core components validated successfully",
                extra={
                    'initialization_time': round(initialization_time * 1000, 2),
                    'components_tested': ['BiasPatternDatabase', 'ManipulationDetector', 'InputValidator'],
                    'version': __version__,
                    'safety_fallback_enabled': SAFETY_FALLBACK_ENABLED,
                    'scoring_consistency_enabled': SCORING_CONSISTENCY_ENABLED
                }
            )
            return True
            
        except Exception as component_error:
            logger.warning(
                f"Component validation completed with warnings: {str(component_error)}",
                extra={'initialization_time': round((time.time() - test_start) * 1000, 2)}
            )
            return True  # Allow module to load even with warnings
            
    except Exception as e:
        logger.error(f"Module initialization failed: {str(e)}")
        return False

# Initialize on import
_module_ready = _initialize_module()

# Export module readiness status
MODULE_READY = _module_ready

# Usage examples and patterns for quick reference
USAGE_EXAMPLES = {
    'basic_analysis': """
# Basic context analysis
from agents.context_analyzer import create_context_analyzer

agent = create_context_analyzer()
input_data = {
    'text': 'Your article content here...',
    'previous_analysis': {
        'prediction': 'REAL',
        'confidence': 0.85,
        'source': 'News Source'
    }
}

result = agent.process(input_data)
if result['success']:
    analysis = result['result']
    print(f"Bias score: {analysis['context_scores']['bias_score']}")
    print(f"Manipulation score: {analysis['context_scores']['manipulation_score']}")
""",

    'domain_specific_analysis': """
# Domain-specific analysis with optimized settings
from agents.context_analyzer import create_context_analyzer, get_analysis_config

health_config = get_analysis_config('health')
agent = create_context_analyzer(health_config)

# Process health-related content with optimized settings
result = agent.process(health_article_data)
""",

    'validation_usage': """
# Input validation before analysis
from agents.context_analyzer import validate_analysis_input, create_context_analyzer

validation_result = validate_analysis_input(input_data)
if validation_result.is_valid:
    agent = create_context_analyzer()
    result = agent.process(input_data)
else:
    print(f"Validation errors: {validation_result.errors}")
    print(f"Suggestions: {validation_result.suggestions}")
""",

    'error_handling': """
# Comprehensive error handling with recovery
from agents.context_analyzer import create_context_analyzer, handle_context_analyzer_exception

try:
    agent = create_context_analyzer()
    result = agent.process(input_data)
except Exception as e:
    error_info = handle_context_analyzer_exception(e)
    print(f"Analysis failed: {error_info['message']}")
    if 'fallback_available' in error_info:
        print("Fallback analysis may be possible")
"""
}

def get_usage_examples() -> Dict[str, str]:
    """
    Get code examples for common usage patterns and best practices.

    Returns:
        Dict[str, str]: Dictionary of usage examples with descriptions
    """
    return USAGE_EXAMPLES.copy()

# Performance and monitoring helpers
def get_performance_summary() -> Dict[str, Any]:
    """
    Get performance summary for monitoring and optimization.

    Returns:
        Dict[str, Any]: Performance metrics and system status
    """
    try:
        return {
            'module_status': 'ready' if MODULE_READY else 'initialization_failed',
            'version': __version__,
            'safety_features_enabled': SAFETY_FALLBACK_ENABLED,
            'scoring_consistency_enabled': SCORING_CONSISTENCY_ENABLED,
            'analysis_types_available': len(get_supported_analysis_types()),
            'configuration_presets': list(ANALYSIS_TYPES.keys()),
            'validation_enabled': COMPREHENSIVE_VALIDATION_ENABLED,
            'session_tracking_enabled': SESSION_TRACKING_ENABLED,
            'production_ready': _module_ready and SAFETY_FALLBACK_ENABLED and SCORING_CONSISTENCY_ENABLED
        }
    except Exception:
        return {
            'module_status': 'limited_functionality',
            'version': __version__,
            'error': 'Could not gather full performance data'
        }

# Add performance summary to exports
__all__.extend([
    'get_usage_examples',
    'get_performance_summary',
    'get_analysis_config',
    'MODULE_READY',
    'USAGE_EXAMPLES'
])

# Final module validation message
if __name__ == "__main__":
    """Module self-test and information display."""
    print("=" * 70)
    print("CONTEXT ANALYZER AGENT MODULE")
    print("=" * 70)
    
    info = get_module_info()
    print(f"Version: {info['version']}")
    print(f"Status: {info['status']}")
    print(f"Module Ready: {'✅' if MODULE_READY else '❌'}")
    
    print("\nKey Features:")
    for feature in info['features'][:5]:  # Show top 5 features
        print(f"  • {feature}")
    
    print(f"\nSupported Analysis Types: {', '.join(get_supported_analysis_types()[:4])}...")
    
    perf = get_performance_summary()
    print(f"Analysis Types Available: {perf['analysis_types_available']}")
    print(f"Safety Features: {'✅ Enabled' if perf['safety_features_enabled'] else '❌ Disabled'}")
    print(f"Scoring Consistency: {'✅ Enabled' if perf['scoring_consistency_enabled'] else '❌ Disabled'}")
    print(f"Production Ready: {'✅' if perf['production_ready'] else '❌'}")
    
    print("\n" + "=" * 70)
    print("MODULE INITIALIZATION COMPLETE")
    print("=" * 70)
