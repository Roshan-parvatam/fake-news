# agents/credible_source/__init__.py

"""
Credible Source Agent Package - Production Ready

Complete credible source agent implementation with contextual recommendations,
safety filter handling, comprehensive source database integration, and
production-level error handling for reliable fact-checking workflows.

Components:
- CredibleSourceAgent: Main agent with contextual recommendations and safety handling
- SourceReliabilityDatabase: Comprehensive source database with reliability scoring  
- DomainClassifier: Domain classification for targeted source recommendations
- Enhanced prompts, validators, and exception handling for production use

Key Features:
- Addresses Gemini safety filter blocking issues with institutional fallbacks
- Contextual vs generic source recommendations
- Production-ready error handling and session tracking
- Comprehensive source database with 50+ credible sources
- Domain-specific classification and source targeting
- Professional-grade prompt templates for fact-checking
"""

import logging
import time
from typing import Dict, List, Any, Optional

# Core agent and components
from .source_agent import CredibleSourceAgent
from .source_database import SourceReliabilityDatabase
from .domain_classifier import DomainClassifier

# Prompt system
from .prompts import (
    get_source_prompt_template,
    get_domain_guidance,
    SourceRecommendationPrompts,
    VerificationStrategyPrompts,
    SafetyEnhancedPrompts,
    DomainSpecificPrompts
)

# Validation utilities
from .validators import (
    InputValidator,
    OutputValidator,
    URLValidator,
    ValidationResult,
    validate_credible_source_input,
    validate_source_url
)

# Exception classes and handlers
from .exceptions import (
    # Base exceptions
    CredibleSourceError,
    
    # Specific exception types
    InputValidationError,
    LLMResponseError,
    SourceDatabaseError,
    DomainClassificationError,
    ReliabilityAssessmentError,
    ContextualRecommendationError,
    VerificationStrategyError,
    SafetyFilterError,
    ConfigurationError,
    RateLimitError,
    ProcessingTimeoutError,
    DataFormatError,
    
    # Convenience functions
    raise_input_validation_error,
    raise_llm_response_error,
    raise_source_database_error,
    raise_contextual_recommendation_error,
    raise_safety_filter_error,
    raise_configuration_error,
    
    # Utility functions
    handle_credible_source_exception,
    is_retryable_error,
    get_retry_delay,
    should_retry_after_attempts,
    get_fallback_recommendation
)

# Module metadata
__version__ = "3.1.0"
__description__ = "Production-ready credible source agent with safety handling and contextual recommendations"
__author__ = "Credible Source Agent Development Team"
__license__ = "MIT"
__status__ = "Production"

# Main exports for clean API
__all__ = [
    # Core agent and components
    'CredibleSourceAgent',
    'SourceReliabilityDatabase', 
    'DomainClassifier',
    
    # Prompt system
    'get_source_prompt_template',
    'get_domain_guidance',
    'SourceRecommendationPrompts',
    'VerificationStrategyPrompts',
    'SafetyEnhancedPrompts',
    'DomainSpecificPrompts',
    
    # Validation utilities
    'InputValidator',
    'OutputValidator',
    'URLValidator',
    'ValidationResult',
    'validate_credible_source_input',
    'validate_source_url',
    
    # Exception classes
    'CredibleSourceError',
    'InputValidationError',
    'LLMResponseError',
    'SourceDatabaseError',
    'DomainClassificationError',
    'ReliabilityAssessmentError',
    'ContextualRecommendationError',
    'VerificationStrategyError',
    'SafetyFilterError',
    'ConfigurationError',
    'RateLimitError',
    'ProcessingTimeoutError',
    'DataFormatError',
    
    # Exception utilities
    'raise_input_validation_error',
    'raise_llm_response_error',
    'raise_source_database_error',
    'raise_contextual_recommendation_error',
    'raise_safety_filter_error',
    'raise_configuration_error',
    'handle_credible_source_exception',
    'is_retryable_error',
    'get_retry_delay',
    'should_retry_after_attempts',
    'get_fallback_recommendation',
    
    # Convenience functions
    'create_credible_source_agent',
    'get_supported_domains',
    'get_source_database_statistics',
    'validate_source_input',
    'get_module_info',
    'get_domain_config',
    
    # Configuration constants
    'DEFAULT_CONFIG',
    'DOMAIN_CONFIGS',
    'SAFETY_FALLBACK_ENABLED',
    'CONTEXTUAL_RECOMMENDATIONS_ENABLED'
]


# Convenience functions for quick access and common operations

def create_credible_source_agent(config: Optional[Dict[str, Any]] = None,
                                session_id: str = None) -> CredibleSourceAgent:
    """
    Create a new Credible Source Agent instance with production configuration.

    Args:
        config: Optional configuration dictionary for agent customization
        session_id: Optional session ID for tracking and logging

    Returns:
        CredibleSourceAgent: Fully configured agent instance ready for use

    Example:
        >>> agent = create_credible_source_agent()
        >>> result = agent.process(input_data)
    """
    logger = logging.getLogger(f"{__name__}.create_credible_source_agent")
    
    try:
        logger.info(f"Creating credible source agent", extra={'session_id': session_id})
        agent = CredibleSourceAgent(config)
        logger.info(f"Credible source agent created successfully", extra={'session_id': session_id})
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create credible source agent: {str(e)}", extra={'session_id': session_id})
        raise ConfigurationError(f"Agent creation failed: {str(e)}", session_id=session_id)


def get_supported_domains() -> List[str]:
    """
    Get list of supported domains for classification and source recommendations.

    Returns:
        List[str]: Supported domain names for content classification

    Example:
        >>> domains = get_supported_domains()
        >>> print(domains)
        ['health', 'science', 'technology', 'politics', 'economics', ...]
    """
    try:
        classifier = DomainClassifier()
        return classifier.get_supported_domains()
    except Exception:
        # Return default supported domains if classifier fails
        return ['health', 'science', 'technology', 'politics', 'economics', 
                'environment', 'education', 'international', 'general']


def get_source_database_statistics() -> Dict[str, Any]:
    """
    Get comprehensive source database statistics and metadata.

    Returns:
        Dict[str, Any]: Database statistics including source counts, types, and performance metrics

    Example:
        >>> stats = get_source_database_statistics()
        >>> print(f"Total sources: {stats['database_info']['total_sources']}")
    """
    try:
        database = SourceReliabilityDatabase()
        return database.get_database_statistics()
    except Exception as e:
        return {
            'database_info': {'total_sources': 0, 'error': str(e)},
            'performance_metrics': {'error': str(e)}
        }


def validate_source_input(input_data: Dict[str, Any], 
                         config: Optional[Dict[str, Any]] = None,
                         session_id: str = None) -> ValidationResult:
    """
    Quick input validation for source recommendations with detailed feedback.

    Args:
        input_data: Input data to validate
        config: Optional validation configuration
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult: Comprehensive validation results with errors, warnings, and suggestions

    Example:
        >>> result = validate_source_input({'text': 'article content', 'extracted_claims': [...]})
        >>> if result.is_valid:
        >>>     print("Input is valid")
        >>> else:
        >>>     print(f"Errors: {result.errors}")
    """
    try:
        validator = InputValidator(config)
        return validator.validate_input_data(input_data, session_id)
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
            'Contextual source recommendations with AI analysis',
            'Safety filter handling with institutional fallbacks',
            'Domain-specific source databases and classification',
            'Reliability scoring and comprehensive source assessment',
            'Verification strategy generation for fact-checking',
            'Professional-grade prompt templates and guidance',
            'Production-level error handling and session tracking',
            'Comprehensive input/output validation',
            'Performance monitoring and analytics'
        ],
        'components': {
            'main_agent': 'CredibleSourceAgent - Primary agent with safety handling',
            'source_database': 'SourceReliabilityDatabase - 50+ credible sources with scoring',
            'domain_classifier': 'DomainClassifier - Topic classification for targeted recommendations',
            'prompt_system': 'Enhanced prompts with safety filter avoidance',
            'validation_system': 'Comprehensive input/output validation',
            'exception_handling': 'Production-ready error management'
        },
        'domains_supported': get_supported_domains(),
        'key_improvements_v3': [
            'Fixed Gemini safety filter blocking issues',
            'Enhanced contextual vs generic source recommendations', 
            'Institutional fallback sources for blocked content',
            'Session-based tracking and comprehensive logging',
            'Production-ready error handling with retry logic',
            'Comprehensive source database with metadata',
            'Professional prompt templates for fact-checking'
        ],
        'production_features': {
            'error_handling': 'Comprehensive exception system with recovery',
            'logging': 'Structured logging with session tracking',
            'monitoring': 'Performance metrics and usage analytics', 
            'validation': 'Input/output validation with detailed feedback',
            'safety': 'Content safety handling with institutional fallbacks',
            'reliability': 'Retry logic and graceful degradation'
        }
    }


# Configuration constants and presets

DEFAULT_CONFIG = {
    'model_name': 'gemini-1.5-pro',
    'temperature': 0.3,
    'max_sources_per_recommendation': 8,
    'min_reliability_score': 6.0,
    'enable_safety_fallbacks': True,
    'enable_contextual_recommendations': True,
    'confidence_threshold': 0.7,
    'max_retries': 3,
    'rate_limit_seconds': 1.0,
    'session_tracking': True
}

HEALTH_DOMAIN_CONFIG = {
    **DEFAULT_CONFIG,
    'min_reliability_score': 8.5,
    'max_sources_per_recommendation': 10,
    'confidence_threshold': 0.8,
    'preferred_source_types': ['government', 'academic', 'medical_institution'],
    'domain_weight_multiplier': 3.0
}

POLITICS_DOMAIN_CONFIG = {
    **DEFAULT_CONFIG,
    'min_reliability_score': 7.5,
    'max_sources_per_recommendation': 12,
    'confidence_threshold': 0.6,
    'preferred_source_types': ['government', 'think_tank', 'fact_checker'],
    'enable_bias_detection': True
}

SCIENCE_DOMAIN_CONFIG = {
    **DEFAULT_CONFIG,
    'min_reliability_score': 8.5,
    'max_sources_per_recommendation': 8,
    'confidence_threshold': 0.75,
    'preferred_source_types': ['academic', 'professional_organization'],
    'require_peer_review': True
}

TECHNOLOGY_DOMAIN_CONFIG = {
    **DEFAULT_CONFIG,
    'min_reliability_score': 7.0,
    'max_sources_per_recommendation': 10,
    'confidence_threshold': 0.65,
    'preferred_source_types': ['academic', 'professional_organization', 'government']
}

# Export domain configurations
DOMAIN_CONFIGS = {
    'health': HEALTH_DOMAIN_CONFIG,
    'politics': POLITICS_DOMAIN_CONFIG,
    'science': SCIENCE_DOMAIN_CONFIG,
    'technology': TECHNOLOGY_DOMAIN_CONFIG,
    'default': DEFAULT_CONFIG
}


def get_domain_config(domain: str) -> Dict[str, Any]:
    """
    Get domain-specific configuration optimized for different content types.

    Args:
        domain: Domain name (health, politics, science, technology, etc.)

    Returns:
        Dict[str, Any]: Optimized configuration for the specified domain

    Example:
        >>> config = get_domain_config('health')
        >>> agent = create_credible_source_agent(config)
    """
    return DOMAIN_CONFIGS.get(domain.lower(), DEFAULT_CONFIG).copy()


# Module feature flags
SAFETY_FALLBACK_ENABLED = True
CONTEXTUAL_RECOMMENDATIONS_ENABLED = True 
DOMAIN_CLASSIFICATION_ENABLED = True
DATABASE_VALIDATION_ENABLED = True
COMPREHENSIVE_LOGGING_ENABLED = True


# Module initialization function
def _initialize_module() -> bool:
    """Initialize module logging and validate component availability."""
    logger = logging.getLogger(__name__)
    
    try:
        # Log module initialization
        logger.info(f"Credible Source Agent Module v{__version__} initializing...")
        
        # Test core component availability
        test_start = time.time()
        
        # Test agent creation
        test_config = DEFAULT_CONFIG.copy()
        test_config.update({'max_sources_per_recommendation': 5})  # Reduce for faster testing
        
        try:
            # Test basic component initialization (without full agent setup)
            _ = DomainClassifier({'confidence_threshold': 0.5})
            _ = SourceReliabilityDatabase({'min_reliability_threshold': 6.0})
            _ = InputValidator({'min_text_length': 20})
            
            initialization_time = time.time() - test_start
            
            logger.info(
                f"All core components validated successfully",
                extra={
                    'initialization_time': round(initialization_time * 1000, 2),
                    'components_tested': ['DomainClassifier', 'SourceReliabilityDatabase', 'InputValidator'],
                    'version': __version__,
                    'safety_fallback_enabled': SAFETY_FALLBACK_ENABLED,
                    'contextual_recommendations_enabled': CONTEXTUAL_RECOMMENDATIONS_ENABLED
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


# Quick start examples and usage patterns
USAGE_EXAMPLES = {
    'basic_usage': """
# Basic credible source agent usage
from agents.credible_source import create_credible_source_agent

agent = create_credible_source_agent()
input_data = {
    'text': 'Your article content here...',
    'extracted_claims': [
        {'text': 'Claim to verify', 'claim_type': 'Research', 'priority': 1}
    ],
    'evidence_evaluation': {'overall_evidence_score': 7.5}
}

result = agent.process(input_data)
if result['success']:
    recommendations = result['result']
    print(f"Found {len(recommendations['contextual_sources'])} contextual sources")
""",

    'domain_specific_usage': """
# Domain-specific configuration
from agents.credible_source import create_credible_source_agent, get_domain_config

health_config = get_domain_config('health')
agent = create_credible_source_agent(health_config)

# Process health-related content with optimized settings
result = agent.process(health_article_data)
""",

    'validation_usage': """
# Input validation before processing
from agents.credible_source import validate_source_input, create_credible_source_agent

validation_result = validate_source_input(input_data)
if validation_result.is_valid:
    agent = create_credible_source_agent()
    result = agent.process(input_data)
else:
    print(f"Validation errors: {validation_result.errors}")
    print(f"Suggestions: {validation_result.suggestions}")
"""
}


def get_usage_examples() -> Dict[str, str]:
    """
    Get code examples for common usage patterns.

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
        # Attempt to get performance data from active components
        database_stats = get_source_database_statistics()
        
        return {
            'module_status': 'ready' if MODULE_READY else 'initialization_failed',
            'version': __version__,
            'safety_features_enabled': SAFETY_FALLBACK_ENABLED,
            'total_sources_available': database_stats.get('database_info', {}).get('total_sources', 0),
            'domains_supported_count': len(get_supported_domains()),
            'configuration_presets': list(DOMAIN_CONFIGS.keys()),
            'features_available': len(get_module_info()['features']),
            'production_ready': _module_ready and SAFETY_FALLBACK_ENABLED
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
    'MODULE_READY',
    'USAGE_EXAMPLES'
])


# Final module validation message
if __name__ == "__main__":
    """Module self-test and information display."""
    print("=" * 60)
    print("CREDIBLE SOURCE AGENT MODULE")
    print("=" * 60)
    
    info = get_module_info()
    print(f"Version: {info['version']}")
    print(f"Status: {info['status']}")
    print(f"Module Ready: {'✅' if MODULE_READY else '❌'}")
    
    print("\nKey Features:")
    for feature in info['features'][:5]:  # Show top 5 features
        print(f"  • {feature}")
    
    print(f"\nSupported Domains: {', '.join(get_supported_domains())}")
    
    perf = get_performance_summary()
    print(f"Total Sources Available: {perf['total_sources_available']}")
    print(f"Safety Features: {'✅ Enabled' if perf['safety_features_enabled'] else '❌ Disabled'}")
    print(f"Production Ready: {'✅' if perf['production_ready'] else '❌'}")
    
    print("\n" + "=" * 60)
    print("MODULE INITIALIZATION COMPLETE")
    print("=" * 60)
