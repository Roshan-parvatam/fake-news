# agents/claim_extractor/__init__.py

"""
Claim Extractor Agent Package - Production Ready

Production-ready claim extraction agent with modular architecture.
Provides comprehensive claim identification, verification analysis,
and prioritization with enhanced error handling and performance tracking.

Components:
- ClaimExtractorAgent: Main extraction agent with AI integration
- ClaimPatternDatabase: Pattern-based claim detection and analysis
- ClaimParser: Structured claim parsing with multiple fallback methods
- Enhanced prompt templates, validators, and exception handling

Features:
- AI-powered claim extraction with Gemini integration
- Pattern-based pre-analysis and claim detection
- Multiple parsing strategies with intelligent fallbacks
- Verification analysis and claim prioritization
- Comprehensive error handling and recovery
- Performance tracking and quality metrics
- Configuration-driven behavior
- Safety filter avoidance with institutional language
"""

# Core agent and components
from .extractor_agent import ClaimExtractorAgent
from .patterns import ClaimPatternDatabase
from .parsers import ClaimParser

# Prompt system
from .prompts import (
    ClaimExtractionPrompts,
    ClaimCategorizationPrompts,
    StructuredOutputPrompts,
    get_claim_prompt_template,
    validate_claim_extraction_output,
    get_domain_guidance,
    get_prompt_statistics
)

# Validation utilities
from .validators import (
    InputValidator,
    OutputValidator,
    ValidationResult,
    validate_input,
    validate_output,
    validate_claims
)

# Exception classes and utilities
from .exceptions import (
    # Base exception
    ClaimExtractorError,
    
    # Specific exceptions
    InputValidationError,
    LLMResponseError,
    ClaimParsingError,
    ClaimExtractionError,
    VerificationAnalysisError,
    PrioritizationError,
    PatternAnalysisError,
    ConfigurationError,
    RateLimitError,
    ProcessingTimeoutError,
    DataFormatError,
    
    # Convenience functions
    raise_input_validation_error,
    raise_llm_response_error,
    raise_claim_parsing_error,
    raise_claim_extraction_error,
    raise_configuration_error,
    
    # Utility functions
    handle_claim_extractor_exception,
    is_recoverable_error,
    get_retry_delay,
    get_fallback_recommendation,
    log_exception_with_context
)

# Module metadata
__version__ = "3.1.0"
__description__ = "Production-ready modular claim extraction agent with AI integration and pattern analysis"
__author__ = "Claim Extractor Development Team"
__license__ = "MIT"

# Main exports
__all__ = [
    # Core agent and components
    'ClaimExtractorAgent',
    'ClaimPatternDatabase',
    'ClaimParser',
    
    # Prompt system
    'ClaimExtractionPrompts',
    'ClaimCategorizationPrompts',
    'StructuredOutputPrompts',
    'get_claim_prompt_template',
    'validate_claim_extraction_output',
    'get_domain_guidance',
    'get_prompt_statistics',
    
    # Validation utilities
    'InputValidator',
    'OutputValidator',
    'ValidationResult',
    'validate_input',
    'validate_output',
    'validate_claims',
    
    # Exception classes
    'ClaimExtractorError',
    'InputValidationError',
    'LLMResponseError',
    'ClaimParsingError',
    'ClaimExtractionError',
    'VerificationAnalysisError',
    'PrioritizationError',
    'PatternAnalysisError',
    'ConfigurationError',
    'RateLimitError',
    'ProcessingTimeoutError',
    'DataFormatError',
    
    # Exception utilities
    'raise_input_validation_error',
    'raise_llm_response_error',
    'raise_claim_parsing_error',
    'raise_claim_extraction_error',
    'raise_configuration_error',
    'handle_claim_extractor_exception',
    'is_recoverable_error',
    'get_retry_delay',
    'get_fallback_recommendation',
    'log_exception_with_context',
    
    # Convenience functions
    'create_claim_extractor',
    'extract_claims_quick',
    'validate_article_input',
    'get_claim_extraction_info',
    'get_extraction_config',
    
    # Module info
    '__version__',
    '__description__'
]


# Convenience functions for quick access

def create_claim_extractor(config=None, session_id=None):
    """
    Create a new Claim Extractor Agent instance with enhanced configuration.

    Args:
        config: Optional configuration dictionary
        session_id: Optional session ID for tracking

    Returns:
        ClaimExtractorAgent: Configured agent instance with all components initialized

    Example:
        >>> agent = create_claim_extractor({'max_claims_per_article': 10})
        >>> result = agent.process({'text': 'Sample article text...'})
    """
    try:
        agent = ClaimExtractorAgent(config)
        
        if hasattr(agent, 'logger'):
            agent.logger.info(f"Claim extractor created via convenience function", 
                            extra={'session_id': session_id})
        
        return agent
    
    except Exception as e:
        raise ConfigurationError(f"Failed to create claim extractor: {str(e)}")


def extract_claims_quick(article_text: str, max_claims: int = 5, session_id: str = None):
    """
    Quick claim extraction using pattern-based detection only.
    
    Bypasses AI processing for fast extraction using pattern matching.

    Args:
        article_text: Article content to analyze
        max_claims: Maximum number of claims to extract
        session_id: Optional session ID for tracking

    Returns:
        List of potential claim texts

    Example:
        >>> claims = extract_claims_quick("Study shows 85% improvement in patients...")
        >>> print(f"Found {len(claims)} potential claims")
    """
    try:
        pattern_db = ClaimPatternDatabase()
        return pattern_db.extract_potential_claims(article_text, max_claims, session_id)
    
    except Exception as e:
        if session_id:
            log_exception_with_context(e, session_id, {'operation': 'quick_extraction'})
        return [f"Error extracting claims: {str(e)}"]


def validate_article_input(article_text: str, min_length: int = 50, session_id: str = None):
    """
    Quick validation of article text input with enhanced feedback.

    Args:
        article_text: Article text to validate
        min_length: Minimum required length
        session_id: Optional session ID for tracking

    Returns:
        ValidationResult: Comprehensive validation results with suggestions

    Example:
        >>> result = validate_article_input("Short text")
        >>> if not result.is_valid:
        ...     print(f"Validation failed: {result.errors}")
    """
    try:
        validator = InputValidator({'min_text_length': min_length})
        return validator.validate_article_text(article_text, session_id)
    
    except Exception as e:
        if session_id:
            log_exception_with_context(e, session_id, {'operation': 'input_validation'})
        
        return ValidationResult(
            False, 
            [f"Validation error: {str(e)}"], 
            [], 
            0.0, 
            {}, 
            ["Check input format and try again"]
        )


def get_claim_extraction_info():
    """
    Get comprehensive information about claim extraction capabilities.

    Returns:
        dict: Module capabilities, configuration options, and usage examples

    Example:
        >>> info = get_claim_extraction_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Features: {len(info['features'])}")
    """
    return {
        'version': __version__,
        'description': __description__,
        'features': [
            'AI-powered claim extraction with Gemini integration',
            'Pattern-based pre-analysis and claim detection',
            'Multiple parsing strategies with intelligent fallbacks',
            'Verification analysis and claim prioritization',
            'Comprehensive error handling and recovery',
            'Performance tracking and quality metrics',
            'Configuration-driven behavior',
            'Safety filter avoidance with institutional language',
            'Session tracking and audit trails',
            'Production-ready monitoring and alerting'
        ],
        'claim_types_supported': [
            'Statistical', 'Event', 'Attribution', 'Research',
            'Policy', 'Causal', 'Other'
        ],
        'components': {
            'main_agent': {
                'class': 'ClaimExtractorAgent',
                'description': 'Main extraction agent with AI integration',
                'features': ['LLM integration', 'retry logic', 'safety handling']
            },
            'pattern_database': {
                'class': 'ClaimPatternDatabase',
                'description': 'Pattern-based claim detection',
                'features': ['150+ patterns', '7 claim types', 'performance tracking']
            },
            'claim_parser': {
                'class': 'ClaimParser',
                'description': 'Multi-strategy claim parsing',
                'features': ['5 parsing methods', 'intelligent fallbacks', 'quality assessment']
            },
            'prompt_system': {
                'description': 'Enhanced prompt templates',
                'features': ['institutional language', '7 prompt types', 'safety optimized']
            },
            'validation': {
                'description': 'Input/output validators',
                'features': ['security checks', 'quality assessment', 'detailed feedback']
            },
            'error_handling': {
                'description': 'Comprehensive exception system',
                'features': ['12 exception types', 'recovery strategies', 'detailed context']
            }
        },
        'key_improvements': [
            'Enhanced safety handling with institutional language',
            'Comprehensive pattern database with 150+ detection patterns',
            'Multiple parsing fallback strategies for reliability',
            'Production-ready error handling with recovery recommendations',
            'Detailed performance tracking and quality metrics',
            'Clean, modular architecture with pluggable components',
            'Session tracking for complete audit trails',
            'Configuration-driven behavior for flexibility'
        ],
        'usage_examples': {
            'basic_usage': '''
                from agents.claim_extractor import ClaimExtractorAgent
                
                agent = ClaimExtractorAgent()
                result = agent.process({
                    'text': 'Your article text here...',
                    'bert_results': {'prediction': 'REAL', 'confidence': 0.85}
                })
            ''',
            'quick_extraction': '''
                from agents.claim_extractor import extract_claims_quick
                
                claims = extract_claims_quick("Article text...", max_claims=5)
            ''',
            'with_configuration': '''
                from agents.claim_extractor import create_claim_extractor
                
                config = {
                    'max_claims_per_article': 10,
                    'enable_verification_analysis': True,
                    'temperature': 0.2
                }
                agent = create_claim_extractor(config)
            '''
        },
        'production_features': [
            'Rate limiting and API management',
            'Comprehensive error recovery',
            'Performance monitoring and metrics',
            'Security validation and filtering',
            'Session tracking and audit logs',
            'Configuration management',
            'Quality assessment and scoring',
            'Fallback strategies for reliability'
        ]
    }


# Configuration templates for different use cases

DEFAULT_EXTRACTION_CONFIG = {
    'model_name': 'gemini-1.5-pro',
    'temperature': 0.3,
    'max_claims_per_article': 8,
    'min_claim_length': 10,
    'enable_verification_analysis': True,
    'enable_claim_prioritization': True,
    'enable_pattern_preprocessing': True,
    'pattern_confidence_threshold': 0.5,
    'claim_richness_threshold': 5.0,
    'rate_limit_seconds': 1.0,
    'max_retries': 3
}

QUICK_EXTRACTION_CONFIG = {
    **DEFAULT_EXTRACTION_CONFIG,
    'max_claims_per_article': 5,
    'enable_verification_analysis': False,
    'enable_claim_prioritization': False,
    'enable_pattern_preprocessing': True,
    'temperature': 0.2
}

COMPREHENSIVE_EXTRACTION_CONFIG = {
    **DEFAULT_EXTRACTION_CONFIG,
    'max_claims_per_article': 12,
    'enable_verification_analysis': True,
    'enable_claim_prioritization': True,
    'enable_pattern_preprocessing': True,
    'verification_analysis_claim_limit': 8,
    'temperature': 0.3
}

HIGH_ACCURACY_CONFIG = {
    **DEFAULT_EXTRACTION_CONFIG,
    'max_claims_per_article': 6,
    'temperature': 0.1,
    'min_claim_length': 15,
    'pattern_confidence_threshold': 0.7,
    'claim_richness_threshold': 7.0,
    'max_retries': 5
}

# Export configuration templates
EXTRACTION_CONFIGS = {
    'default': DEFAULT_EXTRACTION_CONFIG,
    'quick': QUICK_EXTRACTION_CONFIG,
    'comprehensive': COMPREHENSIVE_EXTRACTION_CONFIG,
    'high_accuracy': HIGH_ACCURACY_CONFIG
}


def get_extraction_config(config_type: str = 'default'):
    """
    Get predefined extraction configuration for different use cases.

    Args:
        config_type: Type of configuration ('default', 'quick', 'comprehensive', 'high_accuracy')

    Returns:
        dict: Configuration dictionary for the specified type

    Example:
        >>> config = get_extraction_config('comprehensive')
        >>> agent = create_claim_extractor(config)
    """
    if config_type not in EXTRACTION_CONFIGS:
        available_types = ', '.join(EXTRACTION_CONFIGS.keys())
        raise ValueError(f"Unknown config type: {config_type}. Available: {available_types}")
    
    return EXTRACTION_CONFIGS[config_type].copy()


# Add config helper to exports
__all__.extend(['get_extraction_config', 'EXTRACTION_CONFIGS'])


# Module initialization and validation
def _initialize_module():
    """Initialize module and validate components for production readiness."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Test component initialization
        logger.info(f"Initializing Claim Extractor Module v{__version__}")
        
        # Test pattern database
        test_pattern_db = ClaimPatternDatabase()
        pattern_stats = test_pattern_db.get_pattern_statistics()
        logger.info(f"Pattern database ready: {pattern_stats['database_composition']['total_claim_patterns']} patterns")
        
        # Test parser
        test_parser = ClaimParser()
        logger.info("Claim parser ready")
        
        # Test validators
        test_input_validator = InputValidator()
        test_output_validator = OutputValidator()
        logger.info("Validators ready")
        
        # Validate configuration templates
        for config_name, config in EXTRACTION_CONFIGS.items():
            if not isinstance(config, dict) or not config.get('model_name'):
                logger.warning(f"Invalid configuration template: {config_name}")
                return False
        
        logger.info(f"All claim extractor components initialized successfully")
        logger.info(f"Available configurations: {list(EXTRACTION_CONFIGS.keys())}")
        
        return True

    except Exception as e:
        logger.error(f"Component initialization failed: {str(e)}")
        return False


# Initialize on import
_module_ready = _initialize_module()

if _module_ready:
    import logging
    logging.getLogger(__name__).info(f"🎯 Claim Extractor v{__version__} ready for production use")
else:
    import logging
    logging.getLogger(__name__).warning("⚠️ Claim Extractor initialization completed with warnings")


# Production readiness check
def check_production_readiness():
    """
    Check if the claim extractor is ready for production use.

    Returns:
        dict: Readiness status with detailed component checks

    Example:
        >>> status = check_production_readiness()
        >>> if status['ready']:
        ...     print("✅ Ready for production")
    """
    checks = {
        'module_initialized': _module_ready,
        'components_available': True,
        'configurations_valid': True,
        'dependencies_satisfied': True
    }
    
    # Check component availability
    try:
        ClaimExtractorAgent()
        ClaimPatternDatabase()
        ClaimParser()
        InputValidator()
        OutputValidator()
    except Exception as e:
        checks['components_available'] = False
        checks['component_error'] = str(e)
    
    # Check configurations
    try:
        for config_type in EXTRACTION_CONFIGS:
            get_extraction_config(config_type)
    except Exception as e:
        checks['configurations_valid'] = False
        checks['config_error'] = str(e)
    
    # Check dependencies
    try:
        import google.generativeai
        import re
        import time
        import logging
    except ImportError as e:
        checks['dependencies_satisfied'] = False
        checks['dependency_error'] = str(e)
    
    checks['ready'] = all([
        checks['module_initialized'],
        checks['components_available'],
        checks['configurations_valid'],
        checks['dependencies_satisfied']
    ])
    
    return checks


# Add production readiness check to exports
__all__.append('check_production_readiness')
