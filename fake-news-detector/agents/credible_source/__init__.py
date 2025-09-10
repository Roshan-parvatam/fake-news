# agents/credible_source/__init__.py

"""
Credible Source Agent Package

Production-ready credible source agent with contextual recommendations,
safety filter handling, and comprehensive source database integration.
Provides modular architecture for fact-checking and source verification.

Components:
- CredibleSourceAgent: Main agent with contextual recommendations
- SourceReliabilityDatabase: Comprehensive source database with reliability scoring
- DomainClassifier: Domain classification for targeted source recommendations
- Enhanced prompts, validators, and exception handling
"""

from .source_agent import CredibleSourceAgent
from .source_database import SourceReliabilityDatabase
from .domain_classifier import DomainClassifier
from .prompts import (
    get_source_prompt_template,
    get_domain_guidance,
    SourceRecommendationPrompts,
    VerificationStrategyPrompts,
    SafetyEnhancedPrompts
)
from .validators import (
    InputValidator,
    OutputValidator,
    ValidationResult
)
from .exceptions import (
    CredibleSourceError,
    InputValidationError,
    LLMResponseError,
    SafetyFilterError,
    ContextualRecommendationError,
    SourceDatabaseError,
    DomainClassificationError,
    ConfigurationError,
    handle_credible_source_exception,
    is_recoverable_error
)

# Module metadata
__version__ = "2.0.0"
__description__ = "Credible source agent with safety handling and contextual recommendations"
__author__ = "Credible Source Team"
__license__ = "MIT"

# Main exports
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
    
    # Validation utilities
    'InputValidator',
    'OutputValidator',
    'ValidationResult',
    
    # Exception classes
    'CredibleSourceError',
    'InputValidationError',
    'LLMResponseError',
    'SafetyFilterError',
    'ContextualRecommendationError',
    'SourceDatabaseError',
    'DomainClassificationError',
    'ConfigurationError',
    'handle_credible_source_exception',
    'is_recoverable_error',
    
    # Module info
    '__version__',
    '__description__'
]

# Convenience functions for quick access
def create_credible_source_agent(config=None):
    """
    Create a new Credible Source Agent instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        CredibleSourceAgent: Configured agent instance
    """
    return CredibleSourceAgent(config)

def get_supported_domains():
    """
    Get list of supported domains for classification.
    
    Returns:
        List[str]: Supported domain names
    """
    classifier = DomainClassifier()
    return classifier.get_supported_domains()

def get_source_database_statistics():
    """
    Get source database statistics.
    
    Returns:
        dict: Database statistics and metadata
    """
    database = SourceReliabilityDatabase()
    return database.get_database_statistics()

def validate_source_input(input_data, config=None):
    """
    Quick input validation for source recommendations.
    
    Args:
        input_data: Input data to validate
        config: Optional validation configuration
        
    Returns:
        ValidationResult: Validation results
    """
    validator = InputValidator(config)
    return validator.validate_article_text(input_data.get('text', ''))

def get_module_info():
    """
    Get module information and capabilities.
    
    Returns:
        dict: Module metadata and feature information
    """
    return {
        'version': __version__,
        'description': __description__,
        'features': [
            'Contextual source recommendations',
            'Safety filter handling with institutional fallbacks',
            'Domain-specific source databases',
            'Reliability scoring and assessment',
            'Verification strategy generation',
            'Fact-checking guidance',
            'Production-level error handling'
        ],
        'components': {
            'agent': 'CredibleSourceAgent',
            'database': 'SourceReliabilityDatabase',
            'classifier': 'DomainClassifier',
            'validators': 'Input/Output validation',
            'prompts': 'Safety-enhanced prompt system'
        },
        'domains_supported': get_supported_domains(),
        'key_improvements': [
            'Fixed safety filter blocking issue',
            'Contextual vs generic source recommendations',
            'Institutional fallback sources',
            'Clean architecture without emoji spam',
            'Production-ready error handling'
        ]
    }

# Module initialization
def _initialize_module():
    """Initialize module logging and check component availability."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Credible Source Agent Module v{__version__} initialized")
    
    # Validate core components are available
    try:
        test_agent = CredibleSourceAgent()
        test_database = SourceReliabilityDatabase()
        test_classifier = DomainClassifier()
        logger.info("All core components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Component initialization failed: {str(e)}")
        return False

# Initialize on import
_module_ready = _initialize_module()

# Module constants
SAFETY_FALLBACK_ENABLED = True
CONTEXTUAL_RECOMMENDATIONS_ENABLED = True
DOMAIN_CLASSIFICATION_ENABLED = True
DATABASE_VALIDATION_ENABLED = True

# Quick access to common configurations
DEFAULT_CONFIG = {
    'model_name': 'gemini-1.5-pro',
    'temperature': 0.3,
    'max_sources_per_recommendation': 8,
    'min_reliability_score': 6.0,
    'enable_safety_fallbacks': True,
    'enable_contextual_recommendations': True,
    'confidence_threshold': 0.7
}

HEALTH_DOMAIN_CONFIG = {
    **DEFAULT_CONFIG,
    'min_reliability_score': 8.0,
    'max_sources_per_recommendation': 10,
    'preferred_source_types': ['government', 'academic', 'medical_institution']
}

POLITICS_DOMAIN_CONFIG = {
    **DEFAULT_CONFIG,
    'min_reliability_score': 7.5,
    'max_sources_per_recommendation': 12,
    'preferred_source_types': ['government', 'think_tank', 'fact_checker']
}

SCIENCE_DOMAIN_CONFIG = {
    **DEFAULT_CONFIG,
    'min_reliability_score': 8.5,
    'max_sources_per_recommendation': 8,
    'preferred_source_types': ['academic', 'professional_organization']
}

# Export configurations
DOMAIN_CONFIGS = {
    'health': HEALTH_DOMAIN_CONFIG,
    'politics': POLITICS_DOMAIN_CONFIG, 
    'science': SCIENCE_DOMAIN_CONFIG,
    'default': DEFAULT_CONFIG
}

def get_domain_config(domain: str):
    """Get domain-specific configuration."""
    return DOMAIN_CONFIGS.get(domain, DEFAULT_CONFIG)

# Add domain config helper to exports
__all__.extend([
    'create_credible_source_agent',
    'get_supported_domains', 
    'get_source_database_statistics',
    'validate_source_input',
    'get_module_info',
    'get_domain_config',
    'DEFAULT_CONFIG',
    'DOMAIN_CONFIGS'
])
