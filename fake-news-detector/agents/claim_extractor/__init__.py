# agents/claim_extractor/__init__.py

"""
Claim Extractor Agent Package

Production-ready claim extraction agent with modular architecture.
Provides comprehensive claim identification, verification analysis,
and prioritization with enhanced error handling and performance tracking.

Components:
- ClaimExtractorAgent: Main extraction agent with AI integration
- ClaimPatternDatabase: Pattern-based claim detection and analysis
- ClaimParser: Structured claim parsing with multiple fallback methods
- Enhanced prompt templates, validators, and exception handling
"""

from .extractor_agent import ClaimExtractorAgent
from .patterns import ClaimPatternDatabase
from .parsers import ClaimParser
from .prompts import (
    ClaimExtractionPrompts,
    ClaimCategorizationPrompts, 
    StructuredOutputPrompts,
    get_claim_prompt_template
)
from .validators import (
    InputValidator,
    OutputValidator,
    ValidationResult
)
from .exceptions import (
    ClaimExtractorError,
    InputValidationError,
    LLMResponseError,
    ClaimParsingError,
    ClaimExtractionError,
    VerificationAnalysisError,
    PrioritizationError,
    PatternAnalysisError,
    ConfigurationError,
    handle_claim_extractor_exception,
    is_recoverable_error
)

# Module metadata
__version__ = "2.0.0"
__description__ = "Modular claim extraction agent with AI integration and pattern analysis"
__author__ = "Claim Extractor Team"
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
    
    # Validation utilities
    'InputValidator',
    'OutputValidator',
    'ValidationResult',
    
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
    'handle_claim_extractor_exception',
    'is_recoverable_error',
    
    # Module info
    '__version__',
    '__description__'
]

# Convenience functions for quick access
def create_claim_extractor(config=None):
    """
    Create a new Claim Extractor Agent instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ClaimExtractorAgent: Configured agent instance
    """
    return ClaimExtractorAgent(config)

def extract_claims_quick(article_text: str, max_claims: int = 5):
    """
    Quick claim extraction using pattern-based detection only.
    
    Args:
        article_text: Article content to analyze
        max_claims: Maximum number of claims to extract
        
    Returns:
        List of potential claim texts
    """
    pattern_db = ClaimPatternDatabase()
    return pattern_db.extract_potential_claims(article_text, max_claims)

def validate_article_input(article_text: str, min_length: int = 50):
    """
    Quick validation of article text input.
    
    Args:
        article_text: Article text to validate
        min_length: Minimum required length
        
    Returns:
        ValidationResult: Validation results
    """
    validator = InputValidator({'min_text_length': min_length})
    return validator.validate_article_text(article_text)

def get_claim_extraction_info():
    """
    Get information about claim extraction capabilities.
    
    Returns:
        dict: Module capabilities and configuration options
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
            'Configuration-driven behavior'
        ],
        'claim_types_supported': [
            'Statistical', 'Event', 'Attribution', 'Research', 
            'Policy', 'Causal', 'Other'
        ],
        'components': {
            'main_agent': 'ClaimExtractorAgent',
            'pattern_database': 'ClaimPatternDatabase', 
            'claim_parser': 'ClaimParser',
            'prompt_system': 'Enhanced prompt templates',
            'validation': 'Input/output validators',
            'error_handling': 'Comprehensive exception system'
        },
        'key_improvements': [
            'Removed emoji spam and verbose logging',
            'Enhanced pattern database with 90+ patterns',
            'Multiple parsing fallback strategies',
            'Production-ready error handling',
            'Comprehensive performance tracking',
            'Clean, modular architecture'
        ]
    }

# Add convenience functions to exports
__all__.extend([
    'create_claim_extractor',
    'extract_claims_quick',
    'validate_article_input',
    'get_claim_extraction_info'
])

# Module initialization
def _initialize_module():
    """Initialize module and validate components."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Claim Extractor Module v{__version__} initialized")
    
    try:
        # Test component initialization
        test_extractor = ClaimExtractorAgent()
        test_pattern_db = ClaimPatternDatabase()
        test_parser = ClaimParser()
        logger.info("All claim extractor components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Component initialization failed: {str(e)}")
        return False

# Initialize on import
_module_ready = _initialize_module()

# Configuration templates
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
    'rate_limit_seconds': 1.0
}

QUICK_EXTRACTION_CONFIG = {
    **DEFAULT_EXTRACTION_CONFIG,
    'max_claims_per_article': 5,
    'enable_verification_analysis': False,
    'enable_claim_prioritization': False,
    'enable_pattern_preprocessing': True
}

COMPREHENSIVE_EXTRACTION_CONFIG = {
    **DEFAULT_EXTRACTION_CONFIG,
    'max_claims_per_article': 12,
    'enable_verification_analysis': True,
    'enable_claim_prioritization': True,
    'enable_pattern_preprocessing': True,
    'verification_analysis_claim_limit': 8
}

# Export configuration templates
EXTRACTION_CONFIGS = {
    'default': DEFAULT_EXTRACTION_CONFIG,
    'quick': QUICK_EXTRACTION_CONFIG,
    'comprehensive': COMPREHENSIVE_EXTRACTION_CONFIG
}

def get_extraction_config(config_type: str = 'default'):
    """Get predefined extraction configuration."""
    return EXTRACTION_CONFIGS.get(config_type, DEFAULT_EXTRACTION_CONFIG)

# Add config helper to exports
__all__.extend(['get_extraction_config', 'EXTRACTION_CONFIGS'])
