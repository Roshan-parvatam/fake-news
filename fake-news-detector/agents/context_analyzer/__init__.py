# agents/context_analyzer/__init__.py

"""
Context Analyzer Agent Module

Production-ready context analysis system for detecting bias, manipulation,
framing techniques, and propaganda in news articles with LLM-driven consistent scoring.

Features:
- Comprehensive bias detection (political, emotional, selection, linguistic)
- Manipulation and propaganda technique identification
- Logical fallacy detection and reasoning quality analysis
- LLM-powered analysis with consistent numerical scoring
- Configurable detection thresholds and scoring weights
- Input/output validation and error handling
- Production-level monitoring and performance tracking
"""

from .analyzer_agent import ContextAnalyzerAgent
from .bias_patterns import BiasPatternDatabase
from .manipulation_detection import ManipulationDetector
from .prompts import get_context_prompt_template, validate_context_analysis_output
from .validators import (
    validate_context_input,
    validate_bias_analysis,
    validate_context_output
)
from .exceptions import (
    ContextAnalyzerError,
    InputValidationError,
    LLMResponseError,
    BiasDetectionError,
    ManipulationDetectionError,
    ScoringConsistencyError,
    ConfigurationError
)

# Module metadata
__version__ = "2.0.0"
__description__ = "Context analysis agent with bias detection and LLM-driven consistent scoring"
__author__ = "Context Analyzer Team"

# Main exports
__all__ = [
    # Core agent
    'ContextAnalyzerAgent',
    
    # Analysis components
    'BiasPatternDatabase',
    'ManipulationDetector',
    
    # Prompt system
    'get_context_prompt_template',
    'validate_context_analysis_output',
    
    # Validation utilities
    'validate_context_input',
    'validate_bias_analysis',
    'validate_context_output',
    
    # Exception classes
    'ContextAnalyzerError',
    'InputValidationError',
    'LLMResponseError',
    'BiasDetectionError',
    'ManipulationDetectionError',
    'ScoringConsistencyError',
    'ConfigurationError',
    
    # Module info
    '__version__',
    '__description__'
]

# Convenience functions for quick access
def create_context_analyzer(config=None):
    """
    Create a new Context Analyzer Agent instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ContextAnalyzerAgent: Configured agent instance
    """
    return ContextAnalyzerAgent(config)

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
            'Political and emotional bias detection',
            'Manipulation and propaganda analysis',
            'Logical fallacy identification',
            'LLM-driven consistent scoring',
            'Framing and narrative analysis',
            'Production-level error handling'
        ],
        'components': {
            'agent': 'ContextAnalyzerAgent',
            'bias_patterns': 'BiasPatternDatabase',
            'manipulation_detector': 'ManipulationDetector',
            'validators': 'Input/Output validation',
            'prompts': 'Industry-standard prompt system'
        },
        'key_improvements': [
            'Fixed scoring consistency issue',
            'Removed emoji spam and verbose logging',
            'Added comprehensive validation system',
            'Industry-standard prompt engineering',
            'Production-ready error handling'
        ]
    }

# Module initialization
def _initialize_module():
    """Initialize module logging and configuration."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Context Analyzer Module v{__version__} initialized")

# Initialize on import
_initialize_module()
