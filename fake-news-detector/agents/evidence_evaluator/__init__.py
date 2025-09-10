# agents/evidence_evaluator/__init__.py

"""
Evidence Evaluator Agent Module

Production-ready evidence evaluation system for assessing news article credibility,
source quality, and logical consistency with specific verification source generation.

Features:
- Comprehensive evidence quality assessment
- Logical fallacy detection and reasoning analysis  
- LLM-powered verification source generation with URL specificity
- Configurable scoring weights and thresholds
- Input/output validation and error handling
- Production-level monitoring and logging
"""

from .evaluator_agent import EvidenceEvaluatorAgent
from .criteria import EvidenceQualityCriteria
from .fallacy_detection import LogicalFallacyDetector
from .prompts import get_prompt_template, PromptValidator
from .validators import (
    validate_evidence_input,
    validate_url_specificity, 
    validate_verification_output
)
from .exceptions import (
    EvidenceEvaluatorError,
    InputValidationError,
    LLMResponseError,
    VerificationSourceError,
    ConfigurationError
)

# Module metadata
__version__ = "2.0.0"
__description__ = "Evidence evaluation agent with LLM-powered verification and URL specificity"
__author__ = "Evidence Evaluator Team"

# Main exports
__all__ = [
    # Core agent
    'EvidenceEvaluatorAgent',
    
    # Assessment components
    'EvidenceQualityCriteria',
    'LogicalFallacyDetector',
    
    # Prompt system
    'get_prompt_template',
    'PromptValidator',
    
    # Validation utilities
    'validate_evidence_input',
    'validate_url_specificity',
    'validate_verification_output',
    
    # Exception classes
    'EvidenceEvaluatorError',
    'InputValidationError', 
    'LLMResponseError',
    'VerificationSourceError',
    'ConfigurationError',
    
    # Module info
    '__version__',
    '__description__'
]

# Convenience functions for quick access
def create_evidence_evaluator(config=None):
    """
    Create a new Evidence Evaluator Agent instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        EvidenceEvaluatorAgent: Configured agent instance
    """
    return EvidenceEvaluatorAgent(config)

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
            'Evidence quality assessment',
            'Logical fallacy detection',
            'LLM-powered verification sources',
            'URL specificity validation',
            'Configurable scoring system',
            'Production-level error handling'
        ],
        'components': {
            'agent': 'EvidenceEvaluatorAgent',
            'criteria': 'EvidenceQualityCriteria', 
            'fallacy_detector': 'LogicalFallacyDetector',
            'validators': 'Input/Output validation',
            'prompts': 'Industry-standard prompt system'
        }
    }

# Module initialization
def _initialize_module():
    """Initialize module logging and configuration."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Evidence Evaluator Module v{__version__} initialized")

# Initialize on import
_initialize_module()
