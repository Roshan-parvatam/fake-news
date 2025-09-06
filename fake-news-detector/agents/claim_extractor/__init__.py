# agents/claim_extractor/__init__.py
"""
Claim Extractor Agent Package with Config Integration

Enhanced claim extraction agent with modular architecture and
configuration integration for identifying and extracting verifiable claims
from news articles.

Components:
- ClaimExtractorAgent: Main claim extraction agent with config support
- ClaimPatternDatabase: Pattern-based claim detection system
- ClaimParser: AI output parsing and structuring utilities
"""

from .extractor_agent import ClaimExtractorAgent
from .patterns import ClaimPatternDatabase
from .parsers import ClaimParser

# ✅ OPTIONAL: EXPOSE CONFIG FUNCTIONS FOR CONVENIENCE
try:
    from config import get_model_config, get_prompt_template, get_settings
    
    def get_claim_extractor_config():
        """Convenience function to get claim extractor configuration"""
        return get_model_config('claim_extractor')
    
    def get_claim_extraction_prompts():
        """Convenience function to get claim extraction prompt templates"""
        return {
            'claim_extraction': get_prompt_template('claim_extractor', 'claim_extraction'),
            'verification_analysis': get_prompt_template('claim_extractor', 'verification_analysis'),
            'claim_prioritization': get_prompt_template('claim_extractor', 'claim_prioritization')
        }
    
    def get_system_settings():
        """Convenience function to get system settings"""
        return get_settings()
    
    # Add to exports
    __all__ = [
        'ClaimExtractorAgent',
        'ClaimPatternDatabase',
        'ClaimParser',
        'get_claim_extractor_config',
        'get_claim_extraction_prompts',
        'get_system_settings'
    ]
    
except ImportError:
    # Config not available, use basic exports
    __all__ = [
        'ClaimExtractorAgent',
        'ClaimPatternDatabase',
        'ClaimParser'
    ]

# Version and metadata
__version__ = "2.0.0"
__description__ = "Claim extractor with modular architecture and config integration"
__config_integrated__ = True
