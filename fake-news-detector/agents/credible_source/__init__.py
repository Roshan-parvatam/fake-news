# agents/credible_source/__init__.py
"""
Credible Source Agent Package with Config Integration

Enhanced credible source agent with modular architecture and
configuration integration for providing source recommendations and
credibility assessments for fact-checking and verification.

Components:
- CredibleSourceAgent: Main source recommendation agent with config support
- SourceReliabilityDatabase: Comprehensive source database with reliability scoring
- DomainClassifier: Domain classification for domain-specific source recommendations
"""

from .source_agent import CredibleSourceAgent
from .source_database import SourceReliabilityDatabase
from .domain_classifier import DomainClassifier

# ✅ OPTIONAL: EXPOSE CONFIG FUNCTIONS FOR CONVENIENCE
try:
    from config import get_model_config, get_prompt_template, get_settings
    
    def get_credible_source_config():
        """Convenience function to get credible source agent configuration"""
        return get_model_config('credible_source')
    
    def get_source_recommendation_prompts():
        """Convenience function to get source recommendation prompt templates"""
        return {
            'source_recommendations': get_prompt_template('credible_source', 'source_recommendations'),
            'reliability_assessment': get_prompt_template('credible_source', 'reliability_assessment'),
            'verification_strategy': get_prompt_template('credible_source', 'verification_strategy'),
            'fact_check_guidance': get_prompt_template('credible_source', 'fact_check_guidance')
        }
    
    def get_system_settings():
        """Convenience function to get system settings"""
        return get_settings()
    
    # Add to exports
    __all__ = [
        'CredibleSourceAgent',
        'SourceReliabilityDatabase',
        'DomainClassifier',
        'get_credible_source_config',
        'get_source_recommendation_prompts',
        'get_system_settings'
    ]
    
except ImportError:
    # Config not available, use basic exports
    __all__ = [
        'CredibleSourceAgent',
        'SourceReliabilityDatabase',
        'DomainClassifier'
    ]

# Version and metadata
__version__ = "2.0.0"
__description__ = "Credible source agent with modular architecture and config integration"
__config_integrated__ = True
