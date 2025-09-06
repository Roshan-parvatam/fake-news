# agents/evidence_evaluator/__init__.py
"""
Evidence Evaluator Agent Package with Config Integration

Enhanced evidence evaluation agent with modular architecture and
configuration integration for assessing evidence quality, source credibility,
and logical consistency in news articles.

Components:
- EvidenceEvaluatorAgent: Main evidence evaluation agent with config support
- EvidenceQualityCriteria: Systematic evidence quality assessment system
- LogicalFallacyDetector: Logical fallacy and reasoning quality detection
"""

from .evaluator_agent import EvidenceEvaluatorAgent
from .criteria import EvidenceQualityCriteria
from .fallacy_detection import LogicalFallacyDetector

# ✅ OPTIONAL: EXPOSE CONFIG FUNCTIONS FOR CONVENIENCE
try:
    from config import get_model_config, get_prompt_template, get_settings
    
    def get_evidence_evaluator_config():
        """Convenience function to get evidence evaluator configuration"""
        return get_model_config('evidence_evaluator')
    
    def get_evidence_evaluation_prompts():
        """Convenience function to get evidence evaluation prompt templates"""
        return {
            'evidence_evaluation': get_prompt_template('evidence_evaluator', 'evidence_evaluation'),
            'source_quality': get_prompt_template('evidence_evaluator', 'source_quality'),
            'logical_consistency': get_prompt_template('evidence_evaluator', 'logical_consistency'),
            'evidence_gaps': get_prompt_template('evidence_evaluator', 'evidence_gaps')
        }
    
    def get_system_settings():
        """Convenience function to get system settings"""
        return get_settings()
    
    # Add to exports
    __all__ = [
        'EvidenceEvaluatorAgent',
        'EvidenceQualityCriteria',
        'LogicalFallacyDetector',
        'get_evidence_evaluator_config',
        'get_evidence_evaluation_prompts',
        'get_system_settings'
    ]
    
except ImportError:
    # Config not available, use basic exports
    __all__ = [
        'EvidenceEvaluatorAgent',
        'EvidenceQualityCriteria',
        'LogicalFallacyDetector'
    ]

# Version and metadata
__version__ = "2.0.0"
__description__ = "Evidence evaluator with modular architecture and config integration"
__config_integrated__ = True
