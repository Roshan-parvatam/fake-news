# agents/context_analyzer/__init__.py
"""
Context Analyzer Agent Package with Config Integration

Enhanced context analysis agent with modular architecture and
configuration integration for detecting bias, manipulation, and framing
in news articles.

Components:
- ContextAnalyzerAgent: Main context analysis agent with config support
- BiasPatternDatabase: Pattern-based bias detection system
- ManipulationDetector: Manipulation and propaganda detection utilities
"""

from .analyzer_agent import ContextAnalyzerAgent
from .bias_patterns import BiasPatternDatabase
from .manipulation_detection import ManipulationDetector

# ✅ OPTIONAL: EXPOSE CONFIG FUNCTIONS FOR CONVENIENCE
try:
    from config import get_model_config, get_prompt_template, get_settings
    
    def get_context_analyzer_config():
        """Convenience function to get context analyzer configuration"""
        return get_model_config('context_analyzer')
    
    def get_context_analysis_prompts():
        """Convenience function to get context analysis prompt templates"""
        return {
            'bias_detection': get_prompt_template('context_analyzer', 'bias_detection'),
            'framing_analysis': get_prompt_template('context_analyzer', 'framing_analysis'),
            'emotional_manipulation': get_prompt_template('context_analyzer', 'emotional_manipulation'),
            'propaganda_detection': get_prompt_template('context_analyzer', 'propaganda_detection')
        }
    
    def get_system_settings():
        """Convenience function to get system settings"""
        return get_settings()
    
    # Add to exports
    __all__ = [
        'ContextAnalyzerAgent',
        'BiasPatternDatabase',
        'ManipulationDetector',
        'get_context_analyzer_config',
        'get_context_analysis_prompts',
        'get_system_settings'
    ]
    
except ImportError:
    # Config not available, use basic exports
    __all__ = [
        'ContextAnalyzerAgent',
        'BiasPatternDatabase',
        'ManipulationDetector'
    ]

# Version and metadata
__version__ = "2.0.0"
__description__ = "Context analyzer with modular architecture and config integration"
__config_integrated__ = True
