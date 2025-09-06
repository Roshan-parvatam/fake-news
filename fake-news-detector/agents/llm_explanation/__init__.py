# agents/llm_explanation/__init__.py
"""
LLM Explanation Agent Package with Config Integration

Enhanced LLM explanation agent with modular architecture and
configuration integration for generating human-readable explanations
of fake news detection results.

Components:
- LLMExplanationAgent: Main explanation generation agent with config support
"""

from .explanation_agent import LLMExplanationAgent

# ✅ OPTIONAL: EXPOSE CONFIG FUNCTIONS FOR CONVENIENCE
try:
    from config import get_model_config, get_prompt_template, get_settings
    
    def get_llm_config():
        """Convenience function to get LLM explanation agent configuration"""
        return get_model_config('llm_explanation')
    
    def get_explanation_prompts():
        """Convenience function to get explanation prompt templates"""
        return {
            'main_explanation': get_prompt_template('llm_explanation', 'main_explanation'),
            'detailed_analysis': get_prompt_template('llm_explanation', 'detailed_analysis'),
            'confidence_analysis': get_prompt_template('llm_explanation', 'confidence_analysis')
        }
    
    def get_system_settings():
        """Convenience function to get system settings"""
        return get_settings()
    
    # Add to exports
    __all__ = [
        'LLMExplanationAgent',
        'get_llm_config',
        'get_explanation_prompts',
        'get_system_settings'
    ]
    
except ImportError:
    # Config not available, use basic exports
    __all__ = [
        'LLMExplanationAgent'
    ]

# Version and metadata
__version__ = "2.0.0"
__description__ = "LLM explanation agent with modular architecture and config integration"
__config_integrated__ = True
